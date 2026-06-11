"""
history_loader.py
=================

Loads conversation history for a given session_id from S3 and returns
records in a schema that is BYTE-COMPATIBLE with the iField Mongo
`/api/userSession/history/{sessionId}` response envelope.

Source-of-truth priority (highest to lowest):
  1. S3 session-level manifest at
       s3://{bucket}/rag-agent/conversation_sessions/{session_id}.json
     (this file already grows per-turn; see session_tracker._push)
  2. S3 per-turn archival files at
       s3://{bucket}/sessions/{user_id}/{project_id}/{agent_id}/*.json
     filtered to kind=="turn" AND session_id matches, sorted by captured_at
  3. iField Mongo fallback (only if both S3 sources fail) — preserves
     current behavior so we are never worse than today.

Critical invariants:
  - Existing write path is UNCHANGED.
  - Mongo continues to receive POSTs from push_session_turn unchanged.
  - This module is READ-ONLY against S3 + Mongo.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import requests
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-driven; safe defaults)
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_SESSION_BUCKET", "agentic-ai-production")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
SESSION_MANIFEST_PREFIX = os.getenv(
    "S3_SESSION_MANIFEST_PREFIX", "rag-agent/conversation_sessions"
)
PER_TURN_PREFIX_TEMPLATE = os.getenv(
    "S3_PER_TURN_PREFIX_TEMPLATE", "sessions/{user_id}/{project_id}/{agent_id}"
)
IFIELD_MONGO_BASE = os.getenv(
    "IFIELD_MONGO_BASE", "https://mongo.ifieldsmart.com/api"
)
PRESIGN_TTL_SECONDS = int(os.getenv("PRESIGN_TTL_SECONDS", "3600"))
FALLBACK_TO_MONGO = os.getenv("HISTORY_FALLBACK_TO_MONGO", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Lazy clients (so tests / cold imports don't require AWS creds)
# ---------------------------------------------------------------------------
_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )
    return _s3_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_session_history(
    session_id: str,
    user_id: int,
    project_id: int,
    agent_id: str = "drawing-agent",
    include_answer: bool = True,
    presign: bool = True,
) -> Dict[str, Any]:
    """
    Returns a Mongo-API-compatible envelope:
      {"success": True, "message": "...", "data": [ {turn...}, ... ]}

    Never raises on individual loader failures — falls back to next source.
    """
    started = time.time()
    source_used = None
    turns: List[Dict[str, Any]] = []

    # Source 1 (preferred): manifest gives us the canonical ORDERED turn_id list,
    # and we resolve each turn_id to its rich per-turn file (full schema).
    try:
        turns = _load_via_manifest_directed_per_turn_fetch(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
        )
        if turns:
            source_used = "s3_manifest_directed"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[history-loader] manifest-directed read failed: %s", exc)

    # Source 2: pure-manifest message grouping (when per-turn files missing)
    if not turns:
        try:
            turns = _load_from_session_manifest(session_id)
            if turns:
                source_used = "s3_manifest_grouped"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[history-loader] manifest grouped read failed: %s", exc)

    # Source 3: brute-force per-turn file listing (no manifest at all)
    if not turns:
        try:
            turns = _load_from_per_turn_files(
                session_id=session_id,
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
            )
            if turns:
                source_used = "s3_per_turn"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[history-loader] per-turn listing failed: %s", exc)

    # Source 3: iField Mongo fallback (last resort, preserves today's behavior)
    if not turns and FALLBACK_TO_MONGO:
        try:
            turns = _load_from_ifield_mongo(session_id)
            if turns:
                source_used = "ifield_mongo_fallback"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[history-loader] mongo fallback failed: %s", exc)

    # Normalize -> Mongo-compatible schema
    normalized = [
        _to_mongo_compat_item(
            t,
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            include_answer=include_answer,
            presign=presign,
        )
        for t in turns
    ]

    # Sort chronologically (defensive)
    normalized.sort(key=lambda r: r.get("capturedAt") or "")

    elapsed_ms = int((time.time() - started) * 1000)
    logger.info(
        "[history-loader] session=%s source=%s turns=%d elapsed_ms=%d",
        session_id,
        source_used or "none",
        len(normalized),
        elapsed_ms,
    )

    return {
        "success": True,
        "message": "History fetched successfully"
        if normalized
        else "No history found",
        "data": normalized,
        # Diagnostic-only fields (UI ignores; helpful during rollout)
        "sessionId": session_id,
        "source": source_used or "none",
    }


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------
def _load_from_session_manifest(session_id: str) -> List[Dict[str, Any]]:
    """Read s3://{bucket}/{SESSION_MANIFEST_PREFIX}/{session_id}.json and
    return a list of turn dicts.

    The manifest's real schema is:
        {session_id, created_at, last_accessed, messages: [...], context, ...}
    where each message is {role, content, timestamp, tokens, metadata: {turn_id, ...}}.
    We group consecutive user+assistant messages by turn_id.

    Falls back to checking other field names in case the schema evolves.
    """
    body = _read_manifest_body(session_id)
    if body is None:
        return []

    # Allow future schemas with a pre-grouped 'turns'/'history' array
    for candidate in ("turns", "history", "conversation"):
        v = body.get(candidate)
        if isinstance(v, list) and v and isinstance(v[0], dict) and (
            v[0].get("turn_id") or v[0].get("turnId")
        ):
            return v

    # Current real schema: group flat messages by turn_id
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return []
    return _group_messages_into_turns(messages)


def _load_via_manifest_directed_per_turn_fetch(
    session_id: str,
    user_id: int,
    project_id: int,
    agent_id: str,
) -> List[Dict[str, Any]]:
    """Use the manifest to learn the ORDERED list of turn_ids, then fetch
    each per-turn file s3://.../sessions/{u}/{p}/{a}/{turn_id}.json which
    has the full Mongo-shape payload (follow-ups + source_documents).

    If a per-turn file is missing, falls back to the manifest's grouped
    message pair for that turn_id so we still return a useful record.
    """
    body = _read_manifest_body(session_id)
    if body is None:
        return []

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return []

    # Build an ordered, deduped list of turn_ids and a fallback map.
    grouped = _group_messages_into_turns(messages)
    if not grouped:
        return []

    prefix = PER_TURN_PREFIX_TEMPLATE.format(
        user_id=user_id, project_id=project_id, agent_id=agent_id
    ).rstrip("/")

    enriched: List[Dict[str, Any]] = []
    for turn in grouped:
        tid = turn.get("turn_id") or turn.get("turnId")
        if not tid:
            continue
        key = f"{prefix}/{tid}.json"
        per_turn = _read_per_turn_file(key)
        if per_turn:
            # Per-turn file is authoritative; carry session_id forward
            per_turn.setdefault("session_id", session_id)
            enriched.append(per_turn)
        else:
            # Per-turn file missing -> use manifest-grouped fallback
            turn.setdefault("session_id", session_id)
            enriched.append(turn)
    return enriched


# --- helpers --------------------------------------------------------------
def _read_manifest_body(session_id: str) -> Optional[Dict[str, Any]]:
    key = f"{SESSION_MANIFEST_PREFIX}/{session_id}.json"
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found(exc):
            return None
        logger.warning("[history-loader] manifest fetch error: %s", exc)
        return None
    try:
        return json.loads(obj["Body"].read())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[history-loader] manifest parse error: %s", exc)
        return None


def _read_per_turn_file(key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
    except Exception:
        return None
    try:
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def _group_messages_into_turns(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group flat conversation messages by their metadata.turn_id.

    A turn = (user message, assistant message) sharing one turn_id.
    Returns turn dicts in first-seen order.
    """
    if not messages:
        return []

    by_turn: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        tid = meta.get("turn_id") or meta.get("turnId")
        if not tid:
            continue
        if tid not in by_turn:
            by_turn[tid] = {"_user": None, "_asst": None}
            order.append(tid)
        role = (m.get("role") or "").lower()
        if role == "user":
            by_turn[tid]["_user"] = m
        elif role in ("assistant", "ai", "system"):
            by_turn[tid]["_asst"] = m

    turns: List[Dict[str, Any]] = []
    for tid in order:
        pair = by_turn[tid]
        u = pair.get("_user") or {}
        a = pair.get("_asst") or {}
        u_meta = u.get("metadata") or {}
        a_meta = a.get("metadata") or {}
        captured = u.get("timestamp") or a.get("timestamp")
        question = u.get("content") or ""
        answer = a.get("content") or ""
        turns.append(
            {
                "turn_id": tid,
                "question": question,
                "title": question[:80] if question else "",
                "answer": answer,
                "captured_at": captured,
                "follow_up_questions": a_meta.get("follow_up_questions")
                or u_meta.get("follow_up_questions")
                or [],
                "source_documents": a_meta.get("source_documents")
                or u_meta.get("source_documents")
                or [],
                "project_id": u_meta.get("project_id") or a_meta.get("project_id"),
                "session_id": u_meta.get("session_id") or a_meta.get("session_id"),
            }
        )
    return turns


def _load_from_per_turn_files(
    session_id: str,
    user_id: int,
    project_id: int,
    agent_id: str,
) -> List[Dict[str, Any]]:
    """List per-turn files under sessions/{user}/{project}/{agent}/ and filter."""
    prefix = PER_TURN_PREFIX_TEMPLATE.format(
        user_id=user_id, project_id=project_id, agent_id=agent_id
    ).rstrip("/") + "/"

    paginator = _s3().get_paginator("list_objects_v2")
    matches: List[Dict[str, Any]] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for entry in page.get("Contents") or []:
            key = entry["Key"]
            if not key.endswith(".json"):
                continue
            try:
                obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
                body = json.loads(obj["Body"].read())
            except Exception as exc:  # noqa: BLE001
                logger.debug("[history-loader] skip %s: %s", key, exc)
                continue
            if body.get("kind") != "turn":
                continue
            if body.get("session_id") != session_id:
                continue
            matches.append(body)

    matches.sort(key=lambda b: b.get("captured_at") or "")
    return matches


def _load_from_ifield_mongo(session_id: str) -> List[Dict[str, Any]]:
    """Proxy to iField Mongo's existing endpoint (last-resort fallback)."""
    url = f"{IFIELD_MONGO_BASE}/userSession/history/{session_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data") or []
    # Mongo items are already mostly Mongo-schema-compatible; we re-normalize
    # below to guarantee identical shape across all three sources.
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Schema mapper -> exact iField Mongo /history shape
# ---------------------------------------------------------------------------
def _to_mongo_compat_item(
    src: Dict[str, Any],
    *,
    session_id: str,
    user_id: int,
    project_id: int,
    agent_id: str,
    include_answer: bool,
    presign: bool,
) -> Dict[str, Any]:
    """Map any source dict (S3 turn, S3 manifest turn, Mongo item) to one
    canonical schema-identical record. Field names mirror iField Mongo."""

    # Pull turn_id (allow either snake or camel)
    turn_id = (
        src.get("turnId")
        or src.get("turn_id")
        or src.get("turnID")
        or src.get("_id")
        or ""
    )

    # Pull question/title/answer with multiple aliases
    question = (
        src.get("question")
        or src.get("query")
        or src.get("user_query")
        or ""
    )
    title = src.get("title") or question[:80]
    answer = (
        src.get("answer")
        or src.get("final_answer")
        or src.get("response")
        or ""
    )

    follow_up = (
        src.get("followUpQuestions")
        or src.get("follow_up_questions")
        or src.get("followups")
        or []
    )
    source_docs = (
        src.get("sourceDocuments")
        or src.get("source_documents")
        or src.get("sources")
        or []
    )

    is_active = src.get("isActive")
    if is_active is None:
        is_active = src.get("is_active", True)

    captured_at_iso = _iso_z(
        src.get("capturedAt")
        or src.get("captured_at")
        or src.get("createdAt")
        or src.get("created_at")
    )

    # Real iField Mongo stores `s3BucketPath` as a FULL presigned URL (not a
    # bare key). Mint one ourselves when missing so the UI's existing fetch
    # logic works unchanged.
    s3_bucket_path = src.get("s3BucketPath") or src.get("s3_bucket_path") or ""
    if not s3_bucket_path or not s3_bucket_path.startswith("http"):
        # Treat anything non-URL-shaped as a key; presign it.
        key = s3_bucket_path or (
            f"sessions/{user_id}/{project_id}/{agent_id}/{turn_id}.json"
            if turn_id else ""
        )
        if presign and key:
            s3_bucket_path = _presign(key) or s3_bucket_path or key
        else:
            s3_bucket_path = key

    # Deterministic ObjectId-shape _id from turn_id so the UI always sees
    # a stable id and re-renders idempotently.
    _id = src.get("_id") or _deterministic_oid(turn_id or session_id)

    # STRICT iField Mongo schema parity — exact 14 fields in the exact
    # naming/casing they use (snake_case for the two arrays, camelCase for
    # the rest, plus the Mongoose version key __v).
    item: Dict[str, Any] = {
        "__v": int(src.get("__v") or 0),
        "_id": _id,
        "agent": agent_id,
        "createdAt": captured_at_iso,
        "follow_up_questions": list(follow_up) if isinstance(follow_up, list) else [],
        "isActive": bool(is_active),
        "projectId": int(project_id),
        "s3BucketPath": s3_bucket_path,
        "sessionId": session_id,
        "source_documents": list(source_docs) if isinstance(source_docs, list) else [],
        "title": title,
        "turnId": turn_id,
        "updatedAt": captured_at_iso,
        "userId": int(user_id),
    }

    # Additive, opt-in: include rendered answer text so clients can skip
    # the S3 fetch when displaying. UI's JSON parser ignores unknown keys.
    if include_answer:
        item["answer"] = answer
        item["question"] = question  # explicit, separate from `title`

    return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iso_z(value: Optional[str]) -> str:
    """Coerce timestamps to ISO 8601 with millisecond precision and 'Z' suffix."""
    if not value:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            v = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        ms = int(dt.microsecond / 1000)
        return dt.strftime(f"%Y-%m-%dT%H:%M:%S.{ms:03d}Z")
    except Exception:  # noqa: BLE001
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _deterministic_oid(seed: str) -> str:
    """Return a 24-hex-char string shaped like a Mongo ObjectId."""
    if not seed:
        seed = "empty"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:24]


def _presign(key: str) -> Optional[str]:
    try:
        return _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=PRESIGN_TTL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[history-loader] presign failed for %s: %s", key, exc)
        return None


def _is_not_found(exc: Exception) -> bool:
    code = getattr(getattr(exc, "response", {}), "get", lambda *_: None)("Error") or {}
    if isinstance(code, dict):
        return code.get("Code") in {"NoSuchKey", "404", "NotFound"}
    return False
