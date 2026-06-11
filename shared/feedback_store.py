"""Feedback storage layer for the thumbs-up/down feature (Phase 1).

Two surfaces:
  1. Mongo collection ``agent_feedback`` — immutable per-event audit log.
     Vote changes APPEND a new doc; we never overwrite. Each doc carries
     `previous_feedback_id` linking to its predecessor in the chain.

  2. S3 session manifest — under ``feedback.{turn_id}``, a rollup carrying
     the current vote + history pointer so the UI can render the thumb
     state without round-tripping Mongo.

Schema (Mongo doc):

    {
      _id:                ObjectId,
      feedback_id:        "fb_<uuid>"           # public stable id
      session_id:         str,
      turn_id:            str,
      user_id:            int,
      project_id:         int,
      agent_id:           str,                  # "drawing-agent" etc.
      vote:               "up" | "down",
      categories:         [str],                # optional, meaningful for "down"
      note:               str | None,           # ≤1000 chars, plain text in v1
      note_is_pii:        bool,                 # tag for Phase 2 encryption
      user_query:         str | None,           # snapshot for audit
      agent_response:     str | None,           # snapshot for audit
      created_at:         datetime,
      previous_feedback_id: str | None,         # set when this is a vote change
      is_vote_change:     bool,
      schema_version:     "feedback.v1"
    }

Indexes (created lazily on first write):
  - feedback_id (unique)
  - (session_id, turn_id, created_at desc)
  - (user_id, agent_id, created_at desc)
  - (project_id, created_at desc)

No deletes from this layer; deletion is a separate D3 endpoint we'll
build after Phase 1.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger("agentic_rag.feedback_store")


# -------- Constants -------------------------------------------------

MONGO_COLLECTION = "agent_feedback"
SCHEMA_VERSION   = "feedback.v1"
NOTE_MAX_CHARS   = 1000
VALID_VOTES      = ("up", "down")
VALID_CATEGORIES = (
    "wrong_facts",
    "poor_format",
    "incomplete",
    "irrelevant",
    "tone",
    "other",
)


# -------- Connection helpers ----------------------------------------

_mongo_client: Optional[MongoClient] = None
_s3_client = None


def _get_mongo_db() -> Database:
    """Return iField Mongo database, opening a connection lazily."""
    global _mongo_client
    if _mongo_client is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI not set")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _mongo_client.iField


def _get_collection() -> Collection:
    return _get_mongo_db()[MONGO_COLLECTION]


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


# -------- Indexes ---------------------------------------------------

_indexes_ensured = False


def _ensure_indexes() -> None:
    """Idempotent index creation. Called once per process on first write."""
    global _indexes_ensured
    if _indexes_ensured:
        return
    coll = _get_collection()
    coll.create_index("feedback_id", unique=True, name="feedback_id_uniq")
    coll.create_index(
        [("session_id", ASCENDING), ("turn_id", ASCENDING), ("created_at", DESCENDING)],
        name="session_turn_created",
    )
    coll.create_index(
        [("user_id", ASCENDING), ("agent_id", ASCENDING), ("created_at", DESCENDING)],
        name="user_agent_created",
    )
    coll.create_index(
        [("project_id", ASCENDING), ("created_at", DESCENDING)],
        name="project_created",
    )
    _indexes_ensured = True
    logger.info("agent_feedback indexes ensured")


# -------- Validation ------------------------------------------------

def validate_payload(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Light validation. Returns (ok, error_message)."""
    if not isinstance(payload, dict):
        return False, "payload must be an object"

    for required in ("user_id", "project_id", "agent_id", "vote"):
        if payload.get(required) in (None, ""):
            return False, f"missing required field: {required}"

    vote = payload.get("vote")
    if vote not in VALID_VOTES:
        return False, f"vote must be one of {VALID_VOTES}, got {vote!r}"

    cats = payload.get("categories") or []
    if not isinstance(cats, list):
        return False, "categories must be a list of strings"
    for c in cats:
        if c not in VALID_CATEGORIES:
            return False, f"unknown category: {c!r}; valid={VALID_CATEGORIES}"

    note = payload.get("note")
    if note is not None:
        if not isinstance(note, str):
            return False, "note must be a string"
        if len(note) > NOTE_MAX_CHARS:
            return False, f"note exceeds {NOTE_MAX_CHARS} chars (got {len(note)})"

    # v3.4.4 (2026-05-26) — note is required for thumbs-down votes.
    # Thumbs-up notes remain optional (existing behaviour).
    if vote == "down":
        note_text = (note or "").strip() if isinstance(note, str) else ""
        if not note_text:
            return False, "note is required for thumbs-down votes"

    try:
        int(payload["user_id"])
        int(payload["project_id"])
    except (TypeError, ValueError):
        return False, "user_id and project_id must be integers"

    return True, None


# -------- Public API ------------------------------------------------

def get_latest_feedback(session_id: str, turn_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent feedback doc for a turn, or None if no feedback yet."""
    _ensure_indexes()
    return _get_collection().find_one(
        {"session_id": session_id, "turn_id": turn_id},
        sort=[("created_at", DESCENDING)],
        projection={"_id": 0},
    )


def get_feedback_history(session_id: str, turn_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the full vote chain for a turn, newest first."""
    _ensure_indexes()
    cursor = _get_collection().find(
        {"session_id": session_id, "turn_id": turn_id},
        projection={"_id": 0},
        sort=[("created_at", DESCENDING)],
    ).limit(limit)
    return list(cursor)


def get_session_feedback(session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Return all feedback for a session (newest first)."""
    _ensure_indexes()
    cursor = _get_collection().find(
        {"session_id": session_id},
        projection={"_id": 0},
        sort=[("created_at", DESCENDING)],
    ).limit(limit)
    return list(cursor)


def submit_feedback(
    *,
    session_id: str,
    turn_id: str,
    user_id: int,
    project_id: int,
    agent_id: str,
    vote: str,
    categories: Optional[List[str]] = None,
    note: Optional[str] = None,
    user_query: Optional[str] = None,
    agent_response: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    feedback_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a feedback event.

    Returns the inserted doc (with feedback_id, previous_feedback_id,
    is_vote_change). Caller is responsible for updating the S3 manifest
    rollup — see ``apply_feedback_to_manifest``.
    """
    _ensure_indexes()

    prev = get_latest_feedback(session_id, turn_id)
    is_change = bool(prev) and prev.get("vote") != vote

    now = datetime.now(timezone.utc)
    # Caller may pre-mint feedback_id (needed when attachments are uploaded
    # to S3 before the Mongo insert so paths can include the id).
    if not feedback_id:
        feedback_id = f"fb_{uuid.uuid4().hex[:24]}"
    doc: Dict[str, Any] = {
        "feedback_id":          feedback_id,
        "session_id":           session_id,
        "turn_id":              turn_id,
        "user_id":              int(user_id),
        "project_id":           int(project_id),
        "agent_id":             agent_id,
        "vote":                 vote,
        "categories":           list(categories or []),
        "note":                 (note or None),
        "note_is_pii":          bool(note),
        "user_query":           (user_query or None),
        "agent_response":       (agent_response or None),
        "attachments":          list(attachments or []),
        "attachment_count":     len(attachments or []),
        "created_at":           now,
        "previous_feedback_id": (prev.get("feedback_id") if prev else None),
        "is_vote_change":       is_change,
        "schema_version":       SCHEMA_VERSION,
    }

    _get_collection().insert_one(doc)
    # insert_one mutates `_id`; drop it before returning
    doc.pop("_id", None)
    logger.info(
        "feedback recorded session=%s turn=%s vote=%s change=%s feedback_id=%s",
        session_id, turn_id, vote, is_change, feedback_id,
    )
    return doc


# -------- S3 manifest rollup ----------------------------------------

def _session_manifest_key(user_id: int, project_id: int, agent_id: str, session_id: str) -> str:
    """Match the layout used by shared/session_tracker.py."""
    return f"sessions/{int(user_id)}/{int(project_id)}/{agent_id}/_session-start_{session_id}.json"


def _s3_bucket() -> str:
    return os.environ.get("S3_BUCKET_NAME", "agentic-ai-production")


def apply_feedback_to_manifest(feedback_doc: Dict[str, Any]) -> bool:
    """Update the S3 session manifest with a per-turn feedback rollup.

    Best-effort: failures are logged but never raised. The Mongo doc is the
    canonical record; the manifest is a convenience cache for the UI.
    """
    try:
        bucket = _s3_bucket()
        key = _session_manifest_key(
            feedback_doc["user_id"],
            feedback_doc["project_id"],
            feedback_doc["agent_id"],
            feedback_doc["session_id"],
        )
        s3 = _get_s3_client()
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            import json
            manifest = json.loads(resp["Body"].read())
        except s3.exceptions.NoSuchKey:
            logger.warning("manifest missing for feedback rollup: bucket=%s key=%s", bucket, key)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("manifest read failed for feedback rollup (%s)", exc)
            return False

        feedback_map = manifest.setdefault("feedback", {})
        turn_id = feedback_doc["turn_id"]
        existing = feedback_map.get(turn_id) or {"vote_history": []}
        existing["current_vote"] = feedback_doc["vote"]
        existing["current_feedback_id"] = feedback_doc["feedback_id"]
        existing["categories"] = feedback_doc.get("categories") or []
        existing["note_present"] = bool(feedback_doc.get("note"))
        existing["updated_at"] = feedback_doc["created_at"].isoformat() \
            if hasattr(feedback_doc["created_at"], "isoformat") else str(feedback_doc["created_at"])
        existing["vote_history"] = (existing.get("vote_history") or [])[-9:] + [{
            "feedback_id": feedback_doc["feedback_id"],
            "vote": feedback_doc["vote"],
            "at":   existing["updated_at"],
            "is_change": feedback_doc.get("is_vote_change", False),
        }]
        feedback_map[turn_id] = existing

        import json
        s3.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(manifest, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("manifest feedback rollup applied: session=%s turn=%s vote=%s",
                    feedback_doc["session_id"], feedback_doc["turn_id"], feedback_doc["vote"])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("apply_feedback_to_manifest failed (%s)", exc)
        return False
