"""Session tracker — push agent session metadata to S3 + iField Mongo.

Used by all three live agents:
  - drawing-agent          (agents/rag-engine)
  - scope-of-work          (agents/doc-generator)
  - constructability-review (agents/review-pipeline)

Two public entrypoints:

  push_session_start(user_id, project_id, agent_id, client_session_id, **meta)
      Called once when a UI session begins. Writes a tiny S3 marker file and
      POSTs to the userSession API.

  push_session_call(user_id, project_id, agent_id, client_session_id, payload)
      Called once per agent call (chat turn, scope-gap run, etc.). Writes the
      full request/response payload to S3 and POSTs to the userSession API so
      iField records a Mongo row pointing at the S3 object.

Both are best-effort + non-blocking: on any failure they log a warning, never
raise, and never block the user-facing flow. Two events share the same
``client_session_id`` (caller-supplied UUID) in their S3 payloads, which is how
calls are linked back to their session-start row; the API has no parent/child
field of its own.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Literal
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


VALID_AGENT_IDS = frozenset({
    "drawing-agent",
    "scope-of-work",
    "constructability-review",
})

USERSESSION_URL = "https://mongo.ifieldsmart.com/api/userSession"
DEFAULT_S3_BUCKET = "agentic-ai-production"
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 100_000  # truncate large agent responses before S3 PUT


def _userSession_url() -> str:
    return os.environ.get("IFIELD_USERSESSION_URL", USERSESSION_URL)


def _userSession_timeout_s() -> float:
    return float(os.environ.get(
        "IFIELD_USERSESSION_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS,
    ))


def _s3_bucket() -> str:
    return os.environ.get("S3_BUCKET", DEFAULT_S3_BUCKET)


def _validate_agent_id(agent_id: str) -> None:
    if agent_id not in VALID_AGENT_IDS:
        raise ValueError(
            f"unknown agent_id {agent_id!r}; "
            f"must be one of {sorted(VALID_AGENT_IDS)}"
        )


def _truncate_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``response`` truncated to MAX_RESPONSE_BYTES."""
    out = dict(payload)
    response = out.get("response")
    if isinstance(response, str) and len(response.encode("utf-8", "ignore")) > MAX_RESPONSE_BYTES:
        encoded = response.encode("utf-8", "ignore")[:MAX_RESPONSE_BYTES]
        out["response"] = encoded.decode("utf-8", "ignore")
        out["response_truncated"] = True
    return out


_SLIM_SOURCE_FIELDS: frozenset[str] = frozenset({
    "s3_path", "file_name", "display_title", "drawing_name",
    "drawing_title", "page", "csi_division", "source_document_type",
    "doc_number",
})


def _slim_source_documents(source_documents: list | None) -> list[dict]:
    """Return new source-doc dicts with only display fields.

    Drops heavy / expiring fields (download_url, text_excerpt, bbox_*,
    text_blocks, page dims, parent_id, fragment_count) — those live in the
    S3 manifest. Never mutates the input (immutability rule).
    """
    if not source_documents:
        return []
    slim: list[dict] = []
    for doc in source_documents:
        if not isinstance(doc, dict):
            continue
        slim.append({k: doc[k] for k in _SLIM_SOURCE_FIELDS if k in doc})
    return slim


def _s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _write_s3_json(*, bucket: str, key: str, payload: dict[str, Any]) -> bool:
    try:
        s3 = _s3_client()
        body = json.dumps(payload, default=str, indent=2).encode("utf-8")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return True
    except Exception as e:
        logger.warning(
            "[session] S3 write failed for %s/%s: %s: %s",
            bucket, key, type(e).__name__, e,
        )
        return False


def _post_user_session(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    s3_bucket_path: str,
    presigned_url: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    title: str | None = None,
    question: str | None = None,
    is_active: bool | None = None,
    follow_up_questions: list | None = None,
    source_documents: list | None = None,
) -> str | None:
    """POST to userSession; return ifield-issued sessionId on 200, else None.

    Core fields (userId/projectId/agent/s3BucketPath) always sent. Per-turn
    fields are added only when supplied, so minimal-body callers are unchanged.
    """
    try:
        _body: dict[str, Any] = {
            "userId":       int(user_id),
            "projectId":    int(project_id),
            "agent":        agent_id,
            "s3BucketPath": s3_bucket_path,
        }
        if presigned_url:
            _body["presignedUrl"] = presigned_url
        if session_id is not None:
            _body["sessionId"] = session_id
        if turn_id is not None:
            _body["turnId"] = turn_id
        if title is not None:
            _body["title"] = title
        if question is not None:
            _body["question"] = question
        if is_active is not None:
            _body["isActive"] = bool(is_active)
        if follow_up_questions is not None:
            _body["follow_up_questions"] = follow_up_questions
        if source_documents is not None:
            _body["source_documents"] = source_documents
        resp = requests.post(
            _userSession_url(),
            json=_body,
            timeout=_userSession_timeout_s(),
        )
        if resp.status_code != 200:
            logger.warning(
                "[session] userSession API returned %s: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        body = resp.json()
        return ((body or {}).get("data") or {}).get("sessionId")
    except Exception as e:
        logger.warning(
            "[session] userSession API call failed: %s: %s",
            type(e).__name__, e,
        )
        return None


def _push(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    client_session_id: str,
    event_uuid: str,
    kind: Literal["session_start", "call"],
    payload: dict[str, Any],
) -> dict[str, Any]:
    _validate_agent_id(agent_id)

    bucket = _s3_bucket()
    if kind == "session_start":
        key = (
            f"sessions/{int(user_id)}/{int(project_id)}/{agent_id}/"
            f"_session-start_{client_session_id}.json"
        )
    else:
        key = (
            f"sessions/{int(user_id)}/{int(project_id)}/{agent_id}/"
            f"{event_uuid}.json"
        )
    s3_full = f"{bucket}/{key}"

    full_payload = {
        "kind":               kind,
        "client_session_id":  client_session_id,
        "event_id":           event_uuid,
        "user_id":            int(user_id),
        "project_id":         int(project_id),
        "agent_id":           agent_id,
        "captured_at":        datetime.now(timezone.utc).isoformat(),
        **_truncate_response(payload or {}),
    }

    s3_ok = _write_s3_json(bucket=bucket, key=key, payload=full_payload)
    # v3.4 — only the once-per-session register pushes to iField Mongo.
    # Per-turn audit files stay in S3 but do not create extra Mongo rows
    # (so byAgent returns 1 row per chat, not 1 per turn). The dedicated
    # push_session_register() helper still posts to Mongo with the presigned
    # URL embedded in s3BucketPath.
    ifield_id = None
    if kind == "session_start":
        ifield_id = _post_user_session(
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            s3_bucket_path=s3_full,
        )

    if ifield_id:
        logger.info(
            "[session] pushed kind=%s agent=%s client_session=%s ifield=%s",
            kind, agent_id, client_session_id, ifield_id,
        )

    return {
        "client_session_id": client_session_id,
        "event_id":          event_uuid,
        "s3_bucket_path":    s3_full,
        "ifield_session_id": ifield_id,
        "s3_written":        s3_ok,
        "ifield_pushed":     bool(ifield_id),
    }


def push_session_start(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    client_session_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a new UI session for an agent.

    Returns a dict with ``client_session_id`` (use it for subsequent
    push_session_call invocations to link calls back to this session).
    """
    sid = client_session_id or str(uuid.uuid4())
    return _push(
        user_id=user_id,
        project_id=project_id,
        agent_id=agent_id,
        client_session_id=sid,
        event_uuid=sid,
        kind="session_start",
        payload=payload or {},
    )


def push_session_call(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    client_session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Record one agent call (chat turn / pipeline run) inside an existing session."""
    if not client_session_id:
        raise ValueError("client_session_id is required for push_session_call")
    return _push(
        user_id=user_id,
        project_id=project_id,
        agent_id=agent_id,
        client_session_id=client_session_id,
        event_uuid=str(uuid.uuid4()),
        kind="call",
        payload=payload,
    )


def push_session_turn(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    session_id: str,
    turn_id: str,
    title: str,
    question: str,
    answer: str | None = None,
    follow_up_questions: list | None = None,
    source_documents: list | None = None,
    is_active: bool = True,
) -> dict[str, Any]:
    """Record ONE chat turn: full payload to S3 + slim row to iField Mongo.

    - S3 key: sessions/{user}/{project}/{agent}/{turn_id}.json (FULL sources).
    - Mongo POST: slim sources only (display fields); backend upserts the
      turn on (userId,projectId,agent,sessionId,turnId) and the session
      summary on (userId,projectId,agent,sessionId).
    Best-effort + non-blocking: logs and returns on any failure, never raises
    — except for an invalid ``agent_id``, which raises ValueError (a
    programming error, consistent with the other push_* helpers).
    """
    if not session_id or not turn_id:
        logger.warning("[session-turn] missing session_id/turn_id; skipping write")
        return {"session_id": session_id, "turn_id": turn_id,
                "s3_bucket_path": None, "ifield_pushed": False, "s3_written": False}
    _validate_agent_id(agent_id)
    bucket = _s3_bucket()
    key = f"sessions/{int(user_id)}/{int(project_id)}/{agent_id}/{turn_id}.json"
    s3_full = f"{bucket}/{key}"

    full_payload = {
        "kind":                "turn",
        "session_id":          session_id,
        "turn_id":             turn_id,
        "user_id":             int(user_id),
        "project_id":          int(project_id),
        "agent_id":            agent_id,
        "title":               title,
        "question":            question,
        "answer":              answer,
        "follow_up_questions": follow_up_questions or [],
        "source_documents":    source_documents or [],
        "captured_at":         datetime.now(timezone.utc).isoformat(),
    }

    s3_ok = False
    try:
        s3_ok = _write_s3_json(bucket=bucket, key=key, payload=full_payload)
    except Exception as e:
        logger.warning("[session-turn] S3 write failed: %s: %s", type(e).__name__, e)

    # 7-day pre-signed GET URL so the frontend can open the S3 payload directly
    # via the returned s3BucketPath. NOTE: expires after _PRESIGN_TTL_SECONDS;
    # per-turn rows are not refreshed, so durable access should re-sign on read.
    presigned_url = _generate_presigned_get_url(bucket, key)

    # answer/full geometry intentionally omitted from the POST — they live in
    # the S3 manifest; Mongo holds only slim display fields.
    ifield_id = _post_user_session(
        user_id=user_id, project_id=project_id, agent_id=agent_id,
        s3_bucket_path=presigned_url or s3_full,
        presigned_url=presigned_url,
        session_id=session_id, turn_id=turn_id, title=title,
        question=question, is_active=is_active,
        follow_up_questions=follow_up_questions or [],
        source_documents=_slim_source_documents(source_documents),
    )

    logger.info(
        "[session-turn] user=%s project=%s agent=%s session=%s turn=%s ifield=%s s3=%s",
        user_id, project_id, agent_id, session_id, turn_id, ifield_id, s3_ok,
    )
    return {
        "session_id":     session_id,
        "turn_id":        turn_id,
        "s3_bucket_path": s3_full,
        "presigned_url":  presigned_url,
        "ifield_pushed":  bool(ifield_id),
        "s3_written":     bool(s3_ok),
    }


# ── read-side helpers (list + detail) ────────────────────────────────


def list_sessions(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
) -> dict[str, Any]:
    """Return all userSession rows for one (user, project, agent).

    Proxies ``GET https://mongo.ifieldsmart.com/api/userSession/list``.
    Result: ``{"success": bool, "data": [<record>, ...], "count": int}``.

    On error returns ``{"success": False, "error": "...", "data": []}`` —
    never raises (best-effort, matches push semantics).
    """
    _validate_agent_id(agent_id)
    base = _userSession_url().rstrip("/")
    url = f"{base}/list"
    try:
        resp = requests.get(
            url,
            params={
                "userId":    int(user_id),
                "projectId": int(project_id),
                "agent":     agent_id,
            },
            timeout=_userSession_timeout_s(),
        )
        if resp.status_code != 200:
            logger.warning(
                "[session] list_sessions HTTP %s: %s",
                resp.status_code, resp.text[:200],
            )
            return {"success": False, "error": f"HTTP {resp.status_code}", "data": []}
        body = resp.json() or {}
        data = body.get("data") or []
        return {"success": True, "data": data, "count": len(data)}
    except Exception as e:
        logger.warning(
            "[session] list_sessions failed: %s: %s",
            type(e).__name__, e,
        )
        return {"success": False, "error": f"{type(e).__name__}: {e}", "data": []}


def get_session_history(*, session_id: str, user_id: int | None = None, project_id: int | None = None) -> dict[str, Any]:
    """Return all turn rows for one session via GET /userSession/history/{id}.

    Result: {"success": bool, "data": [<turn>, ...], "count": int}.
    Optionally scopes by user_id/project_id (query params) for access control.
    Best-effort — never raises.
    """
    base = _userSession_url().rstrip("/")
    url = f"{base}/history/{quote(session_id, safe='')}"
    params: dict[str, Any] = {}
    if user_id is not None:
        params["userId"] = int(user_id)
    if project_id is not None:
        params["projectId"] = int(project_id)
    try:
        resp = requests.get(url, params=params or None, timeout=_userSession_timeout_s())
        if resp.status_code != 200:
            logger.warning(
                "[session] get_session_history HTTP %s: %s",
                resp.status_code, resp.text[:200],
            )
            return {"success": False, "error": f"HTTP {resp.status_code}", "data": []}
        body = resp.json() or {}
        data = body.get("data") or []
        return {"success": True, "data": data, "count": len(data)}
    except Exception as e:
        logger.warning(
            "[session] get_session_history failed: %s: %s",
            type(e).__name__, e,
        )
        return {"success": False, "error": f"{type(e).__name__}: {e}", "data": []}


def resolve_session_title(*, session_id: str, question: str) -> str:
    """Return the canonical title for a session.

    Reuse the title from the first existing turn (so every turn shares one
    title). If no row exists yet (turn-1) or lookup fails, derive from the
    question. Title is generated once and reused verbatim thereafter.
    """
    try:
        hist = get_session_history(session_id=session_id)
        for row in (hist.get("data") or []):
            existing = row.get("title")
            if existing:  # skip blank/None titles — treat as missing
                return existing
    except Exception:
        pass
    return _truncate_title(question)


def fetch_session_payload(s3_bucket_path: str) -> dict[str, Any] | None:
    """Fetch the audit JSON for one session from S3 (``s3BucketPath`` in Mongo).

    Returns ``None`` if the object isn't found or the read fails. Never raises.
    """
    if not s3_bucket_path or "/" not in s3_bucket_path:
        return None
    bucket, _, key = s3_bucket_path.partition("/")
    if not bucket or not key:
        return None
    try:
        s3 = _s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        return json.loads(body.decode("utf-8"))
    except Exception as e:
        logger.warning(
            "[session] fetch_session_payload failed for %s: %s: %s",
            s3_bucket_path, type(e).__name__, e,
        )
        return None


def get_session_by_id(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    session_id: str,
    include_payload: bool = True,
) -> dict[str, Any] | None:
    """Return a single session row + (optionally) its S3 payload.

    Enforces user ownership: the lookup is filtered by ``user_id`` via the
    byAgent API, so a session created by user A cannot be fetched by user B
    (no record returned).

    Shape:
        ``{"record": {<mongo doc>}, "payload": {<s3 blob>} | None}``
    or ``None`` if no record matches.
    """
    res = list_sessions(
        user_id=user_id, project_id=project_id, agent_id=agent_id,
    )
    if not res.get("success"):
        return None
    match = next(
        (r for r in res["data"] if r.get("sessionId") == session_id),
        None,
    )
    if match is None:
        return None
    out: dict[str, Any] = {"record": match, "payload": None}
    if include_payload:
        out["payload"] = fetch_session_payload(match.get("s3BucketPath", ""))
    return out




# ---------------------------------------------------------------------------
# v3.4 (2026-05-20) — ChatGPT-style sidebar helpers
# ---------------------------------------------------------------------------
# AWS S3 presigned-URL hard limit is 7 days (604_800 s) with IAM-user creds —
# there is no "no-expiry" option. We always generate the max and refresh on
# every chat turn so active sessions stay valid; idle sessions get a fresh
# URL on next activity (or via the on-demand refresh helper below).

_PRESIGN_TTL_SECONDS = 604800  # 7 days, the AWS-documented max


def _session_manifest_key(user_id: int, project_id: int, agent_id: str,
                          client_session_id: str) -> str:
    """Stable S3 key for the per-session manifest (1 file per chat)."""
    return (
        f"sessions/{int(user_id)}/{int(project_id)}/{agent_id}/"
        f"_session-start_{client_session_id}.json"
    )


def _truncate_title(text: str, max_chars: int = 80) -> str:
    s = (text or "").strip().replace("\n", " ")
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def _embed_title_in_url(presigned_url: str | None, title: str | None) -> str | None:
    """Append #title=<url-encoded title> fragment to a presigned URL.

    Browsers and fetch() strip URL fragments before sending the request,
    so the AWS signature stays valid. UI clients extract the title from
    the fragment to render sidebar entries without fetching every manifest.
    """
    if not presigned_url:
        return presigned_url
    if not title:
        return presigned_url
    base = presigned_url.split('#', 1)[0]
    encoded = quote(_truncate_title(str(title)), safe='')
    return f"{base}#title={encoded}"


def _generate_presigned_get_url(bucket: str, key: str,
                                 expires: int = _PRESIGN_TTL_SECONDS) -> str | None:
    """Pre-signed GET URL for a manifest object. None on failure."""
    try:
        s3 = _s3_client()
        return s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires),
        )
    except Exception as exc:
        logger.warning(
            "[presigned] generate failed bucket=%s key=%s: %s: %s",
            bucket, key, type(exc).__name__, exc,
        )
        return None


def push_session_register(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    client_session_id: str,
    conversation_session_id: str,
    title: str,
    messages: list,
    extra: dict | None = None,
) -> dict:
    """Fire ONCE per chat session: posts iField Mongo + writes S3 manifest.

    Includes a 7-day presigned URL for the manifest in:
      (a) the iField POST payload (extra ``presignedUrl`` field) — UI consumes
          via /api/userSession/byAgent
      (b) the manifest body itself (self-referential, lets any consumer with
          access to the manifest see its own canonical URL).

    Returns: {client_session_id, ifield_session_id, s3BucketPath, presigned_url}.
    """
    _validate_agent_id(agent_id)
    bucket = _s3_bucket()
    key = _session_manifest_key(user_id, project_id, agent_id, client_session_id)
    s3_path = f"{bucket}/{key}"
    now_iso = datetime.now(timezone.utc).isoformat()

    presigned_url = _generate_presigned_get_url(bucket, key)

    manifest = {
        "kind":                    "session_register",
        "schema_version":          "1",
        "user_id":                 int(user_id),
        "project_id":              int(project_id),
        "agent_id":                agent_id,
        "client_session_id":       client_session_id,
        "conversation_session_id": conversation_session_id,
        "title":                   _truncate_title(title),
        "message_count":           len(messages or []),
        "created_at":              now_iso,
        "updated_at":              now_iso,
        "presigned_url":           presigned_url,
        "presigned_url_expires_at": (
            (datetime.now(timezone.utc) + timedelta(seconds=_PRESIGN_TTL_SECONDS)).isoformat()
            if presigned_url else None
        ),
        "s3BucketPath":            s3_path,
        "messages":                messages or [],
        **(extra or {}),
    }
    s3_ok = _write_s3_json(bucket=bucket, key=key, payload=manifest)
    # Option B: embed the presigned URL in the s3BucketPath field so iField
    # stores it without needing a schema change. UI consumes the field directly
    # as a URL. Fall back to the literal S3 path if URL generation failed.
    # v3.5 (2026-05-21): also embed title as a URL fragment so UI can render
    # sidebar entries without fetching each manifest.
    s3path_for_mongo = _embed_title_in_url(presigned_url, title) if presigned_url else s3_path
    ifield_id = _post_user_session(
        user_id=user_id, project_id=project_id, agent_id=agent_id,
        s3_bucket_path=s3path_for_mongo,
        presigned_url=presigned_url,
    )
    logger.info(
        "[session-register] user=%s project=%s agent=%s cid=%s csid=%s "
        "ifield=%s s3=%s presigned=%s",
        user_id, project_id, agent_id, client_session_id,
        conversation_session_id, ifield_id, s3_ok, bool(presigned_url),
    )
    return {
        "client_session_id": client_session_id,
        "ifield_session_id": ifield_id,
        "s3BucketPath":      s3_path,
        "presigned_url":     presigned_url,
    }


def refresh_session_manifest(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    client_session_id: str,
    messages: list,
    extra: dict | None = None,
) -> dict:
    """Fire on every subsequent turn: overwrite S3 manifest + rotate presigned URL.

    No Mongo call — that row already exists from push_session_register.
    Returns: {s3BucketPath, presigned_url, ok}.
    """
    _validate_agent_id(agent_id)
    bucket = _s3_bucket()
    key = _session_manifest_key(user_id, project_id, agent_id, client_session_id)
    s3_path = f"{bucket}/{key}"
    now_iso = datetime.now(timezone.utc).isoformat()

    created_at = now_iso
    conv_sid = None
    try:
        existing = _read_s3_json(bucket=bucket, key=key)
        if existing:
            created_at = existing.get("created_at", now_iso)
            conv_sid = existing.get("conversation_session_id")
    except Exception:
        pass

    presigned_url = _generate_presigned_get_url(bucket, key)

    manifest = {
        "kind":                    "session_register",
        "schema_version":          "1",
        "user_id":                 int(user_id),
        "project_id":              int(project_id),
        "agent_id":                agent_id,
        "client_session_id":       client_session_id,
        "conversation_session_id": conv_sid,
        "title":                   _truncate_title(
            (messages or [{}])[0].get("content", "") if messages else ""
        ),
        "message_count":           len(messages or []),
        "created_at":              created_at,
        "updated_at":              now_iso,
        "presigned_url":           presigned_url,
        "presigned_url_expires_at": (
            (datetime.now(timezone.utc) + timedelta(seconds=_PRESIGN_TTL_SECONDS)).isoformat()
            if presigned_url else None
        ),
        "s3BucketPath":            s3_path,
        "messages":                messages or [],
        **(extra or {}),
    }
    ok = _write_s3_json(bucket=bucket, key=key, payload=manifest)
    return {"s3BucketPath": s3_path, "presigned_url": presigned_url, "ok": ok}


def _read_s3_json(*, bucket: str, key: str) -> dict | None:
    try:
        s3 = _s3_client()
        resp = s3.get_object(Bucket=bucket, Key=key)
        import json as _json
        return _json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return None


__all__ = [
    "VALID_AGENT_IDS",
    "push_session_start",
    "push_session_call",
    "push_session_turn",
    "list_sessions",
    "get_session_history",
    "resolve_session_title",
    "fetch_session_payload",
    "get_session_by_id",
    "push_session_register",
    "refresh_session_manifest",
]
