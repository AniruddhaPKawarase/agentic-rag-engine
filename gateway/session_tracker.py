"""Thin session tracker — push agent session metadata to iField Mongo.

For every agent call, this module:
  1. Writes a small JSON file to S3 with the call's request / response /
     timestamps / cost.
  2. POSTs the metadata to ``https://mongo.ifieldsmart.com/api/userSession``
     so iField records (userId, projectId, agent, s3BucketPath) in their
     Mongo session collection.

Used by all live agents:
  • ConstructibilityReview  (review_pipeline)
  • ProjectQnARAG           (unified-rag-agent-v31)
  • ScopeGapAnalysis        (construction-intelligence-agent)

Best-effort + non-blocking semantics — if the Mongo API or S3 write fails,
we log a warning and the agent call continues. Session tracking never
breaks the user-facing flow.

Intentionally light:
  • No Postgres index, no in-process buffer, no idle reaper.
  • One API call + one S3 PUT per session push.
  • Agent name is REQUIRED — the s3BucketPath always carries it
    (ifieldsmart/sessions/{userId}/{projectId}/{agent}/{session_uuid}.json)
    so the audit log is filterable by agent in S3 directly.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── env-tunable constants ─────────────────────────────────────────────


def _userSession_url() -> str:
    return os.environ.get(
        "IFIELD_USERSESSION_URL",
        "https://mongo.ifieldsmart.com/api/userSession",
    )


def _userSession_timeout_s() -> float:
    return float(os.environ.get("IFIELD_USERSESSION_TIMEOUT_SECONDS", "5"))


def _s3_bucket() -> str:
    return os.environ.get("S3_BUCKET", "ifieldsmart")


# ── S3 writer ─────────────────────────────────────────────────────────


def _s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _write_s3_json(*, bucket: str, key: str, payload: dict) -> bool:
    """Write a JSON payload to s3://{bucket}/{key}. Returns True on success."""
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
            f"[session] S3 write failed for {bucket}/{key}: "
            f"{type(e).__name__}: {e}"
        )
        return False


# ── userSession API client ────────────────────────────────────────────


def _post_user_session(
    *, user_id: int, project_id: int, agent: str, s3_bucket_path: str,
) -> str | None:
    """POST to mongo.ifieldsmart.com/api/userSession; return sessionId or None."""
    try:
        import requests
        resp = requests.post(
            _userSession_url(),
            json={
                "userId":       int(user_id),
                "projectId":    int(project_id),
                "agent":        agent,
                "s3BucketPath": s3_bucket_path,
            },
            timeout=_userSession_timeout_s(),
        )
        if resp.status_code != 200:
            logger.warning(
                f"[session] userSession API returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            return None
        body = resp.json()
        return ((body or {}).get("data") or {}).get("sessionId")
    except Exception as e:
        logger.warning(
            f"[session] userSession API call failed: "
            f"{type(e).__name__}: {e}"
        )
        return None


# ── public entrypoint ─────────────────────────────────────────────────


def push_session(
    *,
    user_id: int,
    project_id: int,
    agent: str,
    payload: dict[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Push one session record to S3 + iField Mongo.

    Args:
        user_id:    iFieldSmart user identifier.
        project_id: iFieldSmart project identifier.
        agent:      MUST be one of the canonical agent names:
                    'ConstructibilityReview', 'ProjectQnARAG',
                    'ScopeGapAnalysis' (or any new one — but always required).
        payload:    Arbitrary JSON-serializable dict to write to S3 — typically
                    the agent call's request, response, costs, run_id, etc.
        session_id: Optional client-supplied UUID. When omitted we generate
                    one. Either way the same id appears in the S3 key and
                    is included in the payload as ``"session_id"``.

    Returns:
        ``{"session_id": ..., "s3_bucket_path": "...",
           "ifield_session_id": ..., "s3_written": bool, "ifield_pushed": bool}``
        — never raises. Caller can log or ignore.
    """
    if not agent or not str(agent).strip():
        raise ValueError("agent is required for session push")

    sid = session_id or str(uuid.uuid4())
    bucket = _s3_bucket()
    key = (
        f"sessions/{int(user_id)}/{int(project_id)}/{agent}/{sid}.json"
    )
    s3_full = f"{bucket}/{key}"  # what iField stores in their Mongo doc

    # Stamp the session payload with provenance fields so a future reader
    # of the S3 file can identify it without context.
    full_payload = {
        "session_id":   sid,
        "user_id":      int(user_id),
        "project_id":   int(project_id),
        "agent":        agent,
        "captured_at":  datetime.now(timezone.utc).isoformat(),
        **(payload or {}),
    }

    s3_ok = _write_s3_json(bucket=bucket, key=key, payload=full_payload)
    ifield_id = _post_user_session(
        user_id=user_id, project_id=project_id, agent=agent,
        s3_bucket_path=s3_full,
    )
    return {
        "session_id":        sid,
        "s3_bucket_path":    s3_full,
        "ifield_session_id": ifield_id,
        "s3_written":        s3_ok,
        "ifield_pushed":     bool(ifield_id),
    }


__all__ = ["push_session"]
