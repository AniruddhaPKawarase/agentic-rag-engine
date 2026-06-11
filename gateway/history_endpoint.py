"""
history_endpoint.py
===================

FastAPI router that mirrors the iField Mongo endpoints under the same
path prefix (/api/userSession/...) so the UI only needs to change the
base host. Schema is byte-compatible with the current Mongo response.

Routes exposed:
    GET /api/userSession/history/{session_id}
    GET /api/userSession/byAgent          (optional, behind same flag)

Env flags:
    ENABLE_S3_HISTORY_ENDPOINT      — master kill switch (default: false)
    ENABLE_S3_BYAGENT_ENDPOINT      — opt-in for the list endpoint  (default: false)
    HISTORY_INCLUDE_ANSWER          — include `answer` field inline (default: true)
    HISTORY_PRESIGN                 — mint presigned S3 urls (default: true)

Safety:
    - When ENABLE_S3_HISTORY_ENDPOINT != "true", routes return 404 so the
      UI continues to use Mongo as today. Pure dark-launch deploy.
    - This module performs NO writes. Existing write path (router.py
      /query handler -> push_session_turn) is untouched.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from shared.history_loader import load_session_history

logger = logging.getLogger(__name__)

router = APIRouter(tags=["session-history"])


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


@router.get("/api/userSession/history/{session_id}")
async def get_session_history(
    session_id: str,
    userId: int = Query(..., description="iField user id"),
    projectId: int = Query(..., description="iField project id"),
    agent: str = Query("drawing-agent", description="Agent slug"),
):
    """
    Schema-identical replacement for
    `https://mongo.ifieldsmart.com/api/userSession/history/{session_id}`.

    UI change required: ONE base URL. Path, query params, response
    shape are all preserved.
    """
    if not _flag("ENABLE_S3_HISTORY_ENDPOINT"):
        # Dark-launched. UI keeps calling Mongo as today.
        raise HTTPException(status_code=404, detail="Not Found")

    if not session_id or not session_id.startswith("sess_"):
        raise HTTPException(status_code=400, detail="invalid session_id")

    include_answer = _flag("HISTORY_INCLUDE_ANSWER", default="true")
    presign = _flag("HISTORY_PRESIGN", default="true")

    try:
        result = await asyncio.to_thread(
            load_session_history,
            session_id=session_id,
            user_id=userId,
            project_id=projectId,
            agent_id=agent,
            include_answer=include_answer,
            presign=presign,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[history-endpoint] unexpected failure: %s", exc)
        # Preserve Mongo envelope even on error so UI's parser does not crash.
        return {"success": False, "message": "History fetch failed", "data": []}

    # STRICT iField Mongo envelope: {success, message, data} — nothing else.
    # Diagnostic keys (sessionId, source) only included when debug flag is on.
    if _flag("HISTORY_DEBUG_FIELDS", default="false"):
        return result

    return {
        "success": result.get("success", True),
        "message": result.get("message", "History fetched successfully"),
        "data": result.get("data") or [],
    }


@router.get("/api/userSession/byAgent")
async def list_sessions_by_agent(
    userId: int = Query(...),
    projectId: int = Query(...),
    agent: str = Query("drawing-agent"),
):
    """
    Mirror of `/api/userSession/byAgent` — returns the session list.

    Currently a passthrough to iField Mongo because the session list bug
    (if any) was not confirmed in the diagnostic. Wrapped here so we can
    swap to S3-backed listing later without another UI change.

    Gated behind its own flag (default off).
    """
    if not _flag("ENABLE_S3_BYAGENT_ENDPOINT"):
        raise HTTPException(status_code=404, detail="Not Found")

    # Passthrough — verbatim. Same response shape as today.
    import requests

    base = os.getenv("IFIELD_MONGO_BASE", "https://mongo.ifieldsmart.com/api")
    url = f"{base}/userSession/byAgent"
    try:
        resp = await asyncio.to_thread(
            requests.get,
            url,
            params={"userId": userId, "projectId": projectId, "agent": agent},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[history-endpoint] byAgent passthrough failed: %s", exc)
        return {"success": False, "message": "List fetch failed", "data": []}


@router.get("/api/userSession/_health")
async def history_health():
    """Lightweight liveness check for the new endpoint, regardless of flag."""
    return {
        "service": "session-history",
        "enabled": _flag("ENABLE_S3_HISTORY_ENDPOINT"),
        "byagent_enabled": _flag("ENABLE_S3_BYAGENT_ENDPOINT"),
        "include_answer": _flag("HISTORY_INCLUDE_ANSWER", default="true"),
        "presign": _flag("HISTORY_PRESIGN", default="true"),
    }
