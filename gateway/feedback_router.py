"""FastAPI router for the Phase 1 feedback feature.

Endpoints:
  POST /sessions/{session_id}/messages/{turn_id}/feedback
  GET  /sessions/{session_id}/messages/{turn_id}/feedback
  GET  /sessions/{session_id}/feedback

Mounted only when FEEDBACK_ENABLED=true (the caller — gateway/app.py —
checks this flag before importing). When the flag is off, this router
is not registered and these paths return 404.

Schema is described in docs/persona_feedback/03_ARCHITECTURE.md.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from shared.feedback_attachments import (
    upload_attachment,
    validate_uploads,
    refresh_presigned_url,
)
from shared.feedback_store import (
    VALID_CATEGORIES,
    VALID_VOTES,
    apply_feedback_to_manifest,
    get_feedback_history,
    get_latest_feedback,
    get_session_feedback,
    submit_feedback,
    validate_payload,
)

logger = logging.getLogger("agentic_rag.feedback_router")

router = APIRouter(tags=["feedback"])


# -------- Request / response models ---------------------------------

class FeedbackSubmitRequest(BaseModel):
    user_id: int
    project_id: int
    agent_id: str = "drawing-agent"
    vote: str = Field(..., description="'up' or 'down'")
    categories: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    user_query: Optional[str] = None
    agent_response: Optional[str] = None


# -------- POST ------------------------------------------------------

async def _post_feedback_json(
    session_id: str,
    turn_id: str,
    body: FeedbackSubmitRequest,
    request: Request,
) -> dict:
    """Submit thumbs up/down feedback for a single assistant turn.

    Vote changes are allowed any time (decision A1/B3) — each submission
    appends a new immutable row in Mongo. The S3 session manifest is
    updated with a current-state rollup for fast UI render.
    """
    payload = body.model_dump()

    ok, err = validate_payload(payload)
    if not ok:
        raise HTTPException(status_code=422, detail=err)

    try:
        feedback_doc = submit_feedback(
            session_id=session_id,
            turn_id=turn_id,
            user_id=int(payload["user_id"]),
            project_id=int(payload["project_id"]),
            agent_id=payload["agent_id"],
            vote=payload["vote"],
            categories=payload.get("categories") or [],
            note=payload.get("note"),
            user_query=payload.get("user_query"),
            agent_response=payload.get("agent_response"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("feedback submit failed")
        raise HTTPException(status_code=500, detail=f"feedback store failure: {exc}")

    # Manifest rollup is best-effort (Mongo is canonical)
    manifest_ok = apply_feedback_to_manifest(feedback_doc)

    return {
        "success":              True,
        "feedback_id":          feedback_doc["feedback_id"],
        "session_id":           session_id,
        "turn_id":              turn_id,
        "vote":                 feedback_doc["vote"],
        "previous_feedback_id": feedback_doc.get("previous_feedback_id"),
        "is_vote_change":       feedback_doc.get("is_vote_change", False),
        "manifest_updated":     manifest_ok,
    }



import logging as _logging
_logger_dispatch = _logging.getLogger("agentic_rag.feedback_router.dispatch")


@router.post("/sessions/{session_id}/messages/{turn_id}/feedback")
async def post_feedback_dispatcher(
    session_id: str,
    turn_id: str,
    request: Request,
) -> dict:
    """Dispatcher: routes by Content-Type.

    - application/json        -> _post_feedback_json (existing behaviour, no attachments)
    - multipart/form-data     -> _post_feedback_multipart (new: thumbs + optional files)
    """
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/form-data"):
        return await _post_feedback_multipart(session_id, turn_id, request)
    # Default: JSON path. We must parse the body manually since FastAPI's
    # auto-binding requires a body param in the signature.
    import json as _json
    try:
        raw = await request.body()
        payload = _json.loads(raw) if raw else {}
        body_obj = FeedbackSubmitRequest(**payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid JSON body: {exc}")
    return await _post_feedback_json(session_id, turn_id, body_obj, request)


# v3.4.4 (2026-05-26) - multipart attachment support
async def _post_feedback_multipart(
    session_id: str,
    turn_id: str,
    request: Request,
) -> dict:
    """Submit thumbs feedback with optional file attachments.

    Same write path as the JSON endpoint, plus:
      - validates and uploads each file to S3 under
        sessions/{user_id}/{project_id}/{agent_id}/{session_id}/feedback/{turn_id}/{feedback_id}/<file>
      - embeds the per-file metadata into the agent_feedback Mongo doc
        under `attachments`

    Note is REQUIRED for vote=="down" (same rule as JSON path now).
    """
    import uuid as _uuid

    form = await request.form()

    def _get(name, default=""):
        v = form.get(name)
        return v if v is not None else default

    # Form fields
    try:
        user_id    = int(_get("user_id"))
        project_id = int(_get("project_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="user_id and project_id must be integers")

    agent_id        = _get("agent_id") or "drawing-agent"
    vote            = _get("vote")
    note            = (_get("note") or "").strip()
    user_query      = _get("user_query") or None
    agent_response  = _get("agent_response") or None

    # categories: accept either repeated form field or single comma-separated string
    raw_cats = form.getlist("categories") if hasattr(form, "getlist") else []
    if not raw_cats:
        cs = _get("categories")
        raw_cats = [c.strip() for c in cs.split(",") if c.strip()] if cs else []
    if len(raw_cats) == 1 and "," in raw_cats[0]:
        raw_cats = [c.strip() for c in raw_cats[0].split(",") if c.strip()]
    categories = [c for c in raw_cats if c]

    # Files
    files = []
    for f in (form.getlist("files") if hasattr(form, "getlist") else []):
        if hasattr(f, "filename"):                    # filter out non-file fields named 'files'
            files.append(f)

    # Domain validation BEFORE touching S3
    payload_for_validation = {
        "user_id":     user_id,
        "project_id":  project_id,
        "agent_id":    agent_id,
        "vote":        vote,
        "categories":  categories,
        "note":        note or None,
    }
    ok, err = validate_payload(payload_for_validation)
    if not ok:
        raise HTTPException(status_code=422, detail=err)

    # Validate uploads (raises 422 on first failure; no S3 puts yet)
    file_bytes = await validate_uploads(files)

    # Pre-mint feedback_id so attachments share a single S3 sub-folder
    feedback_id = f"fb_{_uuid.uuid4().hex[:24]}"

    # Upload each file (rolls back any successful uploads on failure)
    uploaded: list = []
    try:
        for data, up in zip(file_bytes, files):
            meta = upload_attachment(
                data=data,
                content_type=(up.content_type or "").lower(),
                original_filename=up.filename or "unnamed",
                user_id=user_id, project_id=project_id, agent_id=agent_id,
                session_id=session_id, turn_id=turn_id, feedback_id=feedback_id,
            )
            uploaded.append(meta)
    except Exception as exc:
        # Best-effort rollback
        _logger_dispatch.warning("attachment upload failed; rolling back: %s", exc)
        import boto3 as _b3
        try:
            s3 = _b3.client("s3")
            for m in uploaded:
                try:
                    s3.delete_object(Bucket=m["s3_bucket"], Key=m["s3_key"])
                except Exception:
                    pass
        finally:
            raise HTTPException(status_code=500, detail=f"attachment upload failed: {exc}")

    # Persist
    try:
        feedback_doc = submit_feedback(
            session_id=session_id,
            turn_id=turn_id,
            user_id=user_id, project_id=project_id, agent_id=agent_id,
            vote=vote,
            categories=categories,
            note=note or None,
            user_query=user_query,
            agent_response=agent_response,
            attachments=uploaded,
            feedback_id=feedback_id,
        )
    except Exception as exc:
        _logger_dispatch.exception("feedback submit (multipart) failed")
        # Roll back uploaded files since we can't link them
        import boto3 as _b3
        try:
            s3 = _b3.client("s3")
            for m in uploaded:
                try: s3.delete_object(Bucket=m["s3_bucket"], Key=m["s3_key"])
                except Exception: pass
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"feedback store failure: {exc}")

    manifest_ok = apply_feedback_to_manifest(feedback_doc)

    return {
        "success":              True,
        "feedback_id":          feedback_doc["feedback_id"],
        "session_id":           session_id,
        "turn_id":              turn_id,
        "vote":                 feedback_doc["vote"],
        "previous_feedback_id": feedback_doc.get("previous_feedback_id"),
        "is_vote_change":       feedback_doc.get("is_vote_change", False),
        "manifest_updated":     manifest_ok,
        "attachments":          [
            {k: v for k, v in a.items() if k != "uploaded_at"}  # ISO-encode datetime cleanly
            for a in uploaded
        ],
    }


# -------- GET single turn -------------------------------------------

@router.get("/sessions/{session_id}/messages/{turn_id}/feedback")
async def get_turn_feedback(session_id: str, turn_id: str) -> dict:
    """Return the latest feedback for a turn + the full vote chain.

    v3.4.4 - refreshes presigned URLs on attachments older than 1 day so
    the UI never gets a 403 on a stale S3 link.
    """
    latest = get_latest_feedback(session_id, turn_id)
    history = get_feedback_history(session_id, turn_id, limit=50)

    def _refresh(row):
        if not row:
            return row
        for att in (row.get("attachments") or []):
            refresh_presigned_url(att)
        return row

    _refresh(latest)
    for h in history:
        _refresh(h)

    return {
        "success":      True,
        "session_id":   session_id,
        "turn_id":      turn_id,
        "current":      latest,
        "history":      history,
        "vote_count":   len(history),
    }


# -------- GET whole session -----------------------------------------

@router.get("/sessions/{session_id}/feedback")
async def get_all_session_feedback(session_id: str) -> dict:
    """Return every feedback row for the session, newest first."""
    feedback = get_session_feedback(session_id, limit=500)
    by_turn: dict = {}
    for row in feedback:
        by_turn.setdefault(row["turn_id"], []).append(row)
    return {
        "success":    True,
        "session_id": session_id,
        "feedback":   feedback,
        "by_turn":    by_turn,
        "count":      len(feedback),
    }


# -------- Helpers ---------------------------------------------------

@router.get("/feedback/categories")
async def list_categories() -> dict:
    """Static list of valid down-vote categories — for UI dropdowns."""
    return {
        "categories": list(VALID_CATEGORIES),
        "votes":      list(VALID_VOTES),
    }
