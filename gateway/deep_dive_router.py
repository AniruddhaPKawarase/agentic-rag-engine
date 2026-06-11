"""
gateway.deep_dive_router
========================

FastAPI router for the Deep Dive feature (v3.3).

Three endpoints, all mounted under ``/deep-dive``:

* ``POST /deep-dive``         — non-streaming JSON Deep Dive trigger
* ``POST /deep-dive/stream``  — SSE-streamed Deep Dive trigger
* ``GET  /deep-dive/models``  — list configured + credentialled vision models

The router is mounted alongside the existing ``/query`` router by the
gateway app. The existing ``/query`` and ``/query/stream`` endpoints are
NOT touched — Deep Dive is purely additive.

Hard feature flag: ``DEEP_DIVE_ENABLED=true`` must be set in the service
env for any of these endpoints to do real work. When the flag is off
they return 503 ``feature_disabled`` so the UI can hide the button.

Schema preservation
-------------------
The Deep Dive response shape is identical to the normal /query response
shape. The only differences are:
  - ``turn_type == "deep_dive"`` instead of ``"rag"``
  - ``parent_turn_id`` is populated (points to the original RAG turn)
  - ``answer`` starts with the literal string ``"**Deep Dive Analysis:**"``
    (set by the prompt — clients can use this as a UI label)

Old clients that don't know about ``turn_type`` / ``parent_turn_id``
just ignore the extra fields.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

logger = logging.getLogger("agentic_rag.deep_dive_router")

router = APIRouter(prefix="/deep-dive", tags=["deep-dive"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class DeepDiveRequest(BaseModel):
    """Body for ``POST /deep-dive`` and ``POST /deep-dive/stream``.

    User-story workflow (2026-05-15):
      1. User asks a question; RAG returns an answer + N cited documents.
      2. User picks a subset of those documents (1, 2, or all of them)
         from the UI.
      3. UI sends ``selected_pdf_names`` (preferred) or ``selected_indices``
         indicating the picks. Deep Dive runs the vision call against
         ONLY those picks instead of the full cited set.

    When neither selection field is present we fall back to the full
    grounded-filter output of the parent turn (default behavior).
    """

    session_id: str = Field(..., description="Session that owns the parent turn")
    turn_id: Optional[str] = Field(
        None,
        description=(
            "UUID of the parent turn (RAG or prior Deep Dive). When omitted, "
            "the most recent turn in the session is used — supports multi-turn "
            "Deep Dive conversations where the UI doesn't need to track turn_id "
            "after the first call."
        ),
    )
    project_id: Optional[int] = Field(None, description="Project scope (optional; resolved from parent turn if absent)")

    # --- Question override + multi-turn continuation ---
    # validation_alias accepts both {"question": ...} (canonical) and
    # {"query": ...} (Angular UI legacy key). First match wins per AliasChoices order.
    question: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("question", "query"),
        description=(
            "User's question for this Deep Dive call. When omitted (first-turn "
            "deep-dive), defaults to the parent turn's original question. "
            "Provide a new question to continue the conversation on the same "
            "picked documents without re-running RAG. Also accepts the legacy "
            "Angular UI key query for backward compatibility."
        ),
    )
    reuse_last_selection: bool = Field(
        True,
        description=(
            "When the request omits selected_pdf_names AND selected_indices, "
            "reuse the document selection from the most recent Deep Dive turn "
            "in this session. Lets the user continue asking new questions on "
            "the same docs they already picked. Set false to fall back to the "
            "parent turn's full source_documents."
        ),
    )
    user_id: Optional[str] = Field(None, description="UI-supplied user identifier (logged only)")
    history_turns: int = Field(1, ge=0, le=20, description="Number of prior turn pairs to include in context")
    max_images: Optional[int] = Field(None, ge=1, le=25, description="Cap on images sent to the vision model; defaults to env DEEP_DIVE_MAX_IMAGES")
    primary_model: Optional[str] = Field(None, description="Preferred vision model name (gpt-4.1, claude-sonnet-4, gemini-1.5-pro, etc.)")
    fallback_models: Optional[List[str]] = Field(None, description="Ordered fallback ladder; defaults to env DEEP_DIVE_FALLBACK_MODELS")

    # --- Document selection (Mode A from the user story) ---
    selected_pdf_names: Optional[List[str]] = Field(
        None,
        description=(
            "Pick-list of pdf_name strings (from the parent turn's "
            "source_documents) the user wants analyzed. When provided, "
            "Deep Dive runs vision against ONLY these documents. "
            "Mutually inclusive with selected_indices (a doc matched by "
            "either is included)."
        ),
    )
    selected_indices: Optional[List[int]] = Field(
        None,
        description=(
            "Alternative selection by 0-based index into the parent turn's "
            "source_documents. Useful when pdf_names contain characters "
            "the UI doesn't want to round-trip."
        ),
    )

    # --- Source format (2026-05-15: use both pdf + png as needed) ---
    source_format: str = Field(
        "auto",
        pattern="^(auto|png|pdf|both)$",
        description=(
            "Which asset variant to feed the vision model per selected "
            "document. 'auto' (default) = PNG with PDF fallback. "
            "'png' = PNG only (fastest). 'pdf' = PDF only (best for "
            "vector-text-heavy specs; only GPT-4.1/Gemini accept). "
            "'both' = send PNG AND PDF per doc (max accuracy, 2x image "
            "budget). Pick 'both' when the user explicitly wants the "
            "most accurate extraction and cost is not a concern."
        ),
    )


# ---------------------------------------------------------------------------
# Feature-flag gate
# ---------------------------------------------------------------------------


def _disabled_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": "feature_disabled",
            "message": "Deep Dive is not enabled in this environment.",
        },
    )


# ---------------------------------------------------------------------------
# Response builder — uses the SAME shape as /query
# ---------------------------------------------------------------------------


_UNCERTAINTY_MARKERS = (
    # First-person uncertainty
    "i cannot clearly read",
    "i cannot read",
    "cannot determine",
    "cannot verify",
    "unable to verify",
    "unable to determine",
    "unable to confirm",
    "i don't see",
    "i do not see",
    # Negation phrases the model uses for missing info
    "not legible",
    "[unreadable]",
    "not visible",
    "not shown",
    "not stated",
    "not specified",
    "not indicated",
    "not provided",
    "not given",
    "is not present",
    "at this resolution",
    "too small to read",
    "too blurry",
)


def _confidence_for(answer: str) -> tuple[str, float]:
    """H5 fix (2026-06-02): derive confidence from answer content instead of
    a hardcoded 'high if non-empty'. When the model explicitly says it
    cannot read or verify something, downgrade to medium/low so the UI
    doesn't mislead the user."""
    if not answer:
        return "low", 0.0
    a = answer.lower()
    hits = sum(1 for m in _UNCERTAINTY_MARKERS if m in a)
    if hits >= 3:
        return "low", 0.35
    if hits >= 1:
        return "medium", 0.65
    return "high", 0.9


def _result_to_response_dict(result: Any) -> Dict[str, Any]:
    """Translate a ``DeepDiveResult`` into the standard /query response shape.

    Every existing /query response field is present (with safe defaults
    where the Deep Dive call doesn't have a value). Only the additive
    ``turn_type`` / ``parent_turn_id`` / ``deep_dive_id`` fields are new.
    """
    confidence, conf_score = _confidence_for(result.deep_dive_answer or "")
    return {
        # core answer — answer == deep_dive_answer
        "query": result.original_question,
        "answer": result.deep_dive_answer,
        "rag_answer": result.deep_dive_answer,
        "web_answer": None,
        # retrieval / scoring — mirror the parent turn's vibe
        "retrieval_count": len(result.source_documents or []),
        "average_score": conf_score,
        "confidence": confidence,
        "confidence_score": conf_score,
        "is_clarification": False,
        # follow-ups (Deep Dive doesn't generate these)
        "follow_up_questions": [],
        "improved_queries": [],
        "query_tips": [],
        # model
        "model_used": result.model_used or "",
        "token_usage": {
            "total_tokens": (result.input_tokens + result.output_tokens),
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
        },
        "token_tracking": None,
        # sources — same shape as /query
        "s3_paths": [
            (sd.get("s3_path") or "")
            for sd in (result.source_documents or [])
            if isinstance(sd, dict) and sd.get("s3_path")
        ],
        "s3_path_count": sum(
            1 for sd in (result.source_documents or [])
            if isinstance(sd, dict) and sd.get("s3_path")
        ),
        "source_documents": result.source_documents or [],
        "sources": [
            (sd.get("display_title")
             or sd.get("drawing_name")
             or sd.get("pdf_name") or "")
            for sd in (result.source_documents or [])
            if isinstance(sd, dict)
        ],
        "retrieved_chunks": [],
        # debug
        "debug_info": {
            "deep_dive_id": result.deep_dive_id,
            "provider": result.provider,
            "fallback_used": result.fallback_used,
            "image_count": len(result.image_urls_used or []),
            "processing_status": result.status,
        },
        # identity / timing
        "processing_time_ms": result.processing_ms,
        "project_id": result.project_id,
        "session_id": result.session_id,
        "session_stats": None,
        # engine metadata
        "search_mode": "deep_dive_vision",
        "web_sources": [],
        "web_source_count": 0,
        "pin_status": None,
        "needs_document_selection": False,
        "available_documents": [],
        "scoped_to": None,
        "success": result.status == "completed",
        "engine_used": "deep_dive_vision",
        "fallback_used": result.fallback_used,
        "agentic_confidence": confidence,
        "error": result.error,
        # v3.3 turn identity
        "turn_id": result.turn_id,
        "turn_type": "deep_dive",
        "parent_turn_id": result.parent_turn_id,
        "deep_dive_id": result.deep_dive_id,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/models")
async def deep_dive_models(request: Request) -> JSONResponse:
    """List configured vision models + which have credentials present.

    Useful for UI / Postman to pre-flight which providers are usable.
    Always returns 200 — does NOT require the feature flag to be on
    so operators can verify config before flipping the flag.
    """
    try:
        from agentic.generation.vision_client import available_models
        return JSONResponse(content={
            "deep_dive_enabled": _is_enabled(),
            "models": available_models(),
            "primary_default": _primary_default(),
            "fallback_default": _fallback_default(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("deep_dive_models failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "models_introspection_failed", "message": str(exc)},
        )


@router.post("")
async def deep_dive(body: DeepDiveRequest, request: Request) -> JSONResponse:
    """Non-streaming Deep Dive trigger."""
    if not _is_enabled():
        return _disabled_response()
    try:
        from agentic.generation.deep_dive_agent import run_deep_dive

        handle = run_deep_dive(
            session_id=body.session_id,
            parent_turn_id=body.turn_id,
            project_id=body.project_id,
            history_turns=body.history_turns,
            max_images=body.max_images,
            primary_model=body.primary_model,
            fallback_models=body.fallback_models,
            selected_pdf_names=body.selected_pdf_names,
            selected_indices=body.selected_indices,
            source_format=body.source_format,
            user_question=body.question,
            reuse_last_selection=body.reuse_last_selection,
            stream=False,
        )
        # Drain the stream to populate the result; we do NOT yield to client.
        result = handle.collect()
        return JSONResponse(content=_result_to_response_dict(result))
    except Exception as exc:  # noqa: BLE001
        logger.error("deep_dive endpoint failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "deep_dive_failed", "message": str(exc)},
        )


@router.post("/stream")
async def deep_dive_stream(body: DeepDiveRequest, request: Request) -> StreamingResponse:
    """SSE-streamed Deep Dive trigger.

    Event types emitted:
      * ``event: status``    — phase updates ("collecting_sources",
                               "fetching_images", "vision_inference")
      * ``event: token``     — streamed text chunk
      * ``event: done``      — final response dict (same shape as /deep-dive)
      * ``event: error``     — terminal error (rare; per-chunk errors
                               are surfaced inside the token stream)
    """
    if not _is_enabled():
        return StreamingResponse(
            iter([_sse("error", {"error": "feature_disabled"})]),
            media_type="text/event-stream",
        )

    body_dict = body.model_dump()

    def _iter() -> Any:
        from agentic.generation.deep_dive_agent import run_deep_dive

        # Emit the FIRST status frame immediately — this is the "< 2s first
        # event" budget. Everything below this line can take its time.
        # H6 fix (2026-06-02): redact the request echo — previously the
        # entire request body (including session_id, user_id, and any
        # future auth tokens) was broadcast back as the first SSE frame.
        _redacted = {
            "session_id_prefix": (body_dict.get("session_id") or "")[:8],
            "project_id": body_dict.get("project_id"),
            "selected_count": len(body_dict.get("selected_pdf_names") or []),
            "has_question": bool(body_dict.get("question") or body_dict.get("query")),
            "source_format": body_dict.get("source_format"),
        }
        yield _sse("status", {"phase": "received", "request_summary": _redacted})

        try:
            handle = run_deep_dive(
                session_id=body.session_id,
                parent_turn_id=body.turn_id,
                project_id=body.project_id,
                history_turns=body.history_turns,
                max_images=body.max_images,
                primary_model=body.primary_model,
                fallback_models=body.fallback_models,
                selected_pdf_names=body.selected_pdf_names,
                selected_indices=body.selected_indices,
                source_format=body.source_format,
                user_question=body.question,
                reuse_last_selection=body.reuse_last_selection,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"error": "deep_dive_init_failed", "message": str(exc)})
            return

        yield _sse("status", {
            "phase": "collecting_sources",
            "image_count": len(handle.image_urls),
            "parent_turn_id": handle.parent_turn_id,
        })
        yield _sse("status", {"phase": "vision_inference", "deep_dive_id": handle.deep_dive_id})

        # Stream tokens
        try:
            for chunk in handle.tokens():
                if chunk:
                    yield _sse("token", {"delta": chunk})
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"error": "vision_stream_failed", "message": str(exc)})
            return

        # Done — emit the final dict
        result = handle.result()
        yield _sse("done", _result_to_response_dict(result))

    return StreamingResponse(_iter(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    try:
        from agentic.generation.deep_dive_agent import is_enabled
        return is_enabled()
    except Exception:  # noqa: BLE001
        return False


def _primary_default() -> str:
    import os
    return os.getenv("DEEP_DIVE_PRIMARY_MODEL", "gpt-4.1")


def _fallback_default() -> List[str]:
    import os
    raw = os.getenv(
        "DEEP_DIVE_FALLBACK_MODELS",
        "claude-sonnet-4,gemini-1.5-pro,gpt-4o",
    )
    return [m.strip() for m in raw.split(",") if m.strip()]


def _sse(event: str, data: Dict[str, Any]) -> bytes:
    """Format a single SSE frame."""
    line = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {line}\n\n".encode("utf-8")
