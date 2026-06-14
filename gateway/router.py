"""
Gateway Router - all 18 endpoints for the Unified RAG Agent.

The orchestrator is accessed via ``request.app.state.orchestrator``.
All engine imports are lazy (inside functions, wrapped in try/except)
so the router works even if one engine fails to import.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.models import QueryRequest, UnifiedResponse

# --- Hallucination Guard v2 -- Pillar 11 adversarial pre-filter (env-flag gated) ---
try:
    from agentic.hallucination_guard.pillar_11_adversarial import (
        pre_filter_query as _hg_p11_filter,
        applied_metadata as _hg_p11_meta,
    )
    from agentic.hallucination_guard import get_active_pillars as _hg_active_pillars
except Exception:  # pragma: no cover -- package absent -> all-allow stub
    class _HG_AllowDecision:
        action = "allow"
        reason_class = "package_unavailable"
        matched_patterns = []
        refusal_text = None
        prompt_addendum = None
        regulatory_reframe_detected = False
        pillar_applied = False
    def _hg_p11_filter(q, has_history=False, force_enable=None):
        return _HG_AllowDecision()
    def _hg_p11_meta(decision=None):
        return {"pillar": 11, "applied": False}
    def _hg_active_pillars():
        return set()


# --- Hallucination Guard v2 -- Pillar 1 (L1 anaphora resolver) -------------
try:
    from agentic.hallucination_guard.pillar_1_anaphora import (
        resolve_anaphora as _hg_p1_resolve,
    )
except Exception:  # pragma: no cover -- package absent -> identity stub
    def _hg_p1_resolve(query, history=None, anthropic_client=None, force_enable=None):
        return query


def _hg_build_refusal_response(decision, body, resolved_session_id=None) -> dict:
    """Build a /query response dict for a Pillar 11 injection refusal.

    Returns a dict matching the /query response shape; skips retrieval + synthesis.
    """
    return {
        "answer": decision.refusal_text,
        "final_answer": decision.refusal_text,
        "session_id": resolved_session_id,
        "engine_used": "hallucination_guard_pillar_11",
        "sources": [],
        "all_retrieved_sources": [],
        "verification_meta": {
            "refused": True,
            "reason_class": decision.reason_class,
            "active_pillars": sorted(_hg_active_pillars()),
            "pillar_11": _hg_p11_meta(decision),
        },
    }


logger = logging.getLogger(__name__)

router = APIRouter()

# --- session history endpoint (additive, env-flag gated) -----------------
try:
    from gateway.history_endpoint import router as _history_router
    router.include_router(_history_router)
except Exception as _hist_exc:  # noqa: BLE001
    import logging as _hist_logging
    _hist_logging.getLogger(__name__).warning(
        "[history-endpoint] not loaded: %s: %s",
        type(_hist_exc).__name__, _hist_exc,
    )

# --- projects listing endpoint (additive) — GET /projects over drawings_v3 -
try:
    from gateway.projects_endpoint import router as _projects_router
    router.include_router(_projects_router)
except Exception as _proj_exc:  # noqa: BLE001
    import logging as _proj_logging
    _proj_logging.getLogger(__name__).warning(
        "[projects-endpoint] not loaded: %s: %s",
        type(_proj_exc).__name__, _proj_exc,
    )
# -------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Streaming helpers (shared SSE protocol with /deep-dive/stream)
#
# IMPORTANT: the streaming endpoints must NEVER run blocking I/O directly on
# the event loop. All heavy work goes through orchestrator.query (which
# offloads via asyncio.to_thread internally) or asyncio.to_thread wrappers.
# A prior version did synchronous S3/session I/O on the loop and, under
# concurrent traffic with --workers 1, stalled the whole service (504s).
# ---------------------------------------------------------------------------

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data: Any) -> str:
    """Format a named Server-Sent Event frame (same protocol as deep-dive)."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _run_query_offloop(orchestrator: Any, **kwargs: Any) -> dict:
    """Run orchestrator.query FULLY off the server event loop.

    orchestrator.query is async but contains synchronous on-loop work (intent
    classifier LLM call, S3-backed session reads/writes). Awaiting it directly
    would block the single-worker event loop under concurrent streaming load
    (observed: /health stalls). We instead run the whole coroutine inside a
    worker thread with its OWN event loop (asyncio.run), so the server loop
    stays free to serve other requests (/health, /deep-dive, other streams).
    """
    def _runner() -> Any:
        return asyncio.run(orchestrator.query(**kwargs))
    result = await asyncio.to_thread(_runner)
    return result if isinstance(result, dict) else {}


def _iter_word_chunks(text: str) -> "list[str]":
    """Split text into word-sized chunks that re-concatenate to the original.

    ``"".join(_iter_word_chunks(t)) == t`` always holds. Pure/cheap CPU.
    """
    if not text:
        return []
    import re
    return [m.group(0) for m in re.finditer(r"\S+\s*|\s+", text)]


def _typewriter_delay(n_chunks: int) -> float:
    """Per-chunk delay (seconds) for the typewriter effect.

    A real delay is REQUIRED: with sleep(0) the server emits every token in
    the same event-loop tick, so the client (and any proxy) receives them as
    one burst at the end — looking like 'whole answer appears at once'.

    Targets a TOTAL typing time of ~3.5s regardless of length: delay = 3.5/n,
    clamped to [2ms, 30ms] per word. The low 2ms floor (was 8ms) keeps long
    answers near the target — an 800-word answer now types in ~3.5s instead
    of ~6.4s — while short answers stay visible (30ms/word cap).
    """
    if n_chunks <= 0:
        return 0.02
    return max(0.002, min(0.03, 3.5 / n_chunks))


def _load_turn(session_id: str, turn_id: str) -> Optional[dict]:
    """Load (question, answer, source_documents) for a turn from memory.

    Synchronous (reads MemoryManager / S3-backed session). MUST be called
    via asyncio.to_thread from async endpoints so it never blocks the loop.
    """
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
    except ImportError:
        try:
            from shared.session import get_memory_manager  # type: ignore
        except ImportError:
            return None
    try:
        session = get_memory_manager().get_session(session_id)
    except Exception:
        return None
    if session is None:
        return None
    messages = getattr(session, "messages", []) or []
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        meta = getattr(msg, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        if meta.get("turn_id") == turn_id and getattr(msg, "role", None) == "assistant":
            user_text = ""
            for back in range(idx - 1, -1, -1):
                pmsg = messages[back]
                if getattr(pmsg, "role", None) == "user":
                    user_text = getattr(pmsg, "content", "") or ""
                    break
            return {
                "user_text": user_text,
                "assistant_text": getattr(msg, "content", "") or "",
                "source_documents": meta.get("source_documents", []) or [],
                "project_id": meta.get("project_id"),
                "set_id": meta.get("set_id"),
            }
    return None


# Feedback categories — lock-step with the frontend's mapFeedbackCategory().
_FEEDBACK_CATEGORIES: dict = {
    "wrong_facts": (
        "The previous answer contained inaccurate information. Carefully "
        "re-verify every claim against the retrieved source documents and "
        "correct any wrong facts."
    ),
    "incomplete": (
        "The previous answer was incomplete. Identify what was missing and "
        "provide a thorough, complete answer covering all relevant details."
    ),
    "irrelevant": (
        "The previous answer did not address the question. Focus precisely "
        "on what the user actually asked and answer that directly."
    ),
    "hallucination": (
        "The previous answer included made-up or unsupported data. State "
        "ONLY facts grounded in the retrieved documents; if something is "
        "unknown, say so explicitly rather than inventing it."
    ),
    "formatting": (
        "The factual content was acceptable but the formatting was poor. "
        "Re-present the same information with clear structure (headings, "
        "bullet lists, and tables where appropriate). Do not change the facts."
    ),
}

_FEEDBACK_LABEL_ALIASES: dict = {
    "inaccurate information": "wrong_facts",
    "incomplete answer": "incomplete",
    "not relevant to my question": "irrelevant",
    "hallucination / made up data": "hallucination",
    "poor formatting": "formatting",
}


def _normalize_category(raw: Optional[str]) -> Optional[str]:
    """Map a single category code or human label to a canonical code."""
    if not raw:
        return None
    key = raw.strip().lower()
    if key in _FEEDBACK_CATEGORIES:
        return key
    return _FEEDBACK_LABEL_ALIASES.get(key)


def _normalize_categories(raw: Union[str, List[str], None]) -> List[str]:
    """Normalise one OR many category values into a deduped list of codes."""
    if not raw:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: List[str] = []
    for item in items:
        code = _normalize_category(item if isinstance(item, str) else str(item))
        if code and code not in out:
            out.append(code)
    return out


def _persist_feedback(
    session_id: str, turn_id: str, user_id: Any,
    vote: str, comment: Optional[str], categories: Optional[List[str]] = None,
) -> None:
    """Best-effort feedback persistence. Never raises. Call via to_thread."""
    categories = categories or []
    record = {
        "session_id": session_id, "turn_id": turn_id, "user_id": user_id,
        "vote": vote, "categories": categories, "comment": comment,
    }
    logger.info("feedback_recorded: %s", json.dumps(record, default=str))
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
        recorder = getattr(get_memory_manager(), "record_feedback", None)
        if callable(recorder):
            recorder(
                session_id=session_id, turn_id=turn_id,
                vote=vote, comment=comment, user_id=user_id, categories=categories,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_orchestrator(request: Request) -> Any:
    """Retrieve the orchestrator from app state."""
    return request.app.state.orchestrator


def _get_config_summary() -> dict:
    """Return config summary without secrets."""
    try:
        from shared.config import get_config
        cfg = get_config()
        return {
            "host": cfg.host,
            "port": cfg.port,
            "log_level": cfg.log_level,
            "agentic_model": cfg.agentic_model,
            "agentic_model_fallback": cfg.agentic_model_fallback,
            "agentic_max_steps": cfg.agentic_max_steps,
            "traditional_model": cfg.traditional_model,
            "traditional_embedding_model": cfg.traditional_embedding_model,
            "fallback_enabled": cfg.fallback_enabled,
            "fallback_timeout_seconds": cfg.fallback_timeout_seconds,
            "faiss_lazy_load": cfg.faiss_lazy_load,
            "storage_backend": cfg.storage_backend,
            "mongo_db": cfg.mongo_db,
        }
    except Exception as exc:
        logger.warning("Failed to load config summary: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Root / Info
# ---------------------------------------------------------------------------

@router.get("/")
async def root() -> dict:
    """API info and available endpoints."""
    return {
        "service": "Unified RAG Agent",
        "version": "1.0.0",
        "engines": ["agentic", "traditional"],
        "endpoints": {
            "query": "POST /query",
            "stream": "POST /query/stream",
            "quick_query": "POST /quick-query",
            "web_search": "POST /web-search",
            "health": "GET /health",
            "config": "GET /config",
            "sessions": "GET /sessions",
        },
    }


# ---------------------------------------------------------------------------
# Health + Config
# ---------------------------------------------------------------------------

@router.get("/health")
async def health(request: Request) -> dict:
    """Check health of both engines."""
    orchestrator = _get_orchestrator(request)
    status = {
        "status": "healthy",
        "engines": {
            "agentic": {
                "initialized": orchestrator.agentic._initialized,
            },
            "traditional": {
                "faiss_loaded": orchestrator.traditional.is_loaded,
            },
        },
        "fallback_enabled": orchestrator.fallback_enabled,
    }
    return status


@router.get("/config")
async def config_endpoint() -> dict:
    """Return config summary (no secrets)."""
    return _get_config_summary()


# ---------------------------------------------------------------------------
# Gateway mode routing (multi-source fan-out)
# ---------------------------------------------------------------------------

_GATEWAY_MODES = {"rag", "web", "email", "meeting", "rfi", "hybrid"}
_LEGACY_ENGINE_MODES = {"agentic", "traditional"}
# [V19-COLLECTION-MODES] user-selectable collection scoping (additive; rides RAG path)
_COLLECTION_MODE_ALIASES = {
    "drawing": "drawing", "drawings": "drawing",
    "specification": "specification", "specifications": "specification", "spec": "specification",
}


def _v19_request_scope(tadr_scope, collection_scope):
    """Merge TADR scope dict + collection scope into the request scope (or None)."""
    merged = dict(tadr_scope) if tadr_scope else {}
    if collection_scope:
        merged["_collection_scope"] = collection_scope
    return merged or None


def _parse_collection_scope(search_mode) -> str | None:
    """Return "drawing" | "specification" | None from search_mode string."""
    import os as _os_cm
    if _os_cm.getenv("COLLECTION_MODES_ENABLED", "true").lower() != "true":
        return None
    if not search_mode:
        return None
    for tok in str(search_mode).split(","):
        hit = _COLLECTION_MODE_ALIASES.get(tok.strip().lower())
        if hit:
            return hit
    return None


def _parse_modes(search_mode: Optional[str]) -> set[str]:
    """Return set of gateway sources to query from search_mode string.

    None / legacy engine modes -> RAG-only (opt-in fan-out)
    'hybrid'                   -> {rag, email, meeting}
    'rag,email' / 'email' etc. -> intersection with _GATEWAY_MODES
    """
    if not search_mode or search_mode == "hybrid" or search_mode in _LEGACY_ENGINE_MODES:
        return {"rag", "email", "meeting", "rfi"}
    parsed = {m.strip().lower() for m in search_mode.split(",")}
    # [V19-MODE-NORM] collection modes are RAG-path modes
    parsed = {("rag" if m in _COLLECTION_MODE_ALIASES else m) for m in parsed}
    if "hybrid" in parsed:
        return {"rag", "email", "meeting", "rfi"}
    valid = parsed & _GATEWAY_MODES
    return valid if valid else {"rag"}


# ---------------------------------------------------------------------------
# Core Query Endpoints
# ---------------------------------------------------------------------------

@router.post("/query")
async def query(request: Request, body: QueryRequest) -> dict:
    """Route query to selected sources, return per-source answers + synthesized final_answer."""
    # --- Pillar 11 pre-filter (Hallucination Guard v2) ---
    _hg_has_history = bool(body.conversation_history)
    _hg_decision = _hg_p11_filter(body.query, has_history=_hg_has_history)
    if _hg_decision.action == "refuse_injection":
        logger.info(
            "[hg-p11] /query refused -- reason=%s patterns=%d",
            _hg_decision.reason_class, len(_hg_decision.matched_patterns),
        )
        return _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)

    # --- Pillar 1 anaphora resolver (Hallucination Guard v2) ---
    # Rewrites follow-up queries with pronouns/references into self-contained queries.
    # No-op when HG_PILLAR_1=false or history empty.
    _hg_resolved_query = _hg_p1_resolve(body.query, history=body.conversation_history or [])

    from gateway.email_hybrid_search import search_emails
    from gateway.external_sources import search_meetings, search_rfis
    from gateway.synthesizer import synthesize_answer

    orchestrator = _get_orchestrator(request)
    project_id_str = str(body.project_id)
    modes = _parse_modes(body.search_mode)
    # [V17-MODE-INIT-FIX] init before rag-block so email/rfi/meeting-only modes do not crash
    _tadr_phase3_skip_extras = False
    _tadr_scope = None
    # V19-SCOPE-COMPUTE: user-selected collection scope (drawing | specification | None)
    _collection_scope = _parse_collection_scope(body.search_mode)
    if _collection_scope:
        logger.info("[v19-mode] collection_scope=%s", _collection_scope)
    # Orchestrator only needs search_mode when 'web'; otherwise it runs default (agentic)
    orchestrator_search_mode = body.search_mode if body.search_mode == "web" else None

    # v3.4 — if UI sent client_session_id without session_id, look up existing
    # conversation session so all turns of one chat use the same Track A id.
    # v3.4.1 (2026-05-21) — accept session_id as the UI stable chat handle
    # when client_session_id is absent.
    # v3.4.2 (2026-05-25) — auto-generate session_id when UI omits it
    # so the UI's first POST need only carry query + project_id (+ optional user_id);
    # we mint the session_id, write Track B / iField Mongo with it, and echo it
    # in the response. UI captures it from the response and re-sends it on turn 2+.
    import uuid as _uuid
    effective_client_session_id = body.client_session_id or body.session_id
    _auto_generated_sid = False
    if not effective_client_session_id:
        effective_client_session_id = f"sess_{_uuid.uuid4().hex[:16]}"
        _auto_generated_sid = True
        logger.info(
            "[session] auto-generated session_id=%s (user_id=%s project_id=%s)",
            effective_client_session_id, body.user_id, body.project_id,
        )

    resolved_session_id = body.session_id
    if not resolved_session_id:
        if _auto_generated_sid:
            # Use the auto-generated id as the orchestrator's session handle too,
            # so response.session_id == effective_client_session_id and the
            # Track B path uses one consistent id end-to-end.
            resolved_session_id = effective_client_session_id
        else:
            resolved_session_id = _resolve_session_from_client_id(
                body.user_id, body.project_id, effective_client_session_id,
            )

    # [CG-V1] conversational gate — greetings/small-talk answered instantly
    # with full schema parity; no retrieval, no LLM. Ambiguous input falls
    # through. Flag CONVERSATIONAL_GATEWAY_ENABLED (default true).
    try:
        from gateway.conversational_gateway import (
            detect as _cg_detect, build_response as _cg_build,
        )
        _cg_intent = _cg_detect(body.query)
    except Exception as _cg_exc:  # noqa: BLE001
        logger.warning("[cg-v1] detect failed (%s); pipeline continues", _cg_exc)
        _cg_intent = None
    if _cg_intent:
        logger.info("[cg-v1] /query intercepted intent=%s mode=%s", _cg_intent, body.search_mode)
        return _cg_build(_cg_intent, body.query, project_id=body.project_id,
                         session_id=resolved_session_id, search_mode=body.search_mode)

    coros: dict[str, Any] = {}
    if "rag" in modes or "web" in modes:
        # [TADR-INSERT-MARKER] Trade-Aware Drawing Router — additive, env-flag gated
        try:
            from gateway.trade_router import classify as _tadr_classify
            _tadr_router_result = _tadr_classify(body.query)
            if not _tadr_router_result.is_noop:
                logger.info(
                    "[tadr] route=%s conf=%.2f tier=%d latency_ms=%d rationale=%s",
                    _tadr_router_result.primary,
                    _tadr_router_result.primary_confidence,
                    _tadr_router_result.classifier_tier,
                    _tadr_router_result.latency_ms,
                    _tadr_router_result.rationale[:60],
                )
        except Exception as _tadr_exc:  # noqa: BLE001
            logger.warning("[tadr] router degraded (no-op): %s: %s",
                           type(_tadr_exc).__name__, _tadr_exc)


        # [TADR-PHASE3-MARKER] TADR-aware retrieval shortcuts (skip RRF + external_sources)
        _tadr_phase3_skip_extras = False
        _tadr_phase3_sentinel_scope = None
        try:
            import os as _tadr_os
            _tadr_p3_enabled = _tadr_os.environ.get('TADR_PHASE3_ENABLED', 'true').strip().lower() in ('1','true','yes','on')
            _tadr_p3_threshold = float(_tadr_os.environ.get('TADR_PHASE3_HIGH_CONF', '0.90'))
            _trr = locals().get('_tadr_router_result')
            if (_tadr_p3_enabled and _trr is not None and not _trr.is_noop
                and _trr.primary_confidence >= _tadr_p3_threshold
                and _trr.primary[0] not in ('unknown', 'G')):
                _tadr_phase3_skip_extras = True
                _tadr_phase3_sentinel_scope = {
                    '_tadr_high_conf': True,
                    '_tadr_trade': _trr.primary[0],
                    '_tadr_role': _trr.primary[1],
                }
                logger.info('[tadr-p3] skipping RRF + external_sources (trade=%s role=%s conf=%.2f)',
                            _trr.primary[0], _trr.primary[1], _trr.primary_confidence)
        except Exception as _tadr_p3_exc:
            logger.warning('[tadr-p3] gate computation failed: %s', _tadr_p3_exc)

        # [COGUP-MARKER-TADR-SCOPE-FIX] Wire TADR sentinel scope as rrf_hint to bias agent retrieval
        _tadr_scope = None
        try:
            import os as _os_ts
            if (_os_ts.getenv('TADR_SCOPE_HINT_ENABLED', 'true').lower() == 'true'
                    and _collection_scope != 'specification'):  # [V19-TADR-SKIP]
                _trr_local = locals().get('_tadr_router_result')
                if _trr_local is not None and not _trr_local.is_noop:
                    _t = _trr_local.primary[0] if _trr_local.primary else 'unknown'
                    _ro = _trr_local.primary[1] if _trr_local.primary else 'unknown'
                    _conf = float(_trr_local.primary_confidence or 0.0)
                    if _conf >= 0.60 and _t not in ('unknown', 'G'):
                        _trade_names = {'G':'General/Cover','C':'Civil','L':'Landscape',
                                        'AS':'Architectural Site','A':'Architectural','ID':'Interiors',
                                        'S':'Structural','M':'Mechanical (HVAC)','P':'Plumbing',
                                        'E':'Electrical','FA':'Fire Alarm','FP':'Fire Protection',
                                        'T':'Telecom/Low-Voltage','V':'Vertical Transportation',
                                        'K':'Kitchen/Food Service','PE':'Process/Specialty Equipment'}
                        _role_names = {'Plan':'Plan','Section':'Section','Elevation':'Elevation',
                                       'Detail':'Detail','Schedule':'Schedule','Diagram':'Diagram',
                                       'Notes':'Notes','RCP':'Reflected Ceiling Plan (RCP)'}
                        _tn = _trade_names.get(_t, _t)
                        _rn = _role_names.get(_ro, _ro)
                        # Build hint: authoritative-drawing focus for the agent
                        # [COGUP-MARKER-TADR-SCOPE-FIX-V2] strengthen hint into HARD constraint
                        _hint_lines = [
                            f'### TADR AUTHORITATIVE-DRAWING CONSTRAINT (confidence {_conf:.2f}) ###',
                            f'',
                            f'You MUST cite a drawing whose sheetNumber starts with the {_t}- prefix',
                            f'(i.e. {_tn} trade) AND whose drawingTitle indicates it is a {_rn}.',
                            f'',
                            f'**DO NOT cite ANY of these for this question, even if their text mentions',
                            f'the topic keywords**:',
                            f'- Life Safety Plans (A-021, A-0##, *LIFE SAFETY*)',
                            f'- Site Plans (A-001, *SITE PLAN*)',
                            f'- Cover Sheets / Index (G-###, CS-###)',
                            f'- Electrical drawings (E-, EL-, EE-) unless the question is electrical',
                            f'- Fire Protection (FP-) unless the question is fire protection',
                            f'- Interiors (I-, ID-) furnishings unless the question is interior finishes',
                            f'',
                            f'**Required sequence**:',
                            f'1. Call legacy_list_drawings OR list_drawings_v3 first to ENUMERATE',
                            f'   drawings matching the trade+role constraint above.',
                            f'2. From the candidates, pick the one whose drawingTitle best matches',
                            f'   the {_rn} role. If multiple match, prefer the more specific scope',
                            f'   (typical corridor → corridor RCP, not full-floor RCP).',
                            f'3. Call v3_get_drawing_full_text or legacy_get_text on THAT sheet to',
                            f'   extract the answer value.',
                            f'',
                            f'# [V15-SOFT-FALLBACK]',
                            f'If NO drawing matches the trade+role constraint after enumeration,',
                            f'FALL BACK to your normal search strategy across ALL project drawings',
                            f'(any trade or role) and answer from the best available evidence.',
                            f'When you do fall back, state in the answer which drawing class the',
                            f'value came from (e.g. "per the interior finish plan, not a dedicated',
                            f'{_rn} drawing"). Only refuse if you find no relevant content anywhere',
                            f'in the project after the fallback search.',
                        ]
                        _tadr_scope = {
                            'rrf_hint': '\n'.join(_hint_lines),
                            '_tadr_trade': _t,
                            '_tadr_role': _ro,
                            '_tadr_confidence': _conf,
                            '_original_query': body.query,  # [COGUP-MARKER-V8-ORIG-QUERY] for V8 ordinal expansion
                        }
                        logger.info('[tadr-scope] hint scope built: trade=%s role=%s conf=%.2f', _t, _ro, _conf)
        except Exception as _tadr_scope_exc:
            logger.warning('[tadr-scope] scope build failed: %s', _tadr_scope_exc)

        coros["rag"] = orchestrator.query(
            query=_hg_resolved_query,
            project_id=body.project_id,
            engine=body.engine,
            session_id=resolved_session_id,
            user_id=body.user_id,
            client_session_id=effective_client_session_id,
            set_id=body.set_id,
            conversation_history=body.conversation_history,
            search_mode=orchestrator_search_mode,
            generate_document=body.generate_document,
            filter_source_type=body.filter_source_type,
            filter_drawing_name=body.filter_drawing_name,
            docqa_document=body.docqa_document,
            mode_hint=body.mode_hint,
            skip_multi_query_rrf=_tadr_phase3_skip_extras,
            scope=_v19_request_scope(_tadr_scope, _collection_scope),  # [V19-SCOPE-MERGE]
        )
    if not _tadr_phase3_skip_extras:
        if "email" in modes:
            coros["email"] = search_emails(body.query, project_id_str, session_id=resolved_session_id)
        if "meeting" in modes:
            coros["meeting"] = search_meetings(body.query, project_id_str)
        if "rfi" in modes:
            coros["rfi"] = search_rfis(body.query, project_id_str)

    keys = list(coros.keys())
    results_list = await asyncio.gather(*coros.values(), return_exceptions=True)
    results: dict[str, Any] = dict(zip(keys, results_list))

    rag_result: dict[str, Any] = results.get("rag", {})
    if isinstance(rag_result, Exception):
        logger.error("RAG query raised: %s", rag_result)
        rag_result = {"success": False, "answer": "", "sources": [], "error": str(rag_result)}
    elif not rag_result:
        rag_result = {}

    source_answers: dict[str, str] = {}
    rag_answer = (rag_result.get("answer") or "").strip()
    if rag_answer and "rag" in results and not isinstance(results["rag"], Exception):
        source_answers["rag"] = rag_answer

    # Meeting (always set key, default None)
    rag_result.setdefault("meeting_answer", None)
    meeting_result = results.get("meeting")
    if meeting_result and not isinstance(meeting_result, Exception):
        m_answer = (meeting_result.get("answer") or "").strip()
        rag_result["meeting_answer"] = m_answer or None
        if m_answer:
            source_answers["meeting"] = m_answer
        rag_result.setdefault("sources", []).extend(
            {"source_type": "meeting", **s} for s in meeting_result.get("sources", [])
        )
    elif isinstance(meeting_result, Exception):
        logger.warning("Meeting search raised: %s", meeting_result)

    # Email (always set key, default None)
    rag_result.setdefault("email_answer", None)
    email_result = results.get("email")
    if email_result and not isinstance(email_result, Exception):
        e_answer = (email_result.get("answer") or "").strip()
        rag_result["email_answer"] = e_answer or None
        if e_answer:
            source_answers["email"] = e_answer
        rag_result.setdefault("sources", []).extend(
            email_result.get("sources", [])
        )
        # Persist email Q&A to session — mirrors RAG's MemoryManager persistence.
        # After gather (serial), so no race with orchestrator session writes.
        if e_answer:
            try:
                from traditional.memory_manager import get_memory_manager, estimate_tokens
                _mm = get_memory_manager()
                _session = _mm.get_session(resolved_session_id)
                if not _session:
                    # Email-only mode: orchestrator did not run, create session here.
                    _mm.create_session(
                        user_query=body.query,
                        project_id=body.project_id,
                        session_id=resolved_session_id,
                    )
                    _mm.add_to_session(
                        resolved_session_id, "user", body.query,
                        tokens=estimate_tokens(body.query),
                        metadata={"search_mode": "email", "project_id": body.project_id},
                    )
                _mm.add_to_session(
                    resolved_session_id, "assistant", e_answer,
                    tokens=estimate_tokens(e_answer),
                    metadata={"source": "email", "project_id": body.project_id},
                )
            except Exception as _exc:
                logger.warning("Email session save failed: %s", _exc)
    elif isinstance(email_result, Exception):
        logger.warning("Email search raised: %s", email_result)

    rfi_result = results.get("rfi")
    if rfi_result and not isinstance(rfi_result, Exception):
        r_answer = (rfi_result.get("answer") or "").strip()
        rag_result["rfi_answer"] = r_answer or None
        if r_answer:
            source_answers["rfi"] = r_answer
        rag_result.setdefault("sources", []).extend(
            {"source_type": "rfi", **s} for s in rfi_result.get("sources", [])
        )
    elif isinstance(rfi_result, Exception):
        logger.warning("RFI search raised: %s", rfi_result)

    # Determine final_answer � single source returned directly, multi-source synthesized
    if not source_answers:
        final_answer = rag_result.get("answer", "")
    elif len(source_answers) == 1:
        final_answer = next(iter(source_answers.values()))
    else:
        synth_model = "gpt-4.1"
        try:
            from shared.config import get_config
            synth_model = get_config().agentic_model
        except Exception:
            pass
        final_answer = await synthesize_answer(body.query, source_answers, model=synth_model)

    rag_result["final_answer"] = final_answer
    rag_result["sources_used"] = sorted(source_answers.keys()) if source_answers else sorted(modes)

    await _push_call_audit(
        user_id=body.user_id,
        project_id=body.project_id,
        client_session_id=effective_client_session_id,
        query_text=body.query,
        result=rag_result,
    )

    # v3.4 — ChatGPT-style sidebar manifest sync (1 row per chat session)
    # v3.4.2 — use effective_client_session_id (auto-generated above if body
    # omitted both session_id and client_session_id) so the manifest writes
    # every time regardless of whether the UI minted a handle.
    await _register_or_refresh_session_manifest(
        user_id=body.user_id,
        project_id=body.project_id,
        client_session_id=effective_client_session_id,
        conversation_session_id=rag_result.get('session_id'),
        query_text=body.query,
        answer_text=rag_result.get('final_answer') or rag_result.get('answer') or '',
    )

    # Per-turn Mongo row (slim) + presigned S3 URL — powers /list and /history.
    try:
        from shared.session_tracker import push_session_turn, resolve_session_title
        _turn_id = rag_result.get("turn_id")
        if _turn_id and body.user_id is not None:
            _title = await asyncio.to_thread(
                resolve_session_title,
                session_id=effective_client_session_id,
                question=body.query,
            )
            await asyncio.to_thread(
                push_session_turn,
                user_id=body.user_id,
                project_id=body.project_id,
                agent_id="drawing-agent",
                session_id=effective_client_session_id,
                turn_id=_turn_id,
                title=_title,
                question=body.query,
                answer=rag_result.get("final_answer") or rag_result.get("answer") or "",
                follow_up_questions=rag_result.get("follow_up_questions") or [],
                source_documents=rag_result.get("source_documents") or [],
            )
    except Exception as _e:
        logger.warning("[session-turn] /query write skipped: %s: %s", type(_e).__name__, _e)

    # v3.4.3 (2026-05-26) - record turns to MemoryManager for deep-dive parent lookup
    # The v3.1 agentic chain bypasses MemoryManager.add_to_session, leaving
    # session.messages empty. Deep-dive's _load_parent_turn() reads ONLY from
    # MemoryManager, so without this it always returns parent_turn_not_found.
    # Here we record both the user query and the assistant answer so the
    # in-process cache stays consistent with what the user actually saw.
    # NOTE: in-process only (single-worker uvicorn) — durable persistence
    # would require also recording to the manifest's messages[].metadata.
    try:
        _mem_session_id = rag_result.get("session_id") or resolved_session_id
        _mem_turn_id    = rag_result.get("turn_id")
        if _mem_session_id and _mem_turn_id:
            from traditional.memory_manager import get_memory_manager  # type: ignore
            _mm = get_memory_manager()
            # Ensure the session exists in MemoryManager. create_session is
            # idempotent on session_id collision when explicit id is passed.
            if _mm.get_session(_mem_session_id) is None:
                _mm.create_session(
                    user_query=body.query,
                    project_id=body.project_id,
                    session_id=_mem_session_id,
                    user_id=body.user_id,
                    client_session_id=effective_client_session_id,
                )
            _meta_user = {
                "turn_id":    _mem_turn_id,
                "project_id": body.project_id,
                "set_id":     body.set_id,
            }
            _meta_assist = {
                "turn_id":           _mem_turn_id,
                "project_id":        body.project_id,
                "set_id":            body.set_id,
                "source_documents":  rag_result.get("source_documents") or [],
                "engine_used":       rag_result.get("engine_used"),
            }
            _mm.add_to_session(_mem_session_id, role="user", content=body.query, metadata=_meta_user)
            _mm.add_to_session(
                _mem_session_id, role="assistant",
                content=rag_result.get("final_answer") or rag_result.get("answer") or "",
                metadata=_meta_assist,
            )
    except Exception as _exc:  # noqa: BLE001
        logger.warning("[session] memory record failed (deep-dive may miss this turn): %s: %s",
                       type(_exc).__name__, _exc)
					   
    rag_result.setdefault("session_id", resolved_session_id)
    # v32 observe-only guardrails (additive verification_meta; flag-gated,
    # default OFF). Does NOT change the answer, retrieval, or latency materially.
    import os as _os
    if _os.getenv("ENABLE_V32_GUARDRAILS", "false").lower() == "true":
        try:
            from gateway.v32_guardrails import compute as _v32_compute
            rag_result["verification_meta"] = _v32_compute(
                body.query,
                rag_result.get("final_answer") or rag_result.get("answer") or "",
                rag_result.get("source_documents") or [],
            )
            # [COGUP-MARKER-V13-CORRID-P11] Phase 0.2 correlation_id + Phase 1.1 Pillar 11 record
            try:
                _tp = request.headers.get("traceparent") if request else None
                from shared.correlation import correlation_id as _corr_id
                rag_result["verification_meta"]["correlation_id"] = _corr_id(_tp)
            except Exception:
                pass
            try:
                _p11 = {
                    "action": getattr(_hg_decision, "action", "allow"),
                    "reason_class": getattr(_hg_decision, "reason_class", None),
                    "matched_patterns": list(getattr(_hg_decision, "matched_patterns", []) or []),
                    "addendum_present": bool(getattr(_hg_decision, "prompt_addendum", None)),
                    "regulatory_reframe": bool(getattr(_hg_decision, "regulatory_reframe_detected", False)),
                    "pillar_applied": bool(getattr(_hg_decision, "pillar_applied", False)),
                }
                rag_result["verification_meta"]["pillar_11"] = _p11
            except Exception:
                pass
        except Exception:
            pass  # best-effort -- never break the response
    return rag_result


def _resolve_session_from_client_id(
    user_id, project_id, client_session_id,
) -> str | None:
    """Look up the conversation_session_id for an existing chat session by
    its UI-provided client_session_id. Reads the S3 manifest. None if no
    prior session.
    """
    if user_id is None or project_id is None or not client_session_id:
        return None
    try:
        from shared.session_tracker import (
            _read_s3_json, _session_manifest_key, _s3_bucket,
        )
        bucket = _s3_bucket()
        key = _session_manifest_key(int(user_id), int(project_id),
                                    "drawing-agent", client_session_id)
        m = _read_s3_json(bucket=bucket, key=key)
        if m and m.get("conversation_session_id"):
            return m["conversation_session_id"]
    except Exception as exc:
        logger.warning(
            "[session-resolve] failed for cid=%s: %s: %s",
            client_session_id, type(exc).__name__, exc,
        )
    return None


async def _push_call_audit(
    *,
    user_id: Optional[int],
    project_id: Optional[int],
    client_session_id: Optional[str],
    query_text: str,
    result: dict,
) -> None:
    """Best-effort: push one call record to the iField userSession API."""
    if user_id is None or project_id is None or not client_session_id:
        return
    # [COGUP-MARKER-V13-PII-AUDIT] Phase 0.4 - PII redaction at response_audit boundary
    _q_audit = query_text
    _r_audit = result.get("answer", "")
    try:
        import os as _os_p4
        if _os_p4.getenv("PII_REDACTION_ENABLED", "true").lower() == "true":
            from shared.pii_boundaries import redact_for_boundary as _redact
            _q_audit = _redact(_q_audit, boundary="response_audit")
            _r_audit = _redact(_r_audit, boundary="response_audit")
    except Exception:
        pass
    try:
        from shared.session_tracker import push_session_call
        payload = {
            "request":      {"query": _q_audit},
            "response":     _r_audit,
            "run_id":       result.get("session_id"),
            "cost_usd":     result.get("cost_usd"),
            "elapsed_ms":   result.get("elapsed_ms"),
            "engine_used":  result.get("engine_used"),
            "fallback":     result.get("fallback_used"),
            "sources":      result.get("sources", [])[:5],
        }
        await asyncio.to_thread(
            push_session_call,
            user_id=int(user_id),
            project_id=int(project_id),
            agent_id="drawing-agent",
            client_session_id=client_session_id,
            payload=payload,
        )
    except Exception as exc:
        logger.warning("[session] push_session_call failed: %s: %s", type(exc).__name__, exc)


async def _register_or_refresh_session_manifest(
    *,
    user_id,
    project_id,
    client_session_id,
    conversation_session_id,
    query_text: str,
    answer_text: str,
) -> None:
    """v3.4 — keep the user-scoped sidebar manifest in sync.

    Turn 1: write new manifest + POST iField (creates Mongo row).
    Turn N>1: overwrite manifest only (no Mongo POST so we don't dup rows).
    """
    if user_id is None or project_id is None or not client_session_id:
        return
    if not conversation_session_id:
        return
    try:
        from shared.session_tracker import (
            push_session_register,
            refresh_session_manifest,
            _read_s3_json,
            _session_manifest_key,
            _s3_bucket,
        )
        from datetime import datetime, timezone
        import asyncio as _aio

        # Read prior messages from the manifest itself — the v3.1 chain
        # bypasses MemoryManager.add_to_session so we cannot rely on it.
        bucket = _s3_bucket()
        manifest_key = _session_manifest_key(
            int(user_id), int(project_id), "drawing-agent", client_session_id,
        )
        existing = await _aio.to_thread(_read_s3_json, bucket=bucket, key=manifest_key)
        prior_messages = (existing or {}).get("messages") or []

        # Append the current turn (user + assistant) — manifest mirrors
        # the full chat history.
        now_iso = datetime.now(timezone.utc).isoformat()
        new_messages = list(prior_messages) + [
            {"role": "user", "content": query_text, "timestamp": now_iso},
            {"role": "assistant", "content": answer_text, "timestamp": now_iso},
        ]

        is_first_turn = not prior_messages

        if is_first_turn:
            await _aio.to_thread(
                push_session_register,
                user_id=int(user_id),
                project_id=int(project_id),
                agent_id="drawing-agent",
                client_session_id=client_session_id,
                conversation_session_id=conversation_session_id,
                title=query_text,
                messages=new_messages,
            )
        else:
            await _aio.to_thread(
                refresh_session_manifest,
                user_id=int(user_id),
                project_id=int(project_id),
                agent_id="drawing-agent",
                client_session_id=client_session_id,
                messages=new_messages,
            )
    except Exception as exc:
        import logging as _logging
        _logging.getLogger("agentic_rag.session_manifest").warning(
            "manifest sync failed (non-fatal): %s: %s",
            type(exc).__name__, exc,
        )



@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    # --- Pillar 11 pre-filter (Hallucination Guard v2) ---
    _hg_has_history = bool(body.conversation_history)
    _hg_decision = _hg_p11_filter(body.query, has_history=_hg_has_history)
    if _hg_decision.action == "refuse_injection":
        logger.info(
            "[hg-p11] /query/stream refused -- reason=%s patterns=%d",
            _hg_decision.reason_class, len(_hg_decision.matched_patterns),
        )
        _hg_refusal_dict = _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)
        async def _hg_refusal_stream():
            yield _sse("status", {"phase": "refused"})
            yield _sse("token", {"delta": _hg_decision.refusal_text})
            yield _sse("done", _hg_refusal_dict)
        return StreamingResponse(_hg_refusal_stream(), media_type="text/event-stream")

    # --- Pillar 1 anaphora resolver (Hallucination Guard v2) ---
    _hg_resolved_query_stream = _hg_p1_resolve(body.query, history=body.conversation_history or [])
    """SSE streaming query — word-by-word token output.

    Protocol (shared with /deep-dive/stream):
      event: status   data: {"phase": ...}
      event: token    data: {"delta": "<chunk>"}     (word-by-word)
      event: done     data: <full /query response dict>
      data: [DONE]

    Computes the answer via the SAME non-blocking path as /query
    (orchestrator.query offloads all heavy work via asyncio.to_thread), then
    streams it word-by-word. This keeps full synthesizer/stylist quality and
    never blocks the event loop. End-of-stream + disconnect audit preserved.
    """

    async def event_generator() -> Any:
        orchestrator = _get_orchestrator(request)
        final_result: dict[str, Any] = {}
        stream_error: str | None = None
        push_done = False
        try:
            yield _sse("status", {"phase": "received"})
            try:
                # Non-blocking: the entire query runs in a worker thread so the
                # server event loop never stalls under concurrent streams.
                # [STREAM-PROGRESS-MARKER] Real-time progress events from agent loop
                import queue as _q_mod
                _progress_q = _q_mod.Queue()
                def _progress_cb(event_type, data):
                    try:
                        _progress_q.put_nowait((event_type, data))
                    except Exception:
                        pass

                # Kick off the agent in the background
                _agent_task = asyncio.create_task(_run_query_offloop(
                    orchestrator,
                    query=_hg_resolved_query_stream,
                    project_id=body.project_id,
                    user_id=body.user_id,
                    client_session_id=body.client_session_id,
                    session_id=body.session_id,
                    set_id=body.set_id,
                    engine=body.engine,
                    conversation_history=body.conversation_history,
                    search_mode=body.search_mode,
                    docqa_document=body.docqa_document,
                    mode_hint=body.mode_hint,
                    progress_callback=_progress_cb,
                ))

                # Drain progress events + time-based heartbeats while agent runs
                _progress_events_enabled = (
                    __import__("os").environ.get("STREAM_PROGRESS_EVENTS_ENABLED", "true").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                import time as _hb_time
                _hb_start = _hb_time.monotonic()
                _hb_milestones_emitted = set()
                _hb_milestones = [
                    (2.0,  {"phase": "retrieving", "message": "Searching documents…"}),
                    (10.0, {"phase": "analyzing",  "message": "Analyzing retrieved content…"}),
                    (30.0, {"phase": "thinking",   "message": "Still working — composing answer…"}),
                    (60.0, {"phase": "synthesizing", "message": "Almost done…"}),
                ]
                while not _agent_task.done():
                    if not _progress_events_enabled:
                        await asyncio.sleep(0.3)
                        continue
                    # Emit time-based heartbeats
                    _hb_elapsed = _hb_time.monotonic() - _hb_start
                    for _hb_t, _hb_data in _hb_milestones:
                        if _hb_elapsed >= _hb_t and _hb_t not in _hb_milestones_emitted:
                            _hb_milestones_emitted.add(_hb_t)
                            yield _sse("status", {**_hb_data, "elapsed_s": int(_hb_elapsed)})
                    try:
                        event_type, data = await asyncio.to_thread(_progress_q.get, True, 0.5)
                        yield _sse(event_type, data)
                    except _q_mod.Empty:
                        continue

                # Drain any remaining events emitted late
                if _progress_events_enabled:
                    while True:
                        try:
                            event_type, data = _progress_q.get_nowait()
                            yield _sse(event_type, data)
                        except _q_mod.Empty:
                            break

                final_result = _agent_task.result()
                if not isinstance(final_result, dict):
                    final_result = {}

                yield _sse("status", {
                    "phase": "generating",
                    "session_id": final_result.get("session_id"),
                    "engine_used": final_result.get("engine_used"),
                })
                # Stream the finished answer word-by-word with a real per-word
                # delay so each frame flushes to the client incrementally
                # (sleep(0) made them arrive as one burst at the end).
                answer = final_result.get("answer") or ""
                chunks = _iter_word_chunks(answer)
                delay = _typewriter_delay(len(chunks))
                for chunk in chunks:
                    yield _sse("token", {"delta": chunk})
                    await asyncio.sleep(delay)
                yield _sse("done", final_result)
            except Exception as exc:
                stream_error = str(exc)
                logger.exception("query_stream failed: %s", exc)
                yield _sse("error", {"error": "stream_failed", "message": str(exc)})
            # End-of-stream audit push (best-effort).
            await _push_call_audit(
                user_id=body.user_id,
                project_id=body.project_id,
                client_session_id=body.client_session_id,
                query_text=body.query,
                result={**final_result, "stream_error": stream_error} if stream_error else final_result,
            )
            # Per-turn Mongo row (slim) + presigned S3 URL — powers /list and /history.
            try:
                if not stream_error and isinstance(final_result, dict):
                    from shared.session_tracker import push_session_turn, resolve_session_title
                    _turn_id = final_result.get("turn_id")
                    _sess = body.client_session_id or body.session_id or final_result.get("session_id")
                    if _turn_id and _sess and body.user_id is not None:
                        _title = await asyncio.to_thread(
                            resolve_session_title, session_id=_sess, question=body.query)
                        await asyncio.to_thread(
                            push_session_turn,
                            user_id=body.user_id,
                            project_id=body.project_id,
                            agent_id="drawing-agent",
                            session_id=_sess,
                            turn_id=_turn_id,
                            title=_title,
                            question=body.query,
                            answer=final_result.get("final_answer") or final_result.get("answer") or "",
                            follow_up_questions=final_result.get("follow_up_questions") or [],
                            source_documents=final_result.get("source_documents") or [],
                        )
            except Exception as _e:
                logger.warning("[session-turn] /query/stream write skipped: %s: %s", type(_e).__name__, _e)
            push_done = True
            yield "data: [DONE]\n\n"
        finally:
            # Guarantee push fires even if the client disconnected mid-stream.
            if not push_done:
                try:
                    await _push_call_audit(
                        user_id=body.user_id,
                        project_id=body.project_id,
                        client_session_id=body.client_session_id,
                        query_text=body.query,
                        result={
                            **final_result,
                            "stream_error": stream_error or "client disconnect",
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "[session] post-disconnect push failed: %s: %s",
                        type(exc).__name__, exc,
                    )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class FeedbackStreamRequest(BaseModel):
    """Body for ``POST /sessions/{session_id}/messages/{turn_id}/feedback/stream``."""

    user_id: str = Field(..., description="UI-supplied user identifier (logged)")
    project_id: int = Field(..., ge=1, le=999999, description="Project the turn belongs to")
    vote: Optional[str] = Field(None, pattern="^(up|down|neutral)$", description="up | down | neutral")
    category: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "Feedback reason(s) — one value or a list. Codes: wrong_facts | "
            "incomplete | irrelevant | hallucination | formatting. Human labels "
            "(e.g. 'Inaccurate information') accepted too. Steers regeneration."
        ),
    )
    comment: Optional[str] = Field(None, max_length=2000, description="Optional free-text feedback")
    regenerate: Optional[bool] = Field(
        None,
        description=(
            "Force regeneration on/off. Defaults to True when vote='down', a "
            "category, or a comment is provided; otherwise False (ack only)."
        ),
    )
    set_id: Optional[int] = Field(None, description="Drawing-set scope (resolved from the turn if absent)")


@router.post("/sessions/{session_id}/messages/{turn_id}/feedback/stream")
async def feedback_stream(
    session_id: str,
    turn_id: str,
    body: FeedbackStreamRequest,
    request: Request,
) -> StreamingResponse:
    """Submit feedback on a prior answer and stream the response (word-by-word).

    Protocol identical to /query/stream. Records the vote/category/comment,
    then either acknowledges (vote=up, no detail) or regenerates an improved
    answer via the non-blocking orchestrator.query path, streamed word-by-word.
    New turn linked via parent_turn_id=<turn_id>, turn_type='feedback'.
    """
    orchestrator = _get_orchestrator(request)

    async def event_generator() -> Any:
        yield _sse("status", {"phase": "received", "session_id": session_id, "turn_id": turn_id})

        parent = await asyncio.to_thread(_load_turn, session_id, turn_id)
        if parent is None:
            yield _sse("error", {
                "error": "turn_not_found",
                "message": f"No turn '{turn_id}' found in session '{session_id}'.",
            })
            yield "data: [DONE]\n\n"
            return

        categories = _normalize_categories(body.category)
        vote = body.vote or ("down" if (categories or body.comment) else "neutral")

        await asyncio.to_thread(
            _persist_feedback, session_id, turn_id, body.user_id, vote, body.comment, categories,
        )
        yield _sse("status", {"phase": "feedback_recorded", "vote": vote, "categories": categories})

        regen = body.regenerate
        if regen is None:
            regen = vote == "down" or bool(categories) or bool(body.comment)

        if not regen:
            ack = "Thanks for the feedback — it's been recorded."
            for word in ack.split(" "):
                yield _sse("token", {"delta": word + " "})
                await asyncio.sleep(0.03)
            yield _sse("done", {
                "query": parent["user_text"],
                "answer": ack,
                "turn_type": "feedback",
                "parent_turn_id": turn_id,
                "session_id": session_id,
                "project_id": body.project_id,
                "vote": vote,
                "categories": categories,
                "category": categories[0] if categories else None,
                "regenerated": False,
                "success": True,
            })
            yield "data: [DONE]\n\n"
            return

        fb = (body.comment or "").strip()
        guidance = " ".join(_FEEDBACK_CATEGORIES[c] for c in categories)
        reasons = ", ".join(f"'{c}'" for c in categories)
        instruction = (
            f"{parent['user_text']}\n\n"
            f"[The previous answer was rated '{vote}'"
            + (f" with reasons: {reasons}." if categories else ".")
            + (f" {guidance}" if guidance else "")
            + (f" Additional user feedback: {fb}." if fb else "")
            + " Produce an improved answer that directly addresses this feedback.]"
        )
        conv = [
            {"role": "user", "content": parent["user_text"]},
            {"role": "assistant", "content": parent["assistant_text"]},
        ]
        project_id = parent.get("project_id") or body.project_id
        set_id = body.set_id if body.set_id is not None else parent.get("set_id")

        try:
            final_result = await _run_query_offloop(
                orchestrator,
                query=instruction,
                project_id=project_id,
                session_id=session_id,
                set_id=set_id,
                conversation_history=conv,
                search_mode="rag",
            )
            answer = final_result.get("answer") or ""
            chunks = _iter_word_chunks(answer)
            delay = _typewriter_delay(len(chunks))
            for chunk in chunks:
                yield _sse("token", {"delta": chunk})
                await asyncio.sleep(delay)
            final_result["turn_type"] = "feedback"
            final_result["parent_turn_id"] = turn_id
            final_result["regenerated"] = True
            final_result["vote"] = vote
            final_result["categories"] = categories
            final_result["category"] = categories[0] if categories else None
            yield _sse("done", final_result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("feedback_stream regeneration failed: %s", exc)
            yield _sse("error", {"error": "regeneration_failed", "message": str(exc)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/quick-query")
async def quick_query(request: Request, body: QueryRequest) -> dict:
    """Simplified query — returns only answer + sources + confidence."""
    orchestrator = _get_orchestrator(request)
    result = await orchestrator.query(
        query=body.query,
        project_id=body.project_id,
        engine=body.engine,
        session_id=body.session_id,
        set_id=body.set_id,
        scope=_v19_request_scope(None, _parse_collection_scope(body.search_mode)),  # [V19-STREAM-SCOPE]
    )
    await _push_call_audit(
        user_id=body.user_id,
        project_id=body.project_id,
        client_session_id=body.client_session_id,
        query_text=body.query,
        result=result,
    )
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", "low"),
        "engine_used": result.get("engine_used", "unknown"),
    }


@router.post("/web-search")
async def web_search(request: Request, body: QueryRequest) -> dict:
    """Delegate to traditional engine's web search capability."""
    try:
        from traditional.rag.api.generation_unified import generate_response  # type: ignore[import-untyped]
        result = await asyncio.to_thread(
            generate_response,
            query=body.query,
            project_id=body.project_id,
            session_id=body.session_id,
            search_mode="web",
        )
        await _push_call_audit(
            user_id=body.user_id,
            project_id=body.project_id,
            client_session_id=body.client_session_id,
            query_text=body.query,
            result={"engine_used": "web-search", **(result if isinstance(result, dict) else {"raw": str(result)[:1000]})},
        )
        return {"success": True, "result": result}
    except ImportError:
        await _push_call_audit(
            user_id=body.user_id,
            project_id=body.project_id,
            client_session_id=body.client_session_id,
            query_text=body.query,
            result={"engine_used": "web-search", "error": "traditional engine not available"},
        )
        return {"success": False, "error": "Traditional engine not available for web search"}
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        await _push_call_audit(
            user_id=body.user_id,
            project_id=body.project_id,
            client_session_id=body.client_session_id,
            query_text=body.query,
            result={"engine_used": "web-search", "error": str(exc)},
        )
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Session Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/create")
async def create_session(request: Request, body: dict = {}) -> dict:
    """Create a new session.

    Optional body fields ``user_id`` and ``project_id`` trigger a best-effort
    ``push_session_start`` to the iField userSession API so this session is
    audit-logged in Mongo from the moment it begins.
    """
    user_id = body.get("user_id")
    project_id = body.get("project_id")
    client_session_id = body.get("client_session_id")

    try:
        from traditional.memory_manager import MemoryManager  # type: ignore[import-untyped]
        mm = MemoryManager()
        session_id = mm.create_session(project_id=project_id)
        result = {"success": True, "session_id": session_id}
    except ImportError:
        session_id = str(uuid.uuid4())
        logger.warning("MemoryManager not available, returning stub session")
        result = {"success": True, "session_id": session_id, "stub": True}
    except Exception as exc:
        logger.error("Session creation failed: %s", exc)
        return {"success": False, "error": str(exc)}

    if user_id is not None and project_id is not None:
        try:
            from shared.session_tracker import push_session_start
            push_result = await asyncio.to_thread(
                push_session_start,
                user_id=int(user_id),
                project_id=int(project_id),
                agent_id="drawing-agent",
                client_session_id=session_id,
                payload={"local_session_id": session_id},
            )
            result["client_session_id"] = push_result["client_session_id"]
            result["ifield_session_id"] = push_result.get("ifield_session_id")
        except Exception as exc:
            logger.warning("[session] push_session_start failed: %s: %s", type(exc).__name__, exc)
    return result


# ── userSession audit-log endpoints (Mongo + S3) ─────────────────────────


@router.get("/sessions/list")
async def list_audit_sessions(
    request: Request,
    user_id: int,
    project_id: int,
) -> dict:
    """List drawing-agent sessions for one user+project.

    Scope is enforced server-side: the lookup is filtered by ``user_id``
    against the iField userSession API, so a caller can only see their
    own sessions for this agent. Pass through ``data`` shape from Mongo.
    """
    try:
        from shared.session_tracker import list_sessions as _list
        return await asyncio.to_thread(
            _list,
            user_id=int(user_id),
            project_id=int(project_id),
            agent_id="drawing-agent",
        )
    except Exception as exc:
        logger.warning("[session] list failed: %s: %s", type(exc).__name__, exc)
        return {"success": False, "error": str(exc), "data": []}


@router.get("/sessions/detail/{session_id}")
async def get_audit_session_detail(
    request: Request,
    session_id: str,
    user_id: int,
    project_id: int,
) -> dict:
    """Return the full turn history for one session via /userSession/history.

    Returns ``{"success": bool, "data": [<turn>, ...], "count": int}``.
    ``user_id``/``project_id`` are forwarded as ``userId``/``projectId``
    query params to scope the history lookup to the caller's own sessions.
    """
    try:
        from shared.session_tracker import get_session_history
        out = await asyncio.to_thread(
            get_session_history,
            session_id=session_id,
            user_id=int(user_id),
            project_id=int(project_id),
        )
        if not out.get("success"):
            return {"success": False, "error": out.get("error", "lookup failed"), "data": []}
        return out
    except Exception as exc:
        logger.warning("[session] detail failed: %s: %s", type(exc).__name__, exc)
        return {"success": False, "error": str(exc), "data": []}


@router.get("/sessions")
async def list_sessions(
    request: Request,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
) -> dict:
    """List this user+project's drawing-agent sessions.

    Sessions are tracked in the iField userSession store (not the in-process
    MemoryManager, which has no enumerate method), so listing REQUIRES a
    user_id + project_id scope. When either is missing we return an empty
    list with a hint rather than erroring — this mirrors /sessions/list and
    keeps old no-arg callers working.
    """
    if user_id is None or project_id is None:
        return {
            "success": True,
            "sessions": [],
            "note": "pass user_id and project_id query params to list sessions",
        }
    try:
        from shared.session_tracker import list_sessions as _list  # type: ignore
        out = await asyncio.to_thread(
            _list,
            user_id=int(user_id),
            project_id=int(project_id),
            agent_id="drawing-agent",
        )
        # Map session_tracker's {"data": [...]} to this endpoint's {"sessions": [...]}.
        return {
            "success": bool(out.get("success", False)),
            "sessions": out.get("data", []),
            "count": out.get("count", len(out.get("data", []))),
            "error": out.get("error"),
        }
    except Exception as exc:
        logger.error("List sessions failed: %s", exc)
        return {"success": False, "error": str(exc), "sessions": []}


@router.get("/sessions/{session_id}/stats")
async def session_stats(request: Request, session_id: str) -> dict:
    """Get session stats including engine usage."""
    try:
        from shared.session.manager import get_session_stats_extended
        stats = get_session_stats_extended(session_id)
        return {"success": True, **stats}
    except ImportError:
        return {"success": True, "session_id": session_id, "stub": True}
    except Exception as exc:
        logger.error("Session stats failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.get("/sessions/{session_id}/conversation")
async def session_conversation(request: Request, session_id: str) -> dict:
    """Get conversation history for a session."""
    try:
        from traditional.memory_manager import MemoryManager  # type: ignore[import-untyped]
        mm = MemoryManager()
        history = mm.get_conversation(session_id)
        return {"success": True, "session_id": session_id, "conversation": history}
    except ImportError:
        return {"success": True, "session_id": session_id, "conversation": [], "stub": True}
    except Exception as exc:
        logger.error("Get conversation failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.post("/sessions/{session_id}/update")
async def update_session(request: Request, session_id: str, body: dict = {}) -> dict:
    """Update session context."""
    try:
        from traditional.memory_manager import MemoryManager  # type: ignore[import-untyped]
        mm = MemoryManager()
        mm.update_session(session_id, body)
        return {"success": True, "session_id": session_id}
    except ImportError:
        return {"success": True, "session_id": session_id, "stub": True}
    except Exception as exc:
        logger.error("Update session failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict:
    """Delete a session."""
    try:
        from traditional.memory_manager import MemoryManager  # type: ignore[import-untyped]
        mm = MemoryManager()
        mm.delete_session(session_id)
        return {"success": True, "session_id": session_id, "deleted": True}
    except ImportError:
        return {"success": True, "session_id": session_id, "deleted": True, "stub": True}
    except Exception as exc:
        logger.error("Delete session failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.post("/sessions/{session_id}/pin-document")
async def pin_document(request: Request, session_id: str, body: dict = {}) -> dict:
    """Pin documents to a session for persistent context."""
    try:
        from traditional.memory_manager import MemoryManager  # type: ignore[import-untyped]
        mm = MemoryManager()
        mm.pin_document(session_id, body.get("document_ids", []))
        return {"success": True, "session_id": session_id, "pinned": True}
    except ImportError:
        return {"success": True, "session_id": session_id, "pinned": True, "stub": True}
    except Exception as exc:
        logger.error("Pin document failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.delete("/sessions/{session_id}/pin-document")
async def unpin_document(request: Request, session_id: str, body: dict = {}) -> dict:
    """Unpin documents from a session."""
    try:
        from traditional.memory_manager import MemoryManager  # type: ignore[import-untyped]
        mm = MemoryManager()
        mm.unpin_document(session_id, body.get("document_ids", []))
        return {"success": True, "session_id": session_id, "unpinned": True}
    except ImportError:
        return {"success": True, "session_id": session_id, "unpinned": True, "stub": True}
    except Exception as exc:
        logger.error("Unpin document failed: %s", exc)
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Debug Endpoints
# ---------------------------------------------------------------------------

@router.get("/test-retrieve")
async def test_retrieve(
    request: Request,
    query: str = "test",
    project_id: int = 7166,
    top_k: int = 5,
) -> dict:
    """Test FAISS retrieval without running the full pipeline."""
    try:
        from traditional.rag.retrieval.loaders import _load_project  # type: ignore[import-untyped]
        from traditional.retrieve import multi_project_retrieve  # type: ignore[import-untyped]

        _load_project(project_id)
        results = multi_project_retrieve(query, [project_id], top_k=top_k)
        return {
            "success": True,
            "query": query,
            "project_id": project_id,
            "results_count": len(results),
            "results": results[:top_k],
        }
    except ImportError as exc:
        return {"success": False, "error": f"Traditional engine not available: {exc}"}
    except Exception as exc:
        logger.error("Test retrieve failed: %s", exc)
        return {"success": False, "error": str(exc)}


@router.get("/debug-pipeline")
async def debug_pipeline(request: Request) -> dict:
    """Debug info for both engines."""
    orchestrator = _get_orchestrator(request)
    debug_info: dict[str, Any] = {
        "orchestrator": {
            "fallback_enabled": orchestrator.fallback_enabled,
            "fallback_timeout": orchestrator.fallback_timeout,
        },
        "agentic": {
            "initialized": orchestrator.agentic._initialized,
        },
        "traditional": {
            "faiss_loaded": orchestrator.traditional.is_loaded,
        },
    }

    # Try to get agentic module info
    try:
        import agentic  # type: ignore[import-untyped]
        debug_info["agentic"]["module_path"] = str(agentic.__file__)
    except ImportError:
        debug_info["agentic"]["module_path"] = "not importable"

    # Try to get traditional module info
    try:
        import traditional  # type: ignore[import-untyped]
        debug_info["traditional"]["module_path"] = str(traditional.__file__)
    except ImportError:
        debug_info["traditional"]["module_path"] = "not importable"

    return debug_info


# ── v3.5 (2026-05-21) — session title rename ───────────────────────────────


class _SessionTitleUpdate(BaseModel):
    title: str
    user_id: int
    project_id: int
    agent_id: str = "drawing-agent"


@router.post("/sessions/{session_id}/title")
async def rename_session_title(
    session_id: str,
    body: _SessionTitleUpdate,
    request: Request,
) -> dict:
    """Rename a chat session's title in iField Mongo.

    Issues PUT /api/userSession/{sessionId} so /list (sidebar) and /history
    reflect the new title. NOTE: the backend update is keyed on sessionId;
    for multi-turn sessions the backend must update ALL rows (it currently
    updates one) for /list -- which reads the latest turn's title -- to change.
    """
    from fastapi import HTTPException
    from urllib.parse import quote
    import requests
    from shared.session_tracker import (
        _truncate_title, _userSession_url, _userSession_timeout_s,
    )

    new_title = (body.title or "").strip()
    if not new_title:
        raise HTTPException(status_code=422, detail="title cannot be empty")
    new_title = _truncate_title(new_title)

    base = _userSession_url().rstrip("/")
    url = f"{base}/{quote(session_id, safe='')}"
    try:
        resp = await asyncio.to_thread(
            requests.put,
            url,
            json={
                "title": new_title,
                "userId": int(body.user_id),
                "projectId": int(body.project_id),
                "agent": body.agent_id,
            },
            timeout=_userSession_timeout_s(),
        )
        ok = resp.status_code == 200
        if not ok:
            logger.warning(
                "[session] rename PUT %s HTTP %s: %s",
                url, resp.status_code, resp.text[:200],
            )
        return {
            "success": ok,
            "session_id": session_id,
            "new_title": new_title,
            "status_code": resp.status_code,
        }
    except Exception as exc:
        logger.warning("[session] rename failed: %s: %s", type(exc).__name__, exc)
        return {
            "success": False,
            "error": str(exc),
            "session_id": session_id,
            "new_title": new_title,
        }

