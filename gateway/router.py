"""
Gateway Router — all 18 endpoints for the Unified RAG Agent.

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
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.auth import auth_required
from gateway.models import QueryRequest, UnifiedResponse

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(auth_required)])


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
        return {"error": "Configuration unavailable"}


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
# Core Query Endpoints
# ---------------------------------------------------------------------------

@router.post("/query")
async def query(request: Request, body: QueryRequest) -> dict:
    """Run a query through the orchestrator (agentic-first with fallback)."""
    orchestrator = _get_orchestrator(request)
    result = await orchestrator.query(
        query=body.query,
        project_id=body.project_id,
        engine=body.engine,
        session_id=body.session_id,
        set_id=body.set_id,
        conversation_history=body.conversation_history,
        search_mode=body.search_mode,
        generate_document=body.generate_document,
        filter_source_type=body.filter_source_type,
        filter_drawing_name=body.filter_drawing_name,
        docqa_document=body.docqa_document.model_dump() if body.docqa_document else None,
    )
    return result


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """SSE streaming endpoint — tries agentic stream, falls back to traditional."""

    async def event_generator() -> Any:
        orchestrator = _get_orchestrator(request)
        try:
            # Attempt agentic streaming
            from agentic.core.agent import run_agent_stream  # type: ignore[import-untyped]
            async for chunk in run_agent_stream(
                query=body.query,
                project_id=body.project_id,
                set_id=body.set_id,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except (ImportError, AttributeError):
            # Fallback: run full query and stream the result as a single event
            logger.info("Agentic streaming not available, falling back to full query")
            try:
                result = await orchestrator.query(
                    query=body.query,
                    project_id=body.project_id,
                    engine=body.engine,
                    session_id=body.session_id,
                )
                yield f"data: {json.dumps(result)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': 'An internal error occurred. Please try again.'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': 'An internal error occurred. Please try again.'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
        return {"success": True, "result": result}
    except ImportError:
        return {"success": False, "error": "Traditional engine not available for web search"}
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


# ---------------------------------------------------------------------------
# Session Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/create")
async def create_session(request: Request, body: dict = {}) -> dict:
    """Create a new session."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        session_id = mm.create_session(
            user_query=body.get("initial_query", "Session created via API"),
            project_id=body.get("project_id"),
            filter_source_type=body.get("filter_source_type"),
            session_id=body.get("session_id"),
        )
        return {"success": True, "session_id": session_id}
    except ImportError:
        session_id = str(uuid.uuid4())
        logger.warning("MemoryManager not available, returning stub session")
        return {"success": True, "session_id": session_id, "stub": True}
    except Exception as exc:
        logger.error("Session creation failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.get("/sessions")
async def list_sessions(request: Request) -> dict:
    """List all sessions."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        sessions = []
        for sid, session in mm.sessions.items():
            sessions.append({
                "session_id": sid,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "message_count": len(session.messages),
                "total_tokens": session.total_tokens,
                "project_id": session.context.project_id,
            })
        return {"success": True, "count": len(sessions), "sessions": sessions}
    except ImportError:
        return {"success": True, "count": 0, "sessions": [], "stub": True}
    except Exception as exc:
        logger.error("List sessions failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.get("/sessions/{session_id}/stats")
async def session_stats(request: Request, session_id: str) -> dict:
    """Get session stats including engine usage."""
    try:
        from shared.session.manager import get_session_stats_extended
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        # Combine both layers of session stats
        basic_stats = mm.get_session_stats(session_id)
        unified_stats = get_session_stats_extended(session_id)
        return {"success": True, **basic_stats, **unified_stats}
    except ImportError:
        return {"success": True, "session_id": session_id, "stub": True}
    except Exception as exc:
        logger.error("Session stats failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.get("/sessions/{session_id}/conversation")
async def session_conversation(request: Request, session_id: str) -> dict:
    """Get conversation history for a session."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        session = mm.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        history = session.get_full_conversation_history(include_summaries=True)
        return {
            "success": True,
            "session_id": session_id,
            "message_count": len(session.messages),
            "conversation": history,
        }
    except ImportError:
        return {"success": True, "session_id": session_id, "conversation": [], "stub": True}
    except Exception as exc:
        logger.error("Get conversation failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.post("/sessions/{session_id}/update")
async def update_session(request: Request, session_id: str, body: dict = {}) -> dict:
    """Update session context."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        mm.update_context(
            session_id=session_id,
            project_id=body.get("project_id"),
            filter_source_type=body.get("filter_source_type"),
            custom_instructions=body.get("custom_instructions"),
            pinned_documents=body.get("pinned_documents"),
            pinned_titles=body.get("pinned_titles"),
        )
        return {"success": True, "session_id": session_id}
    except ImportError:
        return {"success": True, "session_id": session_id, "stub": True}
    except Exception as exc:
        logger.error("Update session failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict:
    """Delete a session."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        deleted = mm.clear_session(session_id)
        return {"success": True, "session_id": session_id, "deleted": deleted}
    except ImportError:
        return {"success": True, "session_id": session_id, "deleted": True, "stub": True}
    except Exception as exc:
        logger.error("Delete session failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.post("/sessions/{session_id}/pin-document")
async def pin_document(request: Request, session_id: str, body: dict = {}) -> dict:
    """Pin documents to a session for persistent context."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        doc_ids = body.get("document_ids", [])
        mm.update_context(
            session_id=session_id,
            pinned_documents=doc_ids,
            pinned_titles=body.get("document_titles", doc_ids),
        )
        return {"success": True, "session_id": session_id, "pinned": True, "document_count": len(doc_ids)}
    except ImportError:
        return {"success": True, "session_id": session_id, "pinned": True, "stub": True}
    except Exception as exc:
        logger.error("Pin document failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.delete("/sessions/{session_id}/pin-document")
async def unpin_document(request: Request, session_id: str, body: dict = {}) -> dict:
    """Unpin documents from a session."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore[import-untyped]
        mm = get_memory_manager()
        mm.update_context(
            session_id=session_id,
            pinned_documents=[],
            pinned_titles=[],
        )
        return {"success": True, "session_id": session_id, "unpinned": True}
    except ImportError:
        return {"success": True, "session_id": session_id, "unpinned": True, "stub": True}
    except Exception as exc:
        logger.error("Unpin document failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


# ---------------------------------------------------------------------------
# Document Discovery (Angular UI integration)
# ---------------------------------------------------------------------------

@router.get("/projects/{project_id}/documents")
async def discover_documents(request: Request, project_id: int, set_id: int = None) -> dict:
    """List available drawing titles and spec sections for a project.

    Used by the Angular frontend to show document groups when the agent
    cannot answer — the user can select a document to scope queries.
    """
    try:
        orchestrator = _get_orchestrator(request)
        available = await orchestrator._discover_documents(project_id, set_id)
        return {
            "success": True,
            "project_id": project_id,
            "document_count": len(available),
            "documents": available,
        }
    except Exception as exc:
        logger.error("Document discovery failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


# ---------------------------------------------------------------------------
# Session Scope Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/scope")
async def set_scope(request: Request, session_id: str, body: dict = {}) -> dict:
    """Set document scope for a session."""
    try:
        from shared.session.manager import set_document_scope

        # Sanitize inputs — strip control characters, limit length
        import re
        def _sanitize(val: str, max_len: int = 200) -> str:
            if not val:
                return ""
            return re.sub(r'[\x00-\x1f\x7f]', '', val)[:max_len].strip()

        result = set_document_scope(
            session_id=session_id,
            drawing_title=_sanitize(body.get("drawing_title", "")),
            drawing_name=_sanitize(body.get("drawing_name", "")),
            document_type=_sanitize(body.get("document_type", "drawing"), 20),
            section_title=_sanitize(body.get("section_title", "")),
            pdf_name=_sanitize(body.get("pdf_name", "")),
        )
        return {"success": True, "session_id": session_id, "scope": result}
    except ImportError:
        return {"success": True, "session_id": session_id, "scope": {}, "stub": True}
    except Exception as exc:
        logger.error("Set scope failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.delete("/sessions/{session_id}/scope")
async def clear_scope(request: Request, session_id: str) -> dict:
    """Clear document scope, return to full project search."""
    try:
        from shared.session.manager import clear_document_scope
        result = clear_document_scope(session_id)
        return {"success": True, "session_id": session_id, "scope": result}
    except ImportError:
        return {"success": True, "session_id": session_id, "scope": {}, "stub": True}
    except Exception as exc:
        logger.error("Clear scope failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


@router.get("/sessions/{session_id}/scope")
async def get_scope(request: Request, session_id: str) -> dict:
    """Get current document scope state for a session."""
    try:
        from shared.session.manager import get_document_scope, get_meta
        scope = get_document_scope(session_id)
        meta = get_meta(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "scope": scope,
            "previously_scoped": meta.previously_scoped,
        }
    except ImportError:
        return {"success": True, "session_id": session_id, "scope": {}, "stub": True}
    except Exception as exc:
        logger.error("Get scope failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


# ---------------------------------------------------------------------------
# Admin / Maintenance Endpoints
# ---------------------------------------------------------------------------

@router.get("/admin/sessions")
async def admin_list_sessions(request: Request) -> dict:
    """Admin: list all active sessions with scope state."""
    try:
        from shared.session.manager import _session_meta
        sessions = []
        for sid, meta in _session_meta.items():
            sessions.append({
                "session_id": sid,
                "last_engine": meta.last_engine,
                "total_cost_usd": meta.total_cost_usd,
                "scope": meta.scope.to_dict(),
                "engine_usage": meta.engine_usage.to_dict(),
            })
        return {"success": True, "count": len(sessions), "sessions": sessions}
    except ImportError:
        return {"success": True, "count": 0, "sessions": [], "stub": True}


@router.post("/admin/cache/refresh")
async def admin_cache_refresh(request: Request, body: dict = {}) -> dict:
    """Admin: refresh or invalidate title cache."""
    try:
        from gateway.title_cache import invalidate_project, invalidate_all, get_cache_stats
        project_id = body.get("project_id")
        if project_id:
            existed = invalidate_project(int(project_id))
            return {"success": True, "action": "invalidate_project", "project_id": project_id, "existed": existed}
        else:
            count = invalidate_all()
            return {"success": True, "action": "invalidate_all", "cleared": count}
    except ImportError:
        return {"success": True, "action": "noop", "stub": True}
    except Exception as exc:
        logger.error("Cache refresh failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


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
        return {"success": False, "error": "Traditional engine not available"}
    except Exception as exc:
        logger.error("Test retrieve failed: %s", exc)
        return {"success": False, "error": "An internal error occurred. Please try again."}


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
        "title_cache": {},
    }

    # Add title cache stats
    try:
        from gateway.title_cache import get_cache_stats
        debug_info["title_cache"] = get_cache_stats()
    except ImportError:
        debug_info["title_cache"] = {"status": "not available"}

    return debug_info
