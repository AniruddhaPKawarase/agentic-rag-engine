"""
Unified session manager — module-level store for session metadata.

Provides functions to track which engine handled each request,
accumulate costs, and retrieve extended session statistics.
"""

from __future__ import annotations

from shared.session.models import DocumentScope, EngineUsage, UnifiedSessionMeta

# Module-level session metadata store (not persisted across restarts).
_session_meta: dict[str, UnifiedSessionMeta] = {}


def get_meta(session_id: str) -> UnifiedSessionMeta:
    """Return the metadata for *session_id*, creating it if absent."""
    if session_id not in _session_meta:
        _session_meta[session_id] = UnifiedSessionMeta()
    return _session_meta[session_id]


def record_engine_use(
    session_id: str,
    engine: str,
    cost_usd: float = 0.0,
) -> None:
    """Record that *engine* handled a request, adding *cost_usd*."""
    meta = get_meta(session_id)
    meta.engine_usage.record(engine)
    meta.last_engine = engine
    meta.total_cost_usd += cost_usd


def get_session_stats_extended(session_id: str) -> dict:
    """Return a JSON-serialisable dict of extended session statistics."""
    meta = get_meta(session_id)
    return {
        "session_id": session_id,
        "last_engine": meta.last_engine,
        "total_cost_usd": meta.total_cost_usd,
        "engine_usage": meta.engine_usage.to_dict(),
        "scope": meta.scope.to_dict(),
        "previously_scoped": meta.previously_scoped,
    }


def set_document_scope(
    session_id: str,
    drawing_title: str = "",
    drawing_name: str = "",
    document_type: str = "drawing",
    section_title: str = "",
    pdf_name: str = "",
) -> dict:
    """Enter document-scoped mode for a session."""
    meta = get_meta(session_id)
    meta.scope.activate(
        drawing_title=drawing_title,
        drawing_name=drawing_name,
        document_type=document_type,
        section_title=section_title,
        pdf_name=pdf_name,
    )
    # Track in history for quick-access
    entry = {
        "drawing_title": drawing_title,
        "drawing_name": drawing_name,
        "document_type": document_type,
        "section_title": section_title,
    }
    if entry not in meta.previously_scoped:
        meta.previously_scoped.append(entry)
    return meta.scope.to_dict()


def clear_document_scope(session_id: str) -> dict:
    """Exit document-scoped mode, return to full project scope."""
    meta = get_meta(session_id)
    meta.scope.clear()
    return meta.scope.to_dict()


def get_document_scope(session_id: str) -> dict:
    """Get current document scope state."""
    meta = get_meta(session_id)
    return meta.scope.to_dict()


def clear_meta(session_id: str) -> None:
    """Remove metadata for *session_id* (e.g. on session delete)."""
    _session_meta.pop(session_id, None)
