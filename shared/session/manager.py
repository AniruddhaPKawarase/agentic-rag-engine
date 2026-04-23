"""
Unified session manager — module-level store for session metadata.

Provides functions to track which engine handled each request,
accumulate costs, and retrieve extended session statistics.
"""

from __future__ import annotations

from shared.session.models import EngineUsage, UnifiedSessionMeta

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
    }


def clear_meta(session_id: str) -> None:
    """Remove metadata for *session_id* (e.g. on session delete)."""
    _session_meta.pop(session_id, None)
