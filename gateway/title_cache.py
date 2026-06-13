"""Document title cache — in-memory LRU cache for drawing/spec title lists.

Pre-computed at session creation and refreshed hourly. Thread-safe for
up to 3 workers. Supports manual invalidation via maintenance endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache config
_CACHE_TTL_SECONDS = 3600  # 1 hour
_CACHE_MAX_PROJECTS = 100  # LRU eviction after this many projects

# Thread-safe cache storage: {project_id: {"titles": [...], "timestamp": float}}
_cache: dict[int, dict[str, Any]] = {}
_lock = threading.Lock()
_access_order: list[int] = []  # LRU tracking


def get_cached_titles(project_id: int) -> Optional[dict]:
    """Get cached title lists for a project if not stale.

    Returns {"drawings": [...], "specifications": [...], "cached_at": float}
    or None if cache miss or stale.
    """
    with _lock:
        entry = _cache.get(project_id)
        if entry is None:
            return None
        if time.time() - entry["timestamp"] > _CACHE_TTL_SECONDS:
            del _cache[project_id]
            if project_id in _access_order:
                _access_order.remove(project_id)
            return None
        # Update LRU order
        if project_id in _access_order:
            _access_order.remove(project_id)
        _access_order.append(project_id)
        return entry["data"]


def set_cached_titles(
    project_id: int,
    drawings: list[dict],
    specifications: list[dict],
) -> None:
    """Store title lists in cache. Evicts LRU entries if over capacity."""
    with _lock:
        # Evict LRU if at capacity
        while len(_cache) >= _CACHE_MAX_PROJECTS and _access_order:
            evict_id = _access_order.pop(0)
            _cache.pop(evict_id, None)
            logger.debug("Evicted title cache for project %d (LRU)", evict_id)

        _cache[project_id] = {
            "data": {
                "drawings": drawings,
                "specifications": specifications,
                "cached_at": time.time(),
            },
            "timestamp": time.time(),
        }
        if project_id in _access_order:
            _access_order.remove(project_id)
        _access_order.append(project_id)
        logger.info(
            "Cached titles for project %d: %d drawings, %d specs",
            project_id, len(drawings), len(specifications),
        )


def invalidate_project(project_id: int) -> bool:
    """Invalidate cache for a specific project. Returns True if entry existed."""
    with _lock:
        existed = project_id in _cache
        _cache.pop(project_id, None)
        if project_id in _access_order:
            _access_order.remove(project_id)
        if existed:
            logger.info("Invalidated title cache for project %d", project_id)
        return existed


def invalidate_all() -> int:
    """Invalidate all cached titles. Returns count of cleared entries."""
    with _lock:
        count = len(_cache)
        _cache.clear()
        _access_order.clear()
        logger.info("Invalidated all title caches (%d entries)", count)
        return count


def get_cache_stats() -> dict:
    """Return cache statistics for monitoring."""
    with _lock:
        return {
            "cached_projects": len(_cache),
            "max_projects": _CACHE_MAX_PROJECTS,
            "ttl_seconds": _CACHE_TTL_SECONDS,
            "project_ids": list(_cache.keys()),
        }
