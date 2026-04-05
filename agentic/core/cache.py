"""
Two-level caching for AgenticRAG.

1. Tool cache: Caches MongoDB query results (TTL 5 min, 500 entries)
2. Agent cache: Caches complete agent responses (TTL 10 min, 200 entries)

Prevents redundant MongoDB queries and OpenAI API calls for repeated questions.
"""

import hashlib
import logging
import threading
from typing import Any, Optional

from cachetools import TTLCache

logger = logging.getLogger("agentic_rag.cache")

# Tool-level cache: MongoDB results (short TTL, many entries)
_tool_cache: TTLCache = TTLCache(maxsize=500, ttl=300)
_tool_lock = threading.Lock()

# Agent-level cache: Full agent responses (longer TTL, fewer entries)
_agent_cache: TTLCache = TTLCache(maxsize=200, ttl=600)
_agent_lock = threading.Lock()


def _make_key(*args: Any) -> str:
    """Create a deterministic cache key from arguments."""
    raw = str(args).lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Tool cache ────────────────────────────────────────────────────────

def get_tool_result(tool_name: str, **kwargs: Any) -> Optional[Any]:
    """Get cached tool result, or None if not cached."""
    key = _make_key(tool_name, kwargs)
    with _tool_lock:
        result = _tool_cache.get(key)
    if result is not None:
        logger.debug(f"Tool cache HIT: {tool_name}")
    return result


def set_tool_result(tool_name: str, result: Any, **kwargs: Any) -> None:
    """Cache a tool result."""
    key = _make_key(tool_name, kwargs)
    with _tool_lock:
        _tool_cache[key] = result


# ── Agent cache ───────────────────────────────────────────────────────

def get_agent_result(query: str, project_id: int, set_id: int = None) -> Optional[Any]:
    """Get cached agent result, or None if not cached."""
    normalized = query.strip().lower()
    key = _make_key("agent", normalized, project_id, set_id)
    with _agent_lock:
        result = _agent_cache.get(key)
    if result is not None:
        logger.info(f"Agent cache HIT: project={project_id}")
    return result


def set_agent_result(
    query: str,
    project_id: int,
    result: Any,
    set_id: int = None,
) -> None:
    """Cache an agent result (only if confidence is not 'low')."""
    if hasattr(result, "confidence") and result.confidence == "low":
        return  # Don't cache low-confidence results
    normalized = query.strip().lower()
    key = _make_key("agent", normalized, project_id, set_id)
    with _agent_lock:
        _agent_cache[key] = result
    logger.info(f"Agent cache SET: project={project_id}")


def clear_all() -> None:
    """Clear both caches."""
    with _tool_lock:
        _tool_cache.clear()
    with _agent_lock:
        _agent_cache.clear()
    logger.info("All caches cleared")
