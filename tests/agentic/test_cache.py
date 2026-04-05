"""Unit tests for core/cache.py — two-level TTL cache."""

from unittest.mock import MagicMock

from core.cache import (
    _make_key,
    clear_all,
    get_agent_result,
    get_tool_result,
    set_agent_result,
    set_tool_result,
)


class TestToolCache:
    """Tests for tool-level cache (MongoDB result caching)."""

    def setup_method(self) -> None:
        """Clear caches before each test to avoid cross-contamination."""
        clear_all()

    def test_tool_cache_set_and_get(self) -> None:
        """Setting a value and getting it returns the same value."""
        set_tool_result("list_drawings", result={"count": 5}, project_id=7166)
        result = get_tool_result("list_drawings", project_id=7166)
        assert result == {"count": 5}

    def test_tool_cache_miss(self) -> None:
        """Getting a non-existent key returns None."""
        result = get_tool_result("nonexistent_tool", project_id=9999)
        assert result is None

    def test_tool_cache_different_args_different_keys(self) -> None:
        """Different kwargs produce different cache entries."""
        set_tool_result("search", result="result_a", query="plumbing")
        set_tool_result("search", result="result_b", query="electrical")
        assert get_tool_result("search", query="plumbing") == "result_a"
        assert get_tool_result("search", query="electrical") == "result_b"


class TestAgentCache:
    """Tests for agent-level cache (full response caching)."""

    def setup_method(self) -> None:
        """Clear caches before each test."""
        clear_all()

    def test_agent_cache_set_and_get(self) -> None:
        """Setting an agent result and getting it returns the same value."""
        result_obj = {"answer": "The panel is rated 200A", "confidence": "high"}
        set_agent_result("What is the panel rating?", project_id=7166, result=result_obj)
        cached = get_agent_result("What is the panel rating?", project_id=7166)
        assert cached == result_obj

    def test_agent_cache_skips_low_confidence(self) -> None:
        """Low-confidence results with .confidence attribute are NOT cached."""
        low_conf = MagicMock()
        low_conf.confidence = "low"
        set_agent_result("vague question", project_id=7166, result=low_conf)
        cached = get_agent_result("vague question", project_id=7166)
        assert cached is None

    def test_agent_cache_stores_high_confidence(self) -> None:
        """High-confidence results with .confidence attribute ARE cached."""
        high_conf = MagicMock()
        high_conf.confidence = "high"
        set_agent_result("clear question", project_id=7166, result=high_conf)
        cached = get_agent_result("clear question", project_id=7166)
        assert cached is high_conf

    def test_agent_cache_normalizes_query(self) -> None:
        """Queries are normalized (lowercase, stripped) for cache key."""
        set_agent_result("  ELECTRICAL Panel  ", project_id=7166, result="answer")
        cached = get_agent_result("electrical panel", project_id=7166)
        assert cached == "answer"


class TestCacheKeyGeneration:
    """Tests for _make_key deterministic hashing."""

    def test_cache_key_same_args_same_key(self) -> None:
        """Identical arguments produce identical cache keys."""
        key1 = _make_key("tool", {"project_id": 7166})
        key2 = _make_key("tool", {"project_id": 7166})
        assert key1 == key2

    def test_cache_key_different_args_different_key(self) -> None:
        """Different arguments produce different cache keys."""
        key1 = _make_key("tool", {"project_id": 7166})
        key2 = _make_key("tool", {"project_id": 7201})
        assert key1 != key2

    def test_cache_key_is_sha256_hex(self) -> None:
        """Cache key is a 64-character hex string (SHA-256)."""
        key = _make_key("test")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
