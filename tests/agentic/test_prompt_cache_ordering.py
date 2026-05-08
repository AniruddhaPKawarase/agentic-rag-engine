"""Phase 1.3: message ordering for OpenAI auto-cache.

Cacheable prefix = [system, ...history(stable), user_query_last].
Dynamic injections (RRF hints) must go into the LAST user message, not
prepended to history — otherwise cache busts every turn.
"""
from __future__ import annotations

import pytest


def test_build_react_messages_system_first():
    from core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[{"role": "user", "content": "prev"}],
        user_query="current",
        rrf_hint=None,
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYSTEM"


def test_build_react_messages_query_last():
    from core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[{"role": "user", "content": "prev"}],
        user_query="current",
        rrf_hint=None,
    )
    assert msgs[-1]["role"] == "user"
    assert "current" in msgs[-1]["content"]


def test_rrf_hint_appended_to_last_user_message_not_prepended():
    from core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[{"role": "user", "content": "prev"}],
        user_query="current",
        rrf_hint="SEARCH_HINT: try 'X' or 'Y'",
    )
    # Hint goes into LAST message, preserving earlier-message cache
    assert "SEARCH_HINT" in msgs[-1]["content"]
    assert "SEARCH_HINT" not in msgs[0]["content"]
    for m in msgs[:-1]:
        assert "SEARCH_HINT" not in (m.get("content") or "")


def test_build_react_messages_no_history_ok():
    from core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=None,
        user_query="hello",
        rrf_hint=None,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hello"


def test_build_react_messages_empty_history_ok():
    from core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[],
        user_query="hello",
        rrf_hint=None,
    )
    assert len(msgs) == 2


def test_cache_hit_rate_metric_logged(caplog):
    """Ensure the usage.cached_tokens from OpenAI is logged at INFO level."""
    import logging
    from core.agent import _log_cache_metrics
    caplog.set_level(logging.INFO, logger="agentic_rag.agent")
    _log_cache_metrics({"cached_tokens": 1536, "prompt_tokens": 2048})
    matched = [r for r in caplog.records if "cached_tokens" in r.message and "1536" in r.message]
    assert matched, f"No log record with cached_tokens=1536: {[r.message for r in caplog.records]}"


def test_cache_metric_zero_total_does_not_crash(caplog):
    from core.agent import _log_cache_metrics
    # Should not divide-by-zero
    _log_cache_metrics({"cached_tokens": 0, "prompt_tokens": 0})
    # Just assert no exception raised


def test_cache_metric_missing_keys_does_not_crash():
    from core.agent import _log_cache_metrics
    _log_cache_metrics({})  # entirely missing
    _log_cache_metrics({"prompt_tokens": 100})  # missing cached_tokens
