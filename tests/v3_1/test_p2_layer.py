"""
Tests for the v3.1 P2 generation-chain front-end:

* recall_agent.py — Memory Recall (Agent 1)
* query_rewriter.py — Query Rewriter (Agent 2)

All external dependencies are mocked. No real OpenAI / Atlas calls.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Make the v3.1 worktree importable.
# ---------------------------------------------------------------------------
_WORKTREE = Path(__file__).resolve().parent.parent.parent
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(role: str, content: str) -> SimpleNamespace:
    """Minimal stand-in for traditional.memory_manager.Message."""
    return SimpleNamespace(
        role=role,
        content=content,
        timestamp=time.time(),
        tokens=0,
        metadata=None,
    )


def _make_session(
    *,
    messages: List[SimpleNamespace] | None = None,
    rolling_summary: str | None = None,
    custom_instructions: str = "",
) -> SimpleNamespace:
    """Minimal stand-in for ConversationSession."""
    ctx = SimpleNamespace(
        custom_instructions=custom_instructions,
        rolling_summary=rolling_summary,
    )
    return SimpleNamespace(
        session_id="session_test",
        messages=messages or [],
        context=ctx,
    )


def _enable_recall(monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "true")


def _enable_rewriter(monkeypatch, *, heuristic: bool = True):
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "true")
    monkeypatch.setenv(
        "QUERY_REWRITER_SKIP_HEURISTIC", "true" if heuristic else "false"
    )


# ===========================================================================
# recall_agent.py
# ===========================================================================


def test_recall_returns_empty_when_session_id_none(monkeypatch):
    _enable_recall(monkeypatch)
    from agentic.memory.recall_agent import recall

    out = recall(session_id=None, user_query="anything")

    assert out["had_context"] is False
    assert out["rolling_summary"] is None
    assert out["recent_turns"] == []
    assert out["semantic_turns"] == []
    assert out["topic_tags"] == []


def test_recall_returns_empty_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "false")
    from agentic.memory.recall_agent import recall

    # Even with a real-looking session_id we should bail.
    out = recall(session_id="session_abc", user_query="anything")

    assert out["had_context"] is False


def test_recall_returns_empty_when_session_missing(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod

    fake_mm = MagicMock()
    fake_mm.get_session.return_value = None
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)

    from agentic.memory.recall_agent import recall

    out = recall(session_id="session_nope", user_query="anything")

    assert out["had_context"] is False
    fake_mm.get_session.assert_called_once_with("session_nope")


def test_recall_pulls_recent_turns_from_memory_manager(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod

    msgs = [
        _make_message("user", "What's drawing S-101 about?"),
        _make_message("assistant", "S-101 covers structural framing."),
        _make_message("user", "And M101?"),
        _make_message("assistant", "M101 is the mechanical layout."),
    ]
    session = _make_session(messages=msgs)
    fake_mm = MagicMock()
    fake_mm.get_session.return_value = session
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)

    # Stub semantic recall to a controlled empty list.
    from agentic.memory import recall_agent
    monkeypatch.setattr(
        recall_agent, "_extract_semantic_turns", lambda **_kw: []
    )

    out = recall_agent.recall(
        session_id="session_abc",
        user_query="what about that one",
        top_k_recent=3,
    )

    assert out["had_context"] is True
    assert len(out["recent_turns"]) == 3  # last 3 of 4
    assert [t["turn_index"] for t in out["recent_turns"]] == [1, 2, 3]
    assert out["recent_turns"][-1]["role"] == "assistant"


def test_recall_dedupes_semantic_against_recent(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod

    msgs = [
        _make_message("user", f"turn {i}") for i in range(8)
    ]
    session = _make_session(messages=msgs)
    fake_mm = MagicMock()
    fake_mm.get_session.return_value = session
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)

    # Recent will be turns 2..7 (top_k_recent=6). Semantic returns
    # turn_index 3 (dup) and turn_index 0 (new) — only 0 should remain.
    from agentic.memory import embeddings as emb_mod
    from agentic.memory import vector_store as vs_mod

    monkeypatch.setattr(emb_mod, "embed_text", lambda _t: [0.1] * 1536)

    fake_store = MagicMock()
    fake_store.search.return_value = [
        {"turn_index": 3, "role": "user", "text_excerpt": "dup", "score": 0.9},
        {"turn_index": 0, "role": "user", "text_excerpt": "new", "score": 0.8},
    ]
    monkeypatch.setattr(
        vs_mod, "SessionVectorStore", lambda *a, **kw: fake_store
    )

    from agentic.memory.recall_agent import recall

    out = recall(
        session_id="session_abc",
        user_query="what about it",
        top_k_recent=6,
        top_k_semantic=2,
    )

    assert len(out["semantic_turns"]) == 1
    assert out["semantic_turns"][0]["turn_index"] == 0


def test_recall_extracts_drawing_name_topic_tags(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod
    from agentic.memory import recall_agent

    msgs = [
        _make_message("user", "Show me drawing S-101 and M101"),
        _make_message("assistant", "S-101 is structural; M101 is mechanical."),
        _make_message("user", "What about E604a?"),
    ]
    session = _make_session(messages=msgs)
    fake_mm = MagicMock()
    fake_mm.get_session.return_value = session
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)
    monkeypatch.setattr(
        recall_agent, "_extract_semantic_turns", lambda **_kw: []
    )

    out = recall_agent.recall(
        session_id="session_abc", user_query="anything"
    )

    tags_lower = {t.lower() for t in out["topic_tags"]}
    assert "s-101" in tags_lower
    assert "m101" in tags_lower
    assert "e604a" in tags_lower


def test_recall_extracts_trade_topic_tags(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod
    from agentic.memory import recall_agent

    msgs = [
        _make_message(
            "user",
            "I need the mechanical and electrical specs for the HVAC system",
        ),
        _make_message(
            "assistant",
            "Mechanical specs cover ducts; electrical covers wiring.",
        ),
    ]
    session = _make_session(messages=msgs)
    fake_mm = MagicMock()
    fake_mm.get_session.return_value = session
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)
    monkeypatch.setattr(
        recall_agent, "_extract_semantic_turns", lambda **_kw: []
    )

    out = recall_agent.recall(
        session_id="session_abc", user_query="anything"
    )

    tags_lower = {t.lower() for t in out["topic_tags"]}
    assert "mechanical" in tags_lower
    assert "electrical" in tags_lower
    assert "hvac" in tags_lower


def test_recall_swallows_errors_gracefully(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod

    def explode():
        raise RuntimeError("DB on fire")

    monkeypatch.setattr(mm_mod, "get_memory_manager", explode)

    from agentic.memory.recall_agent import recall

    # Must not raise.
    out = recall(session_id="session_abc", user_query="anything")

    assert out["had_context"] is False
    assert out["recent_turns"] == []


def test_recall_uses_rolling_summary_when_present(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod
    from agentic.memory import recall_agent

    session = _make_session(
        messages=[_make_message("user", "hi")],
        rolling_summary="Discussion about S-101 specifications.",
        custom_instructions="legacy-instructions-should-not-win",
    )
    fake_mm = MagicMock()
    fake_mm.get_session.return_value = session
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)
    monkeypatch.setattr(
        recall_agent, "_extract_semantic_turns", lambda **_kw: []
    )

    out = recall_agent.recall(
        session_id="session_abc", user_query="anything"
    )

    assert out["rolling_summary"] == "Discussion about S-101 specifications."


def test_recall_falls_back_to_custom_instructions_when_no_summary(monkeypatch):
    _enable_recall(monkeypatch)
    import traditional.memory_manager as mm_mod
    from agentic.memory import recall_agent

    session = _make_session(
        messages=[_make_message("user", "hi")],
        rolling_summary=None,
        custom_instructions="Legacy summary about HVAC.",
    )
    fake_mm = MagicMock()
    fake_mm.get_session.return_value = session
    monkeypatch.setattr(mm_mod, "get_memory_manager", lambda: fake_mm)
    monkeypatch.setattr(
        recall_agent, "_extract_semantic_turns", lambda **_kw: []
    )

    out = recall_agent.recall(
        session_id="session_abc", user_query="anything"
    )

    assert out["rolling_summary"] == "Legacy summary about HVAC."


# ===========================================================================
# query_rewriter.py
# ===========================================================================


def _ctx_with_history() -> Dict[str, Any]:
    return {
        "rolling_summary": "User has been asking about drawings S-101 and M101.",
        "recent_turns": [
            {"role": "user", "content": "Tell me about S-101.", "turn_index": 0},
            {"role": "assistant", "content": "S-101 is structural framing.", "turn_index": 1},
            {"role": "user", "content": "And M101?", "turn_index": 2},
            {"role": "assistant", "content": "M101 is the mechanical layout.", "turn_index": 3},
        ],
        "semantic_turns": [
            {"role": "user", "text_excerpt": "Earlier S-101 question", "turn_index": 0},
        ],
        "topic_tags": ["S-101", "M101"],
        "had_context": True,
    }


def test_rewriter_skips_when_flag_off(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "false")
    from agentic.memory.query_rewriter import rewrite

    out = rewrite(user_query="what about it?", memory_context=_ctx_with_history())

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "flag_off"
    assert out["contextualized_query"] == "what about it?"


def test_rewriter_skips_when_no_session_context(monkeypatch):
    _enable_rewriter(monkeypatch)
    from agentic.memory.query_rewriter import rewrite

    empty_ctx = {
        "rolling_summary": None,
        "recent_turns": [],
        "semantic_turns": [],
        "topic_tags": [],
        "had_context": False,
    }
    out = rewrite(user_query="what about it?", memory_context=empty_ctx)

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "no_session_context"


def test_rewriter_skips_when_no_anaphora_markers_in_substantive_query(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=True)
    from agentic.memory import query_rewriter

    # Sentry — fail loudly if the LLM is called at all.
    def _no_call(**_kw):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(
        "agentic.generation.llm_client.generate", _no_call, raising=False
    )

    out = query_rewriter.rewrite(
        user_query="What are the fire safety requirements for atrium smoke control?",
        memory_context=_ctx_with_history(),
    )

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "no_anaphora_markers"


def test_rewriter_proceeds_when_anaphora_markers_present(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=True)
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    monkeypatch.setattr(
        llm_client,
        "generate",
        lambda **_kw: "What is drawing M101?",
    )

    out = query_rewriter.rewrite(
        user_query="what about the second one?",
        memory_context=_ctx_with_history(),
    )

    assert out["was_rewritten"] is True
    assert out["skip_reason"] is None
    assert out["contextualized_query"] == "What is drawing M101?"


def test_rewriter_returns_pass_through_when_already_self_contained(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=False)  # force LLM path
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    # LLM echoes the query back unchanged.
    monkeypatch.setattr(
        llm_client,
        "generate",
        lambda **_kw: "What is M101?",
    )

    out = query_rewriter.rewrite(
        user_query="What is M101?",
        memory_context=_ctx_with_history(),
    )

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "already_self_contained"
    assert out["contextualized_query"] == "What is M101?"


def test_rewriter_resolves_pronouns_using_recent_turns(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=True)
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    captured: Dict[str, Any] = {}

    def fake_generate(**kw):
        captured.update(kw)
        return "What is the mechanical layout for drawing M101?"

    monkeypatch.setattr(llm_client, "generate", fake_generate)

    # Use a longer follow-up so the legitimate expansion stays under
    # the 3x output-length sanity guard.
    user_query = "ok so what about that one — can you tell me more?"
    out = query_rewriter.rewrite(
        user_query=user_query,
        memory_context=_ctx_with_history(),
    )

    assert out["was_rewritten"] is True
    assert "M101" in out["contextualized_query"]
    # Verify the prompt actually packed the recent turns.
    assert "M101" in captured["user_prompt"]
    assert "Current question:" in captured["user_prompt"]


def test_rewriter_falls_back_when_output_too_long(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=False)
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    monkeypatch.setattr(
        llm_client,
        "generate",
        lambda **_kw: "x " * 500,  # way more than 3x the query length
    )

    short_query = "what about it?"
    out = query_rewriter.rewrite(
        user_query=short_query, memory_context=_ctx_with_history()
    )

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "rewriter_output_invalid"
    assert out["contextualized_query"] == short_query


def test_rewriter_falls_back_when_output_has_forbidden_pattern(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=False)
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    monkeypatch.setattr(
        llm_client,
        "generate",
        lambda **_kw: "Here is the rewritten query: What is M101?",
    )

    out = query_rewriter.rewrite(
        user_query="what about it?", memory_context=_ctx_with_history()
    )

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "rewriter_output_invalid"
    assert out["contextualized_query"] == "what about it?"


def test_rewriter_strips_quotes_from_output(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=True)
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    monkeypatch.setattr(
        llm_client,
        "generate",
        lambda **_kw: '"What is drawing M101?"',
    )

    out = query_rewriter.rewrite(
        user_query="what about it?",
        memory_context=_ctx_with_history(),
    )

    assert out["was_rewritten"] is True
    assert out["contextualized_query"] == "What is drawing M101?"


def test_rewriter_swallows_llm_errors(monkeypatch):
    _enable_rewriter(monkeypatch, heuristic=True)
    from agentic.generation import llm_client
    from agentic.memory import query_rewriter

    def explode(**_kw):
        raise RuntimeError("Anthropic 503")

    monkeypatch.setattr(llm_client, "generate", explode)

    out = query_rewriter.rewrite(
        user_query="what about it?",
        memory_context=_ctx_with_history(),
    )

    assert out["was_rewritten"] is False
    assert out["skip_reason"] == "rewriter_llm_error"
    assert out["contextualized_query"] == "what about it?"
