"""Tests for Phase P4 — gateway.generation_chain.

Pure orchestration tests. Every sub-agent (recall, rewrite, run_agent,
synthesize, stylize, reexpress_cached, MemoryWriter) is mocked at its
import point inside ``gateway.generation_chain``.

Coverage targets:
* Branch A (cache hit) — flag-off and flag-on paths.
* Branch B (full pipeline) — happy path, every skip rule, every failure
  fallback, fire-and-forget writer, streaming protocol.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Make the v3.1 worktree importable regardless of pytest's CWD.
# ---------------------------------------------------------------------------
_WORKTREE = Path(__file__).resolve().parent.parent.parent
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from gateway import generation_chain as gc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    """Mimics agentic.core.agent.AgentResult."""

    answer: str = "raw answer from ReAct retrieval " * 20  # >200 chars
    sources: list = field(default_factory=lambda: ["S-101", "M-204"])
    source_docs: list = field(
        default_factory=lambda: [
            {"drawing_name": "S-101", "page": 1, "text_excerpt": "Foo"},
            {"drawing_name": "M-204", "page": 3, "text_excerpt": "Bar"},
        ]
    )
    confidence: str = "high"
    total_cost_usd: float = 0.012
    total_steps: int = 4
    total_input_tokens: int = 1200
    total_output_tokens: int = 350
    model: str = "gpt-4.1"
    follow_up_questions: list = field(default_factory=list)


@pytest.fixture
def all_flags_on(monkeypatch):
    """Enable every v3.1 sub-flag so the chain runs the full pipeline."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "true")
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_SYNTHESIZER_ENABLED", "true")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "true")
    monkeypatch.setenv("CACHE_REEXPRESSION_ENABLED", "true")


@pytest.fixture
def patch_chain(monkeypatch):
    """Patch every sub-agent inside ``gateway.generation_chain``.

    Returns a dict of MagicMocks the test can configure / assert on.
    """
    fake_recall_mod = MagicMock()
    fake_recall_mod.recall = MagicMock(
        return_value={
            "rolling_summary": "We discussed S-101 fire safety.",
            "recent_turns": [
                {"role": "user", "content": "Earlier user msg", "turn_index": 0},
                {"role": "assistant", "content": "Earlier assistant reply", "turn_index": 1},
            ],
            "semantic_turns": [],
            "topic_tags": ["S-101"],
            "had_context": True,
        }
    )

    fake_rewriter_mod = MagicMock()
    fake_rewriter_mod.rewrite = MagicMock(
        return_value={
            "contextualized_query": "rewritten standalone query",
            "was_rewritten": True,
            "skip_reason": None,
        }
    )

    fake_agent_mod = MagicMock()
    fake_agent_mod.run_agent = MagicMock(return_value=FakeAgentResult())

    fake_synth_mod = MagicMock()
    fake_synth_mod.synthesize = MagicMock(return_value="synthesized draft")

    fake_stylist_mod = MagicMock()
    fake_stylist_mod.stylize = MagicMock(return_value="polished final answer")
    fake_stylist_mod.reexpress_cached = MagicMock(return_value="re-expressed answer")

    fake_writer_instance = MagicMock()
    fake_writer_instance.write_turn_async = MagicMock()
    fake_writer_mod = MagicMock()
    fake_writer_mod.MemoryWriter = MagicMock(return_value=fake_writer_instance)

    monkeypatch.setitem(sys.modules, "agentic.memory.recall_agent", fake_recall_mod)
    monkeypatch.setitem(sys.modules, "agentic.memory.query_rewriter", fake_rewriter_mod)
    monkeypatch.setitem(sys.modules, "agentic.core.agent", fake_agent_mod)
    monkeypatch.setitem(sys.modules, "agentic.generation.synthesizer", fake_synth_mod)
    monkeypatch.setitem(sys.modules, "agentic.generation.stylist", fake_stylist_mod)
    monkeypatch.setitem(sys.modules, "agentic.memory.writer", fake_writer_mod)

    return {
        "recall": fake_recall_mod.recall,
        "rewrite": fake_rewriter_mod.rewrite,
        "run_agent": fake_agent_mod.run_agent,
        "synthesize": fake_synth_mod.synthesize,
        "stylize": fake_stylist_mod.stylize,
        "reexpress_cached": fake_stylist_mod.reexpress_cached,
        "memory_writer_cls": fake_writer_mod.MemoryWriter,
        "memory_writer_instance": fake_writer_instance,
    }


def _run(coro):
    """Helper: run an async chain call to completion."""
    return asyncio.run(coro)


async def _collect(stream):
    """Collect every event from an async generator."""
    out: List[dict] = []
    async for evt in stream:
        out.append(evt)
    return out


# ===========================================================================
# 1. Master kill switch
# ===========================================================================


def test_chain_disabled_falls_through_to_legacy(monkeypatch):
    """Default-off behaviour: chain_enabled() is False unless explicitly set."""
    monkeypatch.delenv("V31_CHAIN_ENABLED", raising=False)
    assert gc.chain_enabled() is False
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    assert gc.chain_enabled() is True
    monkeypatch.setenv("V31_CHAIN_ENABLED", "FALSE")
    assert gc.chain_enabled() is False


# ===========================================================================
# 2. Branch B — happy full pipeline
# ===========================================================================


def test_chain_full_pipeline_no_cache(all_flags_on, patch_chain):
    """All flags on → recall → rewrite → ReAct → synth → stylist → writer."""
    resp = _run(
        gc.run_generation_chain(
            user_query="What about the second drawing?",
            session_id="sess-1",
            project_id=42,
            set_id=7,
            scope=None,
        )
    )

    assert resp["answer"] == "polished final answer"
    assert resp["memory_context_used"] is True
    assert resp["query_rewritten"] is True
    assert resp["synthesizer_used"] is True
    assert resp["stylist_used"] is True
    assert resp["cache_reexpressed"] is False
    assert resp["engine_used"] == "agentic"
    assert resp["success"] is True
    # All six agents fired.
    patch_chain["recall"].assert_called_once()
    patch_chain["rewrite"].assert_called_once()
    patch_chain["run_agent"].assert_called_once()
    patch_chain["synthesize"].assert_called_once()
    patch_chain["stylize"].assert_called_once()
    # ReAct received the *rewritten* query, not the original.
    assert (
        patch_chain["run_agent"].call_args.kwargs["query"]
        == "rewritten standalone query"
    )


# ===========================================================================
# 3. Branch A — cache hit
# ===========================================================================


def test_chain_cache_hit_with_reexpression_disabled_returns_cached_unchanged(
    monkeypatch, patch_chain
):
    """CACHE_REEXPRESSION_ENABLED=false → cached answer untouched."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("CACHE_REEXPRESSION_ENABLED", "false")

    cached = {
        "answer": "cached body",
        "session_id": "sess-1",
        "source_documents": [{"foo": 1}],
        "engine_used": "agentic",
    }

    resp = _run(
        gc.run_generation_chain(
            user_query="follow-up?",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
            cached_result=cached,
        )
    )

    assert resp["answer"] == "cached body"
    assert resp["cache_reexpressed"] is False
    assert resp["source_documents"] == [{"foo": 1}]
    patch_chain["reexpress_cached"].assert_not_called()
    patch_chain["recall"].assert_not_called()


def test_chain_cache_hit_with_reexpression_enabled_runs_stylist(
    monkeypatch, patch_chain
):
    """CACHE_REEXPRESSION_ENABLED=true → cached answer is re-expressed."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "true")
    monkeypatch.setenv("CACHE_REEXPRESSION_ENABLED", "true")

    cached = {
        "answer": "cached body",
        "session_id": "sess-1",
        "source_documents": [{"foo": 1}],
        "engine_used": "agentic",
    }

    resp = _run(
        gc.run_generation_chain(
            user_query="follow-up?",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
            cached_result=cached,
        )
    )

    assert resp["answer"] == "re-expressed answer"
    assert resp["cache_reexpressed"] is True
    # Source docs preserved through re-expression.
    assert resp["source_documents"] == [{"foo": 1}]
    patch_chain["reexpress_cached"].assert_called_once()
    # ReAct was NOT run.
    patch_chain["run_agent"].assert_not_called()


# ===========================================================================
# 4. Skip rules
# ===========================================================================


def test_chain_skips_recall_when_no_session_id(all_flags_on, patch_chain):
    """No session_id → skip Memory Recall (don't call the function)."""
    resp = _run(
        gc.run_generation_chain(
            user_query="standalone query",
            session_id=None,
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    patch_chain["recall"].assert_not_called()
    assert resp["memory_context_used"] is False


def test_chain_skips_rewriter_when_no_anaphora(monkeypatch, patch_chain):
    """QUERY_REWRITER_ENABLED=false → skip rewriter, ReAct uses original."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "true")
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "false")
    monkeypatch.setenv("ANSWER_SYNTHESIZER_ENABLED", "false")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "false")

    resp = _run(
        gc.run_generation_chain(
            user_query="What are the fire-safety zones?",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    patch_chain["rewrite"].assert_not_called()
    assert resp["query_rewritten"] is False
    # ReAct received the original query.
    assert (
        patch_chain["run_agent"].call_args.kwargs["query"]
        == "What are the fire-safety zones?"
    )


def test_chain_skips_synthesizer_when_answer_short(monkeypatch, patch_chain):
    """ANSWER_SYNTHESIZER_ENABLED=false → skip synthesizer."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("ANSWER_SYNTHESIZER_ENABLED", "false")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "false")

    resp = _run(
        gc.run_generation_chain(
            user_query="hi",
            session_id=None,
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    patch_chain["synthesize"].assert_not_called()
    assert resp["synthesizer_used"] is False


def test_chain_skips_stylist_when_draft_short(monkeypatch, patch_chain):
    """STYLE_REWRITER_ENABLED=false → skip stylist."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "false")

    resp = _run(
        gc.run_generation_chain(
            user_query="hi",
            session_id=None,
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    patch_chain["stylize"].assert_not_called()
    assert resp["stylist_used"] is False


# ===========================================================================
# 5. Memory writer fire-and-forget
# ===========================================================================


def test_chain_dispatches_writer_after_response(all_flags_on, patch_chain):
    """Writer is dispatched once with the final answer + original query."""
    _run(
        gc.run_generation_chain(
            user_query="orig question",
            session_id="sess-1",
            project_id=42,
            set_id=7,
            scope=None,
        )
    )

    writer = patch_chain["memory_writer_instance"]
    writer.write_turn_async.assert_called_once()
    kwargs = writer.write_turn_async.call_args.kwargs
    assert kwargs["session_id"] == "sess-1"
    # Writer receives the *original* user query (not the rewritten one)
    # so memory reflects what the user actually said.
    assert kwargs["user_text"] == "orig question"
    assert kwargs["assistant_text"] == "polished final answer"
    assert kwargs["project_id"] == 42
    assert kwargs["set_id"] == "7"


def test_chain_writer_is_fire_and_forget(all_flags_on, patch_chain):
    """If the writer raises, the chain still returns successfully."""
    patch_chain["memory_writer_instance"].write_turn_async.side_effect = RuntimeError(
        "exec pool gone"
    )

    resp = _run(
        gc.run_generation_chain(
            user_query="orig question",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    assert resp["success"] is True
    assert resp["answer"] == "polished final answer"


# ===========================================================================
# 6. Telemetry / metadata
# ===========================================================================


def test_chain_returns_metadata_about_skipped_agents(monkeypatch, patch_chain):
    """The response carries flags for which v3.1 agents fired."""
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "true")
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "false")
    monkeypatch.setenv("ANSWER_SYNTHESIZER_ENABLED", "true")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "false")

    resp = _run(
        gc.run_generation_chain(
            user_query="question?",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    assert resp["memory_context_used"] is True
    assert resp["query_rewritten"] is False
    assert resp["synthesizer_used"] is True
    assert resp["stylist_used"] is False
    assert resp["cache_reexpressed"] is False


# ===========================================================================
# 7. Streaming
# ===========================================================================


def test_chain_streams_tokens_when_stream_true(all_flags_on, patch_chain):
    """stream=True yields token events as the stylist streams chunks."""

    def stylize_stream(*, draft_answer, user_query, last_assistant_turn, stream):
        if stream:
            return iter(["polished ", "final ", "answer"])
        return "polished final answer"

    patch_chain["stylize"].side_effect = stylize_stream

    def synth_stream(*, raw_answer, user_query, source_docs, rolling_summary, stream):
        if stream:
            return iter(["synth-chunk-1 ", "synth-chunk-2"])
        return "synthesized draft"

    patch_chain["synthesize"].side_effect = synth_stream

    async def go():
        events = []
        async for evt in gc.run_generation_chain(
            user_query="streaming please",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
            stream=True,
        ):
            events.append(evt)
        return events

    events = _run(go())

    assert events[0]["event"] == "metadata"
    token_chunks = [e["delta"] for e in events if e["event"] == "token"]
    # When stylist is on, only stylist chunks are streamed (synth tokens are buffered).
    assert "polished " in token_chunks
    assert "final " in token_chunks
    assert "answer" in token_chunks
    assert events[-1]["event"] == "done"


def test_chain_yields_metadata_then_tokens_then_done(all_flags_on, patch_chain):
    """Streaming protocol: metadata first, tokens middle, done last."""

    def stylize_stream(*, draft_answer, user_query, last_assistant_turn, stream):
        if stream:
            return iter(["a", "b", "c"])
        return "abc"

    patch_chain["stylize"].side_effect = stylize_stream

    def synth_stream(*, raw_answer, user_query, source_docs, rolling_summary, stream):
        if stream:
            return iter(["x"])
        return "x"

    patch_chain["synthesize"].side_effect = synth_stream

    async def go():
        events = []
        async for evt in gc.run_generation_chain(
            user_query="ordering test",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
            stream=True,
        ):
            events.append(evt)
        return events

    events = _run(go())

    kinds = [e["event"] for e in events]
    assert kinds[0] == "metadata"
    assert kinds[-1] == "done"
    # No tokens after `done`.
    assert "token" not in kinds[kinds.index("done") + 1 :]
    # Done event carries the final response dict.
    assert events[-1]["metadata"]["answer"] == "abc"


# ===========================================================================
# 8. Failure resilience
# ===========================================================================


def test_chain_handles_react_failure_gracefully(all_flags_on, patch_chain):
    """If ReAct raises, the chain still returns a response (with error set)."""
    patch_chain["run_agent"].side_effect = RuntimeError("DB timeout")

    resp = _run(
        gc.run_generation_chain(
            user_query="question?",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    assert resp["success"] is False
    assert "DB timeout" in (resp["error"] or "")
    # Chain didn't crash; downstream agents (synth/stylist) saw empty answer
    # and short-circuited, so the final answer is empty string.
    assert resp["answer"] == ""


def test_chain_recall_failure_doesnt_break_pipeline(all_flags_on, patch_chain):
    """A blowup inside Memory Recall is swallowed; pipeline continues."""
    patch_chain["recall"].side_effect = RuntimeError("Atlas down")

    resp = _run(
        gc.run_generation_chain(
            user_query="question?",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    assert resp["success"] is True
    assert resp["memory_context_used"] is False
    # Chain still produced a final answer through the rest of the pipeline.
    assert resp["answer"] == "polished final answer"


def test_chain_rewriter_failure_falls_back_to_original_query(all_flags_on, patch_chain):
    """Rewriter exception → ReAct receives the *original* user query."""
    patch_chain["rewrite"].side_effect = RuntimeError("model 500")

    _run(
        gc.run_generation_chain(
            user_query="original user query verbatim",
            session_id="sess-1",
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    assert (
        patch_chain["run_agent"].call_args.kwargs["query"]
        == "original user query verbatim"
    )


# ===========================================================================
# 9. Additional sanity
# ===========================================================================


def test_chain_response_carries_session_and_project(all_flags_on, patch_chain):
    resp = _run(
        gc.run_generation_chain(
            user_query="q",
            session_id="sess-9",
            project_id=42,
            set_id=None,
            scope=None,
        )
    )
    assert resp["session_id"] == "sess-9"
    assert resp["project_id"] == 42
    assert "processing_time_ms" in resp
    assert resp["processing_time_ms"] >= 0
