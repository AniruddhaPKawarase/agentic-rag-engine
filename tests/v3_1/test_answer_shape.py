"""Tests for Agent 0.5 — Answer Shape Classifier.

Covers:
* Master flag off → default shape (no behaviour change downstream).
* Image override fast-path.
* All 5 regex fast-path buckets.
* LLM fallback (mocked at boundary — no real API calls).
* Env-var overrides for target lengths.
* Synthesizer + Stylist consumes shape rule in system prompt.
* Chain threads the shape dict through to both downstream agents.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the v3.1 worktree is importable regardless of pytest CWD.
_WORKTREE = Path(__file__).resolve().parent.parent.parent
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from agentic.generation import answer_shape as shape_mod  # noqa: E402
from agentic.generation import stylist as stylist_mod  # noqa: E402
from agentic.generation import synthesizer as synth_mod  # noqa: E402
from gateway import generation_chain as gc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shape_on(monkeypatch):
    """Enable the master flag for the classifier."""
    monkeypatch.setenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", "true")


# ---------------------------------------------------------------------------
# 1. Skip rule — flag off
# ---------------------------------------------------------------------------


def test_disabled_flag_returns_default_shape(monkeypatch):
    """Flag explicitly false → returns default with skip_reason='flag_off'.

    This is the contract that downstream agents rely on for byte-identical
    behaviour when the classifier is off. Note: in v3.1 the *default* is
    on, so callers must set the env var to "false" explicitly to disable.
    """
    monkeypatch.setenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", "false")
    out = shape_mod.classify_shape(user_query="How many doors?")
    assert out["shape"] == "default"
    assert out["target_length_chars"] == 0
    assert out["target_word_count"] == 0
    assert out["was_llm_classified"] is False
    assert out["confidence"] == 0.0
    assert out["skip_reason"] == "flag_off"

    monkeypatch.setenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", "FALSE")
    out2 = shape_mod.classify_shape(user_query="How many doors?")
    assert out2["shape"] == "default"
    assert out2["skip_reason"] == "flag_off"


def test_default_flag_is_on_in_v31(monkeypatch):
    """Unset env var → default is now ON (v3.1 explicit choice)."""
    monkeypatch.delenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", raising=False)
    out = shape_mod.classify_shape(user_query="How many doors are there?")
    # "How many" is a count regex hit → should NOT be 'default'
    assert out["shape"] == "count"
    assert out["skip_reason"] is None


# ---------------------------------------------------------------------------
# 2. Image override fast-path
# ---------------------------------------------------------------------------


def test_image_factoid_override(shape_on):
    """has_image=True + short query + level/floor → forced factoid."""
    out = shape_mod.classify_shape(
        user_query="What level is this", has_image=True
    )
    assert out["shape"] == "factoid"
    assert out["was_llm_classified"] is False
    assert out["confidence"] >= 0.9
    assert out["target_length_chars"] == 80


# ---------------------------------------------------------------------------
# 3. Regex fast-path — one test per bucket
# ---------------------------------------------------------------------------


def test_regex_count_pattern(shape_on):
    out = shape_mod.classify_shape(user_query="How many DOAS units are there?")
    assert out["shape"] == "count"
    assert out["was_llm_classified"] is False
    assert out["confidence"] == 0.9
    assert out["target_length_chars"] == 60


def test_regex_factoid_size(shape_on):
    out = shape_mod.classify_shape(user_query="What is the duct size?")
    assert out["shape"] == "factoid"
    assert out["was_llm_classified"] is False


def test_regex_factoid_height(shape_on):
    out = shape_mod.classify_shape(user_query="What is the ceiling height?")
    assert out["shape"] == "factoid"


def test_regex_factoid_level(shape_on):
    out = shape_mod.classify_shape(user_query="What level is this drawing for?")
    assert out["shape"] == "factoid"


def test_regex_list_pattern(shape_on):
    out = shape_mod.classify_shape(user_query="List all electrical panels")
    assert out["shape"] == "list"
    assert out["target_length_chars"] == 300


def test_regex_comparison_pattern(shape_on):
    out = shape_mod.classify_shape(user_query="Compare A-101 with S-101")
    assert out["shape"] == "comparison"
    assert out["target_length_chars"] == 400


def test_regex_explanation_pattern(shape_on):
    out = shape_mod.classify_shape(
        user_query="Explain how the HVAC system works"
    )
    assert out["shape"] == "explanation"
    assert out["target_length_chars"] == 600


# ---------------------------------------------------------------------------
# 4. LLM fallback
# ---------------------------------------------------------------------------


def test_llm_fallback_when_no_regex_match(shape_on, monkeypatch):
    """Mock llm_client.generate at the import boundary inside answer_shape.

    The classifier uses a lazy import, so we patch the function as it
    will be imported by ``classify_shape``.
    """
    import agentic.generation.llm_client as llm_client_mod

    calls = {"count": 0, "kwargs": None}

    def fake_generate(**kwargs):
        calls["count"] += 1
        calls["kwargs"] = kwargs
        return "factoid"

    monkeypatch.setattr(llm_client_mod, "generate", fake_generate)

    # Query that doesn't match any regex (no count/factoid/list/etc keywords).
    query = "I want to understand the broader context here please"
    out = shape_mod.classify_shape(user_query=query)
    assert out["shape"] == "factoid"
    assert out["was_llm_classified"] is True
    assert out["confidence"] == 0.7
    assert calls["count"] == 1
    # System prompt requires the classifier instruction.
    assert "classify" in calls["kwargs"]["system_prompt"].lower()
    assert calls["kwargs"]["max_tokens"] == 20
    assert calls["kwargs"]["temperature"] == 0.0


def test_llm_fallback_invalid_response_returns_default(shape_on, monkeypatch):
    """Mock returns a label not in the taxonomy → falls back to default."""
    import agentic.generation.llm_client as llm_client_mod

    monkeypatch.setattr(
        llm_client_mod, "generate", lambda **_kw: "weird unknown text"
    )

    query = "I want to understand the broader context here please"
    out = shape_mod.classify_shape(user_query=query)
    assert out["shape"] == "default"
    assert out["was_llm_classified"] is False
    assert out["skip_reason"] == "no_match"


def test_llm_fallback_swallows_errors(shape_on, monkeypatch):
    """Mock raises an exception → classifier returns default, never crashes."""
    import agentic.generation.llm_client as llm_client_mod

    def boom(**_kw):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(llm_client_mod, "generate", boom)

    query = "I want to understand the broader context here please"
    out = shape_mod.classify_shape(user_query=query)
    assert out["shape"] == "default"
    assert out["skip_reason"] == "no_match"


# ---------------------------------------------------------------------------
# 5. Env-tunable target lengths
# ---------------------------------------------------------------------------


def test_target_lengths_respect_env_overrides(shape_on, monkeypatch):
    monkeypatch.setenv("SHAPE_FACTOID_MAX_CHARS", "42")
    out = shape_mod.classify_shape(user_query="What is the duct size?")
    assert out["shape"] == "factoid"
    assert out["target_length_chars"] == 42
    assert out["target_word_count"] == 42 // 5


# ---------------------------------------------------------------------------
# 6. Synthesizer + Stylist consume the shape rule
# ---------------------------------------------------------------------------


def test_synthesizer_uses_shape_in_prompt(mocker):
    """When answer_shape is set, system prompt carries ANSWER_SHAPE block + reduced max_tokens."""
    fake = mocker.patch.object(synth_mod, "generate", return_value="ok")

    long_raw = "X" * 500
    shape = {
        "shape": "factoid",
        "target_length_chars": 80,
        "target_word_count": 16,
        "was_llm_classified": False,
        "confidence": 0.9,
        "skip_reason": None,
    }
    synth_mod.synthesize(
        raw_answer=long_raw,
        user_query="What is the duct size?",
        source_docs=[],
        answer_shape=shape,
    )

    kwargs = fake.call_args.kwargs
    assert "ANSWER_SHAPE: factoid" in kwargs["system_prompt"]
    assert "LENGTH_BUDGET" in kwargs["system_prompt"]
    assert "FORMAT_RULE" in kwargs["system_prompt"]
    # max_tokens reduced relative to legacy 400 default — factoid budget is tight.
    assert kwargs["max_tokens"] < 400
    assert kwargs["max_tokens"] >= 80


def test_stylist_uses_shape_in_prompt(mocker):
    fake = mocker.patch.object(stylist_mod, "generate", return_value="ok")

    long_draft = "Y" * 500
    shape = {
        "shape": "list",
        "target_length_chars": 300,
        "target_word_count": 60,
        "was_llm_classified": False,
        "confidence": 0.9,
        "skip_reason": None,
    }
    stylist_mod.stylize(
        draft_answer=long_draft,
        user_query="List all panels",
        answer_shape=shape,
    )
    kwargs = fake.call_args.kwargs
    assert "ANSWER_SHAPE: list" in kwargs["system_prompt"]
    assert "FORMAT_RULE" in kwargs["system_prompt"]
    # max_tokens scales with target word count.
    assert kwargs["max_tokens"] == max(80, 60 * 2)


def test_synthesizer_default_shape_does_not_modify_prompt(mocker):
    """shape='default' or None must produce the legacy system prompt verbatim."""
    fake_default = mocker.patch.object(synth_mod, "generate", return_value="ok")
    long_raw = "Z" * 500

    # No answer_shape kwarg — legacy behaviour.
    synth_mod.synthesize(
        raw_answer=long_raw,
        user_query="long story",
        source_docs=[],
    )
    legacy_prompt = fake_default.call_args.kwargs["system_prompt"]
    legacy_max_tokens = fake_default.call_args.kwargs["max_tokens"]

    # answer_shape with shape='default' — must be byte-identical.
    fake_default.reset_mock()
    default_shape = {
        "shape": "default",
        "target_length_chars": 0,
        "target_word_count": 0,
        "was_llm_classified": False,
        "confidence": 0.0,
        "skip_reason": "flag_off",
    }
    synth_mod.synthesize(
        raw_answer=long_raw,
        user_query="long story",
        source_docs=[],
        answer_shape=default_shape,
    )
    assert fake_default.call_args.kwargs["system_prompt"] == legacy_prompt
    assert fake_default.call_args.kwargs["max_tokens"] == legacy_max_tokens


# ---------------------------------------------------------------------------
# 7. Chain integration
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    answer: str = "raw answer from ReAct retrieval " * 20
    sources: list = field(default_factory=lambda: ["S-101"])
    source_docs: list = field(
        default_factory=lambda: [{"drawing_name": "S-101", "page": 1, "text_excerpt": "x"}]
    )
    confidence: str = "high"
    total_cost_usd: float = 0.01
    total_steps: int = 2
    total_input_tokens: int = 100
    total_output_tokens: int = 50
    model: str = "gpt-4o-mini"
    follow_up_questions: list = field(default_factory=list)


def test_chain_passes_shape_through_to_synth_and_stylist(monkeypatch):
    """End-to-end chain wiring: classify_shape result threads into synth + stylist calls."""
    # Enable everything needed for the full pipeline.
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "false")
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "false")
    monkeypatch.setenv("ANSWER_SYNTHESIZER_ENABLED", "true")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "true")
    monkeypatch.setenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", "true")

    # Mock the agent + sub-agents.
    fake_agent_mod = MagicMock()
    fake_agent_mod.run_agent = MagicMock(return_value=FakeAgentResult())

    fake_synth_mod = MagicMock()
    fake_synth_mod.synthesize = MagicMock(return_value="synthesized draft")

    fake_stylist_mod = MagicMock()
    fake_stylist_mod.stylize = MagicMock(return_value="polished final")
    fake_stylist_mod.reexpress_cached = MagicMock(return_value="re-expressed")

    fake_writer_mod = MagicMock()
    fake_writer_mod.MemoryWriter = MagicMock(
        return_value=MagicMock(write_turn_async=MagicMock())
    )

    monkeypatch.setitem(sys.modules, "agentic.core.agent", fake_agent_mod)
    monkeypatch.setitem(sys.modules, "agentic.generation.synthesizer", fake_synth_mod)
    monkeypatch.setitem(sys.modules, "agentic.generation.stylist", fake_stylist_mod)
    monkeypatch.setitem(sys.modules, "agentic.memory.writer", fake_writer_mod)

    resp = asyncio.run(
        gc.run_generation_chain(
            user_query="How many DOAS units are in the project?",
            session_id=None,
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    # Synth + stylist were called WITH answer_shape kwarg (count bucket).
    fake_synth_mod.synthesize.assert_called_once()
    synth_kwargs = fake_synth_mod.synthesize.call_args.kwargs
    assert "answer_shape" in synth_kwargs
    assert synth_kwargs["answer_shape"]["shape"] == "count"

    fake_stylist_mod.stylize.assert_called_once()
    stylize_kwargs = fake_stylist_mod.stylize.call_args.kwargs
    assert "answer_shape" in stylize_kwargs
    assert stylize_kwargs["answer_shape"]["shape"] == "count"

    # Response surfaces shape telemetry.
    assert resp["answer_shape"] == "count"
    assert resp["target_length_chars"] == 60


def test_chain_with_shape_disabled_does_not_pass_shape_kwarg(monkeypatch):
    """When the classifier flag is off, downstream agents must NOT receive the kwarg.

    This is the byte-identical-behaviour guarantee. Existing test mocks
    whose signatures predate the answer_shape kwarg must keep working.
    """
    monkeypatch.setenv("V31_CHAIN_ENABLED", "true")
    monkeypatch.setenv("MEMORY_RECALL_ENABLED", "false")
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "false")
    monkeypatch.setenv("ANSWER_SYNTHESIZER_ENABLED", "true")
    monkeypatch.setenv("STYLE_REWRITER_ENABLED", "true")
    # v3.1 default is now ON, so explicitly disable for this test.
    monkeypatch.setenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", "false")

    fake_agent_mod = MagicMock()
    fake_agent_mod.run_agent = MagicMock(return_value=FakeAgentResult())
    fake_synth_mod = MagicMock()
    fake_synth_mod.synthesize = MagicMock(return_value="synthesized draft")
    fake_stylist_mod = MagicMock()
    fake_stylist_mod.stylize = MagicMock(return_value="polished")
    fake_stylist_mod.reexpress_cached = MagicMock()
    fake_writer_mod = MagicMock()
    fake_writer_mod.MemoryWriter = MagicMock(
        return_value=MagicMock(write_turn_async=MagicMock())
    )
    monkeypatch.setitem(sys.modules, "agentic.core.agent", fake_agent_mod)
    monkeypatch.setitem(sys.modules, "agentic.generation.synthesizer", fake_synth_mod)
    monkeypatch.setitem(sys.modules, "agentic.generation.stylist", fake_stylist_mod)
    monkeypatch.setitem(sys.modules, "agentic.memory.writer", fake_writer_mod)

    asyncio.run(
        gc.run_generation_chain(
            user_query="How many doors?",
            session_id=None,
            project_id=1,
            set_id=None,
            scope=None,
        )
    )

    # Critical: when classifier flag is off, do NOT thread the kwarg through.
    synth_kwargs = fake_synth_mod.synthesize.call_args.kwargs
    assert "answer_shape" not in synth_kwargs
    stylize_kwargs = fake_stylist_mod.stylize.call_args.kwargs
    assert "answer_shape" not in stylize_kwargs
