"""Tests for the v3.1 generation chain (P3): llm_client + synthesizer + stylist.

All provider SDKs are mocked at the boundary — no live API calls.
"""

from __future__ import annotations

import sys
import types
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from agentic.generation import llm_client, stylist, synthesizer


# ── Shared fixtures ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Ensure each test starts with no cached clients and known env vars."""
    llm_client._reset_clients_for_tests()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anth-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    yield
    llm_client._reset_clients_for_tests()


def _make_anthropic_message(text: str, in_tok: int = 10, out_tok: int = 20):
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    usage = MagicMock()
    usage.input_tokens = in_tok
    usage.output_tokens = out_tok
    msg.usage = usage
    return msg


def _make_openai_completion(text: str, in_tok: int = 10, out_tok: int = 20):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    completion.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = in_tok
    usage.completion_tokens = out_tok
    completion.usage = usage
    return completion


def _install_fake_anthropic(monkeypatch, *, messages_create_side_effect=None,
                            stream_text_chunks=None, final_message=None):
    """Install a fake `anthropic` module in sys.modules.

    Returns the messages mock so tests can assert on calls.
    """
    fake_module = types.ModuleType("anthropic")

    client_mock = MagicMock(name="AnthropicClient")
    if messages_create_side_effect is not None:
        client_mock.messages.create.side_effect = messages_create_side_effect

    if stream_text_chunks is not None:
        # Build a context-manager mock that yields the chunks.
        stream_ctx = MagicMock()
        stream_ctx.text_stream = iter(stream_text_chunks)
        stream_ctx.get_final_message.return_value = (
            final_message or _make_anthropic_message("".join(stream_text_chunks))
        )
        cm = MagicMock()
        cm.__enter__.return_value = stream_ctx
        cm.__exit__.return_value = False
        client_mock.messages.stream.return_value = cm

    fake_module.Anthropic = MagicMock(return_value=client_mock)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return client_mock


def _install_fake_openai(monkeypatch, *, chat_create_side_effect=None,
                        stream_events=None):
    """Install a fake `openai` module in sys.modules."""
    fake_module = types.ModuleType("openai")

    client_mock = MagicMock(name="OpenAIClient")
    if chat_create_side_effect is not None:
        client_mock.chat.completions.create.side_effect = chat_create_side_effect
    if stream_events is not None:
        client_mock.chat.completions.create.return_value = iter(stream_events)

    fake_module.OpenAI = MagicMock(return_value=client_mock)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return client_mock


# ── llm_client tests ───────────────────────────────────────────────────


def test_llm_client_routes_to_anthropic_for_claude_model(monkeypatch):
    anth_client = _install_fake_anthropic(
        monkeypatch,
        messages_create_side_effect=[_make_anthropic_message("hello from claude")],
    )
    # Even with openai installed, claude-* must NOT call it.
    oai_client = _install_fake_openai(monkeypatch)

    out = llm_client.generate(
        system_prompt="sys",
        user_prompt="hi",
        model="claude-haiku-4-5",
        fallback_model="gpt-4o-mini",
    )

    assert out == "hello from claude"
    assert anth_client.messages.create.call_count == 1
    assert oai_client.chat.completions.create.call_count == 0
    # Verify provider received the system prompt at the right place.
    kwargs = anth_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "sys"
    assert kwargs["model"] == "claude-haiku-4-5"


def test_llm_client_routes_to_openai_for_gpt_model(monkeypatch):
    oai_client = _install_fake_openai(
        monkeypatch,
        chat_create_side_effect=[_make_openai_completion("hello from gpt")],
    )
    anth_client = _install_fake_anthropic(monkeypatch)

    out = llm_client.generate(
        system_prompt="sys",
        user_prompt="hi",
        model="gpt-4o-mini",
        fallback_model=None,
    )

    assert out == "hello from gpt"
    assert oai_client.chat.completions.create.call_count == 1
    assert anth_client.messages.create.call_count == 0
    msgs = oai_client.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "sys"
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "hi"


def test_llm_client_falls_back_on_anthropic_outage(monkeypatch):
    # Anthropic raises a retryable error on every attempt.
    class FakeOverloadedError(Exception):
        status_code = 503

    anth_client = _install_fake_anthropic(
        monkeypatch,
        messages_create_side_effect=FakeOverloadedError("overloaded"),
    )
    oai_client = _install_fake_openai(
        monkeypatch,
        chat_create_side_effect=[_make_openai_completion("served by openai")],
    )
    # Skip backoff for test speed.
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)

    out = llm_client.generate(
        system_prompt="sys",
        user_prompt="hi",
        model="claude-haiku-4-5",
        fallback_model="gpt-4o-mini",
    )

    assert out == "served by openai"
    # Both retries on anthropic, then 1 successful openai call.
    assert anth_client.messages.create.call_count == llm_client._MAX_ATTEMPTS
    assert oai_client.chat.completions.create.call_count == 1


def test_llm_client_streams_when_requested(monkeypatch):
    chunks = ["hel", "lo ", "world"]
    _install_fake_anthropic(monkeypatch, stream_text_chunks=chunks)
    _install_fake_openai(monkeypatch)

    result = llm_client.generate(
        system_prompt="sys",
        user_prompt="hi",
        model="claude-haiku-4-5",
        fallback_model="gpt-4o-mini",
        stream=True,
    )

    # Must be an iterator, not a string.
    assert not isinstance(result, str)
    collected = list(result)
    assert collected == chunks


def test_llm_client_no_retry_on_4xx_other_than_429(monkeypatch):
    """4xx (non-429) → don't retry on primary; jump to fallback."""
    class FakeBadRequest(Exception):
        status_code = 400

    anth_client = _install_fake_anthropic(
        monkeypatch,
        messages_create_side_effect=FakeBadRequest("bad request"),
    )
    oai_client = _install_fake_openai(
        monkeypatch,
        chat_create_side_effect=[_make_openai_completion("openai-served")],
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)

    out = llm_client.generate(
        system_prompt="sys",
        user_prompt="hi",
        model="claude-haiku-4-5",
        fallback_model="gpt-4o-mini",
    )

    assert out == "openai-served"
    # 400 is not retryable → primary called only once.
    assert anth_client.messages.create.call_count == 1


# ── synthesizer tests ──────────────────────────────────────────────────


def test_synthesizer_passes_through_short_answers(monkeypatch, mocker):
    """Short answer + no garble markers → no LLM call."""
    spy = mocker.patch.object(synthesizer, "generate")

    short = "There are 3 doors on sheet S-101 [S-101 p1]."
    out = synthesizer.synthesize(
        raw_answer=short,
        user_query="how many doors?",
        source_docs=[],
    )

    assert out == short
    spy.assert_not_called()


def test_synthesizer_compresses_long_raw_answer(monkeypatch, mocker):
    """Long raw answer → invokes LLM with synthesizer system prompt."""
    long_raw = "DETAILED DRAWING OCR DUMP. " * 100  # >> 200 chars
    expected = "There are 3 doors per S-101 [S-101 p1]."

    fake_generate = mocker.patch.object(synthesizer, "generate", return_value=expected)

    out = synthesizer.synthesize(
        raw_answer=long_raw,
        user_query="how many doors?",
        source_docs=[
            {"drawing_name": "S-101", "page": 1, "text_excerpt": "3 doors shown"},
        ],
        rolling_summary="Discussing structural drawings for level 1.",
    )

    assert out == expected
    fake_generate.assert_called_once()
    kwargs = fake_generate.call_args.kwargs
    # System prompt asserts core rules.
    assert "shortest" in kwargs["system_prompt"].lower()
    assert "[S-101 p1]" in kwargs["system_prompt"] or "drawing pX" in kwargs["system_prompt"].lower()
    assert "Direct Answer" in kwargs["system_prompt"]  # forbidden header listed
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 400
    # User prompt contains the truncated raw answer + summary + sources JSON.
    assert "USER QUERY" in kwargs["user_prompt"]
    assert "RAW ANSWER" in kwargs["user_prompt"]
    assert "S-101" in kwargs["user_prompt"]
    assert "Discussing structural drawings" in kwargs["user_prompt"]


def test_synthesizer_preserves_citations(monkeypatch, mocker):
    """When LLM returns text with citations, synthesizer returns them as-is."""
    long_raw = "X" * 500  # forces synthesis
    citation_answer = "Beam B-12 sits on grid C [S-201 p3] per the framing plan."
    mocker.patch.object(synthesizer, "generate", return_value=citation_answer)

    out = synthesizer.synthesize(
        raw_answer=long_raw,
        user_query="where is beam B-12?",
        source_docs=[{"drawing_name": "S-201", "page": 3, "text_excerpt": "B-12 grid C"}],
    )

    assert "[S-201 p3]" in out


def test_synthesizer_preserves_tail_citations_after_truncation(monkeypatch, mocker):
    """When raw_answer exceeds MAX_RAW_ANSWER_CHARS and tail citations
    would be dropped by a head-only slice, both head and tail
    citations must survive in the user prompt sent to the LLM.
    """
    fake_generate = mocker.patch.object(synthesizer, "generate", return_value="ok")

    # ~21k chars: "filler " (7 chars) * 3000 = 21000 chars of filler.
    # Tail citations land well past MAX_RAW_ANSWER_CHARS (16000).
    raw = "intro " + ("filler " * 3000) + "[S-101 p1] tail content [E-604 p3]"

    synthesizer.synthesize(
        raw_answer=raw,
        user_query="what survives truncation?",
        source_docs=[],
    )

    fake_generate.assert_called_once()
    user_prompt = fake_generate.call_args.kwargs["user_prompt"]
    assert "[S-101 p1]" in user_prompt
    assert "[E-604 p3]" in user_prompt
    # Sanity: tail-preserving truncation wrapper appears.
    assert "earlier passages omitted" in user_prompt


def test_synthesizer_uses_llm_when_short_but_garbled(monkeypatch, mocker):
    """Short answer with OCR garble → still synthesizes (the LLM cleans up)."""
    fake_generate = mocker.patch.object(synthesizer, "generate", return_value="cleaned")

    out = synthesizer.synthesize(
        raw_answer="short ▯▯ garbled",
        user_query="q?",
        source_docs=[],
    )

    assert out == "cleaned"
    fake_generate.assert_called_once()


# ── stylist tests ──────────────────────────────────────────────────────


def test_stylist_skips_short_polished_answers(monkeypatch, mocker):
    """Short, low-jargon draft → return as-is."""
    spy = mocker.patch.object(stylist, "generate")

    short = "There are three exit doors on level 2."
    out = stylist.stylize(draft_answer=short, user_query="exits?")

    assert out == short
    spy.assert_not_called()


def test_stylist_polishes_technical_dump(monkeypatch, mocker):
    """Long OR jargon-heavy → invokes stylist LLM."""
    polished = "Mechanical units serve floors 1-3 with VAV reheat."
    fake_generate = mocker.patch.object(stylist, "generate", return_value=polished)

    # Trigger via length: 400 chars of plain text > 300-char threshold.
    long_draft = "the system has a mechanical layout. " * 20
    out = stylist.stylize(
        draft_answer=long_draft,
        user_query="mechanical?",
        last_assistant_turn="We were looking at the HVAC layout.",
    )

    assert out == polished
    fake_generate.assert_called_once()
    kwargs = fake_generate.call_args.kwargs
    assert "polish" in kwargs["system_prompt"].lower()
    assert "ChatGPT" in kwargs["system_prompt"]
    assert kwargs["temperature"] == 0.4
    assert "PREVIOUS ASSISTANT TURN" in kwargs["user_prompt"]


def test_stylist_polishes_jargon_heavy_short_draft(monkeypatch, mocker):
    """Even short drafts get polished if jargon-dense."""
    fake_generate = mocker.patch.object(stylist, "generate", return_value="cleaned")

    # Two ALL-CAPS technical tokens within ~60 chars triggers the rule.
    out = stylist.stylize(
        draft_answer="RCP shows VAV with CFM rates [S-101 p1].",
        user_query="rcp?",
    )

    assert out == "cleaned"
    fake_generate.assert_called_once()


def test_reexpress_cached_preserves_facts(monkeypatch, mocker):
    """With session context present, reexpress invokes the LLM."""
    cached = "There are 3 doors on S-101 [S-101 p1]."
    rephrased = "Building on what we found earlier, S-101 shows 3 doors [S-101 p1]."
    fake_generate = mocker.patch.object(stylist, "generate", return_value=rephrased)

    out = stylist.reexpress_cached(
        cached_answer=cached,
        user_query="and the doors again?",
        rolling_summary="Reviewing S-101 doors.",
        last_assistant_turn="We just discussed the door schedule.",
        is_followup=True,
    )

    assert out == rephrased
    fake_generate.assert_called_once()
    kwargs = fake_generate.call_args.kwargs
    assert "re-expressing" in kwargs["system_prompt"].lower()
    assert "cached" in kwargs["system_prompt"].lower()
    assert kwargs["temperature"] == 0.5
    # The cached answer must reach the prompt unchanged.
    assert "[S-101 p1]" in kwargs["user_prompt"]


def test_reexpress_cached_skips_when_no_session_context(monkeypatch, mocker):
    """No prior turn, no summary, not a follow-up → return cached as-is."""
    spy = mocker.patch.object(stylist, "generate")

    cached = "There are 3 doors on S-101 [S-101 p1]."
    out = stylist.reexpress_cached(
        cached_answer=cached,
        user_query="how many doors?",
        rolling_summary=None,
        last_assistant_turn=None,
        is_followup=False,
    )

    assert out == cached
    spy.assert_not_called()


def test_reexpress_cached_acknowledges_followup(monkeypatch, mocker):
    """is_followup=True alone is enough session signal to re-express."""
    fake_generate = mocker.patch.object(stylist, "generate", return_value="re-said")

    out = stylist.reexpress_cached(
        cached_answer="Original cached fact [S-1 p1].",
        user_query="and again?",
        rolling_summary=None,
        last_assistant_turn=None,
        is_followup=True,
    )

    assert out == "re-said"
    fake_generate.assert_called_once()
    kwargs = fake_generate.call_args.kwargs
    assert "FOLLOW-UP: yes" in kwargs["user_prompt"]


def test_stylist_streams_when_requested(monkeypatch, mocker):
    """stream=True path returns the generator from llm_client.generate."""
    def fake_iter() -> Iterator[str]:
        yield "a"
        yield "b"

    mocker.patch.object(stylist, "generate", return_value=fake_iter())

    long_draft = "the system has a mechanical layout. " * 20  # forces non-skip
    result = stylist.stylize(
        draft_answer=long_draft,
        user_query="m?",
        stream=True,
    )

    assert not isinstance(result, str)
    assert list(result) == ["a", "b"]


def test_synthesizer_skip_streams_single_chunk(monkeypatch, mocker):
    """Skip path with stream=True must still yield a chunk (not return str)."""
    spy = mocker.patch.object(synthesizer, "generate")

    short = "Short factual answer [S-1 p1]."
    result = synthesizer.synthesize(
        raw_answer=short,
        user_query="q?",
        source_docs=[],
        stream=True,
    )

    assert not isinstance(result, str)
    assert list(result) == [short]
    spy.assert_not_called()
