"""Unit tests for Fix #4 — ``gateway.self_rag`` claim verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _FakeOpenAI:
    """Stub that cycles through canned responses so one client can back both
    the claim-extraction call and the verification call."""

    def __init__(self, *canned_responses: str) -> None:
        self._queue = list(canned_responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.captured: list[dict] = []

    def _next(self) -> str:
        if not self._queue:
            return ""
        return self._queue.pop(0)

    def _create(self, **kwargs):  # noqa: ANN003, ANN202
        self.captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._next()))]
        )


class _BoomClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._boom))

    def _boom(self, **kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("api down")


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------

def test_extract_claims_parses_json_list() -> None:
    from gateway.self_rag import extract_claims

    client = _FakeOpenAI(json.dumps(["There are 2 DOAS", "DOAS-1 is on roof", "Section 220700 covers plumbing insulation"]))
    claims = extract_claims(answer="The project specifies two DOAS units...", openai_client=client)
    assert len(claims) == 3
    assert "DOAS-1" in claims[1]


def test_extract_claims_returns_empty_on_short_answer() -> None:
    from gateway.self_rag import extract_claims

    assert extract_claims(answer="") == []


def test_extract_claims_handles_markdown_fence() -> None:
    from gateway.self_rag import extract_claims

    client = _FakeOpenAI("```json\n" + json.dumps(["A", "B"]) + "\n```")
    assert extract_claims(answer="some long answer text here", openai_client=client) == ["A", "B"]


def test_extract_claims_returns_empty_on_llm_failure() -> None:
    from gateway.self_rag import extract_claims

    assert extract_claims(answer="hello world", openai_client=_BoomClient()) == []


def test_extract_claims_caps_at_max() -> None:
    from gateway.self_rag import extract_claims

    client = _FakeOpenAI(json.dumps([f"c{i}" for i in range(40)]))
    claims = extract_claims(answer="long answer", max_claims=5, openai_client=client)
    assert len(claims) == 5


# ---------------------------------------------------------------------------
# verify_claims
# ---------------------------------------------------------------------------

def test_verify_claims_marks_supported_and_flagged() -> None:
    from gateway.self_rag import verify_claims

    verify_response = json.dumps([
        {"supported": True,  "reason": "matches S1"},
        {"supported": False, "reason": "no reference in sources"},
    ])
    client = _FakeOpenAI(verify_response)
    claims = ["there are 2 DOAS", "there are 2 chillers"]
    sources = [{"drawing_title": "HVAC SCHEDULE", "text": "DOAS-1, DOAS-2 listed on roof"}]

    out = verify_claims(claims, sources, openai_client=client)
    assert out[0]["supported"] is True
    assert out[1]["supported"] is False
    assert "no reference" in out[1]["reason"].lower()


def test_verify_claims_fail_open_on_llm_error() -> None:
    from gateway.self_rag import verify_claims

    claims = ["x"]
    out = verify_claims(claims, sources=[], openai_client=_BoomClient())
    assert out == [{"claim": "x", "supported": True, "reason": "verifier unavailable"}]


def test_verify_claims_handles_malformed_response() -> None:
    from gateway.self_rag import verify_claims

    client = _FakeOpenAI("not json at all")
    out = verify_claims(["x", "y"], sources=[], openai_client=client)
    assert all(item["supported"] for item in out)
    assert len(out) == 2


def test_verify_claims_pads_short_response() -> None:
    from gateway.self_rag import verify_claims

    client = _FakeOpenAI(json.dumps([{"supported": False, "reason": "no"}]))  # only 1 of 3
    out = verify_claims(["a", "b", "c"], sources=[], openai_client=client)
    assert len(out) == 3
    # Positions 1 and 2 default to supported (fail-open)
    assert out[0]["supported"] is False
    assert out[1]["supported"] is True
    assert out[2]["supported"] is True


# ---------------------------------------------------------------------------
# evaluate_groundedness (end-to-end stub)
# ---------------------------------------------------------------------------

def test_evaluate_groundedness_end_to_end() -> None:
    from gateway.self_rag import evaluate_groundedness

    extract_resp = json.dumps(["there are 2 DOAS", "insulation is 1/2 inch"])
    verify_resp = json.dumps([
        {"supported": True,  "reason": "matches S1"},
        {"supported": False, "reason": "not stated"},
    ])
    client = _FakeOpenAI(extract_resp, verify_resp)

    sources = [{"drawing_title": "MECH SCHEDULE", "text": "DOAS-1, DOAS-2"}]
    result = evaluate_groundedness(
        answer="The project specifies two DOAS and insulation is 1/2 inch thick.",
        sources=sources,
        openai_client=client,
    )
    assert result is not None
    assert result.groundedness_score == 0.5
    assert len(result.flagged) == 1
    assert result.flagged[0]["claim"].startswith("insulation")
    pub = result.to_public()
    assert pub["groundedness_score"] == 0.5
    assert pub["claims_total"] == 2
    assert pub["claims_supported"] == 1
    assert len(pub["flagged_claims"]) == 1


def test_evaluate_groundedness_returns_none_on_short_answer() -> None:
    from gateway.self_rag import evaluate_groundedness

    assert evaluate_groundedness(answer="yes", sources=[]) is None


def test_evaluate_groundedness_returns_none_when_claims_unavailable() -> None:
    from gateway.self_rag import evaluate_groundedness

    # Claim extractor errors → evaluate returns None
    assert evaluate_groundedness(
        answer="This is a long enough answer that would normally be verified",
        sources=[],
        openai_client=_BoomClient(),
    ) is None
