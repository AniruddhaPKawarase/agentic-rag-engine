"""Unit tests for Fix #1 — ``gateway.retrieval_enrichment``.

Runs without the full agent stack. OpenAI calls are stubbed via a fake client.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Stub OpenAI client
# ---------------------------------------------------------------------------

class _FakeOpenAIClient:
    """Minimal stub that returns the canned content on every call."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):  # noqa: ANN003, ANN202
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _RaisingClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._boom))

    def _boom(self, **kwargs):  # noqa: ANN003, ANN202
        raise self._exc


# ---------------------------------------------------------------------------
# decompose_query
# ---------------------------------------------------------------------------

def test_decompose_happy_path_returns_distinct_queries() -> None:
    from gateway.retrieval_enrichment import decompose_query

    payload = json.dumps([
        "stair pressurization fan size level 2",
        "stair pressurization fan schedule",
        "smoke control stair pressurization",
        "stair pressurization Division 23 HVAC",
    ])
    client = _FakeOpenAIClient(payload)
    out = decompose_query("What size are the stair press on level 2?", n=4, openai_client=client)
    assert len(out) == 4
    assert len(set(out)) == 4  # all distinct
    assert "stair pressurization" in out[0].lower()


def test_decompose_handles_markdown_fenced_json() -> None:
    from gateway.retrieval_enrichment import decompose_query

    payload = "```json\n" + json.dumps(["a", "b", "c"]) + "\n```"
    out = decompose_query("foo", n=3, openai_client=_FakeOpenAIClient(payload))
    assert out == ["a", "b", "c"]


def test_decompose_extracts_array_from_prose() -> None:
    from gateway.retrieval_enrichment import decompose_query

    payload = "Sure, here are the variants: [\"x\", \"y\"] -- use these."
    out = decompose_query("foo", n=3, openai_client=_FakeOpenAIClient(payload))
    assert out == ["x", "y"]


def test_decompose_falls_back_on_llm_exception() -> None:
    from gateway.retrieval_enrichment import decompose_query

    client = _RaisingClient(RuntimeError("api down"))
    out = decompose_query("how many valves", n=4, openai_client=client)
    assert out == ["how many valves"]


def test_decompose_falls_back_on_nonsense() -> None:
    from gateway.retrieval_enrichment import decompose_query

    out = decompose_query("x", n=4, openai_client=_FakeOpenAIClient("not json at all"))
    assert out == ["x"]


def test_decompose_dedupes_case_insensitive() -> None:
    from gateway.retrieval_enrichment import decompose_query

    payload = json.dumps(["A B", "a b", "A   B", "c d"])
    out = decompose_query("q", n=4, openai_client=_FakeOpenAIClient(payload))
    assert len(out) == 2


def test_decompose_caps_at_n() -> None:
    from gateway.retrieval_enrichment import decompose_query

    payload = json.dumps([f"q{i}" for i in range(10)])
    out = decompose_query("q", n=3, openai_client=_FakeOpenAIClient(payload))
    assert len(out) == 3


def test_decompose_empty_query_returns_empty() -> None:
    from gateway.retrieval_enrichment import decompose_query

    assert decompose_query("   ", n=3) == []


# ---------------------------------------------------------------------------
# multi_query_retrieve
# ---------------------------------------------------------------------------

def test_multi_query_retrieve_fans_out_and_filters_empty() -> None:
    from gateway.retrieval_enrichment import multi_query_retrieve

    calls: list[tuple[str, int, str, int]] = []

    def spec(project_id: int, search_text: str, limit: int = 10) -> list[dict]:
        calls.append(("spec", project_id, search_text, limit))
        if "empty" in search_text:
            return []
        return [{"pdfName": f"spec_{search_text[:5]}"}]

    def legacy(project_id: int, search_text: str, limit: int = 10) -> list[dict]:
        calls.append(("legacy", project_id, search_text, limit))
        return [{"drawingName": f"d_{search_text[:5]}"}]

    ranked = multi_query_retrieve(
        project_id=7222,
        sub_queries=["valves", "empty query"],
        tools=[("spec", spec), ("legacy", legacy)],
        per_tool_limit=5,
        max_workers=2,
    )
    # 2 sub-queries × 2 tools = 4 slots, but one empty; 3 non-empty lists
    assert len(ranked) == 3
    assert all(isinstance(lst, list) and lst for lst in ranked)
    # All tools got both sub-queries
    assert len(calls) == 4


def test_multi_query_retrieve_isolates_tool_errors() -> None:
    from gateway.retrieval_enrichment import multi_query_retrieve

    def good(project_id: int, search_text: str, limit: int = 10) -> list[dict]:
        return [{"pdfName": "ok"}]

    def broken(project_id: int, search_text: str, limit: int = 10) -> list[dict]:
        raise RuntimeError("boom")

    ranked = multi_query_retrieve(
        project_id=1,
        sub_queries=["q"],
        tools=[("good", good), ("broken", broken)],
    )
    assert len(ranked) == 1
    assert ranked[0][0]["pdfName"] == "ok"


def test_multi_query_retrieve_empty_inputs() -> None:
    from gateway.retrieval_enrichment import multi_query_retrieve

    assert multi_query_retrieve(1, [], tools=[]) == []
    assert multi_query_retrieve(1, ["x"], tools=[]) == []


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

def test_rrf_prefers_items_appearing_in_multiple_lists() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    a = [{"pdfName": "A"}, {"pdfName": "B"}, {"pdfName": "C"}]
    b = [{"pdfName": "B"}, {"pdfName": "D"}]
    c = [{"pdfName": "B"}, {"pdfName": "E"}]
    fused = reciprocal_rank_fusion([a, b, c], top_k=3)
    # B is in all three lists at rank 2, 1, 1 → wins
    assert fused[0]["pdfName"] == "B"
    assert fused[0]["_rrf_support"] == 3


def test_rrf_respects_top_k() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    lists = [[{"pdfName": str(i)} for i in range(10)]]
    fused = reciprocal_rank_fusion(lists, top_k=3)
    assert len(fused) == 3


def test_rrf_adds_score_fields() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[{"pdfName": "X"}]], top_k=1)
    assert "_rrf_score" in fused[0]
    assert "_rrf_support" in fused[0]


def test_rrf_dedup_within_single_list() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    lst = [{"pdfName": "X"}, {"pdfName": "X"}, {"pdfName": "Y"}]
    fused = reciprocal_rank_fusion([lst])
    ids = [f["pdfName"] for f in fused]
    assert ids.count("X") == 1
    assert "Y" in ids


def test_rrf_handles_empty_and_none() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_id_fn_custom() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    a = [{"x": 1}, {"x": 2}]
    b = [{"x": 1}]
    fused = reciprocal_rank_fusion([a, b], id_fn=lambda d: str(d.get("x")))
    # x=1 appears twice, should rank first
    assert fused[0]["x"] == 1


def test_rrf_rejects_nonpositive_k() -> None:
    from gateway.retrieval_enrichment import reciprocal_rank_fusion

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)


# ---------------------------------------------------------------------------
# format_hint_for_agent
# ---------------------------------------------------------------------------

def test_format_hint_empty() -> None:
    from gateway.retrieval_enrichment import format_hint_for_agent

    assert format_hint_for_agent({"fused": []}) == ""


def test_format_hint_populated() -> None:
    from gateway.retrieval_enrichment import format_hint_for_agent

    hint = {
        "sub_queries": ["q1", "q2"],
        "fused": [
            {"pdfName": "PLUMBINGINSULATION", "drawingTitle": "Plumbing Insulation", "_rrf_score": 0.05, "_rrf_support": 3},
            {"pdfName": "STAIRFAN", "drawingTitle": "Stair Pressurization Fan Schedule", "_rrf_score": 0.04, "_rrf_support": 2},
        ],
    }
    text = format_hint_for_agent(hint)
    assert "Pre-retrieved" in text
    assert "Plumbing Insulation" in text
    assert "q1" in text
    assert "spec_get_full_text" in text  # agent guidance present
