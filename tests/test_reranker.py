"""Unit tests for Fix #2 — ``gateway.reranker``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _FakeOpenAI:
    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):  # noqa: ANN003, ANN202
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _BoomClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._boom))

    def _boom(self, **kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("rate limited")


def _docs(n: int) -> list[dict]:
    return [
        {
            "s3_path": f"ifieldsmart/proj/Drawings/pdf{i}",
            "pdf_name": f"doc_{i}",
            "display_title": f"Title {i}",
            "page": i,
        }
        for i in range(n)
    ]


def test_rerank_reorders_by_score() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(4)
    # Model says doc 3 > doc 0 > doc 2 > doc 1
    client = _FakeOpenAI(json.dumps([5, 2, 4, 9]))
    out = rerank_source_documents("q?", docs, openai_client=client, include_score=True)
    titles = [d["display_title"] for d in out]
    assert titles == ["Title 3", "Title 0", "Title 2", "Title 1"]
    # Each got a score (include_score=True)
    assert all("_rerank_score" in d for d in out)


def test_rerank_default_does_not_add_score_field() -> None:
    """UI-contract guard: by default the shape of every source_document
    dict is preserved — no ``_rerank_score`` key leaks."""
    from gateway.reranker import rerank_source_documents

    docs = _docs(3)
    client = _FakeOpenAI(json.dumps([2, 9, 4]))
    out = rerank_source_documents("q?", docs, openai_client=client)
    # Reorder happened
    assert out[0]["display_title"] == "Title 1"
    # But no score key leaked
    assert all("_rerank_score" not in d for d in out)
    # Original keys preserved
    assert set(out[0].keys()) == set(docs[0].keys())


def test_rerank_returns_original_on_llm_error() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(3)
    out = rerank_source_documents("q?", docs, openai_client=_BoomClient())
    assert [d["display_title"] for d in out] == ["Title 0", "Title 1", "Title 2"]


def test_rerank_short_list_pass_through() -> None:
    from gateway.reranker import rerank_source_documents

    assert rerank_source_documents("q", []) == []
    single = [{"pdf_name": "one"}]
    assert rerank_source_documents("q", single) is single  # short-circuit


def test_rerank_parses_fenced_json() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(2)
    client = _FakeOpenAI("```json\n[3, 7]\n```")
    out = rerank_source_documents("q", docs, openai_client=client)
    assert out[0]["display_title"] == "Title 1"


def test_rerank_handles_short_response_by_padding_zeros() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(4)
    # Model returned only 2 scores for 4 docs; the rest get 0 and sort last.
    client = _FakeOpenAI(json.dumps([9, 1]))
    out = rerank_source_documents("q", docs, openai_client=client)
    # Best is doc 0 (score 9), then doc 1 (score 1), then ties at 0 keep original order
    assert out[0]["display_title"] == "Title 0"
    assert out[1]["display_title"] == "Title 1"


def test_rerank_clamps_scores() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(2)
    client = _FakeOpenAI(json.dumps([999, -50]))
    out = rerank_source_documents("q", docs, openai_client=client, include_score=True)
    assert out[0]["_rerank_score"] == 10.0  # clamped from 999
    assert out[1]["_rerank_score"] == 0.0   # clamped from -50


def test_rerank_keep_top_k_truncates() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(5)
    client = _FakeOpenAI(json.dumps([1, 2, 3, 4, 5]))
    out = rerank_source_documents("q", docs, openai_client=client, keep_top_k=2)
    assert len(out) == 2
    assert out[0]["display_title"] == "Title 4"


def test_rerank_preserves_original_dict_fields() -> None:
    from gateway.reranker import rerank_source_documents

    docs = [{"pdf_name": "A", "custom_field": "preserved", "s3_path": "x"}]
    docs.append({"pdf_name": "B", "custom_field": "also", "s3_path": "y"})
    client = _FakeOpenAI(json.dumps([8, 3]))
    out = rerank_source_documents("q", docs, openai_client=client)
    assert out[0]["custom_field"] == "preserved"
    assert out[0]["s3_path"] == "x"


def test_rerank_tail_beyond_cap_kept_unscored() -> None:
    from gateway.reranker import rerank_source_documents

    docs = _docs(5)
    client = _FakeOpenAI(json.dumps([1, 2, 3]))  # only 3 scores
    out = rerank_source_documents("q", docs, openai_client=client, candidate_cap=3, include_score=True)
    assert len(out) == 5
    # First 3 reordered; last 2 appended with _rerank_score=None
    assert out[-1].get("_rerank_score") is None
    assert out[-2].get("_rerank_score") is None


def test_rerank_score_threshold_drops_low_scoring_sources() -> None:
    """Noise-drop: sources below threshold are removed, not just reordered."""
    from gateway.reranker import rerank_source_documents

    docs = _docs(6)
    # Scores: [2, 9, 1, 8, 0, 7] -- threshold 5 keeps only 9, 8, 7
    client = _FakeOpenAI(json.dumps([2, 9, 1, 8, 0, 7]))
    out = rerank_source_documents(
        "q", docs, openai_client=client, score_threshold=5.0, min_keep=1,
    )
    assert len(out) == 3
    # Ordered best-first
    titles = [d["display_title"] for d in out]
    assert titles == ["Title 1", "Title 3", "Title 5"]


def test_rerank_score_threshold_respects_min_keep() -> None:
    """Floor: if all sources fail threshold, at least ``min_keep`` are retained."""
    from gateway.reranker import rerank_source_documents

    docs = _docs(5)
    client = _FakeOpenAI(json.dumps([1, 2, 0, 3, 1]))  # all below threshold 5
    out = rerank_source_documents(
        "q", docs, openai_client=client, score_threshold=5.0, min_keep=2,
    )
    assert len(out) == 2  # padded up to min_keep
    # Ordered by score desc: docs[3]=3, docs[1]=2
    titles = [d["display_title"] for d in out]
    assert titles == ["Title 3", "Title 1"]


def test_rerank_hints_document_kind_in_prompt() -> None:
    """Soft test: ensure spec/drawing kind hints appear in the user prompt."""
    from gateway import reranker

    captured: dict = {}

    class _Capture(_FakeOpenAI):
        def _create(self, **kwargs):  # noqa: ANN003, ANN202
            captured["messages"] = kwargs.get("messages")
            return super()._create(**kwargs)

    docs = [
        {"s3_path": "ifieldsmart/p/Specification/pdf1", "pdf_name": "a"},
        {"s3_path": "ifieldsmart/p/Drawings/pdf2", "pdf_name": "b"},
    ]
    reranker.rerank_source_documents("q", docs, openai_client=_Capture(json.dumps([5, 5])))
    user_msg = captured["messages"][-1]["content"]
    assert "[Specification]" in user_msg
    assert "[Drawing]" in user_msg
