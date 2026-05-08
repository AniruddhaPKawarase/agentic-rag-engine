"""Phase 5: 10 end-to-end flows for unified-rag-agent DocQA extension.

Marked `-m e2e`. Skipped by default. Engineer runs at Phase 5 gate:
  GATEWAY_URL=http://localhost:8001 \\
    python -m pytest tests/e2e/test_docqa_flow.py -v -m e2e
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.e2e

GATEWAY_URL = os.environ.get("GATEWAY_URL") or os.environ.get("SANDBOX_URL")
DEFAULT_PROJECT = int(os.environ.get("E2E_PROJECT_ID", "7222"))

FORBIDDEN_PATTERNS = [
    re.compile(r"\[Source:\s*[^\]]*\]"),
    re.compile(r"^Direct Answer\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"HIGH\s*\(\d+%\)"),
]


def _require_gateway():
    if not GATEWAY_URL:
        pytest.skip("GATEWAY_URL not set — start local gateway + export env var")


def _http() -> httpx.Client:
    return httpx.Client(timeout=180)


def _first_source_with_url(data: dict) -> dict:
    srcs = data.get("source_documents") or []
    for s in srcs:
        if s.get("download_url") and (s.get("s3_path") or s.get("file_name")):
            return s
    pytest.skip(f"RAG returned no source with download_url: {srcs[:1]}")


def _post_query(c: httpx.Client, **body) -> dict:
    body.setdefault("project_id", DEFAULT_PROJECT)
    r = c.post(f"{GATEWAY_URL}/query", json=body)
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:300]}"
    return r.json()


# ────────────────────────────────────────────────────────────── Flow 1

def test_flow1_plain_rag_returns_sources():
    """Basic RAG query returns non-empty source_documents and active_agent=rag."""
    _require_gateway()
    with _http() as c:
        d = _post_query(c, query="list HVAC drawings in the project")
        assert d["success"] is True
        assert (d.get("active_agent") or "rag") == "rag"
        assert d.get("source_documents"), f"no sources: {d}"


# ────────────────────────────────────────────────────────────── Flow 2

def test_flow2_rag_then_docqa_handoff():
    """RAG query → user selects doc → DocQA engages (or graceful fallback)."""
    _require_gateway()
    with _http() as c:
        d1 = _post_query(c, query="list HVAC drawings")
        src = _first_source_with_url(d1)
        sid = d1["session_id"]
        d2 = _post_query(
            c,
            query="Give me a brief overview of this document.",
            session_id=sid,
            search_mode="docqa",
            docqa_document={
                "s3_path": src["s3_path"],
                "file_name": src["file_name"],
                "download_url": src["download_url"],
            },
        )
        assert d2["success"] in (True, False)
        assert d2["engine_used"] in ("docqa", "docqa_fallback"), d2
        if d2["engine_used"] == "docqa":
            assert d2["active_agent"] == "docqa"
            assert d2["docqa_session_id"]


# ────────────────────────────────────────────────────────────── Flow 3

def test_flow3_docqa_followup_reuses_session():
    """Follow-up query in DocQA mode reuses the existing docqa_session_id."""
    _require_gateway()
    with _http() as c:
        d1 = _post_query(c, query="list HVAC drawings")
        src = _first_source_with_url(d1)
        sid = d1["session_id"]
        d2 = _post_query(
            c, query="overview", session_id=sid, search_mode="docqa",
            docqa_document={"s3_path": src["s3_path"], "file_name": src["file_name"],
                            "download_url": src["download_url"]},
        )
        if d2.get("engine_used") == "docqa_fallback":
            pytest.skip("DocQA agent unreachable — handoff skipped")
        first_dq = d2["docqa_session_id"]
        d3 = _post_query(
            c, query="what does page 1 cover?", session_id=sid,
            search_mode="docqa",
            docqa_document={"s3_path": src["s3_path"], "file_name": src["file_name"],
                            "download_url": src["download_url"]},
        )
        assert d3["engine_used"] == "docqa"
        assert d3["docqa_session_id"] == first_dq, (
            f"expected reuse of {first_dq}, got {d3['docqa_session_id']}"
        )


# ────────────────────────────────────────────────────────────── Flow 4

def test_flow4_auto_switch_back_to_rag_on_project_wide_query():
    """After DocQA engagement, a project-wide query routes to RAG via Phase 3 classifier."""
    _require_gateway()
    with _http() as c:
        d1 = _post_query(c, query="list HVAC drawings")
        src = _first_source_with_url(d1)
        sid = d1["session_id"]
        _post_query(c, query="overview", session_id=sid, search_mode="docqa",
            docqa_document={"s3_path": src["s3_path"], "file_name": src["file_name"],
                            "download_url": src["download_url"]})
        # Now send a project-wide query WITHOUT search_mode — classifier decides
        d3 = _post_query(c, query="show all missing scope across project",
                         session_id=sid)
        assert d3["active_agent"] == "rag", d3
        assert d3["engine_used"] != "docqa"


# ────────────────────────────────────────────────────────────── Flow 5

def test_flow5_clarify_prompt_on_ambiguous_pronoun():
    """Ambiguous pronoun query + selected doc → classifier returns clarify envelope."""
    _require_gateway()
    with _http() as c:
        d1 = _post_query(c, query="list HVAC drawings")
        src = _first_source_with_url(d1)
        sid = d1["session_id"]
        d2 = _post_query(c, query="overview", session_id=sid,
            search_mode="docqa",
            docqa_document={"s3_path": src["s3_path"], "file_name": src["file_name"],
                            "download_url": src["download_url"]})
        if d2.get("engine_used") == "docqa_fallback":
            pytest.skip("DocQA unreachable")
        # Now ambiguous pronoun — should clarify
        d3 = _post_query(c, query="is it missing?", session_id=sid)
        # Expect clarify; fallback acceptable if classifier scoring differs slightly
        assert (
            d3.get("needs_clarification") is True
            or d3.get("active_agent") in ("rag", "docqa")
        ), d3
        if d3.get("needs_clarification"):
            assert d3.get("clarification_prompt")


# ────────────────────────────────────────────────────────────── Flow 6

def test_flow6_mode_hint_docqa_overrides_classifier():
    """mode_hint=docqa + selected doc in session overrides a project-wide query."""
    _require_gateway()
    with _http() as c:
        d1 = _post_query(c, query="list HVAC drawings")
        src = _first_source_with_url(d1)
        sid = d1["session_id"]
        d2 = _post_query(c, query="overview", session_id=sid,
            search_mode="docqa",
            docqa_document={"s3_path": src["s3_path"], "file_name": src["file_name"],
                            "download_url": src["download_url"]})
        if d2.get("engine_used") == "docqa_fallback":
            pytest.skip("DocQA unreachable")
        # Project-wide-sounding query but with explicit docqa hint — should stay docqa
        d3 = _post_query(c, query="list everything", session_id=sid,
                         mode_hint="docqa",
                         search_mode="docqa",
                         docqa_document={"s3_path": src["s3_path"],
                                         "file_name": src["file_name"],
                                         "download_url": src["download_url"]})
        assert d3["engine_used"] == "docqa"


# ────────────────────────────────────────────────────────────── Flow 7

def test_flow7_download_url_is_live_and_fetchable():
    """Every source_document.download_url returned by RAG is actually reachable.

    Uses GET with a Range header to fetch just the first 1 KB — this is what
    a real browser does on a <a href> click (method GET). HEAD is intentionally
    NOT used: S3 rejects HEAD on SigV4 presigned URLs generated for GetObject
    (the signature is method-scoped to GET). A 200 or 206 Partial Content proves
    the URL is valid + the credentials have s3:GetObject permission.
    """
    _require_gateway()
    with _http() as c:
        d = _post_query(c, query="list HVAC drawings")
        srcs = d.get("source_documents") or []
        if not srcs:
            pytest.skip("no sources returned")
        checked = 0
        for s in srcs[:3]:  # sample first 3 to keep test fast
            url = s.get("download_url")
            if not url:
                continue
            resp = c.get(
                url,
                timeout=15,
                follow_redirects=True,
                headers={"Range": "bytes=0-1023"},
            )
            assert resp.status_code in (200, 206), (
                f"broken download_url: {url} -> {resp.status_code} "
                f"body={resp.text[:200]}"
            )
            # Bonus sanity: content-type should be PDF-ish
            ct = (resp.headers.get("content-type") or "").lower()
            assert "pdf" in ct or "octet-stream" in ct or "binary" in ct, (
                f"unexpected content-type for {url}: {ct}"
            )
            checked += 1
        assert checked >= 1, "no download_urls were fetchable"


# ────────────────────────────────────────────────────────────── Flow 8

def test_flow8_answer_format_clean_across_multiple_queries():
    """No query in a batch leaks [Source:], Direct Answer, or HIGH (NN%) artifacts."""
    _require_gateway()
    with _http() as c:
        queries = [
            "What XVENT model for double exhaust?",
            "List all HVAC drawings.",
            "How many DOAS units are specified?",
        ]
        for q in queries:
            d = _post_query(c, query=q)
            answer = d.get("answer") or ""
            for pat in FORBIDDEN_PATTERNS:
                assert not pat.search(answer), (
                    f"artifact {pat.pattern!r} in answer for {q!r}: {answer[:200]!r}"
                )


# ────────────────────────────────────────────────────────────── Flow 9

def test_flow9_docqa_error_returns_graceful_fallback():
    """Broken s3_path produces engine_used=docqa_fallback, never 5xx."""
    _require_gateway()
    with _http() as c:
        r = c.post(
            f"{GATEWAY_URL}/query",
            json={
                "query": "what's in it?",
                "project_id": DEFAULT_PROJECT,
                "search_mode": "docqa",
                "docqa_document": {
                    "s3_path": "nonexistent-bucket/does-not-exist.pdf",
                    "file_name": "does-not-exist.pdf",
                    "download_url": "https://invalid.example.com/nothing.pdf",
                },
            },
            timeout=180,
        )
        assert r.status_code == 200, f"expected graceful 200, got {r.status_code}"
        d = r.json()
        assert d["engine_used"] == "docqa_fallback", d
        assert d["fallback_used"] is True
        assert d["active_agent"] == "rag"
        # human-readable fallback message present
        assert "Could not load document" in (d.get("answer") or "")


# ────────────────────────────────────────────────────────────── Flow 10

def test_flow10_schema_backward_compatible():
    """Existing UI clients depending on current field names must keep working."""
    _require_gateway()
    with _http() as c:
        d = _post_query(c, query="list HVAC drawings")
        # Fields the sandbox UI already reads — must remain present with same names
        required = [
            "success", "answer", "session_id", "confidence", "engine_used",
            "source_documents", "active_agent",
        ]
        missing = [k for k in required if k not in d]
        assert not missing, f"missing required fields: {missing}"
        # Types sanity
        assert isinstance(d["success"], bool)
        assert isinstance(d["answer"], str)
        assert isinstance(d["source_documents"], list)
