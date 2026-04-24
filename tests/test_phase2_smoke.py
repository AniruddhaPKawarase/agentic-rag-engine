"""Phase 2 live smoke: full RAG→DocQA handoff against a running gateway.

Requires:
- gateway on $GATEWAY_URL (or $SANDBOX_URL) with DocQA reachable at port 8006
- project 7222 loaded with at least one source document with download_url

Usage:
  # Terminal 1: python -m gateway.app   # wait for "Uvicorn running"
  # Terminal 2:
  GATEWAY_URL=http://localhost:8001 \\
    python -m pytest tests/test_phase2_smoke.py -v -m integration
"""
from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

GATEWAY_URL = os.environ.get("GATEWAY_URL") or os.environ.get("SANDBOX_URL")


def _require_gateway():
    if not GATEWAY_URL:
        pytest.skip("GATEWAY_URL not set — start local gateway and set env var")


def _first_source_with_url(data):
    srcs = data.get("source_documents") or []
    for s in srcs:
        if s.get("download_url") and (s.get("s3_path") or s.get("file_name")):
            return s
    pytest.skip(f"RAG returned no source with download_url: {srcs[:1]}")


def test_handoff_rag_to_docqa_same_session():
    _require_gateway()
    with httpx.Client(timeout=180) as c:
        # --- Turn 1: plain RAG query ---
        r1 = c.post(
            f"{GATEWAY_URL}/query",
            json={"query": "list HVAC drawings in the project", "project_id": 7222},
        )
        r1.raise_for_status()
        d1 = r1.json()
        assert d1.get("success") is True, d1
        assert (d1.get("active_agent") or "rag") == "rag"
        sid = d1.get("session_id")
        assert sid, f"no session_id in response: {d1}"
        src = _first_source_with_url(d1)

        # --- Turn 2: select that doc + ask something scoped to it ---
        r2 = c.post(
            f"{GATEWAY_URL}/query",
            json={
                "query": "Give me a brief overview of this document.",
                "project_id": 7222,
                "session_id": sid,
                "search_mode": "docqa",
                "docqa_document": {
                    "s3_path": src.get("s3_path"),
                    "file_name": src.get("file_name"),
                    "download_url": src.get("download_url"),
                    "pdf_name": src.get("pdf_name") or src.get("file_name"),
                },
            },
        )
        r2.raise_for_status()
        d2 = r2.json()
        assert d2.get("success") is True, d2
        # Either full docqa engagement OR graceful fallback — never 5xx
        assert d2.get("engine_used") in ("docqa", "docqa_fallback"), d2
        if d2["engine_used"] == "docqa":
            assert d2.get("active_agent") == "docqa"
            assert d2.get("docqa_session_id"), d2
            assert d2.get("selected_document"), d2
            assert (d2.get("answer") or "").strip(), "docqa turn returned empty answer"
        else:
            # Fallback path — still a valid UnifiedResponse envelope
            assert d2.get("active_agent") == "rag"
            assert d2.get("fallback_used") is True


def test_handoff_reuses_docqa_session_on_followup():
    """Selecting the same doc twice should reuse the docqa_session_id (no re-upload)."""
    _require_gateway()
    with httpx.Client(timeout=180) as c:
        r1 = c.post(
            f"{GATEWAY_URL}/query",
            json={"query": "list HVAC drawings", "project_id": 7222},
        )
        r1.raise_for_status()
        d1 = r1.json()
        sid = d1["session_id"]
        src = _first_source_with_url(d1)

        body = {
            "query": "overview please",
            "project_id": 7222,
            "session_id": sid,
            "search_mode": "docqa",
            "docqa_document": {
                "s3_path": src["s3_path"],
                "file_name": src["file_name"],
                "download_url": src["download_url"],
            },
        }
        # First select
        r2 = c.post(f"{GATEWAY_URL}/query", json=body)
        r2.raise_for_status()
        d2 = r2.json()
        if d2.get("engine_used") == "docqa_fallback":
            pytest.skip(f"docqa unreachable (fallback): {d2.get('answer')}")
        first_dq_sid = d2["docqa_session_id"]
        assert first_dq_sid

        # Second select — same doc
        body["query"] = "what else is in it?"
        r3 = c.post(f"{GATEWAY_URL}/query", json=body)
        r3.raise_for_status()
        d3 = r3.json()
        assert d3["engine_used"] == "docqa"
        assert d3["docqa_session_id"] == first_dq_sid, (
            f"docqa_session_id should be reused; first={first_dq_sid} second={d3['docqa_session_id']}"
        )


def test_docqa_error_returns_graceful_fallback():
    """Broken s3_path should trigger bridge error handling, not a 5xx."""
    _require_gateway()
    with httpx.Client(timeout=180) as c:
        r = c.post(
            f"{GATEWAY_URL}/query",
            json={
                "query": "what's in it?",
                "project_id": 7222,
                "search_mode": "docqa",
                "docqa_document": {
                    "s3_path": "nonexistent-bucket/does-not-exist.pdf",
                    "file_name": "does-not-exist.pdf",
                    "download_url": "https://invalid.example.com/nothing.pdf",
                },
            },
        )
        # Gateway must NOT 5xx; graceful UnifiedResponse envelope
        assert r.status_code == 200, f"expected 200 graceful fallback, got {r.status_code}: {r.text[:300]}"
        d = r.json()
        assert d.get("engine_used") == "docqa_fallback", d
        assert d.get("success") is False
        assert d.get("fallback_used") is True
        assert "Could not load document" in (d.get("answer") or "") or "error" in (d.get("answer") or "").lower()
