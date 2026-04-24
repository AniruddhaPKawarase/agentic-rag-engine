"""Phase 2.4: orchestrator routes search_mode='docqa' via DocQABridge.

NOTE: _run_docqa is an instance method on gateway.orchestrator.Orchestrator.
      _get_docqa_bridge() is a module-level factory that the method calls,
      so we monkeypatch it at the module level.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_run_docqa_uses_bridge(monkeypatch):
    """_run_docqa should delegate to DocQABridge (ensure_loaded + ask + normalize)."""
    from gateway import orchestrator
    from gateway.orchestrator import Orchestrator

    fake_bridge = MagicMock()
    fake_bridge.ensure_document_loaded = AsyncMock(return_value="dq_xyz")
    fake_bridge.ask = AsyncMock(return_value={
        "answer": "on page 14",
        "session_id": "dq_xyz",
        "source_documents": [{"file_name": "x.pdf", "page": 14}],
    })
    fake_bridge.normalize = MagicMock(return_value={
        "success": True,
        "answer": "on page 14",
        "active_agent": "docqa",
        "docqa_session_id": "dq_xyz",
        "session_id": "rag_s1",
        "engine_used": "docqa",
        "source_documents": [{"file_name": "x.pdf", "page": 14}],
    })

    monkeypatch.setattr(orchestrator, "_get_docqa_bridge", lambda: fake_bridge)

    orch = Orchestrator()
    result = await orch._run_docqa(
        query="where is fire damper?",
        project_id=7222,
        session_id="rag_s1",
        docqa_document={
            "s3_path": "bucket/x.pdf", "file_name": "x.pdf",
            "download_url": "https://presigned",
        },
        start=0.0,
    )
    assert result["engine_used"] == "docqa"
    assert result["active_agent"] == "docqa"
    fake_bridge.ensure_document_loaded.assert_awaited_once()
    fake_bridge.ask.assert_awaited_once_with(
        docqa_session_id="dq_xyz",
        query="where is fire damper?",
    )
    fake_bridge.normalize.assert_called_once()


@pytest.mark.asyncio
async def test_run_docqa_graceful_degrade_on_bridge_exception(monkeypatch):
    """If bridge raises, _run_docqa must return a fallback response via bridge.normalize(error=...)."""
    from gateway import orchestrator
    from gateway.orchestrator import Orchestrator

    fake_bridge = MagicMock()
    fake_bridge.ensure_document_loaded = AsyncMock(side_effect=RuntimeError("s3 down"))
    fake_bridge.normalize = MagicMock(return_value={
        "success": False,
        "answer": "Could not load document for deep-dive. (s3 down)",
        "active_agent": "rag",
        "engine_used": "docqa_fallback",
        "session_id": "rag_s1",
        "fallback_used": True,
    })

    monkeypatch.setattr(orchestrator, "_get_docqa_bridge", lambda: fake_bridge)

    orch = Orchestrator()
    result = await orch._run_docqa(
        query="anything",
        project_id=7222,
        session_id="rag_s1",
        docqa_document={"s3_path": "bucket/x.pdf", "file_name": "x.pdf"},
        start=0.0,
    )
    assert result["success"] is False
    assert result["engine_used"] == "docqa_fallback"
    # normalize was called with error payload
    call_kwargs = fake_bridge.normalize.call_args.kwargs or {}
    assert "error" in (call_kwargs.get("docqa_response") or {})
