"""tests/test_integration.py — Full orchestrator integration tests.

These tests mock both engines (agentic + traditional) and verify
the complete orchestrator flow: routing, fallback logic, error handling.
No real API/DB calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_agentic_success_no_fallback():
    """AgenticRAG returns high confidence -> no fallback triggered."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True)

    mock_result = MagicMock()
    mock_result.answer = "The XVENT THEB-446 models are specified in drawing M-401."
    mock_result.sources = [{"name": "M-401"}]
    mock_result.confidence = "high"
    mock_result.needs_escalation = False
    mock_result.cost_usd = 0.005
    mock_result.total_steps = 2
    mock_result.model = "gpt-4.1"

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(return_value=mock_result)

    result = await orch.query("What XVENT models?", project_id=2361)

    assert result["success"] is True
    assert result["engine_used"] == "agentic"
    assert result["fallback_used"] is False
    assert "XVENT" in result["answer"]
    orch.agentic.query.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_low_confidence_triggers_fallback():
    """AgenticRAG returns low confidence -> Traditional RAG takes over."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True)

    mock_agentic = MagicMock()
    mock_agentic.answer = "I could not find that information in the available data."
    mock_agentic.sources = []
    mock_agentic.confidence = "low"
    mock_agentic.needs_escalation = False

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(return_value=mock_agentic)

    trad_result = {
        "success": True,
        "answer": "Based on the electrical drawings, the panel is 200A rated.",
        "sources": [{"text": "200A panel board"}],
        "confidence": "high",
    }
    orch.traditional.query = AsyncMock(return_value=trad_result)
    orch.traditional._faiss_loaded = True

    result = await orch.query("Panel rating?", project_id=7325)

    assert result["fallback_used"] is True
    assert result["engine_used"] == "traditional"
    assert result["agentic_confidence"] == "low"
    assert "200A" in result["answer"]


@pytest.mark.asyncio
async def test_engine_override_traditional():
    """engine='traditional' skips agentic entirely."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True)

    trad_result = {
        "success": True,
        "answer": "The plumbing fixtures include two lavatories and one water closet.",
        "sources": [{"drawing": "P-101"}],
    }
    orch.traditional.query = AsyncMock(return_value=trad_result)
    orch.traditional._faiss_loaded = True

    # Spy on agentic to confirm it was NOT called
    orch.agentic.query = AsyncMock()

    result = await orch.query("Plumbing fixtures?", project_id=7212, engine="traditional")

    assert result["engine_used"] == "traditional"
    assert result["fallback_used"] is False
    assert "plumbing" in result["answer"].lower()
    orch.agentic.query.assert_not_called()


@pytest.mark.asyncio
async def test_agentic_exception_triggers_fallback():
    """AgenticRAG throws exception -> fallback to Traditional."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True)

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(side_effect=RuntimeError("MongoDB connection lost"))

    trad_result = {
        "success": True,
        "answer": "Found the answer via document search.",
        "sources": [{"drawing": "E-101"}],
    }
    orch.traditional.query = AsyncMock(return_value=trad_result)
    orch.traditional._faiss_loaded = True

    result = await orch.query("Electrical plans?", project_id=7212)

    assert result["fallback_used"] is True
    assert "Found the answer" in result["answer"]


@pytest.mark.asyncio
async def test_both_engines_fail_returns_error():
    """Both engines fail -> return error response."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True, fallback_timeout=5)

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(side_effect=RuntimeError("Agentic down"))
    orch.traditional.query = AsyncMock(side_effect=RuntimeError("Traditional down"))
    orch.traditional._faiss_loaded = True

    result = await orch.query("Test query?", project_id=1)

    # When both fail, the orchestrator returns the agentic result (None)
    # with the fallback error message attached
    assert (
        result["success"] is False
        or "error" in str(result.get("answer", "")).lower()
        or result.get("error") is not None
    )
