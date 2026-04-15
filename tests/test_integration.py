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

    from dataclasses import dataclass, field

    @dataclass
    class MockAgenticResult:
        answer: str = ""
        sources: list = field(default_factory=list)
        confidence: str = "high"
        needs_escalation: bool = False
        cost_usd: float = 0.005
        total_steps: int = 2
        model: str = "gpt-4.1"

    mock_result = MockAgenticResult(
        answer="The XVENT THEB-446 models are specified in drawing M-401.",
        sources=[{"name": "M-401"}],
    )

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(return_value=mock_result)

    result = await orch.query("What XVENT models?", project_id=2361)

    assert result["success"] is True
    assert result["engine_used"] == "agentic"
    assert result["fallback_used"] is False
    assert "XVENT" in result["answer"]
    orch.agentic.query.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_low_confidence_triggers_document_discovery():
    """AgenticRAG returns low confidence -> document discovery instead of FAISS fallback."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True)

    mock_agentic = MagicMock()
    mock_agentic.answer = "I could not find that information in the available data."
    mock_agentic.sources = []
    mock_agentic.confidence = "low"
    mock_agentic.needs_escalation = False
    mock_agentic.follow_up_questions = []

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(return_value=mock_agentic)

    result = await orch.query("Panel rating?", project_id=7325)

    assert result["needs_document_selection"] is True
    assert result["agentic_confidence"] == "low"
    assert isinstance(result["available_documents"], list)


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
async def test_agentic_exception_triggers_document_discovery():
    """AgenticRAG throws exception -> document discovery with available titles."""
    from gateway.orchestrator import Orchestrator

    orch = Orchestrator(fallback_enabled=True)

    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(side_effect=RuntimeError("MongoDB connection lost"))

    result = await orch.query("Electrical plans?", project_id=7212)

    assert result["needs_document_selection"] is True
    assert isinstance(result["available_documents"], list)


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
