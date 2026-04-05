"""tests/test_orchestrator.py — Orchestrator fallback logic tests."""

import pytest
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockAgenticResult:
    answer: str = ""
    sources: list = field(default_factory=lambda: [{"name": "M-101"}])
    confidence: str = "high"
    needs_escalation: bool = False
    cost_usd: float = 0.005
    elapsed_ms: int = 3000
    total_steps: int = 2
    model: str = "gpt-4.1"


def test_should_fallback_on_low_confidence():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(confidence="low", answer="Some answer that is long enough to pass length check", sources=[{"x": 1}])
    assert _should_fallback(result) is True


def test_should_not_fallback_on_high_confidence():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(confidence="high", answer="Good answer with enough detail here", sources=[{"x": 1}])
    assert _should_fallback(result) is False


def test_should_fallback_on_empty_answer():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(answer="", sources=[{"x": 1}])
    assert _should_fallback(result) is True


def test_should_fallback_on_none():
    from gateway.orchestrator import _should_fallback
    assert _should_fallback(None) is True


def test_should_fallback_on_no_sources():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(answer="Answer without sources but long enough text", sources=[])
    assert _should_fallback(result) is True


def test_should_fallback_on_escalation():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(answer="Partial answer that is long enough text", needs_escalation=True)
    assert _should_fallback(result) is True


def test_should_not_fallback_on_medium_confidence():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(confidence="medium", answer="Decent answer with sufficient length", sources=[{"x": 1}])
    assert _should_fallback(result) is False
