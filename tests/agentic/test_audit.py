"""Unit tests for core/audit.py — structured audit logging."""

import json
import logging

import pytest

from core.audit import log_auth_failure, log_query, log_rate_limit


class TestLogQuery:
    """Tests for log_query audit entries."""

    def test_log_query_structured(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify JSON structure has all required fields."""
        with caplog.at_level(logging.INFO, logger="agentic_rag.audit"):
            log_query(
                user_id="user_123",
                project_id=7166,
                query="What are the electrical specs?",
                sources=["drawing_A.pdf", "drawing_B.pdf"],
                confidence="high",
                cost_usd=0.03,
                elapsed_ms=1200,
                steps=3,
                needs_escalation=False,
                client_ip="192.168.1.1",
            )

        assert len(caplog.records) == 1
        payload = json.loads(caplog.records[0].message)

        assert payload["event"] == "query_completed"
        assert payload["user_id"] == "user_123"
        assert payload["project_id"] == 7166
        assert payload["query"] == "What are the electrical specs?"
        assert payload["sources_count"] == 2
        assert payload["confidence"] == "high"
        assert payload["cost_usd"] == 0.03
        assert payload["elapsed_ms"] == 1200
        assert payload["steps"] == 3
        assert payload["needs_escalation"] is False
        assert payload["client_ip"] == "192.168.1.1"
        assert "timestamp" in payload

    def test_log_query_truncates_long_query(self, caplog: pytest.LogCaptureFixture) -> None:
        """Query text longer than 500 chars is truncated in log."""
        long_query = "x" * 1000
        with caplog.at_level(logging.INFO, logger="agentic_rag.audit"):
            log_query(
                user_id="u1",
                project_id=1,
                query=long_query,
                sources=[],
                confidence="low",
                cost_usd=0.0,
                elapsed_ms=0,
                steps=0,
            )

        payload = json.loads(caplog.records[0].message)
        assert len(payload["query"]) == 500

    def test_log_query_caps_sources_at_10(self, caplog: pytest.LogCaptureFixture) -> None:
        """Sources list is capped at 10 entries in the log."""
        sources = [f"file_{i}.pdf" for i in range(20)]
        with caplog.at_level(logging.INFO, logger="agentic_rag.audit"):
            log_query(
                user_id="u1",
                project_id=1,
                query="test",
                sources=sources,
                confidence="medium",
                cost_usd=0.01,
                elapsed_ms=500,
                steps=2,
            )

        payload = json.loads(caplog.records[0].message)
        assert len(payload["sources"]) == 10
        assert payload["sources_count"] == 20


class TestLogAuthFailure:
    """Tests for log_auth_failure audit entries."""

    def test_log_auth_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify auth failure log has required fields."""
        with caplog.at_level(logging.WARNING, logger="agentic_rag.audit"):
            log_auth_failure(
                client_ip="10.0.0.1",
                reason="Invalid API key",
            )

        assert len(caplog.records) == 1
        payload = json.loads(caplog.records[0].message)

        assert payload["event"] == "auth_failure"
        assert payload["client_ip"] == "10.0.0.1"
        assert payload["reason"] == "Invalid API key"
        assert "timestamp" in payload


class TestLogRateLimit:
    """Tests for log_rate_limit audit entries."""

    def test_log_rate_limited(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify rate limit log has required fields."""
        with caplog.at_level(logging.WARNING, logger="agentic_rag.audit"):
            log_rate_limit(
                client_ip="10.0.0.2",
                user_id="user_456",
            )

        assert len(caplog.records) == 1
        payload = json.loads(caplog.records[0].message)

        assert payload["event"] == "rate_limited"
        assert payload["client_ip"] == "10.0.0.2"
        assert payload["user_id"] == "user_456"
        assert "timestamp" in payload

    def test_log_rate_limited_no_user(self, caplog: pytest.LogCaptureFixture) -> None:
        """Rate limit log works without user_id."""
        with caplog.at_level(logging.WARNING, logger="agentic_rag.audit"):
            log_rate_limit(client_ip="10.0.0.3")

        payload = json.loads(caplog.records[0].message)
        assert payload["user_id"] == ""
