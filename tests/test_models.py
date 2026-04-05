"""Tests for gateway.models — QueryRequest and UnifiedResponse."""

import pytest
from pydantic import ValidationError


class TestQueryRequestMinimal:
    """test_query_request_minimal — query + project_id only, defaults correct."""

    def test_query_request_minimal(self) -> None:
        from gateway.models import QueryRequest

        req = QueryRequest(query="What is the HVAC system?", project_id=7166)
        assert req.query == "What is the HVAC system?"
        assert req.project_id == 7166
        assert req.session_id is None
        assert req.search_mode is None
        assert req.generate_document is True
        assert req.filter_source_type is None
        assert req.filter_drawing_name is None
        assert req.set_id is None
        assert req.conversation_history is None
        assert req.engine is None


class TestQueryRequestWithEngineOverride:
    """test_query_request_with_engine_override — engine='traditional' stored."""

    def test_query_request_with_engine_override(self) -> None:
        from gateway.models import QueryRequest

        req = QueryRequest(
            query="Show plumbing drawings",
            project_id=7201,
            engine="traditional",
        )
        assert req.engine == "traditional"
        assert req.project_id == 7201


class TestQueryRequestRejectsEmptyQuery:
    """test_query_request_rejects_empty_query — '' raises validation error."""

    def test_query_request_rejects_empty_query(self) -> None:
        from gateway.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="", project_id=7166)


class TestUnifiedResponseDefaults:
    """test_unified_response_defaults — engine_used='agentic', fallback_used=False."""

    def test_unified_response_defaults(self) -> None:
        from gateway.models import UnifiedResponse

        resp = UnifiedResponse()
        assert resp.success is True
        assert resp.answer == ""
        assert resp.sources == []
        assert resp.confidence == "high"
        assert resp.session_id == ""
        assert resp.follow_up_questions == []
        assert resp.needs_clarification is False
        assert resp.engine_used == "agentic"
        assert resp.fallback_used is False
        assert resp.agentic_confidence is None
        assert resp.cost_usd == 0.0
        assert resp.elapsed_ms == 0
        assert resp.total_steps == 0
        assert resp.model == ""


class TestUnifiedResponseWithFallback:
    """test_unified_response_with_fallback — fallback_used=True, agentic_confidence='low'."""

    def test_unified_response_with_fallback(self) -> None:
        from gateway.models import UnifiedResponse

        resp = UnifiedResponse(
            fallback_used=True,
            agentic_confidence="low",
            engine_used="traditional",
            answer="Fallback answer",
        )
        assert resp.fallback_used is True
        assert resp.agentic_confidence == "low"
        assert resp.engine_used == "traditional"
        assert resp.answer == "Fallback answer"
