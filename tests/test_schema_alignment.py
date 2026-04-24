"""Phase 1 schema-alignment tests. Ensures UnifiedResponse/QueryRequest
match the wire truth emitted by the orchestrator and accept new DocQA
bridge fields. All additions are optional; backward compatibility for
existing clients is asserted."""
from __future__ import annotations

import pytest
from gateway.models import QueryRequest, UnifiedResponse


def test_query_request_accepts_docqa_document():
    req = QueryRequest(
        query="what does it say on page 14?",
        project_id=7222,
        docqa_document={"s3_path": "x/y.pdf", "file_name": "y.pdf"},
        mode_hint="docqa",
    )
    assert req.docqa_document == {"s3_path": "x/y.pdf", "file_name": "y.pdf"}
    assert req.mode_hint == "docqa"


def test_query_request_backward_compatible_without_new_fields():
    req = QueryRequest(query="hello", project_id=7222)
    assert req.docqa_document is None
    assert req.mode_hint is None


def test_unified_response_new_fields_default_null():
    resp = UnifiedResponse()
    assert resp.source_documents is None
    assert resp.active_agent == "rag"
    assert resp.selected_document is None
    assert resp.needs_clarification is False
    assert resp.clarification_prompt is None
    assert resp.docqa_session_id is None
    assert resp.groundedness_score is None
    assert resp.flagged_claims is None


def test_unified_response_backward_compatible_existing_fields():
    """Old clients depend on these — must stay present with same names and defaults."""
    resp = UnifiedResponse(
        success=True, answer="hi", sources=[], confidence="high",
        session_id="s1", engine_used="agentic",
    )
    assert resp.success is True
    assert resp.sources == []
    assert resp.confidence == "high"
    assert resp.session_id == "s1"
    assert resp.engine_used == "agentic"
    # defaults
    assert resp.follow_up_questions == []
    assert resp.fallback_used is False


def test_unified_response_populated_docqa_turn():
    resp = UnifiedResponse(
        active_agent="docqa",
        selected_document={"file_name": "HVAC.pdf"},
        docqa_session_id="dq_xyz",
        source_documents=[{"file_name": "HVAC.pdf", "page": 14}],
    )
    assert resp.active_agent == "docqa"
    assert resp.docqa_session_id == "dq_xyz"
    assert resp.source_documents[0]["page"] == 14


def test_unified_response_populated_clarify_turn():
    resp = UnifiedResponse(
        needs_clarification=True,
        clarification_prompt="Which document?",
    )
    assert resp.needs_clarification is True
    assert resp.clarification_prompt == "Which document?"
