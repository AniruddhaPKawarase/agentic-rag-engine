"""Tests for document scope, title cache, query enhancement, and intent matching."""

from __future__ import annotations

import time
import pytest


# ---------------------------------------------------------------------------
# Title Cache Tests
# ---------------------------------------------------------------------------

class TestTitleCache:
    """Test the in-memory LRU title cache."""

    def setup_method(self):
        from gateway.title_cache import invalidate_all
        invalidate_all()

    def test_cache_miss_returns_none(self):
        from gateway.title_cache import get_cached_titles
        assert get_cached_titles(9999) is None

    def test_set_and_get_cached_titles(self):
        from gateway.title_cache import set_cached_titles, get_cached_titles
        drawings = [{"drawingTitle": "Floor Plan", "drawingName": "A-101"}]
        specs = [{"sectionTitle": "23 31 00", "pdfName": "hvac.pdf"}]
        set_cached_titles(1234, drawings, specs)

        cached = get_cached_titles(1234)
        assert cached is not None
        assert len(cached["drawings"]) == 1
        assert len(cached["specifications"]) == 1
        assert cached["drawings"][0]["drawingTitle"] == "Floor Plan"

    def test_cache_invalidate_project(self):
        from gateway.title_cache import set_cached_titles, get_cached_titles, invalidate_project
        set_cached_titles(100, [{"drawingTitle": "X"}], [])
        assert get_cached_titles(100) is not None

        existed = invalidate_project(100)
        assert existed is True
        assert get_cached_titles(100) is None

    def test_cache_invalidate_nonexistent(self):
        from gateway.title_cache import invalidate_project
        assert invalidate_project(99999) is False

    def test_cache_invalidate_all(self):
        from gateway.title_cache import set_cached_titles, invalidate_all, get_cache_stats
        set_cached_titles(1, [{"drawingTitle": "A"}], [])
        set_cached_titles(2, [{"drawingTitle": "B"}], [])
        count = invalidate_all()
        assert count == 2
        assert get_cache_stats()["cached_projects"] == 0

    def test_cache_stats(self):
        from gateway.title_cache import set_cached_titles, get_cache_stats
        set_cached_titles(42, [{"drawingTitle": "Test"}], [])
        stats = get_cache_stats()
        assert stats["cached_projects"] == 1
        assert 42 in stats["project_ids"]
        assert stats["ttl_seconds"] == 3600
        assert stats["max_projects"] == 100


# ---------------------------------------------------------------------------
# Session Scope Tests
# ---------------------------------------------------------------------------

class TestSessionScope:
    """Test document scope state management."""

    def test_scope_initially_inactive(self):
        from shared.session.manager import get_document_scope
        scope = get_document_scope("test-scope-1")
        assert scope["is_active"] is False

    def test_set_and_get_scope(self):
        from shared.session.manager import set_document_scope, get_document_scope
        result = set_document_scope(
            session_id="test-scope-2",
            drawing_title="Mechanical Floor Plan",
            drawing_name="M-101",
            document_type="drawing",
        )
        assert result["is_active"] is True
        assert result["drawing_title"] == "Mechanical Floor Plan"

        scope = get_document_scope("test-scope-2")
        assert scope["is_active"] is True
        assert scope["drawing_name"] == "M-101"

    def test_clear_scope(self):
        from shared.session.manager import set_document_scope, clear_document_scope, get_document_scope
        set_document_scope("test-scope-3", drawing_title="Test Drawing")
        clear_document_scope("test-scope-3")
        scope = get_document_scope("test-scope-3")
        assert scope["is_active"] is False

    def test_previously_scoped_history(self):
        from shared.session.manager import set_document_scope, get_meta
        set_document_scope("test-scope-4", drawing_title="Plan A", drawing_name="A-1")
        set_document_scope("test-scope-4", drawing_title="Plan B", drawing_name="B-1")
        meta = get_meta("test-scope-4")
        assert len(meta.previously_scoped) == 2

    def test_scope_dedup_in_history(self):
        from shared.session.manager import set_document_scope, get_meta
        set_document_scope("test-scope-5", drawing_title="Same Plan")
        set_document_scope("test-scope-5", drawing_title="Same Plan")
        meta = get_meta("test-scope-5")
        assert len(meta.previously_scoped) == 1

    def test_scope_in_session_stats(self):
        from shared.session.manager import set_document_scope, get_session_stats_extended
        set_document_scope("test-scope-6", drawing_title="Elec Plan")
        stats = get_session_stats_extended("test-scope-6")
        assert "scope" in stats
        assert stats["scope"]["is_active"] is True
        assert "previously_scoped" in stats


# ---------------------------------------------------------------------------
# Document Scope Model Tests
# ---------------------------------------------------------------------------

class TestDocumentScopeModel:
    """Test the DocumentScope dataclass directly."""

    def test_is_active_when_set(self):
        from shared.session.models import DocumentScope
        scope = DocumentScope()
        scope.activate(drawing_title="Test", drawing_name="T-1")
        assert scope.is_active is True

    def test_is_inactive_when_cleared(self):
        from shared.session.models import DocumentScope
        scope = DocumentScope()
        scope.activate(drawing_title="Test")
        scope.clear()
        assert scope.is_active is False

    def test_is_inactive_when_empty(self):
        from shared.session.models import DocumentScope
        scope = DocumentScope()
        assert scope.is_active is False

    def test_touch_updates_timestamp(self):
        from shared.session.models import DocumentScope
        scope = DocumentScope()
        scope.activate(drawing_title="Test")
        old_time = scope.last_query_at
        time.sleep(0.01)
        scope.touch()
        assert scope.last_query_at > old_time

    def test_to_dict(self):
        from shared.session.models import DocumentScope
        scope = DocumentScope()
        scope.activate(drawing_title="Floor Plan", drawing_name="A-101", document_type="drawing")
        d = scope.to_dict()
        assert d["is_active"] is True
        assert d["drawing_title"] == "Floor Plan"
        assert d["drawing_name"] == "A-101"
        assert d["document_type"] == "drawing"


# ---------------------------------------------------------------------------
# Query Enhancement Tests
# ---------------------------------------------------------------------------

class TestQueryEnhancement:
    """Test the generic (non-LLM) query enhancement path."""

    def test_generic_enhance_with_trades(self):
        from gateway.query_enhancer import _generic_enhance
        docs = [
            {"type": "drawing", "drawing_title": "HVAC Plan", "trade": "Mechanical"},
            {"type": "drawing", "drawing_title": "Elec Plan", "trade": "Electrical"},
        ]
        result = _generic_enhance("Check missing items", docs)
        assert len(result["improved_queries"]) > 0
        assert len(result["query_tips"]) > 0
        assert any("Mechanical" in q or "Electrical" in q for q in result["improved_queries"])

    def test_generic_enhance_empty_docs(self):
        from gateway.query_enhancer import _generic_enhance
        result = _generic_enhance("Something", [])
        assert len(result["improved_queries"]) > 0
        assert len(result["query_tips"]) > 0

    def test_parse_enhancement_valid(self):
        from gateway.query_enhancer import _parse_enhancement
        text = "IMPROVED:\n- query one\n- query two\n- query three\nTIPS:\n- tip one\n- tip two\n- tip three"
        result = _parse_enhancement(text)
        assert len(result["improved_queries"]) == 3
        assert len(result["query_tips"]) == 3

    def test_parse_enhancement_empty_falls_back(self):
        from gateway.query_enhancer import _parse_enhancement
        result = _parse_enhancement("")
        assert len(result["improved_queries"]) > 0  # generic fallback
        assert len(result["query_tips"]) > 0


# ---------------------------------------------------------------------------
# Orchestrator _base_response Tests
# ---------------------------------------------------------------------------

class TestBaseResponse:
    """Test the _base_response factory preserves all fields."""

    def test_base_response_has_all_fields(self):
        from gateway.orchestrator import _base_response
        resp = _base_response()
        required_fields = [
            "answer", "rag_answer", "web_answer", "confidence", "confidence_score",
            "follow_up_questions", "improved_queries", "query_tips",
            "source_documents", "s3_paths", "s3_path_count",
            "web_sources", "web_source_count",
            "model_used", "token_usage", "processing_time_ms",
            "session_id", "search_mode", "engine_used", "fallback_used",
            "agentic_confidence", "pin_status", "debug_info",
            "needs_document_selection", "available_documents", "scoped_to",
            "success", "error",
        ]
        for field in required_fields:
            assert field in resp, f"Missing field: {field}"

    def test_base_response_overrides(self):
        from gateway.orchestrator import _base_response
        resp = _base_response(answer="Hello", engine_used="agentic", confidence="high")
        assert resp["answer"] == "Hello"
        assert resp["engine_used"] == "agentic"
        assert resp["confidence"] == "high"

    def test_extract_source_documents(self):
        from gateway.orchestrator import _extract_source_documents
        sources = [
            {"s3_path": "path1.pdf", "name": "Drawing A", "sheet_number": "A-101",
             "pdfName": "plan.pdf", "drawingName": "A-101", "drawingTitle": "Floor Plan", "page": 1},
            "raw_string_source.pdf",
        ]
        docs, paths = _extract_source_documents(sources)
        assert len(docs) == 2
        assert len(paths) == 2
        assert docs[0]["s3_path"] == "path1.pdf"
        assert docs[0]["pdf_name"] == "plan.pdf"
        assert docs[0]["drawing_name"] == "A-101"
        assert docs[0]["drawing_title"] == "Floor Plan"
        assert docs[0]["page"] == 1
        assert docs[1]["s3_path"] == "raw_string_source.pdf"
