"""Phase 2.1: session state for DocQA bridge."""
from __future__ import annotations

from traditional.memory_manager import MemoryManager


def test_session_tracks_active_agent_default():
    mm = MemoryManager(enable_persistence=False)
    sid = mm.create_session("What is this?")
    sess = mm.get_session(sid)
    assert sess.active_agent == "rag"
    assert sess.docqa_session_id is None
    assert sess.selected_documents == []
    assert sess.last_intent_decision is None


def test_session_can_set_docqa_state():
    mm = MemoryManager(enable_persistence=False)
    sid = mm.create_session("What is this?")
    sess = mm.get_session(sid)
    sess.active_agent = "docqa"
    sess.docqa_session_id = "dq_xyz"
    sess.selected_documents.append({
        "s3_path": "bucket/x.pdf",
        "file_name": "x.pdf",
        "docqa_session_id": "dq_xyz",
    })
    sess.last_intent_decision = {
        "target": "docqa", "confidence": 0.85, "reason": "doc-scoped hit",
    }
    # Fetch again and verify persistence within-process
    sess2 = mm.get_session(sid)
    assert sess2.active_agent == "docqa"
    assert sess2.docqa_session_id == "dq_xyz"
    assert len(sess2.selected_documents) == 1
    assert sess2.selected_documents[0]["s3_path"] == "bucket/x.pdf"
    assert sess2.last_intent_decision["target"] == "docqa"


def test_session_selected_documents_is_independent_per_session():
    """Regression guard: list default must not be shared between instances."""
    mm = MemoryManager(enable_persistence=False)
    sid1 = mm.create_session("Query 1")
    sid2 = mm.create_session("Query 2")
    s1 = mm.get_session(sid1)
    s2 = mm.get_session(sid2)
    s1.selected_documents.append({"s3_path": "only-in-s1"})
    assert s2.selected_documents == []
