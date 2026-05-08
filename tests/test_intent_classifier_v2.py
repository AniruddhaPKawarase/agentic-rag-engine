"""Phase 3.1: bidirectional intent classifier tests."""
from __future__ import annotations

import pytest

from gateway.intent_classifier import classify, IntentDecision


class FakeSession:
    def __init__(self, selected_documents=None, active_agent="rag"):
        self.selected_documents = selected_documents or []
        self.active_agent = active_agent


# (query, has_selected_docs, expected_target, note)
CASES = [
    # --- clear DocQA (strong doc-scope + selected doc) ---
    ("what does it say on page 14 of this spec?", True, "docqa", "doc-scoped + on-page"),
    ("in this document, where is fire damper?", True, "docqa", "'in this document'"),
    ("explain section 2.3 of the selected drawing", True, "docqa", "'selected drawing'"),
    ("what is mentioned on page 5", True, "docqa", "'on page N'"),
    ("summarize this pdf", True, "docqa", "'this pdf'"),
    ("in the uploaded doc, is fire damper listed?", True, "docqa", "'uploaded doc'"),
    ("on this page, what's the callout?", True, "docqa", "'on this' + 'page'"),
    ("tell me about page 3", True, "docqa", "'page N'"),
    ("and on page 16?", True, "docqa", "follow-up + page"),
    ("what about this page?", True, "docqa", "'this page'"),

    # --- clear RAG (project-wide signals always win) ---
    ("show all missing scope across project", False, "rag", "'across project'"),
    ("how many DOAS units are in the project", False, "rag", "'how many'"),
    ("list all HVAC drawings", False, "rag", "'list all'"),
    ("compare all mechanical floors", False, "rag", "'compare all'"),
    ("total count of fire dampers", False, "rag", "'total count'"),
    ("project-wide summary", False, "rag", "'project-wide'"),
    ("every drawing with plumbing", False, "rag", "'every drawing'"),
    ("summarize the entire project", False, "rag", "'entire project'"),
    ("generate scope gap report", False, "rag", "'generate scope'"),

    # --- RAG dominance: project-wide beats selected-doc ---
    ("show me all missing scope across project", True, "rag", "project-wide overrides selected"),
    ("list all drawings", True, "rag", "'all drawings'"),
    ("which drawings have fire damper callouts", True, "rag", "'which drawings' plural"),

    # --- clarify needed (ambiguous pronouns or generic nouns with selected doc) ---
    ("is it missing?", True, "clarify", "pure pronoun + selected"),
    ("tell me about fire damper", True, "clarify", "generic noun + selected"),
    ("what does it cover?", True, "clarify", "pronoun + selected"),
    ("any details on HVAC?", True, "clarify", "generic topic + selected"),

    # --- no selected doc => RAG even if query looks doc-scoped ---
    ("what does this say?", False, "rag", "no doc selected"),
    ("explain page 14", False, "rag", "no doc selected"),

    # --- exit signals ---
    ("back to project search", True, "rag", "explicit exit"),
    ("go back", True, "rag", "exit"),
    ("exit document", True, "rag", "exit"),
    ("return to search", True, "rag", "exit"),

    # --- greetings / meta (no signal, no selected doc → RAG) ---
    ("hi", False, "rag", "greeting"),
    ("what can you do?", False, "rag", "meta"),

    # --- construction-domain generic terms (neutral, selected ambiguous) ---
    ("fire damper specs", False, "rag", "no selected, no doc-scope"),
    ("xvent model numbers", False, "rag", "generic"),
    ("structural foundation details", False, "rag", "no selected, generic"),
]


@pytest.mark.parametrize("query,has_docs,expected,note", CASES)
def test_classify_cases(query, has_docs, expected, note):
    sess = FakeSession(
        selected_documents=[{"file_name": "X.pdf"}] if has_docs else []
    )
    decision = classify(query=query, session=sess)
    assert decision.target == expected, (
        f"[{note}] query={query!r} "
        f"got target={decision.target} conf={decision.confidence:.2f} "
        f"reason={decision.reason!r}"
    )


def test_mode_hint_docqa_overrides_classifier():
    sess = FakeSession(selected_documents=[{"file_name": "X.pdf"}])
    d = classify(query="list all drawings", session=sess, mode_hint="docqa")
    assert d.target == "docqa"
    assert d.confidence == 1.0
    assert "mode_hint" in d.reason


def test_mode_hint_rag_overrides_classifier():
    sess = FakeSession(selected_documents=[{"file_name": "X.pdf"}])
    d = classify(query="in this doc, anything?", session=sess, mode_hint="rag")
    assert d.target == "rag"
    assert d.confidence == 1.0


def test_mode_hint_invalid_is_ignored():
    sess = FakeSession()
    d = classify(query="hi", session=sess, mode_hint="gibberish")
    # Invalid hint is ignored; normal scoring runs
    assert d.target == "rag"
    assert d.confidence < 1.0


def test_decision_shape():
    sess = FakeSession()
    d = classify(query="hi", session=sess)
    assert isinstance(d, IntentDecision)
    assert isinstance(d.target, str)
    assert isinstance(d.confidence, float)
    assert 0.0 <= d.confidence <= 1.0
    assert isinstance(d.reason, str)
    # clarification_prompt may be None


def test_clarify_prompt_includes_file_name():
    sess = FakeSession(selected_documents=[{"file_name": "HVAC_spec.pdf"}])
    d = classify(query="tell me about fire damper", session=sess)
    assert d.target == "clarify"
    assert d.clarification_prompt
    assert "HVAC_spec.pdf" in d.clarification_prompt


def test_clarify_prompt_handles_missing_file_name():
    """If selected_documents is non-empty but lacks file_name, prompt should still be built."""
    sess = FakeSession(selected_documents=[{"s3_path": "bucket/x.pdf"}])
    d = classify(query="tell me about it", session=sess)
    assert d.target == "clarify"
    assert d.clarification_prompt
    # Should use a generic fallback ("your selection") or the s3_path — not crash
    assert any(token in d.clarification_prompt for token in ("selection", "x.pdf", "document"))


def test_confidence_is_deterministic():
    """Same input → same confidence every call."""
    sess = FakeSession(selected_documents=[{"file_name": "X.pdf"}])
    d1 = classify(query="in this document, what's on page 3?", session=sess)
    d2 = classify(query="in this document, what's on page 3?", session=sess)
    assert d1.target == d2.target
    assert d1.confidence == d2.confidence
