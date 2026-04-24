"""
Intent classifier v2 — bidirectional RAG ↔ DocQA routing.

Weighted keyword + session-state scoring. Returns an IntentDecision
with (target, confidence, reason, clarification_prompt).

The orchestrator uses confidence to decide:
  ≥ 0.7  → route directly to target
  0.3–0.7 + selected doc → clarify (ask user)
  < 0.3  → default to RAG

`mode_hint="rag"` or `mode_hint="docqa"` fully overrides. Any other
mode_hint value is ignored.

This module has NO side effects and NO external calls — pure scoring.
Tested in isolation; wired into orchestrator by Task 3.2.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ patterns

PROJECT_WIDE_PATTERNS: List[str] = [
    r"\bacross (the )?project\b",
    r"\bacross all\b",
    r"\ball drawings\b",
    r"\ball specifications?\b",
    r"\ball trades\b",
    r"\bproject[- ]wide\b",
    r"\bentire project\b",
    r"\bhow many\b",
    r"\blist all\b",
    r"\bshow all\b",
    r"\bcompare all\b",
    r"\bmissing scope\b",
    r"\bscope gap\b",
    r"\btotal count\b",
    r"\bproject summary\b",
    r"\bproject overview\b",
    r"\bevery (floor|drawing|spec|sheet)\b",
    r"\ball (mechanical|electrical|plumbing|hvac)\b",
    r"\bdifferent drawings\b",
    r"\bwhich drawings\b",
    r"\bgenerate (scope|report)\b",
]

DOC_SCOPED_PATTERNS: List[str] = [
    r"\bthis (document|drawing|spec(ification)?|page|section|file|pdf)\b",
    r"\bin this\b",
    r"\bon this\b",
    r"\bfrom this\b",
    r"\bof this\b",
    r"\bthe (selected|current|uploaded)\b",
    r"\b(on|in) page\s*\d+\b",
    r"\bpage\s*\d+\b",
    r"\bpage (number|no\.?)\b",
    r"\bin section\b",
    r"\bwhat does it say\b",
    r"\bwhat is mentioned\b",
]

EXIT_SIGNAL_PATTERNS: List[str] = [
    r"\bback to (search|project)\b",
    r"\bexit document\b",
    r"\bstop chatting with\b",
    r"\breturn to (rag|search)\b",
    r"\bgo back\b",
    r"\bsearch project\b",
    r"\bproject search\b",
]

PRONOUN_ONLY_PATTERNS: List[str] = [
    r"\bwhat about it\b",
    r"\band the\b",
    r"\btell me more\b",
    r"\bany details\b",
    r"\bwhat(?:'s| is) in\b",
    r"\bis it\b",
    r"\bwhat does it\b",
    r"\btell me about\b",  # common ambiguous frame when doc is selected
]

# ------------------------------------------------------------------ API


@dataclass(frozen=True)
class IntentDecision:
    target: str                         # "rag" | "docqa" | "clarify"
    confidence: float                   # 0.0 – 1.0
    reason: str
    clarification_prompt: Optional[str] = None


# ----------------------------------------------------------------- helpers


def _matches_any(patterns: List[str], q: str) -> List[str]:
    return [p for p in patterns if re.search(p, q, re.IGNORECASE)]


def _selected_name(session: Any) -> Optional[str]:
    docs = getattr(session, "selected_documents", None) or []
    if not docs:
        return None
    last = docs[-1]
    if not isinstance(last, dict):
        return None
    return last.get("file_name") or last.get("s3_path", "").rsplit("/", 1)[-1] or None


# ---------------------------------------------------------------- public API


def classify(
    query: str,
    session: Any,
    mode_hint: Optional[str] = None,
) -> IntentDecision:
    """Classify user intent. See module docstring for scoring rules."""
    q = (query or "").strip().lower()

    # 1. mode_hint override (only "rag" or "docqa" are valid)
    if mode_hint in ("rag", "docqa"):
        return IntentDecision(
            target=mode_hint,
            confidence=1.0,
            reason=f"explicit mode_hint={mode_hint}",
        )

    # 2. Exit signals always win — return user to RAG
    exit_hits = _matches_any(EXIT_SIGNAL_PATTERNS, q)
    if exit_hits:
        return IntentDecision(
            target="rag",
            confidence=1.0,
            reason=f"explicit exit ({exit_hits[0]})",
        )

    # 3. Project-wide signals override selected-doc bias
    project_hits = _matches_any(PROJECT_WIDE_PATTERNS, q)
    if project_hits:
        return IntentDecision(
            target="rag",
            confidence=0.9,
            reason=f"project-wide signal: {project_hits[:2]}",
        )

    # 4. Accumulate DocQA score
    has_selected = bool(getattr(session, "selected_documents", None))
    doc_hits = _matches_any(DOC_SCOPED_PATTERNS, q)
    pronoun_hits = _matches_any(PRONOUN_ONLY_PATTERNS, q)

    score = 0.0
    if has_selected:
        score += 0.3
    if doc_hits:
        score += 0.4
    if pronoun_hits:
        score += 0.2
    score = min(score, 1.0)

    # 5. Decide
    if score >= 0.7:
        return IntentDecision(
            target="docqa",
            confidence=score,
            reason=(
                f"doc-scoped hits={doc_hits[:2]} "
                f"pronoun_hits={pronoun_hits[:2]} selected={has_selected}"
            ),
        )
    if 0.3 <= score < 0.7 and has_selected:
        name = _selected_name(session) or "your selection"
        prompt = (
            f"Should I answer from the selected document ({name}) "
            f"or search the whole project?"
        )
        return IntentDecision(
            target="clarify",
            confidence=score,
            reason=f"ambiguous (score={score:.2f}) with selected document",
            clarification_prompt=prompt,
        )

    # Default to RAG
    return IntentDecision(
        target="rag",
        confidence=max(0.1, 0.4 - score),
        reason="no strong doc-scope signal",
    )


# ----------------------------------------------------------------- legacy API


def classify_intent(query: str, active_agent: str = "rag") -> str:
    """Legacy one-way wrapper kept for backward compatibility.

    DO NOT USE in new code — use classify() which returns IntentDecision.
    This wrapper is retained so any existing caller that imports
    classify_intent does not break.
    """
    # Build a session-like shim from active_agent alone
    class _Shim:
        def __init__(self, active: str):
            self.selected_documents = []
            self.active_agent = active
    decision = classify(query=query, session=_Shim(active_agent))
    # Map new values back to the legacy vocabulary when in docqa mode
    if active_agent == "docqa":
        if decision.target == "rag":
            return "suggest_switch_to_rag"
        return "docqa"
    return decision.target if decision.target in ("rag", "docqa") else "rag"
