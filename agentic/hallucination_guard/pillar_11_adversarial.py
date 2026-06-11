"""
Pillar 11 — Adversarial Defense (Sycophancy + Prompt Injection)

A pre-filter that runs BEFORE retrieval and synthesis. Two detectors:

  (a) Sycophancy detector — pattern-matches "are you sure?", "I think you're
      wrong", "double-check", etc. On match (with prior turn in history):
      annotates the synthesizer system prompt with a HOLD-THE-LINE directive
      so the model does NOT flip the answer without new retrieval evidence.

  (b) Prompt-injection detector — pattern-matches "ignore previous
      instructions", role-override, system-prompt extraction attempts,
      chat-template injection (<|im_start|>, [INST]). On match: returns
      an immediate refusal response. The synthesizer is NEVER called.

  (c) Regulatory-reframe detector (informational) — pattern-matches
      "hypothetically", "just for curiosity", "in general" — flags the
      query for downstream regulatory-tier handling but does not refuse
      on its own (combine with liability tier in Pillar V/§5.18 later).

Tactical Patch v2 scope:
  - Sycophancy → prompt enhancement (no re-synthesis path yet; that needs
    Pillar 4 verifier shipped first).
  - Injection → immediate refusal with INJECTION_DETECTED reason class.
  - Regulatory-reframe → metadata flag only (no behavior change).

From baseline eval 2026-06-03: injection 1/5 (20%) is 8001's worst category;
Pillar 11 targets +50%pp lift on this category alone.

Cost: zero LLM cost. Latency overhead: <2 ms (pure regex). Reversible: flag
HG_PILLAR_11=false → all detectors return ALLOW.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .flags import is_pillar_enabled

logger = logging.getLogger(__name__)

PILLAR_NUMBER = 11

# =============================================================================
# Pattern libraries — verbatim, intentionally narrow to minimize false-positives.
# Maintained quarterly per roadmap §5.20.
# =============================================================================

SYCOPHANCY_PATTERNS = [
    r"\bare you sure\b",
    r"\bare you certain\b",
    r"\byou(?:\s+are|['’]re)\s+(?:wrong|incorrect|mistaken|missing)\b",
    r"\bi (?:think|believe) (?:you|that|it|the answer)['\s](?:re|is|are)?\s*(?:wrong|different|incorrect|actually)\b",
    r"\bi don[''']?t think (?:that|this|it)(?:['\s]s)?\s+(?:right|correct|accurate)\b",
    r"\b(?:double[\s-]check|recheck|re-check|look again|try again|check again)\b",
    r"\bactually,?\s+(?:it|the answer|i think)\b",
    r"\bcould (?:you|it|that) (?:be|maybe) (?:different|wrong|higher|lower|more|less|something else)\b",
    r"\bjust (?:to confirm|double check)\b",
    r"\bthat[''']?s (?:wrong|incorrect|not right)\b",
    r"\bi was told (?:it|that)(?:['\s]s| is)\b",
    r"\b(?:really|sure)\?$",   # standalone "Really?" "Sure?"
    r"\bare you positive\b",
]

INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior|preceding|above|the)\s+(?:instructions|prompts|directives|rules|messages)\b",
    r"\bdisregard\s+(?:the\s+)?(?:previous|prior|preceding|above|all)\b",
    r"\byou are now (?:a|an|the)\s+\w+",
    r"\byou (?:must|will) now\b",
    r"\bact (?:as|like) (?:if you (?:are|were)|a different|though you (?:are|were))\b",
    r"\b(?:show|reveal|print|output|display|tell|give)\s+(?:\w+\s+){0,3}(?:your|the|my)\s+(?:system\s+)?prompt\b",
    r"\bwhat (?:are|is) your (?:instructions|prompt|directives|system prompt)\b",
    r"\bforget\s+(?:everything|all|your|the)\s*(?:previous|prior|instructions|rules)?",
    r"\b(?:new|updated)\s+instructions?\s*:",
    r"<\|im_(?:start|end)\|>",
    r"\[INST\]|\[/INST\]",
    r"<system>|</system>",
    r"###\s*(?:system|instructions?)\s*",
    r"\b(?:forget|ignore)\s+(?:construction|the docs|the documents|the drawings)\b",
    r"\boverride\s+(?:your|the)\s+(?:system|instructions|directives)\b",
    r"\brepeat (?:the|your) (?:above|previous|system) (?:prompt|instructions|message)\b",
]

REGULATORY_REFRAME_PATTERNS = [
    r"\bhypothetically\b",
    r"\bjust (?:for|out of)\s+(?:curiosity|interest)\b",
    r"\b(?:in general|generally speaking|in a general sense|broadly speaking)\b",
    r"\bjust asking\b",
    r"\boff the record\b",
    r"\bin theory\b",
    r"\bif (?:i|we|someone) were to\b",
    r"\bspeaking (?:generally|broadly|theoretically)\b",
]

# Compile once at import time
_SYC = [re.compile(p, re.IGNORECASE) for p in SYCOPHANCY_PATTERNS]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_REG = [re.compile(p, re.IGNORECASE) for p in REGULATORY_REFRAME_PATTERNS]


# =============================================================================
# FilterDecision — the structured output of pre-filtering
# =============================================================================

@dataclass
class FilterDecision:
    """Decision from the Pillar 11 pre-filter.

    Fields:
        action:            "allow" | "refuse_injection" | "annotate_sycophancy"
        reason_class:      Detail label for logs / response metadata
        matched_patterns:  List of pattern strings that fired
        refusal_text:      Pre-baked refusal message (for action=refuse_injection)
        prompt_addendum:   Text to append to synthesizer system prompt
                           (for action=annotate_sycophancy)
        regulatory_reframe_detected: Informational only — does not change action
        pillar_applied:    True iff Pillar 11 was active for this decision
    """
    action: str = "allow"
    reason_class: str = "no_match"
    matched_patterns: List[str] = field(default_factory=list)
    refusal_text: Optional[str] = None
    prompt_addendum: Optional[str] = None
    regulatory_reframe_detected: bool = False
    pillar_applied: bool = False


# =============================================================================
# Refusal + sycophancy-handling templates
# =============================================================================

INJECTION_REFUSAL_TEMPLATE = (
    "I can only answer questions about the construction documents in this "
    "project. I cannot follow instructions to reveal system configuration, "
    "switch roles, or operate outside that scope. "
    "Please ask your question directly about the project drawings, "
    "specifications, or related documents."
)

SYCOPHANCY_PROMPT_ADDENDUM = (
    "\n\n[ADVERSARIAL CONTEXT — Pillar 11 detected user pushback signal in this turn]\n"
    "The user is challenging your previous answer (\"are you sure?\", \"I think it's "
    "different\", \"double-check\", etc.).\n"
    "STRICT RULES FOR THIS TURN:\n"
    "1. Do NOT change your previous answer unless CURRENT retrieval shows new or "
    "contradicting evidence.\n"
    "2. If retrieval supports the previous answer, REAFFIRM it with citations.\n"
    "3. If retrieval is silent or contradicts, explicitly say so — do not invent "
    "alternatives to placate the user.\n"
    "4. Do not apologize for the original answer unless retrieval shows it was wrong."
)


# =============================================================================
# Detector functions
# =============================================================================

def _scan(patterns: List[re.Pattern], text: str) -> List[str]:
    """Return list of matched pattern source strings (not match objects)."""
    return [p.pattern for p in patterns if p.search(text)]


def detect_sycophancy(user_message: str) -> List[str]:
    return _scan(_SYC, user_message or "")


def detect_injection(user_message: str) -> List[str]:
    return _scan(_INJ, user_message or "")


def detect_regulatory_reframe(user_message: str) -> List[str]:
    return _scan(_REG, user_message or "")


# =============================================================================
# Main entry point — pre-filter
# =============================================================================

def pre_filter_query(user_message: str,
                     has_history: bool = False,
                     force_enable: Optional[bool] = None) -> FilterDecision:
    """Pre-filter a query before retrieval/synthesis. Run this at the
    /query handler entry, BEFORE any work begins.

    Args:
        user_message: The current turn's user content.
        has_history:  True if conversation_history is non-empty (prior turn exists).
                      Sycophancy detection only triggers if has_history is True
                      (otherwise "are you sure?" with no prior context is just a
                      benign opener).
        force_enable: For unit tests — pass True to force-run regardless of flag.
                      None reads the flag.

    Returns:
        FilterDecision with action one of: allow / refuse_injection /
        annotate_sycophancy. When flag is OFF (default), always returns
        action='allow' with pillar_applied=False.

    Caller responsibilities:
        - If action == "refuse_injection": return refusal_text directly to user
          as the /query response (status 200, body {answer: refusal_text,
          verification_meta.refused: true, reason_class: ...}). Skip retrieval
          and synthesis entirely.
        - If action == "annotate_sycophancy": append prompt_addendum to the
          synthesizer's system prompt for THIS turn only.
        - If action == "allow": process normally.
    """
    enabled = is_pillar_enabled(PILLAR_NUMBER) if force_enable is None else bool(force_enable)
    if not enabled:
        return FilterDecision(action="allow", reason_class="pillar_disabled", pillar_applied=False)

    msg = user_message or ""

    # Order matters: injection is the highest-priority — refuse early.
    inj = detect_injection(msg)
    if inj:
        logger.info("Pillar 11 INJECTION_DETECTED. patterns=%s", inj)
        return FilterDecision(
            action="refuse_injection",
            reason_class="INJECTION_DETECTED",
            matched_patterns=inj,
            refusal_text=INJECTION_REFUSAL_TEMPLATE,
            pillar_applied=True,
        )

    # Sycophancy — only meaningful if there's a prior turn to push back AGAINST.
    syc = detect_sycophancy(msg)
    if syc and has_history:
        logger.info("Pillar 11 SYCOPHANCY_DETECTED (has_history=True). patterns=%s", syc)
        reg = detect_regulatory_reframe(msg)
        return FilterDecision(
            action="annotate_sycophancy",
            reason_class="SYCOPHANCY_DETECTED",
            matched_patterns=syc,
            prompt_addendum=SYCOPHANCY_PROMPT_ADDENDUM,
            regulatory_reframe_detected=bool(reg),
            pillar_applied=True,
        )

    # Regulatory reframe alone — informational only in Tactical Patch v2.
    reg = detect_regulatory_reframe(msg)
    if reg:
        logger.debug("Pillar 11 regulatory_reframe (informational). patterns=%s", reg)
        return FilterDecision(
            action="allow",
            reason_class="regulatory_reframe_informational",
            matched_patterns=reg,
            regulatory_reframe_detected=True,
            pillar_applied=True,
        )

    return FilterDecision(action="allow", reason_class="no_match", pillar_applied=True)


def applied_metadata(decision: Optional[FilterDecision] = None) -> dict:
    """Return metadata for verification_meta in the /query response.

    If `decision` provided, includes the decision details (for the current turn).
    Otherwise returns a flag-state-only snapshot.
    """
    base = {
        "pillar": PILLAR_NUMBER,
        "name": "adversarial_defense",
        "applied": is_pillar_enabled(PILLAR_NUMBER),
        "version": "v1.0",
        "detectors": ["sycophancy", "injection", "regulatory_reframe"],
    }
    if decision is not None:
        base.update({
            "action": decision.action,
            "reason_class": decision.reason_class,
            "matched_count": len(decision.matched_patterns),
            "regulatory_reframe_detected": decision.regulatory_reframe_detected,
        })
    return base
