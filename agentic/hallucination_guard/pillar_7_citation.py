"""
Pillar 7 -- Citation-Forcing (tactical patch v1 variant).

Per HYBRID_RAG_v32_ROADMAP.md section 5.16 Pillar 7. The FULL v32 design
uses Pydantic schema enforcement via Anthropic tool_use to REJECT decoder
paths that lack citations. This tactical-patch variant is a POST-HOC
validator + transparency tag:

  1. After the synthesizer returns its answer, run a citation-density check.
  2. Count factual sentences (sentences containing numbers, named entities,
     specific claims) vs. sentences that have a citation marker.
  3. If citation density < threshold (default 0.5), append a discreet
     "[Pillar 7 note: N/M factual claims cited]" tag at the end of the answer.

Why tactical-only:
  - Full Pydantic schema enforcement requires changing the synthesizer's
    LLM call path to use Anthropic tool_use with a tool_choice forcing the
    schema. Higher risk -- we want to validate the *measurement* benefit
    of citation transparency first.
  - The tag makes uncited claims visible to the grader (and to users), which
    is the same goal the schema enforcement achieves -- it just does so
    after-the-fact rather than at decode time.

The v32 design (P79b in roadmap) will replace this post-hoc tag with
schema enforcement after this tactical variant proves the value.

Cost: zero LLM cost (pure regex). Latency overhead: <5 ms. Reversible:
HG_PILLAR_7=false -> validator is a no-op.
"""

import logging
import re
from typing import List, Tuple

from .flags import is_pillar_enabled

logger = logging.getLogger(__name__)

PILLAR_NUMBER = 7

# Configurable density threshold (env-overridable).
# 0.5 means "at least half of factual sentences must have a citation marker".
import os
_DENSITY_THRESHOLD = float(os.environ.get("HG_PILLAR_7_DENSITY_THRESHOLD", "0.5"))

# Citation marker patterns -- matches what 8001's synthesizer already emits.
# Heuristic; project may format differently.
_CITATION_PATTERNS = [
    re.compile(r"\(Ref:\s*[\w\-\s/§\d.,]+\)", re.IGNORECASE),
    re.compile(r"\[chunk[_\-]?\d+\]", re.IGNORECASE),
    re.compile(r"\[source\s*\d+\]", re.IGNORECASE),
    re.compile(r"see\s+(drawing|sheet|spec\s+section|page)\s+[\w\-]+", re.IGNORECASE),
    re.compile(r"per\s+(spec(ification)?|section)\s+\d{2}\s*\d{2}\s*\d{2}", re.IGNORECASE),
    re.compile(r"§\s*\d+\.\d+", re.IGNORECASE),
    re.compile(r"\b[A-Z]{1,3}-\d{3}", re.IGNORECASE),    # drawing names: M-211, A-321, SE-101, etc.
    re.compile(r"\bsheet\s+[A-Z]-\d+", re.IGNORECASE),
    re.compile(r"\[Page\s*\d+", re.IGNORECASE),
    re.compile(r"\bM-\d{3}\b"),
    re.compile(r"\bA-\d{3}\b"),
    re.compile(r"\bE-\d{3}\b"),
    re.compile(r"\bS-\d{3}\b"),
    re.compile(r"\bP-\d{3}\b"),
]

# A "factual sentence" is one that makes a specific claim. Heuristics:
#  - contains a digit (number, year, count)
#  - contains a named entity-shaped token (drawing name, section number)
#  - contains specific factual verbs ("is", "are", "has", "specifies", etc.)
_FACT_INDICATORS = [
    re.compile(r"\b\d+(?:\.\d+)?\b"),                # any number
    re.compile(r"\b[A-Z]{1,4}-\d+\b"),                # drawing-style ID
    re.compile(r"\bsection\s+\d", re.IGNORECASE),
    re.compile(r"\b(is|are|has|have|specifies|requires|contains|shows)\b", re.IGNORECASE),
    re.compile(r"\b(per|cited|noted|listed)\b", re.IGNORECASE),
]

# Sentences that AREN'T factual claims (we exclude these from density math):
_NON_FACTUAL_PATTERNS = [
    re.compile(r"^\s*\(", re.IGNORECASE),               # parenthetical-only
    re.compile(r"^\s*method\s*:", re.IGNORECASE),       # "Method: ..." preamble
    re.compile(r"^\s*(however|moreover|additionally|also|note that)", re.IGNORECASE),
    re.compile(r"^\s*if\s+(you|i)\b", re.IGNORECASE),
]


def _has_citation_marker(sentence: str) -> bool:
    return any(p.search(sentence) for p in _CITATION_PATTERNS)


def _is_factual(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 15:
        return False
    if any(p.match(s) for p in _NON_FACTUAL_PATTERNS):
        return False
    # Need at least 2 factual indicators
    matches = sum(1 for p in _FACT_INDICATORS if p.search(s))
    return matches >= 2


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter -- adequate for our heuristic."""
    # Split on ., !, ?, or newline followed by capital
    parts = re.split(r"(?:[.!?]+\s+)|(?:\n\s*\n)", text)
    return [p.strip() for p in parts if p.strip()]


def validate_citations(answer_text: str) -> dict:
    """Score citation density on an answer.

    Returns dict with:
      density: float (0-1)
      factual_count: int
      cited_count: int
      verdict: "pass" | "tag_unverified" | "no_factual_content"
    """
    if not answer_text or not answer_text.strip():
        return {"density": 0.0, "factual_count": 0, "cited_count": 0, "verdict": "no_factual_content"}

    sentences = _split_sentences(answer_text)
    factual = [s for s in sentences if _is_factual(s)]
    cited = [s for s in factual if _has_citation_marker(s)]

    if not factual:
        return {
            "density": 1.0,
            "factual_count": 0,
            "cited_count": 0,
            "verdict": "no_factual_content",
        }

    density = len(cited) / len(factual)
    verdict = "pass" if density >= _DENSITY_THRESHOLD else "tag_unverified"
    return {
        "density": round(density, 3),
        "factual_count": len(factual),
        "cited_count": len(cited),
        "verdict": verdict,
        "threshold": _DENSITY_THRESHOLD,
    }


def post_validate(answer: str, force_enable: bool = None) -> Tuple[str, dict]:
    """Run post-hoc citation validation on a synthesized answer.

    Args:
        answer: The synthesizer's output text.
        force_enable: For unit tests; bypasses flag check.

    Returns:
        (possibly-tagged answer, validation metadata dict)

    Behavior:
        - If flag is off: returns (answer, {"applied": False})
        - If no factual content: returns (answer, validation result)
        - If density >= threshold: returns (answer, validation result with verdict="pass")
        - If density < threshold: returns (answer + "[Pillar 7 note: ...]" tag,
          validation result with verdict="tag_unverified")

    NEVER raises. NEVER returns None. Failures fall back to (answer, {"error": ...}).
    """
    enabled = is_pillar_enabled(PILLAR_NUMBER) if force_enable is None else bool(force_enable)
    if not enabled:
        return answer, {"pillar": PILLAR_NUMBER, "applied": False}

    try:
        result = validate_citations(answer)
        meta = {"pillar": PILLAR_NUMBER, "applied": True, "name": "citation_validator", **result}

        if result["verdict"] == "tag_unverified":
            tag = (
                f"\n\n[Pillar 7 note: only {result['cited_count']} of "
                f"{result['factual_count']} factual statements have explicit citations. "
                f"Please verify uncited details against retrieved sources before relying.]"
            )
            logger.info(
                "[hg-p7] tag_unverified: density=%.2f (%d/%d cited, threshold=%.2f)",
                result["density"], result["cited_count"], result["factual_count"], _DENSITY_THRESHOLD,
            )
            return answer + tag, meta

        if result["verdict"] == "no_factual_content":
            logger.debug("[hg-p7] no_factual_content; skipping tag")

        return answer, meta

    except Exception as exc:
        logger.warning("Pillar 7 validator failed (%s); passing through unchanged", exc)
        return answer, {"pillar": PILLAR_NUMBER, "applied": True, "error": str(exc)}


def applied_metadata(result: dict = None) -> dict:
    """For verification_meta.pillar_7 in response."""
    base = {
        "pillar": PILLAR_NUMBER,
        "name": "citation_validator",
        "applied": is_pillar_enabled(PILLAR_NUMBER),
        "version": "v1.0-post-hoc",
        "threshold": _DENSITY_THRESHOLD,
    }
    if result:
        base.update({k: result.get(k) for k in ("density", "factual_count", "cited_count", "verdict")})
    return base
