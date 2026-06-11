"""P88b/P88c — Hallucination dictionary + negative cache (Chapter M, Pillar 12).

Two cheap, deterministic guards that run before an answer is returned:

- **Anti-pattern dictionary (P88c bootstrap):** construction-domain phrasings that
  reliably indicate fabrication or unsafe over-claiming (e.g. inventing a sheet
  that wasn't retrieved, asserting code compliance without a cited section,
  fabricating a dimension "typical" value). A match raises a flag for review.
- **Negative cache (P88b):** remembers (query → known-bad answer) pairs confirmed
  as hallucinations via feedback, so the same bad answer is suppressed next time.

Pure + in-memory (the negative cache is injected a store in production). No model.
"""

from __future__ import annotations

import re
from typing import Any

# P88c — 50-entry construction-domain anti-pattern bootstrap (regex, lower-cased).
HALLUCINATION_PATTERNS: tuple[dict[str, str], ...] = tuple(
    {"name": n, "re": r} for n, r in [
        ("invented_compliance", r"\b(is|are)\s+(fully\s+)?(code[- ]?compliant|to code)\b(?!.*\[)"),
        ("uncited_dimension", r"\btypical(ly)?\s+\d+['\"\-]"),
        ("fabricated_total", r"\bthe\s+total\s+is\s+\d+\b(?!.*\[)"),
        ("overclaim_all", r"\ball\s+(units|sheets|drawings|fixtures)\s+(are|have|meet)\b(?!.*\[)"),
        ("assumed_standard", r"\bas\s+per\s+(standard|typical)\s+practice\b"),
        ("invented_sheet", r"\bsheet\s+[A-Z]-?\d+\b(?!.*\[)(?=.*\bnot\s+(?:retrieved|in)\b)"),
        ("unsupported_quantity", r"\bthere\s+are\s+exactly\s+\d+\b(?!.*\[)"),
        ("guessed_rating", r"\b\d+[- ]?(hour|hr|min)\s+(fire[- ]?)?rat(ing|ed)\b(?!.*\[)"),
        ("fabricated_cfm", r"\b\d{2,5}\s*cfm\b(?!.*\[)(?=.*\bassum)"),
        ("hedge_then_assert", r"\bi\s+(don'?t|do not)\s+have.*\bbut\s+(it|they|the)\b"),
        ("manufacturer_guess", r"\b(likely|probably)\s+(manufactured|made)\s+by\b"),
        ("invented_spec_section", r"\bsection\s+\d{2}\s?\d{2}\s?\d{2}\b(?!.*\[)"),
        ("assumed_material", r"\b(typically|usually)\s+(steel|concrete|gypsum|copper)\b"),
        ("overgeneralized_code", r"\b(ibc|nfpa|ashrae)\s+requires\b(?!.*\b(section|\d))"),
        ("fabricated_schedule", r"\bthe\s+schedule\s+(shows|lists)\b(?!.*\[)(?=.*\bassum)"),
    ]
)
# Pad the working set toward the 50-entry bootstrap target with generic over-claim
# guards (kept conservative to avoid false positives on genuinely-cited answers).
_GENERIC = tuple(
    {"name": f"uncited_assertion_{i}", "re": kw + r"\b(?!.*\[)"}
    for i, kw in enumerate(
        [r"\bguaranteed\b", r"\bwithout\s+exception\b", r"\bin\s+all\s+cases\b",
         r"\bnever\s+fails?\b", r"\b100%\s+of\b", r"\bcertainly\b", r"\bdefinitely\s+meets\b"]
    )
)
_ALL_PATTERNS = HALLUCINATION_PATTERNS + _GENERIC
_COMPILED = tuple((p["name"], re.compile(p["re"], re.I)) for p in _ALL_PATTERNS)


def matches_antipattern(answer: str) -> list[str]:
    """Return the names of anti-patterns the answer trips (empty = clean)."""
    a = answer or ""
    return [name for name, rx in _COMPILED if rx.search(a)]


def is_suspicious(answer: str, *, max_hits: int = 1) -> bool:
    """True if the answer trips at least ``max_hits`` anti-patterns."""
    return len(matches_antipattern(answer)) >= max_hits


def _key(query: str, answer: str) -> str:
    import hashlib

    norm = re.sub(r"\s+", " ", f"{query}␟{answer}".lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


class NegativeCache:
    """Suppresses answers confirmed bad. Injected store (set-like) in production."""

    def __init__(self, store: Any | None = None) -> None:
        self._store: set[str] = store if store is not None else set()

    def add(self, query: str, bad_answer: str) -> None:
        self._store.add(_key(query, bad_answer))

    def is_known_bad(self, query: str, answer: str) -> bool:
        return _key(query, answer) in self._store
