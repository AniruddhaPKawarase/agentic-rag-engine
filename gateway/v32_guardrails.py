"""v32 observe-only guardrails ported to 8001 (flag-gated, additive).

Computes verification signals over the FINAL answer + source_documents WITHOUT
changing the answer text, retrieval, or latency materially (pure string work, no
network, no new deps). Attached to the /query response as ``verification_meta``
ONLY when ENABLE_V32_GUARDRAILS=true — default off keeps 8001 byte-identical.

Ported (reimplemented against 8001's response shape) from the v32 modules:
quality/citation_grounding (source_precision), quality/entity_verifier,
quality/citation_floor, quality/evidence_coverage.
"""

from __future__ import annotations

import re

# Equipment/sheet tags: FCU-XC-5, AHU-1, M-301, RTU2, DOAS-1, A-211.
_TAG_RE = re.compile(r"\b[A-Z]{1,5}-?\d[A-Za-z0-9.\-]*\b")
_NORM_RE = re.compile(r"[\s\-]+")
_SCHEDULE_RE = re.compile(
    r"\b(schedule|cfm|airflow|efficiency|capacity|tonnage|mbh|gpm|btu|tons?)\b", re.I
)
# Soft-refusal / evasive phrasing (incl. the "cannot" variants v32 fixed).
_EVASIVE_RE = re.compile(
    r"\b(can'?t|cannot|can not|could not|couldn'?t|unable to|"
    r"don'?t have enough|do(?:es)? not (?:contain|provide|include|specify|mention))\b"
    r"|not (?:found|available|present|specified|listed) in",
    re.I,
)


def _norm(s: str) -> str:
    return _NORM_RE.sub(" ", s or "").strip().lower()


def _source_text(src: dict) -> str:
    """Searchable text for one 8001 source_documents item."""
    return " ".join(
        str(src.get(k) or "")
        for k in ("text_excerpt", "drawing_name", "drawing_title", "doc_number")
    )


def _present(tag: str, text: str) -> bool:
    """Word-boundary, separator-insensitive match (FCU-101 == fcu101, not FCU-1010)."""
    nt = _norm(tag)
    if not nt:
        return False
    pat = r"\b" + r"\s*".join(re.escape(t) for t in nt.split()) + r"\b"
    return re.search(pat, _norm(text)) is not None


def entity_grounding(answer: str, sources: list[dict]) -> dict:
    """Tags asserted in the answer that are / aren't present in the retrieved text."""
    hay = " ".join(_source_text(s) for s in sources)
    tags = sorted({m for m in _TAG_RE.findall(answer or "")})
    supported = [t for t in tags if _present(t, hay)]
    unsupported = [t for t in tags if t not in supported]
    return {"total": len(tags), "supported": len(supported), "unsupported": unsupported[:15]}


def source_precision(answer: str, sources: list[dict], min_support: float = 0.12) -> dict:
    """Fraction of CITED sources whose text supports the answer (cited⇒supports)."""
    if not sources:
        return {"precision": 1.0, "supported": 0, "total": 0, "unsupported": [], "below_floor": False}
    terms = {_norm(m) for m in _TAG_RE.findall(answer or "")}
    terms = {t for t in terms if len(t) >= 2}
    supported, unsupported = 0, []
    for s in sources:
        txt = _norm(_source_text(s))
        sid = str(s.get("drawing_name") or s.get("doc_number") or s.get("file_name") or "?")
        hit = bool(terms) and (sum(1 for t in terms if t in txt) / len(terms)) >= min_support
        if hit:
            supported += 1
        else:
            unsupported.append(sid)
    prec = supported / len(sources)
    return {
        "precision": round(prec, 3),
        "supported": supported,
        "total": len(sources),
        "unsupported": unsupported[:10],
        "below_floor": prec < 0.5,
    }


def citation_density(answer: str, n_sources: int) -> dict:
    """Ratio of factual (tag/number-bearing) sentences that could be source-backed."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer or "") if s.strip()]
    factual = [s for s in sents if _TAG_RE.search(s) or re.search(r"\d", s)]
    ratio = (min(n_sources, len(factual)) / len(factual)) if factual else 1.0
    return {
        "factual_sentences": len(factual),
        "n_sources": n_sources,
        "ratio": round(ratio, 3),
        "below_floor": bool(factual) and ratio < 0.5,
    }


def evidence_coverage(query: str, answer: str) -> dict:
    """Honest badge: schedule-oriented question + evasive answer → partial coverage."""
    if _SCHEDULE_RE.search(query or "") and _EVASIVE_RE.search(answer or ""):
        return {
            "coverage": "partial",
            "reason": "schedule_table_not_extracted",
            "note": (
                "Schedule-oriented query; the answer indicates the table data was "
                "not available in the retrieved context."
            ),
        }
    return {"coverage": "full"}


def compute(query: str, answer: str, sources: list[dict]) -> dict:
    """All observe-only guardrails as one verification_meta block (never raises)."""
    # [COGUP-MARKER-V13-REWIRE-PHASE01] hallucination_dict + citation_grounding_v2
    sources = sources or []
    # Phase 1.2 — Pillar 12 hallucination anti-pattern dict (observe-only)
    halluc_matches = []
    try:
        from agentic.quality.hallucination_dict import matches_antipattern as _hd_matches
        halluc_matches = _hd_matches(answer or "")
    except Exception:
        halluc_matches = []
    # Phase 1.3 — citation grounding v2 (observe-only)
    cit_v2 = {"precision": None, "error": "not_computed"}
    try:
        from agentic.quality.citation_grounding_v2 import citation_precision as _cgv2
        cit_v2 = _cgv2(answer or "", sources)
    except Exception as _cgv2_exc:
        cit_v2 = {"precision": None, "error": type(_cgv2_exc).__name__}
    return {
        "entity_check": entity_grounding(answer, sources),
        "source_precision": source_precision(answer, sources),
        "citation_density": citation_density(answer, len(sources)),
        "evidence_coverage": evidence_coverage(query, answer),
        "hallucination_dict": {
            "match_count": len(halluc_matches),
            "matches": halluc_matches,
            "suspicious": len(halluc_matches) >= 1,
        },
        "citation_grounding_v2": cit_v2,
        "guardrails_version": "v32-observe-only-3-cogup",
    }
