"""Citation grounding v2 — observability-first.

Standalone port of v32 quality/citation_grounding.py adapted for 8013's
dict-shape source documents (not Pydantic Chunk).

Two outputs:
1. ``citation_precision()`` — score fraction of CITED sources that actually
   support the answer (1.0 = every citation earned). Pure observability.
2. ``ground_citations_dict()`` — re-orders sources by support; never drops to
   zero so an answer is never source-less.

The 8001 failure mode this addresses: "right answer, wrong source" — the answer
is correct but the attached top source doesn't actually back it.
"""
from __future__ import annotations
import re
from typing import Any

_BRACKET_REF = re.compile(r"\[([A-Za-z0-9.\-_/]+?)\]")
_PAREN_REF = re.compile(r"\(\s*Ref:\s*([A-Za-z0-9.\-_/, ]+?)\s*\)", re.I)
_SHEET = re.compile(r"\b[A-Z]{1,3}-?\d[A-Za-z0-9.\-]*\b")
_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_TAG = re.compile(r"\b(FCU|AHU|VAV|RTU|EF|SF|HW|CHW|EWC|FP|PRV|DFU|VFD|ELV|EM)-?[A-Z0-9.\-]+\b", re.I)

SOURCE_PRECISION_FLOOR = 0.5

def _norm(token: str) -> str:
    t = (token or "").strip().upper()
    t = re.sub(r"_P\d+$", "", t)
    return t

def answer_referenced_ids(answer: str) -> set[str]:
    refs: set[str] = set()
    for m in _BRACKET_REF.findall(answer or ""):
        refs.add(_norm(m))
    for grp in _PAREN_REF.findall(answer or ""):
        for piece in grp.split(","):
            if piece.strip():
                refs.add(_norm(piece))
    return {r for r in refs if r}

def salient_terms(answer: str) -> set[str]:
    a = answer or ""
    terms: set[str] = set()
    for m in _TAG.findall(a):
        terms.add(m.upper())
    for m in _SHEET.findall(a):
        terms.add(m.upper())
    for m in _NUM.findall(a):
        terms.add(m)
    return {t for t in terms if len(t) >= 2}

def _src_id(src: dict[str, Any]) -> str:
    # [V14-CIT-FIELDS] prefer sheet-shaped ids (drawing_name / sheetNumber) over numeric drawing_id —
    # source_documents on 8001/8013 carry the sheet number in drawing_name ("A-701.00")
    return _norm(str(src.get("drawing_name") or src.get("sheetNumber") or src.get("sheet_number")
                  or src.get("source_id") or src.get("pdf_name")
                  or src.get("drawing_id") or src.get("id") or ""))

def _src_sheet(src: dict[str, Any]) -> str:
    return _norm(str(src.get("drawing_name") or src.get("sheetNumber") or src.get("sheet_number") or ""))

def _src_text(src: dict[str, Any]) -> str:
    # [V14-CIT-FIELDS] include text_excerpt + display_title + drawing_title (actual response schema)
    parts = [src.get("content"), src.get("text"), src.get("snippet"),
             src.get("text_excerpt"), src.get("display_title"), src.get("drawing_title")]
    return " ".join(str(p) for p in parts if p)

def _support(src: dict[str, Any], refs: set[str], terms: set[str]) -> float:
    sid = _src_id(src)
    sheet = _src_sheet(src)
    if (sid and sid in refs) or (sheet and sheet in refs):
        return 1.0
    if not terms:
        return 0.0
    hay = (_src_text(src) + " " + sid + " " + sheet).upper()
    hits = sum(1 for t in terms if t in hay)
    return hits / len(terms) if terms else 0.0

def citation_precision(answer: str, sources: list[dict[str, Any]],
                       min_support: float = 0.12) -> dict[str, Any]:
    """Score the precision of the source list against the answer text."""
    if not sources:
        return {"precision": 1.0, "supported": 0, "total": 0,
                "unsupported": [], "below_floor": False}
    refs = answer_referenced_ids(answer)
    terms = salient_terms(answer)
    supported = 0
    unsupported: list[str] = []
    for s in sources:
        if _support(s, refs, terms) >= min_support:
            supported += 1
        else:
            unsupported.append(_src_id(s))
    total = len(sources)
    precision = supported / total
    return {
        "precision": round(precision, 3),
        "supported": supported,
        "total": total,
        "unsupported": unsupported,
        "below_floor": precision < SOURCE_PRECISION_FLOOR,
    }
