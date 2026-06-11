"""
Tier B2 — Deterministic symbol counting via the ``symbolsByKind`` field.

The v3 extraction populates ``drawings_v3.symbolsByKind`` as a flat
``{kind: count}`` dict per drawing page (e.g. ``{"pipe": 12, "fixture": 3,
"detail_callout": 6}``). Audit (2026-05-27) confirmed 96-100% population
across all 6 projects.

These tools answer count / inventory questions **deterministically** with
a single Mongo aggregation — no LLM, no retrieval, no hallucination:
  - "How many fixtures on sheet P-101?"        -> count_symbols_by_kind_v3
  - "Total pipe symbols across all plumbing?"   -> sum_symbols_by_kind_v3
  - "Which drawing has the most ducts?"         -> find_drawings_by_symbol_kind_v3

NOT for visual identification ("what symbol is this?") — that needs vision.
NOT for sub-type granularity ("double-leaf vs single-leaf doors") — the
extractor's categories are top-level (door, pipe, fixture, etc.).

NO Mongo writes — read-only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id

logger = logging.getLogger("agentic_rag.tools.symbol_count_v3")

COLLECTION = "drawings_v3"

_DRAWING_META_PROJ: Dict[str, Any] = {
    "_id": 0,
    "drawingName": 1, "drawingTitle": 1,
    "sheetNumber": 1, "sheet_number": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "discipline": 1, "drawingType": 1, "trade": 1,
    "drawingId": 1, "symbolsCount": 1, "symbolsByKind": 1,
}


def _build_drawing_filter(
    project_id: int,
    sheet_number: Optional[str] = None,
    drawing_name: Optional[str] = None,
    discipline: Optional[str] = None,
    level: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose a Mongo filter from optional narrowing fields. Always
    scoped to project_id."""
    flt: Dict[str, Any] = {"projectId": project_id}
    if sheet_number:
        # Match either sheet_number or sheetNumber (both exist in v3 docs)
        sheet_number = str(sheet_number).strip()
        flt["$or"] = [
            {"sheet_number": sheet_number},
            {"sheetNumber": sheet_number},
        ]
    if drawing_name:
        flt["drawingName"] = str(drawing_name).strip()
    if discipline:
        flt["discipline"] = str(discipline).strip()
    if level:
        flt["level"] = str(level).strip()
    return flt


def count_symbols_by_kind_v3(
    project_id: int,
    sheet_number: Optional[str] = None,
    drawing_name: Optional[str] = None,
    kind: Optional[str] = None,
    discipline: Optional[str] = None,
    level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return per-drawing symbol counts.

    If ``kind`` is given, returns ``[{drawingName, sheet_number, count}]``
    filtered to that kind only (case-insensitive substring match against
    the keys of ``symbolsByKind``).

    Without ``kind``, returns the full ``symbolsByKind`` dict per drawing
    so the agent can pick the relevant category.

    Examples:
      - "fixtures on P-101"     -> count_symbols_by_kind_v3(p, sheet="P-101", kind="fixture")
      - "all symbols on A-101"  -> count_symbols_by_kind_v3(p, sheet="A-101")
    """
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)
    flt = _build_drawing_filter(
        project_id=project_id,
        sheet_number=sheet_number,
        drawing_name=drawing_name,
        discipline=discipline,
        level=level,
    )

    out: List[Dict[str, Any]] = []
    cursor = coll.find(flt, _DRAWING_META_PROJ).limit(200)
    kind_lc = kind.lower().strip() if kind else None

    for doc in cursor:
        sbk = doc.get("symbolsByKind") or {}
        if not isinstance(sbk, dict):
            sbk = {}
        if kind_lc:
            # substring-match the kind against keys (e.g. "door" matches "door_symbol")
            matched_kinds = {k: v for k, v in sbk.items()
                             if isinstance(k, str) and kind_lc in k.lower()}
            count = sum(int(v) for v in matched_kinds.values() if isinstance(v, (int, float)))
            row = {
                "drawingName": doc.get("drawingName"),
                "drawingTitle": doc.get("drawingTitle"),
                "sheet_number": doc.get("sheet_number") or doc.get("sheetNumber"),
                "discipline": doc.get("discipline"),
                "pdfName": doc.get("pdfName"),
                "s3BucketPath": doc.get("s3BucketPath"),
                "page": doc.get("page"),
                "drawingId": doc.get("drawingId"),
                "count": int(count),
                "kind_query": kind,
                "matched_kinds": matched_kinds,
            }
        else:
            row = {
                "drawingName": doc.get("drawingName"),
                "drawingTitle": doc.get("drawingTitle"),
                "sheet_number": doc.get("sheet_number") or doc.get("sheetNumber"),
                "discipline": doc.get("discipline"),
                "pdfName": doc.get("pdfName"),
                "s3BucketPath": doc.get("s3BucketPath"),
                "page": doc.get("page"),
                "drawingId": doc.get("drawingId"),
                "symbolsByKind": sbk,
                "symbolsCount": doc.get("symbolsCount", 0),
            }
        out.append(row)

    logger.info(
        "count_symbols_by_kind_v3: project=%d filter=%r kind=%r hits=%d",
        project_id, {k: v for k, v in flt.items() if k != "projectId"}, kind, len(out),
    )
    return out


def sum_symbols_by_kind_v3(
    project_id: int,
    kind: str,
    discipline: Optional[str] = None,
    level: Optional[str] = None,
    drawing_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Sum a single symbol kind across all drawings matching the filters.

    Returns ``{total, drawings_contributing, kind_query, matched_kinds, top_drawings}``.

    Use for portfolio-level questions like "how many roof drains across
    all level 1 plumbing plans" — Mongo aggregation does the math in one
    round-trip, no LLM math involved.
    """
    project_id = validate_project_id(project_id)
    if not kind or not isinstance(kind, str):
        return {"total": 0, "drawings_contributing": 0, "kind_query": kind,
                "matched_kinds": {}, "top_drawings": []}
    coll = get_collection(COLLECTION)

    flt: Dict[str, Any] = {"projectId": project_id, "symbolsByKind": {"$exists": True}}
    if discipline:
        flt["discipline"] = str(discipline).strip()
    if level:
        flt["level"] = str(level).strip()
    if drawing_type:
        flt["drawingType"] = str(drawing_type).strip()

    kind_lc = kind.lower().strip()
    total = 0
    n_contributing = 0
    matched_kinds_global: Dict[str, int] = {}
    per_drawing: List[Dict[str, Any]] = []

    for doc in coll.find(flt, _DRAWING_META_PROJ).limit(5000):
        sbk = doc.get("symbolsByKind") or {}
        if not isinstance(sbk, dict):
            continue
        matched = {k: v for k, v in sbk.items()
                   if isinstance(k, str) and kind_lc in k.lower()
                   and isinstance(v, (int, float))}
        if not matched:
            continue
        per_drawing_total = sum(int(v) for v in matched.values())
        if per_drawing_total <= 0:
            continue
        total += per_drawing_total
        n_contributing += 1
        for k, v in matched.items():
            matched_kinds_global[k] = matched_kinds_global.get(k, 0) + int(v)
        per_drawing.append({
            "drawingName": doc.get("drawingName"),
            "drawingTitle": doc.get("drawingTitle"),
            "sheet_number": doc.get("sheet_number") or doc.get("sheetNumber"),
            "discipline": doc.get("discipline"),
            "pdfName": doc.get("pdfName"),
            "s3BucketPath": doc.get("s3BucketPath"),
            "page": doc.get("page"),
            "drawingId": doc.get("drawingId"),
            "count": per_drawing_total,
        })

    per_drawing.sort(key=lambda x: x["count"], reverse=True)
    out = {
        "total": int(total),
        "drawings_contributing": int(n_contributing),
        "kind_query": kind,
        "matched_kinds": matched_kinds_global,
        "top_drawings": per_drawing[:20],
    }
    logger.info(
        "sum_symbols_by_kind_v3: project=%d kind=%r filter=%r total=%d drawings=%d",
        project_id, kind, {k: v for k, v in flt.items() if k != "projectId"},
        total, n_contributing,
    )
    return out


def find_drawings_by_symbol_kind_v3(
    project_id: int,
    kind: str,
    limit: int = 10,
    discipline: Optional[str] = None,
    level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return top-N drawings sorted by count of the requested symbol kind.

    Example: "which drawing has the most duct elbows?" ->
    find_drawings_by_symbol_kind_v3(p, kind="duct").
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=50)
    result = sum_symbols_by_kind_v3(
        project_id=project_id, kind=kind,
        discipline=discipline, level=level,
    )
    return result.get("top_drawings", [])[:limit]


def list_symbol_kinds_v3(
    project_id: int,
    discipline: Optional[str] = None,
    top_n: int = 30,
) -> List[Dict[str, Any]]:
    """Discovery helper: list all distinct symbol kinds in this project
    (or discipline) with aggregate counts.

    Useful when the agent doesn't know what kinds were extracted, e.g.
    user asks "what kinds of symbols exist on architectural drawings?".
    """
    project_id = validate_project_id(project_id)
    top_n = validate_limit(top_n, max_limit=200)
    coll = get_collection(COLLECTION)

    match: Dict[str, Any] = {"projectId": project_id,
                             "symbolsByKind": {"$exists": True, "$ne": {}}}
    if discipline:
        match["discipline"] = str(discipline).strip()

    pipe: List[Dict[str, Any]] = [
        {"$match": match},
        {"$project": {"keys": {"$objectToArray": "$symbolsByKind"}}},
        {"$unwind": "$keys"},
        {"$group": {
            "_id": "$keys.k",
            "n_drawings": {"$sum": 1},
            "total": {"$sum": "$keys.v"},
        }},
        {"$sort": {"total": -1}},
        {"$limit": top_n},
    ]
    out = [
        {"kind": r["_id"], "n_drawings": r["n_drawings"], "total": r["total"]}
        for r in coll.aggregate(pipe)
    ]
    logger.info(
        "list_symbol_kinds_v3: project=%d discipline=%r distinct_kinds=%d",
        project_id, discipline, len(out),
    )
    return out
