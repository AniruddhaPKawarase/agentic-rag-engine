"""
Tier B4 — Deterministic titleblock field lookup via ``drawings_v3.titleBlock``.

The v3 extraction populates ``drawings_v3.titleBlock`` as a structured dict
on 96-100% of drawings (audited 2026-05-27):
    {date, scale, revision, architect, sheet_title, project_name, sheet_number}

These tools answer titleblock factoids deterministically with a single
Mongo lookup — no LLM, no retrieval, no hallucination:
  - "What is the scale on A-101?"               -> lookup_titleblock_field_v3
  - "Who is the architect of this project?"     -> get_project_titleblock_info_v3
  - "When was sheet M-301 last revised?"         -> lookup_titleblock_field_v3

NO Mongo writes — read-only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id

logger = logging.getLogger("agentic_rag.tools.titleblock_v3")

COLLECTION = "drawings_v3"

_TB_PROJECTION: Dict[str, Any] = {
    "_id": 0,
    "drawingName": 1, "drawingTitle": 1,
    "sheetNumber": 1, "sheet_number": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "discipline": 1, "drawingType": 1, "trade": 1,
    "drawingId": 1, "titleBlock": 1,
}

# Canonical titleblock field aliases the LLM might pass
_FIELD_ALIASES: Dict[str, str] = {
    "scale": "scale",
    "date": "date",
    "issue_date": "date",
    "revision": "revision",
    "rev": "revision",
    "revision_date": "revision",
    "architect": "architect",
    "engineer_of_record": "architect",
    "design_firm": "architect",
    "title": "sheet_title",
    "sheet_title": "sheet_title",
    "drawing_title": "sheet_title",
    "project": "project_name",
    "project_name": "project_name",
    "sheet": "sheet_number",
    "sheet_number": "sheet_number",
    "sheet_no": "sheet_number",
}


def _resolve_field(field: Optional[str]) -> Optional[str]:
    if not field:
        return None
    key = str(field).strip().lower().replace("-", "_").replace(" ", "_")
    return _FIELD_ALIASES.get(key, key)


def _build_filter(
    project_id: int,
    sheet_number: Optional[str] = None,
    drawing_name: Optional[str] = None,
    discipline: Optional[str] = None,
) -> Dict[str, Any]:
    flt: Dict[str, Any] = {"projectId": project_id}
    if sheet_number:
        sn = str(sheet_number).strip()
        flt["$or"] = [{"sheet_number": sn}, {"sheetNumber": sn}]
    if drawing_name:
        flt["drawingName"] = str(drawing_name).strip()
    if discipline:
        flt["discipline"] = str(discipline).strip()
    return flt


def lookup_titleblock_v3(
    project_id: int,
    sheet_number: Optional[str] = None,
    drawing_name: Optional[str] = None,
    field: Optional[str] = None,
    discipline: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the titleBlock dict (or a single field) for one or more drawings.

    Examples:
      - "scale of A-101"                  -> lookup_titleblock_v3(p, sheet="A-101", field="scale")
      - "titleblock for sheet GS-100"     -> lookup_titleblock_v3(p, sheet="GS-100")
      - "all Plumbing sheet revisions"    -> lookup_titleblock_v3(p, discipline="Plumbing", field="revision")
    """
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)
    flt = _build_filter(project_id, sheet_number, drawing_name, discipline)

    field_key = _resolve_field(field)

    out: List[Dict[str, Any]] = []
    cursor = coll.find(flt, _TB_PROJECTION).limit(200)

    for doc in cursor:
        tb = doc.get("titleBlock") or {}
        if not isinstance(tb, dict):
            tb = {}
        row: Dict[str, Any] = {
            "drawingName": doc.get("drawingName"),
            "drawingTitle": doc.get("drawingTitle"),
            "sheet_number": doc.get("sheet_number") or doc.get("sheetNumber"),
            "discipline": doc.get("discipline"),
            "drawingType": doc.get("drawingType"),
            "pdfName": doc.get("pdfName"),
            "s3BucketPath": doc.get("s3BucketPath"),
            "page": doc.get("page"),
            "drawingId": doc.get("drawingId"),
        }
        if field_key:
            row["field"] = field
            row["value"] = tb.get(field_key)
        else:
            row["titleBlock"] = tb
        out.append(row)

    logger.info(
        "lookup_titleblock_v3: project=%d filter=%r field=%r hits=%d",
        project_id, {k: v for k, v in flt.items() if k != "projectId"},
        field, len(out),
    )
    return out


def get_project_titleblock_info_v3(project_id: int) -> Dict[str, Any]:
    """Return aggregated project-level info derived from titleBlock fields:
    architect(s), project_name(s), scale distribution, revision date range.

    Useful for ""who is the architect?"", ""what's the project name?"" without
    needing to load every drawing.
    """
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)

    architects: Dict[str, int] = {}
    project_names: Dict[str, int] = {}
    scales: Dict[str, int] = {}
    revisions: List[str] = []
    dates: List[str] = []
    n_total = 0
    n_with_titleblock = 0

    for doc in coll.find(
        {"projectId": project_id},
        {"_id": 0, "titleBlock": 1},
    ).limit(5000):
        n_total += 1
        tb = doc.get("titleBlock") or {}
        if not isinstance(tb, dict) or not tb:
            continue
        n_with_titleblock += 1
        for fld, bucket in (
            ("architect", architects), ("project_name", project_names),
            ("scale", scales),
        ):
            v = tb.get(fld)
            if isinstance(v, str) and v.strip():
                bucket[v.strip()] = bucket.get(v.strip(), 0) + 1
        rv = tb.get("revision")
        if isinstance(rv, str) and rv.strip():
            revisions.append(rv.strip())
        dt = tb.get("date")
        if isinstance(dt, str) and dt.strip():
            dates.append(dt.strip())

    def _top(d: Dict[str, int], n: int = 5) -> List[Dict[str, Any]]:
        return [{"value": k, "count": v}
                for k, v in sorted(d.items(), key=lambda x: -x[1])[:n]]

    out = {
        "project_id": project_id,
        "drawings_total": n_total,
        "drawings_with_titleblock": n_with_titleblock,
        "architect_top": _top(architects),
        "project_name_top": _top(project_names),
        "scale_distribution_top": _top(scales, 10),
        "revision_dates_distinct": sorted(set(revisions))[:20],
        "issue_dates_distinct": sorted(set(dates))[:20],
    }
    logger.info(
        "get_project_titleblock_info_v3: project=%d total=%d w/tb=%d architects=%d",
        project_id, n_total, n_with_titleblock, len(architects),
    )
    return out
