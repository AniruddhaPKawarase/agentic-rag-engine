"""
Tier D1 — Deterministic drawing<->spec cross-reference via CSI MasterFormat.

Join key:
  drawings_v3.csiDivisions  (e.g. ["22 - Plumbing", "26 - Electrical"])
  specifications_v3.csiDivision  (e.g. "22", "26")

The 2-digit prefix in drawings_v3.csiDivisions matches specifications_v3.csiDivision.
We derive that prefix at query time so both fields stay untouched.

These tools answer cross-reference questions without LLM guesswork:
  - "Which spec sections govern sheet A-210?"            -> get_specs_for_drawing
  - "Which drawings reference Division 22 (Plumbing)?"   -> get_drawings_for_csi_division
  - "Which sheets are governed by spec 230593?"          -> get_drawings_for_spec

Audit (2026-05-27) confirmed csiDivisions populated on 100% of drawings
across all 6 projects; csiDivision populated on >97% of specs.

NO Mongo writes — read-only.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_project_id, validate_limit

logger = logging.getLogger("agentic_rag.tools.cross_ref_v3")

DRAWING_COLLECTION = "drawings_v3"
SPEC_COLLECTION = "specifications_v3"

_DRAWING_PROJ: Dict[str, Any] = {
    "_id": 0,
    "drawingName": 1, "drawingTitle": 1,
    "sheetNumber": 1, "sheet_number": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "discipline": 1, "drawingType": 1, "trade": 1,
    "drawingId": 1, "csiDivisions": 1,
}

_SPEC_PROJ: Dict[str, Any] = {
    "_id": 0,
    "specificationNumber": 1, "csiDivision": 1, "csi": 1,
    "sectionTitle": 1, "sectionName": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "pageCount": 1,
}

_CSI_PREFIX_RE = re.compile(r"^\s*(\d{1,2})\b")


def _extract_csi_prefix(s: Any) -> Optional[str]:
    """Pull the leading 2-digit CSI division code from a string like
    '22 - Plumbing' -> '22'. Returns None if no leading digits."""
    if not isinstance(s, str):
        return None
    m = _CSI_PREFIX_RE.match(s)
    if not m:
        return None
    # Normalize to 2-digit zero-padded (specifications_v3 uses '01', '22')
    return m.group(1).zfill(2)


def _normalize_csi_division_input(csi: Optional[str]) -> Optional[str]:
    """Accept '22', '22 - Plumbing', '22-Plumbing', 'Division 22', 22, etc.
    Return canonical 2-digit zero-padded prefix."""
    if csi is None:
        return None
    if isinstance(csi, int):
        return str(csi).zfill(2)
    s = str(csi).strip()
    if not s:
        return None
    # Handle "Division 22" or "Div 22"
    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        return m.group(1).zfill(2)
    return None


def get_specs_for_drawing(
    project_id: int,
    sheet_number: Optional[str] = None,
    drawing_name: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Given one drawing, return all spec sections whose csiDivision
    matches any of the drawing's csiDivisions.

    Returns {drawing: {...}, csi_divisions: [...], specs: [...]}.

    Example: "what spec sections apply to sheet A-210?" ->
    get_specs_for_drawing(p, sheet_number='A-210').
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)

    if not sheet_number and not drawing_name:
        return {"error": "must provide sheet_number or drawing_name",
                "drawing": None, "csi_divisions": [], "specs": []}

    coll_d = get_collection(DRAWING_COLLECTION)
    flt: Dict[str, Any] = {"projectId": project_id}
    if sheet_number:
        sn = str(sheet_number).strip()
        flt["$or"] = [{"sheet_number": sn}, {"sheetNumber": sn}]
    elif drawing_name:
        flt["drawingName"] = str(drawing_name).strip()

    drawing = coll_d.find_one(flt, _DRAWING_PROJ)
    if not drawing:
        logger.info("get_specs_for_drawing: project=%d filter=%r — no drawing matched",
                    project_id, flt)
        return {"drawing": None, "csi_divisions": [], "specs": []}

    csi_divisions_raw = drawing.get("csiDivisions") or []
    prefixes: List[str] = []
    for csi in csi_divisions_raw:
        p = _extract_csi_prefix(csi)
        if p:
            prefixes.append(p)
    prefixes = sorted(set(prefixes))

    if not prefixes:
        return {"drawing": drawing, "csi_divisions": csi_divisions_raw,
                "csi_prefixes": [], "specs": []}

    coll_s = get_collection(SPEC_COLLECTION)
    specs = list(
        coll_s.find(
            {"projectId": project_id, "csiDivision": {"$in": prefixes}},
            _SPEC_PROJ,
        ).limit(limit)
    )

    logger.info(
        "get_specs_for_drawing: project=%d drawing=%r prefixes=%r specs=%d",
        project_id, drawing.get("drawingName"), prefixes, len(specs),
    )
    return {
        "drawing": drawing,
        "csi_divisions": csi_divisions_raw,
        "csi_prefixes": prefixes,
        "specs": specs,
        "spec_count": len(specs),
    }


def get_drawings_for_spec(
    project_id: int,
    specification_number: Optional[str] = None,
    csi_division: Optional[str] = None,
    section_title: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Given a spec section (by number, csiDivision, or section title),
    return all drawings whose csiDivisions reference that division.

    Example: "which drawings does Section 230593 govern?" ->
    get_drawings_for_spec(p, specification_number='230593').
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)

    # Resolve to a CSI division prefix
    prefix: Optional[str] = None
    spec_doc: Optional[Dict[str, Any]] = None
    coll_s = get_collection(SPEC_COLLECTION)

    if specification_number:
        sn = str(specification_number).strip()
        spec_doc = coll_s.find_one(
            {"projectId": project_id,
             "$or": [{"specificationNumber": sn}, {"csi": sn}]},
            _SPEC_PROJ,
        )
        if spec_doc:
            prefix = _normalize_csi_division_input(spec_doc.get("csiDivision"))
            # Fall back to first 2 chars of the spec number itself
            if not prefix:
                prefix = sn[:2].zfill(2) if sn[:2].isdigit() else None
    elif csi_division:
        prefix = _normalize_csi_division_input(csi_division)
    elif section_title:
        # find spec by section title fuzzy match
        spec_doc = coll_s.find_one(
            {"projectId": project_id,
             "sectionTitle": {"$regex": re.escape(str(section_title).strip()),
                               "$options": "i"}},
            _SPEC_PROJ,
        )
        if spec_doc:
            prefix = _normalize_csi_division_input(spec_doc.get("csiDivision"))

    if not prefix:
        return {"error": "could not resolve CSI division from inputs",
                "spec": spec_doc, "drawings": []}

    coll_d = get_collection(DRAWING_COLLECTION)
    # csiDivisions on drawings_v3 are strings like "22 - Plumbing",
    # so match prefix at the start.
    drawings = list(
        coll_d.find(
            {"projectId": project_id,
             "csiDivisions": {"$regex": rf"^{prefix}\b"}},
            _DRAWING_PROJ,
        ).limit(limit)
    )
    logger.info(
        "get_drawings_for_spec: project=%d prefix=%r drawings=%d",
        project_id, prefix, len(drawings),
    )
    return {
        "spec": spec_doc,
        "csi_prefix": prefix,
        "drawings": drawings,
        "drawing_count": len(drawings),
    }


def get_drawings_for_csi_division(
    project_id: int,
    csi_division: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """All drawings tagged with a given CSI division.
    Accepts '22', 'Division 22', '22 - Plumbing', etc.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)
    prefix = _normalize_csi_division_input(csi_division)
    if not prefix:
        return []
    coll_d = get_collection(DRAWING_COLLECTION)
    drawings = list(
        coll_d.find(
            {"projectId": project_id,
             "csiDivisions": {"$regex": rf"^{prefix}\b"}},
            _DRAWING_PROJ,
        ).limit(limit)
    )
    logger.info(
        "get_drawings_for_csi_division: project=%d prefix=%r drawings=%d",
        project_id, prefix, len(drawings),
    )
    return drawings


def get_specs_for_csi_division(
    project_id: int,
    csi_division: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """All spec sections under a given CSI division.
    Accepts '22', 'Division 22', etc.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)
    prefix = _normalize_csi_division_input(csi_division)
    if not prefix:
        return []
    coll_s = get_collection(SPEC_COLLECTION)
    specs = list(
        coll_s.find(
            {"projectId": project_id, "csiDivision": prefix},
            _SPEC_PROJ,
        ).limit(limit)
    )
    logger.info(
        "get_specs_for_csi_division: project=%d prefix=%r specs=%d",
        project_id, prefix, len(specs),
    )
    return specs


def list_csi_divisions_v3(project_id: int) -> Dict[str, Any]:
    """Discovery helper — what CSI divisions exist on this project,
    with counts of how many drawings AND specs reference each.

    Returns {divisions: [{prefix, label, n_drawings, n_specs}]}.
    """
    project_id = validate_project_id(project_id)
    coll_d = get_collection(DRAWING_COLLECTION)
    coll_s = get_collection(SPEC_COLLECTION)

    # Aggregate drawing-side csiDivisions
    pipe_d = [
        {"$match": {"projectId": project_id, "csiDivisions": {"$exists": True}}},
        {"$unwind": "$csiDivisions"},
        {"$group": {"_id": "$csiDivisions", "n": {"$sum": 1}}},
    ]
    d_counts: Dict[str, Dict[str, Any]] = {}
    for r in coll_d.aggregate(pipe_d):
        label = r["_id"]
        prefix = _extract_csi_prefix(label) or "??"
        existing = d_counts.setdefault(prefix, {"prefix": prefix, "labels": set(),
                                                "n_drawings": 0, "n_specs": 0})
        existing["labels"].add(label)
        existing["n_drawings"] += r["n"]

    # Aggregate spec-side csiDivision
    pipe_s = [
        {"$match": {"projectId": project_id}},
        {"$group": {"_id": "$csiDivision", "n": {"$sum": 1}}},
    ]
    for r in coll_s.aggregate(pipe_s):
        prefix = _normalize_csi_division_input(r["_id"]) or "??"
        existing = d_counts.setdefault(prefix, {"prefix": prefix, "labels": set(),
                                                "n_drawings": 0, "n_specs": 0})
        existing["n_specs"] += r["n"]

    out_list = []
    for prefix in sorted(d_counts):
        v = d_counts[prefix]
        out_list.append({
            "prefix": prefix,
            "labels": sorted(v["labels"]),
            "n_drawings": v["n_drawings"],
            "n_specs": v["n_specs"],
        })
    logger.info("list_csi_divisions_v3: project=%d divisions=%d",
                project_id, len(out_list))
    return {"project_id": project_id, "divisions": out_list}
