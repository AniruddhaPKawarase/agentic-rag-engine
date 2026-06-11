"""
Tier E (NEW 2026-05-28) — Tools that query the v3 extraction fields:
  - weak_symbol_labels[]   (equipment tags)
  - cfm_callouts[]
  - duct_sizes[]
  - unit_tags_mined[] + unit_types_mined[] + sf_callouts[]
  - cross_sheet_refs[]
  - vlm_element_labels[]   (CKG backfill)
  - keynote_index[]

All tools are read-only. No Mongo writes.
Used by the agent for symbol-count, enumeration, and spatial-pair questions.
"""
from __future__ import annotations
import logging
import math
import re
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id

logger = logging.getLogger("agentic_rag.tools.extracted_data_v3")
COLLECTION = "drawings_v3"


def _build_filter(project_id: int,
                  drawing_name: Optional[str] = None,
                  sheet_pattern: Optional[str] = None) -> Dict[str, Any]:
    f: Dict[str, Any] = {"projectId": project_id}
    if drawing_name:
        f["drawingName"] = drawing_name
    elif sheet_pattern:
        f["drawingName"] = {"$regex": sheet_pattern}
    return f


# ============================================================
# enumerate_equipment_tags
# ============================================================
def enumerate_equipment_tags(
    project_id: int,
    drawing_name: Optional[str] = None,
    kind: Optional[str] = None,
    tag_pattern: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    List every equipment tag matching the criteria. Returns FULL list with bbox.
    Use for: "What FCUs serve common areas?", "List all WC tags", "How many FSDs?"

    Args:
      project_id    : required
      drawing_name  : exact match (e.g. "M-232")  -- optional
      kind          : FCU, FSD, WC, SD, RG, AC, etc.  -- optional
      tag_pattern   : regex applied to tag_text (e.g. "^FCU-XC-" for common-area FCUs)
      limit         : max tags returned
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=2000)
    f = _build_filter(project_id, drawing_name)

    coll = get_collection(COLLECTION)
    tag_re = re.compile(tag_pattern) if tag_pattern else None
    results: List[Dict[str, Any]] = []
    drawings_with_matches = set()

    cursor = coll.find(f, {
        "_id": 0, "drawingName": 1, "drawingTitle": 1, "page": 1,
        "weak_symbol_labels": 1,
    })
    for d in cursor:
        for wl in (d.get("weak_symbol_labels") or []):
            if kind and wl.get("kind") != kind:
                continue
            if tag_re and not tag_re.search(wl.get("tag_text") or ""):
                continue
            results.append({
                "drawing": d.get("drawingName"),
                "drawing_title": d.get("drawingTitle"),
                "page": wl.get("page", 1),
                "tag": wl.get("tag_text"),
                "kind": wl.get("kind"),
                "trade": wl.get("trade"),
                "bbox_pt": wl.get("bbox_pt"),
                "bbox_px": wl.get("bbox_px"),
            })
            drawings_with_matches.add(d.get("drawingName"))
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    by_kind: Dict[str, int] = {}
    for r in results:
        k = r["kind"]
        by_kind[k] = by_kind.get(k, 0) + 1

    return {
        "count": len(results),
        "by_kind": by_kind,
        "drawings_covered": sorted(drawings_with_matches),
        "tags": results,
    }


# ============================================================
# list_cfm_callouts
# ============================================================
def list_cfm_callouts(
    project_id: int,
    drawing_name: Optional[str] = None,
    modifier: Optional[str] = None,  # OA, RA, SA, EA
    value: Optional[int] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """List every CFM callout with bbox. Use for OA/duct CFM questions."""
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=2000)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)

    results = []
    by_value: Dict[int, int] = {}
    by_mod: Dict[str, int] = {}
    cursor = coll.find(f, {"_id": 0, "drawingName": 1, "drawingTitle": 1, "cfm_callouts": 1})
    for d in cursor:
        for c in (d.get("cfm_callouts") or []):
            if modifier and c.get("modifier") != modifier:
                continue
            if value is not None and c.get("value") != value:
                continue
            results.append({
                "drawing": d.get("drawingName"),
                "value": c.get("value"),
                "modifier": c.get("modifier"),
                "raw": c.get("raw"),
                "bbox_pt": c.get("bbox_pt"),
                "bbox_px": c.get("bbox_px"),
                "page": c.get("page"),
            })
            by_value[c.get("value")] = by_value.get(c.get("value"), 0) + 1
            m = c.get("modifier") or "none"
            by_mod[m] = by_mod.get(m, 0) + 1
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return {
        "count": len(results),
        "by_value": dict(sorted(by_value.items())),
        "by_modifier": by_mod,
        "callouts": results,
    }


# ============================================================
# list_duct_sizes
# ============================================================
def list_duct_sizes(
    project_id: int,
    drawing_name: Optional[str] = None,
    modifier: Optional[str] = None,
    limit: int = 300,
) -> Dict[str, Any]:
    """List every duct-size callout. Use for trunk-duct sequence questions."""
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=2000)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)

    results = []
    distinct_sizes = set()
    cursor = coll.find(f, {"_id": 0, "drawingName": 1, "duct_sizes": 1})
    for d in cursor:
        for ds in (d.get("duct_sizes") or []):
            if modifier and ds.get("modifier") != modifier:
                continue
            size_key = f"{ds.get('width')}x{ds.get('height')}"
            distinct_sizes.add(size_key)
            results.append({
                "drawing": d.get("drawingName"),
                "width": ds.get("width"),
                "height": ds.get("height"),
                "modifier": ds.get("modifier"),
                "raw": ds.get("raw"),
                "bbox_pt": ds.get("bbox_pt"),
                "page": ds.get("page"),
            })
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return {
        "count": len(results),
        "distinct_sizes": sorted(distinct_sizes),
        "items": results,
    }


# ============================================================
# get_keynotes
# ============================================================
def get_keynotes(
    project_id: int,
    drawing_name: str,
    kind: Optional[str] = None,  # "key" or "general"
    keynote_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Enumerate keynotes / sheet-notes on a drawing. Returns verbatim text + bbox."""
    project_id = validate_project_id(project_id)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)

    out = []
    cursor = coll.find(f, {"_id": 0, "drawingName": 1, "keynote_index": 1})
    for d in cursor:
        for k in (d.get("keynote_index") or []):
            if kind and k.get("kind") != kind:
                continue
            if keynote_id and str(k.get("id")) != str(keynote_id):
                continue
            out.append({
                "drawing": d.get("drawingName"),
                "id": k.get("id"),
                "kind": k.get("kind"),
                "text": k.get("text"),
                "bbox_pt": k.get("bbox_pt"),
                "bbox_px": k.get("bbox_px"),
                "ckgNodeId": k.get("ckgNodeId"),
            })

    return {
        "drawing": drawing_name,
        "count": len(out),
        "keynotes": out,
    }


# ============================================================
# pair_textblocks_by_proximity
# ============================================================
def _bbox_center(b):
    if not b or len(b) != 4:
        return None
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def pair_textblocks_by_proximity(
    project_id: int,
    drawing_name: str,
    anchor_filter: Dict[str, Any],  # e.g. {"field": "cfm_callouts", "value": 85}
    target_field: str,              # e.g. "unit_types_mined" or "sf_callouts" or "unit_tags_mined"
    radius_pt: float = 150.0,
    limit_pairs: int = 30,
) -> Dict[str, Any]:
    """
    For each anchor (e.g. each CFM callout matching a filter), find the K nearest
    target items (e.g. nearest unit-type code) within radius_pt.

    Anchor filter shape:
      {"field": "cfm_callouts", "value": 85}                  -> CFM callouts with value=85
      {"field": "weak_symbol_labels", "kind": "FCU"}          -> all FCU tags
      {"field": "duct_sizes", "modifier": "OA"}               -> OA duct sizes
    """
    project_id = validate_project_id(project_id)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)
    radius_pt = float(radius_pt)

    d = coll.find_one(f, {"_id": 0,
                          anchor_filter.get("field"): 1,
                          target_field: 1, "drawingName": 1})
    if not d:
        return {"error": "drawing not found", "drawing": drawing_name}

    anchors_raw = d.get(anchor_filter.get("field")) or []
    targets = d.get(target_field) or []

    # Filter anchors
    anchors = []
    for a in anchors_raw:
        ok = True
        for k, v in anchor_filter.items():
            if k == "field":
                continue
            if a.get(k) != v:
                ok = False; break
        if ok:
            anchors.append(a)

    pairs = []
    for a in anchors:
        ac = _bbox_center(a.get("bbox_pt"))
        if not ac:
            continue
        scored = []
        for t in targets:
            tc = _bbox_center(t.get("bbox_pt"))
            if not tc:
                continue
            dist = math.hypot(ac[0] - tc[0], ac[1] - tc[1])
            if dist <= radius_pt:
                scored.append((dist, t))
        scored.sort(key=lambda x: x[0])
        nearest = [{"distance_pt": round(s[0], 1), **{k: v for k, v in s[1].items() if k != "bbox_pt"}} for s in scored[:5]]
        pairs.append({
            "anchor": {k: v for k, v in a.items() if k != "bbox_pt"},
            "anchor_bbox_pt": a.get("bbox_pt"),
            "nearest": nearest,
        })
        if len(pairs) >= limit_pairs:
            break

    return {
        "drawing": drawing_name,
        "anchor_field": anchor_filter.get("field"),
        "anchor_count": len(anchors),
        "target_field": target_field,
        "radius_pt": radius_pt,
        "pairs": pairs,
    }


# ============================================================
# list_unit_inventory
# ============================================================
def list_unit_inventory(
    project_id: int,
    drawing_name: str,
) -> Dict[str, Any]:
    """
    Spatially join unit_tags_mined (U-###) + unit_types_mined (C2.1) + sf_callouts (1086 SF)
    on the same drawing. Each U-### bbox center is paired with the nearest unit-type
    and the nearest SF callout.
    Use for: "What residential units are shown and what are their sizes?"
    """
    project_id = validate_project_id(project_id)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)

    d = coll.find_one(f, {"_id": 0, "drawingName": 1,
                          "unit_tags_mined": 1, "unit_types_mined": 1, "sf_callouts": 1})
    if not d:
        return {"error": "drawing not found", "drawing": drawing_name}

    u_tags = d.get("unit_tags_mined") or []
    u_types = d.get("unit_types_mined") or []
    sf = d.get("sf_callouts") or []

    def nearest(anchor_b, candidates, max_dist=300.0):
        ac = _bbox_center(anchor_b)
        if not ac:
            return None, None
        best = None
        best_d = 1e9
        for c in candidates:
            cc = _bbox_center(c.get("bbox_pt"))
            if not cc:
                continue
            dist = math.hypot(ac[0] - cc[0], ac[1] - cc[1])
            if dist < best_d and dist <= max_dist:
                best_d = dist; best = c
        return best, (best_d if best else None)

    units = []
    for ut in u_tags:
        ub = ut.get("bbox_pt")
        ttype, tdist = nearest(ub, u_types, max_dist=250.0)
        sfcal, sdist = nearest(ub, sf, max_dist=400.0)
        units.append({
            "unit_id": ut.get("unit_id"),
            "unit_type": ttype.get("unit_type") if ttype else None,
            "unit_type_dist_pt": round(tdist, 1) if tdist else None,
            "area_sf": sfcal.get("area_sf") if sfcal else None,
            "area_sf_dist_pt": round(sdist, 1) if sdist else None,
            "page": ut.get("page"),
            "bbox_pt": ub,
        })

    return {
        "drawing": drawing_name,
        "units_count": len(units),
        "units": units,
    }


# ============================================================
# get_vlm_element_labels (CKG backfill query)
# ============================================================
def get_vlm_element_labels(
    project_id: int,
    drawing_name: str,
    element_type: Optional[str] = None,
    trade: Optional[str] = None,
    label_pattern: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    Returns the VLM-classified elements recovered from PostgreSQL CKG.
    Use for: deep equipment listing, schedule entries, structural members, etc.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=1000)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)
    lp = re.compile(label_pattern) if label_pattern else None

    d = coll.find_one(f, {"_id": 0, "drawingName": 1, "vlm_element_labels": 1})
    if not d:
        return {"error": "drawing not found"}

    labels = d.get("vlm_element_labels") or []
    results = []
    by_type: Dict[str, int] = {}
    for el in labels:
        if element_type and el.get("elementType") != element_type:
            continue
        if trade and el.get("trade") != trade:
            continue
        if lp and not lp.search(el.get("label") or ""):
            continue
        results.append(el)
        et = el.get("elementType") or "unknown"
        by_type[et] = by_type.get(et, 0) + 1
        if len(results) >= limit:
            break

    return {
        "drawing": drawing_name,
        "count": len(results),
        "by_elementType": by_type,
        "labels": results,
    }


# ============================================================
# get_cross_sheet_refs
# ============================================================
def get_cross_sheet_refs(
    project_id: int,
    drawing_name: str,
) -> Dict[str, Any]:
    """Returns all 'Detail X/Y-###' and 'Refer to M-###' references on a drawing."""
    project_id = validate_project_id(project_id)
    f = _build_filter(project_id, drawing_name)
    coll = get_collection(COLLECTION)

    d = coll.find_one(f, {"_id": 0, "drawingName": 1, "cross_sheet_refs": 1})
    if not d:
        return {"error": "drawing not found"}

    refs = d.get("cross_sheet_refs") or []
    # Group by target_sheet
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for r in refs:
        t = r.get("target_sheet") or "unknown"
        by_target.setdefault(t, []).append(r)

    return {
        "drawing": drawing_name,
        "total_refs": len(refs),
        "distinct_targets": sorted(by_target.keys()),
        "by_target": by_target,
    }
