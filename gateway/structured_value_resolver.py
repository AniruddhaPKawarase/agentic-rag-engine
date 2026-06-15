"""[SVR-P2] Structured Value Resolver — multi-attribute deterministic answers.

Generalizes UAR-V1 (area) to more attributes, all from Tier-1 structured arrays
in drawings_v3, dispatched by the P1 slot frame:

    unit_type   : unit_tags_mined  <-> unit_types_mined   (bbox-nearest, vote)
    cfm         : cfm_callouts      enumerate on scoped mechanical sheets
    duct_size   : duct_sizes        enumerate on scoped mechanical sheets
    count       : symbols[kind]     count per canonical sheet, sum across level

`area` stays with UAR-V1 (untouched, already on prod) — this resolver runs AFTER
UAR and handles only the new attributes. Anything thin/ambiguous returns None →
caller falls through to the normal agent (zero regression).

Flag: STRUCTURED_VALUE_RESOLVER_ENABLED (default true). Reuses UAR helpers.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_rag.structured_value_resolver")
RESOLVER_VERSION = "svr-p2"
_NEAR_PT = 60.0

# count-target noun -> symbols[].kind
_COUNT_TARGETS = {
    "diffuser": "diffuser", "diffusers": "diffuser",
    "fixture": "fixture", "fixtures": "fixture",
    "door": "door", "doors": "door",
    "window": "window", "windows": "window",
    "valve": "valve", "valves": "valve",
    "damper": "damper", "dampers": "damper",
    "equipment": "equipment",
    "outlet": "fixture", "outlets": "fixture", "receptacle": "fixture",
    "room": "room_label", "rooms": "room_label",
    "unit": "room_label", "units": "room_label",
    "detail": "detail_callout", "details": "detail_callout",
    "keynote": "keynote", "keynotes": "keynote",
    "dimension": "dimension", "dimensions": "dimension",
    "column": "structural_member", "columns": "structural_member",
}


def _enabled() -> bool:
    return os.getenv("STRUCTURED_VALUE_RESOLVER_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on")


def _uar():
    """Reuse UAR helpers (center/dist/sign/collection) — single source of truth."""
    from gateway import unit_area_resolver as u
    return u


def _level_filter(scope: Dict[str, Any]) -> Optional[int]:
    if not isinstance(scope, dict):
        return None
    return scope.get("level")


# [SVR-FIX1] query-time floor derivation from drawingTitle (the stored `level`
# field is unreliable — _parse_level grabbed the sheet-number series digit, so
# all A-7xx = "7"). We derive floor from the TITLE across variants instead, with
# NO DB write (8013-only). Mechanical/overlay sheets whose titles omit the floor
# remain underivable (genuine data gap → those abstain safely).
_ORD_FLOOR = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
              "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
              "ground": 1, "cellar": 0, "basement": 0}


def _derive_floor(title: Optional[str]) -> Optional[int]:
    """Parse a floor int from a drawing TITLE. Returns None if no floor stated."""
    if not title:
        return None
    t = str(title).upper()
    m = re.search(r"\bLEVEL\s*0?(\d{1,2})\b", t) or re.search(r"\bL-?(\d{1,2})\b(?!\d)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{1,2})(?:ST|ND|RD|TH)\s+FLOOR\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|GROUND|CELLAR|BASEMENT)\b\s*FLOOR?", t)
    if m:
        return _ORD_FLOOR.get(m.group(1).lower())
    if re.search(r"\bCELLAR\b|\bBASEMENT\b", t):
        return 0
    return None


def _sheet_floor_map(docs: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """sheetNumber -> derived floor, taking the first floor any variant's title yields."""
    out: Dict[str, Optional[int]] = {}
    for d in docs:
        sn = (d.get("sheetNumber") or d.get("drawingName") or "").strip().upper()
        if not sn:
            continue
        if out.get(sn) is None:
            out[sn] = _derive_floor(d.get("drawingTitle"))
    return out


def _sheet_matches_level(doc: Dict[str, Any], level: Optional[int]) -> bool:
    """Title-derived floor match (ignores the unreliable stored `level` field).
    True when level is None; when level set, matches only if the title-derived
    floor equals it. Unknown-floor sheets do NOT match a specific level (safe —
    never returns wrong-level data)."""
    if level is None:
        return True
    return _derive_floor(doc.get("drawingTitle")) == level


def _build_source_doc(u, d: Dict[str, Any]) -> Dict[str, Any]:
    s3, pdf = d.get("s3BucketPath"), d.get("pdfName")
    return {
        "s3_path": s3 or "", "pdf_name": pdf or "", "file_name": pdf or "",
        "drawing_name": d.get("sheetNumber") or d.get("drawingName") or "",
        "drawing_title": d.get("drawingTitle") or "", "display_title": d.get("drawingTitle") or "",
        "sheet_number": d.get("sheetNumber") or "", "page": d.get("page") or 1,
        "drawing_id": d.get("drawingId"), "trade": d.get("trade") or "",
        "source_document_type": "drawing", "download_url": u._sign_url(s3, pdf),
    }


# ── unit_type: unit tag -> nearest unit-type label (UAR pattern) ────────────
def _resolve_unit_type(frame, project_id: int) -> Optional[Dict[str, Any]]:
    ent = frame.entity or {}
    if ent.get("type") != "unit" or not ent.get("value"):
        return None
    unit_id = ent["value"]
    u = _uar()
    coll = u._get_collection("drawings_v3")
    proj = {"_id": 0, "drawingId": 1, "sheetNumber": 1, "drawingName": 1,
            "drawingTitle": 1, "drawingType": 1, "discipline": 1, "page": 1,
            "pdfName": 1, "s3BucketPath": 1, "trade": 1,
            "unit_tags_mined": 1, "unit_types_mined": 1}
    q = {"projectId": int(project_id),
         "unit_tags_mined.unit_id": unit_id,
         "unit_types_mined.0": {"$exists": True}}
    try:
        docs = list(coll.find(q, proj).limit(60).max_time_ms(9000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p2] unit_type query failed: %s", exc); return None
    if not docs:
        return None
    near_types: List[str] = []
    docs_for_type: Dict[str, List[Dict]] = defaultdict(list)
    for d in docs:
        tags = [t for t in (d.get("unit_tags_mined") or [])
                if str(t.get("unit_id", "")).upper() == unit_id.upper()]
        types = d.get("unit_types_mined") or []
        for t in tags:
            tc = u._center(t.get("bbox_pt"))
            if not tc:
                continue
            best, bestd = None, 1e9
            for ut in types:
                uc = u._center(ut.get("bbox_pt"))
                if not uc:
                    continue
                dd = u._dist(tc, uc)
                if dd < bestd:
                    bestd, best = dd, ut.get("unit_type")
            if best and bestd <= _NEAR_PT:
                near_types.append(best)
                docs_for_type[best].append(d)
    if not near_types:
        return None
    val, votes = Counter(near_types).most_common(1)[0]
    conf = "high" if (votes >= 2 and votes / len(near_types) >= 0.6) else "medium"
    # prefer the cleanest published sheet to cite (reuse UAR scoring; reject overlays)
    cand = docs_for_type.get(val, [])
    src = max(cand, key=u._citation_score) if cand else {}
    return {"attribute": "unit_type", "value": val, "unit": "", "entity": unit_id,
            "confidence": conf, "votes": votes, "n": len(near_types),
            "source_doc": _build_source_doc(u, src),
            "answer": f"Unit {unit_id} is unit type {val}.", "resolver": RESOLVER_VERSION}


# ── count: count symbols of a kind on canonical sheets for the scope ────────
def _resolve_count(frame, project_id: int, query: str) -> Optional[Dict[str, Any]]:
    kind = None
    for tok, k in _COUNT_TARGETS.items():
        if re.search(rf"\b{re.escape(tok)}\b", query, re.I):
            kind = k; target = tok; break
    if not kind:
        return None
    level = _level_filter(frame.scope)
    # [SVR-QA1] count UNITS via distinct unit tags (NOT room_label symbols, which
    # include non-unit rooms and double-count across RCP+floor-plan sheets).
    if target in ("unit", "units", "apartment", "apartments", "apt"):
        ids = _all_unit_ids(project_id, level)
        if not ids:
            return None
        where = f" on level {level}" if level is not None else ""
        u = _uar()
        return {"attribute": "count", "value": len(ids), "unit": "units", "entity": "unit",
                "confidence": "high", "n_sheets": None,
                "source_doc": {},
                "answer": f"There are {len(ids)} units{where} (distinct unit tags in the drawings).",
                "resolver": RESOLVER_VERSION}
    u = _uar()
    coll = u._get_collection("drawings_v3")
    proj = {"_id": 0, "sheetNumber": 1, "drawingName": 1, "drawingTitle": 1,
            "drawingType": 1, "level": 1, "pdfName": 1, "s3BucketPath": 1,
            "drawingId": 1, "symbols": 1}
    q = {"projectId": int(project_id), "symbols.kind": kind}
    try:
        docs = list(coll.find(q, proj).limit(120).max_time_ms(9000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p2] count query failed: %s", exc); return None
    if not docs:
        return None
    # per sheetNumber take the MAX kind-count among its variants (avoid double count),
    # then sum across distinct sheetNumbers that match the level scope.
    per_sheet: Dict[str, int] = {}
    per_sheet_doc: Dict[str, Dict] = {}
    matched = 0
    for d in docs:
        if not _sheet_matches_level(d, level):
            continue
        matched += 1
        sn = (d.get("sheetNumber") or d.get("drawingName") or "").strip().upper()
        if not sn:
            continue
        c = sum(1 for s in (d.get("symbols") or []) if isinstance(s, dict) and s.get("kind") == kind)
        if c > per_sheet.get(sn, -1):
            per_sheet[sn] = c
            per_sheet_doc[sn] = d
    if not per_sheet:
        return None
    total = sum(per_sheet.values())
    if total == 0:
        return None
    where = f" on level {level}" if level is not None else ""
    sheets = sorted(per_sheet, key=lambda s: -per_sheet[s])[:5]
    top_doc = per_sheet_doc.get(sheets[0], {}) if sheets else {}
    return {"attribute": "count", "value": total, "unit": f"{target}(s)", "entity": target,
            "confidence": "medium", "n_sheets": len(per_sheet),
            "per_sheet": {s: per_sheet[s] for s in sheets},
            "source_doc": _build_source_doc(u, top_doc),
            "answer": (f"There are approximately {total} {target}(s){where} "
                       f"across {len(per_sheet)} sheet(s) (e.g. "
                       + ", ".join(f"{s}: {per_sheet[s]}" for s in sheets) + ")."),
            "resolver": RESOLVER_VERSION}


# ── cfm / duct_size: enumerate distinct values on scoped sheets ─────────────
def _resolve_enumerate_callouts(frame, project_id: int, attribute: str,
                                field_name: str, value_key: str, label: str) -> Optional[Dict[str, Any]]:
    level = _level_filter(frame.scope)
    u = _uar()
    coll = u._get_collection("drawings_v3")
    proj = {"_id": 0, "sheetNumber": 1, "drawingName": 1, "drawingTitle": 1,
            "level": 1, "pdfName": 1, "s3BucketPath": 1, "drawingId": 1,
            "trade": 1, field_name: 1}
    q = {"projectId": int(project_id), f"{field_name}.0": {"$exists": True}}
    try:
        docs = list(coll.find(q, proj).limit(80).max_time_ms(9000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p2] %s query failed: %s", attribute, exc); return None
    if not docs:
        return None
    values: List[str] = []
    src = {}
    for d in docs:
        if not _sheet_matches_level(d, level):
            continue
        for c in (d.get(field_name) or []):
            raw = c.get("raw") or (str(c.get(value_key)) if c.get(value_key) is not None else None)
            if raw:
                values.append(str(raw))
                src = src or d
    if not values:
        return None
    uniq = sorted(set(values))
    where = f" on level {level}" if level is not None else ""
    shown = uniq[:25]
    return {"attribute": attribute, "value": shown, "unit": label, "entity": None,
            "confidence": "medium", "n": len(values), "n_unique": len(uniq),
            "source_doc": _build_source_doc(u, src),
            "answer": (f"{label} values found{where}: " + ", ".join(shown)
                       + (f" (+{len(uniq)-len(shown)} more)" if len(uniq) > len(shown) else "") + "."),
            "resolver": RESOLVER_VERSION}


# query keyword -> extracted_schedule_rows attribute key (case-insensitive contains)
_SCHED_ATTR_KEYS = {
    "cfm": ["cfm"], "capacity": ["capacity", "tonnage", "tons", "mbh", "btu"],
    "model": ["model", "model number", "model no"], "size": ["size", "dimension"],
    "outside air": ["oa"], "return air": ["ra"], "voltage": ["volts", "voltage", "volt"],
    "phase": ["phase"], "weight": ["weight"], "efficiency": ["efficiency", "seer", "eer"],
    "refrigerant": ["refrigerant"],
}


def _norm_tag(s: str) -> str:
    return re.sub(r"[\s.]+", "", str(s or "")).upper()


def _resolve_schedule_value(frame, project_id: int, query: str) -> Optional[Dict[str, Any]]:
    """Equipment schedule lookup via extracted_schedule_rows: tag -> attributes.
    Handles schedule_value AND cfm/duct_size when a specific equipment is named."""
    ent = frame.entity or {}
    if ent.get("type") != "equipment" or not ent.get("value"):
        return None
    tag = ent["value"]
    ntag = _norm_tag(tag)
    u = _uar()
    coll = u._get_collection("drawings_v3")
    proj = {"_id": 0, "drawingId": 1, "sheetNumber": 1, "drawingName": 1,
            "drawingTitle": 1, "pdfName": 1, "s3BucketPath": 1, "trade": 1,
            "extracted_schedule_rows": 1}
    q = {"projectId": int(project_id), "extracted_schedule_rows.0": {"$exists": True}}
    try:
        docs = list(coll.find(q, proj).limit(120).max_time_ms(9000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p2] schedule query failed: %s", exc); return None
    # find rows whose tag matches the entity (exact-ish), vote attributes across docs
    matches: List[tuple] = []   # (attributes_dict, doc)
    for d in docs:
        for row in (d.get("extracted_schedule_rows") or []):
            rtag = _norm_tag(row.get("tag"))
            if not rtag:
                continue
            # [SVR-QA1] prefix match only (dropped the loose `ntag in rtag`
            # mid-string rule that could mis-match e.g. CU-1 inside ACCU-15).
            if rtag == ntag or rtag.startswith(ntag) or ntag.startswith(rtag):
                attrs = row.get("attributes") or {}
                if attrs:
                    matches.append((attrs, d, row.get("tag")))
    if not matches:
        return None
    # merge attributes (first non-empty per key wins), pick a clean-sheet source
    merged: Dict[str, str] = {}
    for attrs, _d, _t in matches:
        for k, v in attrs.items():
            if k not in merged and v not in (None, "", "N/A"):
                merged[k] = v
    src = max((d for _a, d, _t in matches), key=u._citation_score)
    matched_tag = matches[0][2]
    # which attribute did the user ask for?
    want = None
    for canon, kws in _SCHED_ATTR_KEYS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", query, re.I) for k in kws):
            want = canon; break
    # find the merged key matching `want`
    val = None; shown_key = None
    if want:
        for k in merged:
            if any(kw.replace(" ", "") in _norm_tag(k).lower().replace(" ", "")
                   for kw in _SCHED_ATTR_KEYS[want]):
                val = merged[k]; shown_key = k; break
    if val is not None:
        answer = f"{matched_tag}: {shown_key} = {val}."
        value_out = val
    else:
        if not merged:
            return None
        pretty = ", ".join(f"{k}: {v}" for k, v in list(merged.items())[:10])
        answer = f"{matched_tag} schedule values — {pretty}."
        value_out = merged
    return {"attribute": "schedule_value", "value": value_out, "unit": "", "entity": matched_tag,
            "confidence": "high" if len(matches) >= 2 else "medium",
            "source_doc": _build_source_doc(u, src), "answer": answer,
            "resolver": RESOLVER_VERSION}


# ── [SVR-P7] aggregate / compare reduce over all unit areas ─────────────────
def _all_unit_areas(project_id: int) -> Dict[str, int]:
    """Return {unit_id: area_sf} for every unit in the project, by pairing each
    unit tag to its nearest sf_callout and voting across variant sheets. Reuses
    the reliable unit_tag<->sf_callout pairing (the same that powers UAR)."""
    u = _uar()
    coll = u._get_collection("drawings_v3")
    q = {"projectId": int(project_id),
         "unit_tags_mined.0": {"$exists": True}, "sf_callouts.0": {"$exists": True}}
    proj = {"_id": 0, "unit_tags_mined": 1, "sf_callouts": 1}
    try:
        # [SVR-QA1] scan ALL unit-bearing sheets (was limit 80 → truncated the
        # building to a 64-unit sample and falsely reported it as "all units").
        docs = list(coll.find(q, proj).limit(800).max_time_ms(15000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p7] all-unit-areas query failed: %s", exc)
        return {}
    per_unit: Dict[str, List[int]] = defaultdict(list)
    for d in docs:
        tags = d.get("unit_tags_mined") or []
        sfs = d.get("sf_callouts") or []
        for t in tags:
            uid = str(t.get("unit_id") or "").upper()
            tc = u._center(t.get("bbox_pt"))
            if not uid or not tc:
                continue
            best, bd = None, 1e9
            for s in sfs:
                sc = u._center(s.get("bbox_pt"))
                if not sc:
                    continue
                dd = u._dist(tc, sc)
                if dd < bd:
                    bd, best = dd, s.get("area_sf")
            if best is not None and bd <= _NEAR_PT:
                per_unit[uid].append(best)
    out: Dict[str, int] = {}
    for uid, areas in per_unit.items():
        out[uid] = Counter(areas).most_common(1)[0][0]
    return out


def _all_unit_ids(project_id: int, level: Optional[int] = None) -> set:
    """Distinct unit ids in the project (optionally on a given floor), regardless
    of whether an area pairs — the honest denominator for counts/coverage."""
    u = _uar()
    coll = u._get_collection("drawings_v3")
    proj = {"_id": 0, "unit_tags_mined": 1, "drawingTitle": 1}
    try:
        docs = list(coll.find({"projectId": int(project_id), "unit_tags_mined.0": {"$exists": True}},
                              proj).limit(800).max_time_ms(15000))
    except Exception:  # noqa: BLE001
        return set()
    ids = set()
    for d in docs:
        if level is not None and _derive_floor(d.get("drawingTitle")) != level:
            continue
        for t in (d.get("unit_tags_mined") or []):
            uid = str(t.get("unit_id") or "").upper()
            if uid:
                ids.add(uid)
    return ids


def _resolve_aggregate(frame, project_id: int, query: str) -> Optional[Dict[str, Any]]:
    """total / sum / average / count over all unit areas (area attribute only)."""
    if frame.attribute != "area":
        return None
    areas = _all_unit_areas(project_id)
    if len(areas) < 2:
        return None
    vals = list(areas.values())
    n_area = len(vals)
    total = sum(vals)
    avg = round(total / n_area)
    n_total = len(_all_unit_ids(project_id))   # honest denominator (incl. units w/o area)
    cov = "" if n_total <= n_area else (
        f" Note: areas were extracted for {n_area} of {n_total} units identified; "
        f"{n_total - n_area} unit(s) had no extracted area label, so totals are a lower bound.")
    src_note = " Based on extracted drawing data — verify against the unit schedule for an authoritative figure."
    if re.search(r"\b(average|avg|mean)\b", query, re.I):
        answer = f"The average area across {n_area} units with extracted data is {avg:,} SF.{cov}"
        value = avg
    elif re.search(r"\b(count|how many)\b", query, re.I):
        answer = f"{n_total} units were identified in the drawings (areas extracted for {n_area})."
        value = n_total
    else:  # total / sum / combined
        answer = (f"The combined area of the {n_area} units with extracted data is {total:,} SF "
                  f"(average {avg:,} SF/unit).{cov}{src_note}")
        value = total
    # [SVR-QA1] don't over-claim confidence when coverage is incomplete
    conf = "high" if (n_area >= 3 and n_total <= n_area) else "medium"
    return {"attribute": "area_aggregate", "value": value, "unit": "SF",
            "entity": "all_units", "confidence": conf,
            "n": n_area, "n_total_units": n_total,
            "source_doc": {}, "answer": answer, "resolver": RESOLVER_VERSION}


def _resolve_compare(frame, project_id: int, query: str) -> Optional[Dict[str, Any]]:
    """largest / smallest / biggest unit by area."""
    if frame.attribute != "area":
        return None
    areas = _all_unit_areas(project_id)
    if len(areas) < 2:
        return None
    smallest = re.search(r"\b(smallest|least|min|minimum)\b", query, re.I)
    pick = min(areas.items(), key=lambda kv: kv[1]) if smallest else max(areas.items(), key=lambda kv: kv[1])
    sup = "smallest" if smallest else "largest"
    # top-3 context
    ranked = sorted(areas.items(), key=lambda kv: kv[1], reverse=not smallest)[:3]
    ctx = ", ".join(f"{k} ({v:,} SF)" for k, v in ranked)
    n_total = len(_all_unit_ids(project_id))
    cov = "" if n_total <= len(areas) else (
        f" (based on {len(areas)} of {n_total} units with extracted area data)")
    answer = (f"The {sup} unit in the extracted drawing data is {pick[0]} at {pick[1]:,} SF"
              f"{cov}. Top by size: {ctx}.")
    # [SVR-QA2] cite the sheet that actually shows the winning unit (reuse UAR)
    src_doc = {}
    try:
        _ua = _uar().resolve_unit_area(project_id, pick[0])
        if _ua and _ua.get("source_doc"):
            src_doc = _ua["source_doc"]
    except Exception:  # noqa: BLE001
        pass
    return {"attribute": "area_compare", "value": {"unit_id": pick[0], "area_sf": pick[1]},
            "unit": "SF", "entity": pick[0], "confidence": "high" if len(areas) >= 3 else "medium",
            "n": len(areas), "source_doc": src_doc, "answer": answer, "resolver": RESOLVER_VERSION}


_FOLLOWUPS = {
    "unit_type": "What is the square footage of this unit?",
    "count": "Can you list these by sheet?",
    "cfm": "Which equipment do these airflow values serve?",
    "duct_size": "Which runs do these duct sizes belong to?",
}


def resolve(frame, project_id: int, query: str) -> Optional[Dict[str, Any]]:
    """Dispatch the slot frame to an attribute handler. Returns a result dict
    (with `answer` + `source_doc`) or None to fall through to the agent."""
    if not _enabled() or frame is None or not getattr(frame, "is_spatial_value", False):
        return None
    if getattr(frame, "confidence", 0) < 0.6:
        return None
    attr = frame.attribute
    ent = frame.entity or {}
    form = getattr(frame, "form", "point")
    result = None
    try:
        # [SVR-P7] aggregate/compare reduce over all unit areas — runs first when
        # the question is a total/average/largest-style question (not a single entity).
        if attr == "area" and form == "aggregate" and not (ent and ent.get("type") == "room"):
            result = _resolve_aggregate(frame, project_id, query)
            if result is not None:
                return _verify_and_audit(result, frame)
        if attr == "area" and form == "compare":
            result = _resolve_compare(frame, project_id, query)
            if result is not None:
                return _verify_and_audit(result, frame)
        if attr == "unit_type":
            result = _resolve_unit_type(frame, project_id)
        elif attr == "schedule_value":
            result = _resolve_schedule_value(frame, project_id, query)
        elif attr in ("cfm", "duct_size"):
            # equipment-specific -> schedule lookup; otherwise enumerate callouts
            if ent.get("type") == "equipment":
                result = _resolve_schedule_value(frame, project_id, query)
            if result is None:
                if attr == "cfm":
                    result = _resolve_enumerate_callouts(frame, project_id, "cfm", "cfm_callouts", "value", "Airflow (CFM)")
                else:
                    result = _resolve_enumerate_callouts(frame, project_id, "duct_size", "duct_sizes", "raw", "Duct size")
        elif attr == "count":
            result = _resolve_count(frame, project_id, query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p2] resolve(%s) error: %s", attr, exc)
        return None
    if result is not None:
        result = _verify_and_audit(result, frame)
    return result


# ── [SVR-P5] value verifier + abstention + audit ────────────────────────────
def _verify_and_audit(result: Dict[str, Any], frame) -> Optional[Dict[str, Any]]:
    """Stamp an auditable value_resolution onto the result's source_doc and
    enforce abstention when cross-variant agreement is too weak. Grounding is
    inherent (values come straight from structured fields, never an LLM), so this
    adds a self-consistency floor + a transparency record — never invents."""
    votes = result.get("votes")
    n = result.get("n")
    agreement = None
    if isinstance(votes, int) and isinstance(n, int) and n > 0:
        agreement = round(votes / n, 2)
        # conflict abstention: a contested single-value answer (point form) is unsafe
        if result.get("attribute") in ("unit_type",) and n >= 2 and agreement < 0.5:
            logger.info("[svr-p5] abstain: %s contested (agreement=%.2f)",
                        result.get("attribute"), agreement)
            return None
    audit = {
        "attribute": result.get("attribute"),
        "value": result.get("value"),
        "confidence": result.get("confidence"),
        "votes": votes, "agreement": agreement,
        "source_sheet": (result.get("source_doc") or {}).get("drawing_name"),
        "resolver": result.get("resolver"),
        "grounded": True, "verifier": "svr-p5",
    }
    sd = result.get("source_doc")
    if isinstance(sd, dict):
        sd["_value_resolution"] = audit
    result["value_resolution"] = audit
    return result
