"""[SVR-P1] Spatial-value-lookup slot parser (OBSERVE-ONLY in P1).

Classifies a query into a typed frame for the Structured Value Resolver:
    {is_spatial_value, attribute, entity, scope, form, confidence, method}

Tier-1 = deterministic regex/gazetteer (free, instant). When a value-intent is
detected but slots can't be filled confidently, an optional Haiku fallback fills
the frame (flag SPATIAL_PARSER_HAIKU_FALLBACK, reuses the anthropic client like
trade_router). In P1 this parser is OBSERVE-ONLY: callers log the frame into
debug_info and change NO retrieval behavior — so it cannot regress answers.

Flags:
  SPATIAL_PARSER_ENABLED          (default true)  — run the parser at all
  SPATIAL_PARSER_HAIKU_FALLBACK   (default false) — allow the Haiku slot-fill
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger("agentic_rag.spatial_query_parser")

PARSER_VERSION = "svr-p1"

# ── attribute lexicon (regex → canonical attribute) ─────────────────────────
_ATTRIBUTE_PATTERNS = [
    ("area",            r"square\s*f(oot|eet|ootage)?|\bsq\.?\s*ft\b|\bsf\b|\bs\.f\.?\b|floor\s*area|\barea\b"),
    ("ceiling_height",  r"ceiling\s*(height|ht)|\bclg\.?\s*(ht|height)\b|\bAFF\b|height\s+of\s+(the\s+)?ceiling|how\s+(high|tall)\s+is\s+the\s+ceiling"),
    ("elevation",       r"slab\s*elevation|floor\s*elevation|\bT\.?O\.?S\.?\b|\bT\.?O\.?\s*slab\b|finish(ed)?\s*floor\s*elev|\bF\.?F\.?E\.?\b|elevation\s+of"),
    ("cfm",             r"\bcfm\b|air\s*flow|airflow|supply\s+air|return\s+air"),
    ("duct_size",       r"duct\s*size|size\s+of\s+(the\s+)?duct|duct\s+dimension"),
    ("door_size",       r"door\s*(size|width|height|dimension)|size\s+of\s+(the\s+)?door"),
    ("window_size",     r"window\s*(size|width|height|dimension)|size\s+of\s+(the\s+)?window"),
    ("dimension",       r"\bdimensions?\b|how\s+(wide|long|big)\s|width\s+of|length\s+of|\bsize\s+of\s+(the\s+)?room\b"),
    ("count",           r"how\s+many|number\s+of|count\s+of|total\s+(number|count)|how\s+much\s+\w+\s+are\s+there"),
    ("unit_type",       r"unit\s*type|type\s+of\s+unit|what\s+type\s+is\s+unit"),
    ("schedule_value",  r"schedule|capacity\s+of|\bmodel\b|model\s+(number|no)|tonnage|rating\s+of|\bspecs?\b|specification?s?\s+(of|for)|data\s+sheet"),
]
_ATTR_COMPILED = [(a, re.compile(p, re.I)) for a, p in _ATTRIBUTE_PATTERNS]

# ── entity extraction ───────────────────────────────────────────────────────
_ENT_UNIT = re.compile(r"\b(?:unit|apartment|apt)?\s*\bU\s*-?\s?(\d{2,4})\b", re.I)
_ENT_UNIT_WORD = re.compile(r"\b(?:unit|apartment|apt)\s*(?:no\.?|#)?\s*(\d{2,4})\b", re.I)
_ENT_EQUIP = re.compile(r"\b([A-Z]{2,4})-?\s?(\d{1,3}(?:\.\d+)?)\b")   # AHU-1, FCU-3, RTU-2
_EQUIP_PREFIXES = {"AHU", "FCU", "RTU", "VAV", "CU", "HP", "EF", "SF", "RF", "WH",
                   "P", "B", "CH", "CT", "UH", "ERV", "DOAS", "ACCU", "MAU", "CRAC",
                   "EVH", "ESD", "MWUH", "CUH", "DX", "ACU", "HRU", "MUA", "TU"}
_ENT_DOORWIN = re.compile(r"\b([WD]\d{1,3}[A-Z]?)\b")
_ROOM_GAZETTEER = [
    "lobby", "corridor", "hallway", "stair", "stairwell", "elevator", "vestibule",
    "lounge", "kitchen", "bath", "bathroom", "bedroom", "living", "dining",
    "closet", "mechanical room", "electrical room", "storage", "office",
    "conference", "restroom", "toilet", "laundry", "garage", "amenity",
    "common area", "fitness", "gym", "pool", "trash", "mail",
]
_ROOM_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in _ROOM_GAZETTEER) + r")\b", re.I)

# ── form detection ──────────────────────────────────────────────────────────
_FORM_ENUM = re.compile(r"\b(list|all|each|every|show\s+me\s+all|enumerate)\b", re.I)
_FORM_AGG = re.compile(r"\b(how\s+many|total|sum|average|avg|count|combined)\b", re.I)
_FORM_CMP = re.compile(r"\b(largest|smallest|biggest|max|min|which\s+is|compare|more\s+than|less\s+than)\b", re.I)

# ── attribute → authoritative drawing type(s) (boost, not filter) ───────────
ATTRIBUTE_SHEET_AUTHORITY: Dict[str, list] = {
    "area":           ["floor_plan", "reflected_ceiling_plan"],
    "ceiling_height": ["reflected_ceiling_plan"],
    "elevation":      ["section", "floor_plan", "detail"],
    "cfm":            ["floor_plan"],          # mechanical floor plan
    "duct_size":      ["floor_plan"],
    "door_size":      ["schedule", "floor_plan"],
    "window_size":    ["schedule", "floor_plan"],
    "dimension":      ["floor_plan", "enlarged_plan"],
    "count":          ["floor_plan", "reflected_ceiling_plan"],
    "unit_type":      ["floor_plan", "reflected_ceiling_plan"],
    "schedule_value": ["schedule"],
}


@dataclass
class SpatialQueryFrame:
    is_spatial_value: bool = False
    attribute: Optional[str] = None
    entity: Optional[Dict[str, Any]] = None      # {"type":..., "value":...}
    scope: Dict[str, Any] = field(default_factory=dict)   # {"level":int|None,"zone":...}
    form: str = "point"
    confidence: float = 0.0
    method: str = "none"                          # regex | haiku | none
    authority_sheet_types: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parser_enabled() -> bool:
    return os.getenv("SPATIAL_PARSER_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _haiku_enabled() -> bool:
    return os.getenv("SPATIAL_PARSER_HAIKU_FALLBACK", "false").strip().lower() in ("1", "true", "yes", "on")


def _detect_attribute(q: str) -> Optional[str]:
    for attr, rx in _ATTR_COMPILED:
        if rx.search(q):
            return attr
    return None


def _detect_entity(q: str) -> Optional[Dict[str, Any]]:
    m = _ENT_UNIT.search(q) or _ENT_UNIT_WORD.search(q)
    if m:
        return {"type": "unit", "value": f"U-{m.group(1)}"}
    for m in _ENT_EQUIP.finditer(q):
        if m.group(1).upper() in _EQUIP_PREFIXES:
            return {"type": "equipment", "value": f"{m.group(1).upper()}-{m.group(2)}"}
    m = _ENT_DOORWIN.search(q)
    if m:
        kind = "door" if m.group(1)[0].upper() == "D" else "window"
        return {"type": kind, "value": m.group(1).upper()}
    m = _ROOM_RE.search(q)
    if m:
        return {"type": "room", "value": m.group(1).lower()}
    return None


def _detect_scope(q: str) -> Dict[str, Any]:
    scope: Dict[str, Any] = {"level": None, "zone": None}
    # reuse sheet_router floor parsing where available
    lvl = None
    m = re.search(r"\blevel\s*0?(\d{1,2})\b", q, re.I) or re.search(r"\bL-?(\d{1,2})\b", q)
    if m:
        lvl = int(m.group(1))
    else:
        ord_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                   "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
                   "ground": 1, "cellar": 0, "basement": 0}
        m2 = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|ground|cellar|basement)\b(?:\s+floor|\s+level)?", q, re.I)
        if m2:
            lvl = ord_map[m2.group(1).lower()]
        else:
            m3 = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+(?:floor|level)\b", q, re.I)
            if m3:
                lvl = int(m3.group(1))
    scope["level"] = lvl
    mz = re.search(r"\bzone\s*(\d+|[A-D])\b", q, re.I)
    if mz:
        scope["zone"] = mz.group(1).upper()
    return scope


def _detect_form(q: str) -> str:
    if _FORM_CMP.search(q):
        return "compare"
    if _FORM_AGG.search(q):
        return "aggregate"
    if _FORM_ENUM.search(q):
        return "enumerate"
    return "point"


def _haiku_fill(query: str) -> Optional[SpatialQueryFrame]:
    """Optional Haiku slot-fill when regex finds value-intent but not enough slots."""
    if not _haiku_enabled():
        return None
    try:
        from anthropic import Anthropic
        client = Anthropic()
        model = os.getenv("SPATIAL_PARSER_MODEL", "claude-haiku-4-5")
        sys_p = (
            "Extract a construction spatial-value-lookup frame from the question. "
            "Return ONLY JSON: {\"is_spatial_value\":bool, \"attribute\":one of "
            "[area,ceiling_height,elevation,cfm,duct_size,door_size,window_size,"
            "dimension,count,unit_type,schedule_value,null], \"entity\":{\"type\":"
            "str,\"value\":str}|null, \"level\":int|null, \"form\":one of "
            "[point,enumerate,aggregate,compare]}. is_spatial_value=false if the "
            "question is not asking for a specific measurable value at a location."
        )
        resp = client.messages.create(
            model=model, max_tokens=200, temperature=0,
            system=sys_p,
            messages=[{"role": "user", "content": query}],
        )
        txt = resp.content[0].text if resp.content else "{}"
        m = re.search(r"\{.*\}", txt, re.S)
        data = json.loads(m.group(0) if m else txt)
        if not data.get("is_spatial_value"):
            return SpatialQueryFrame(is_spatial_value=False, method="haiku", confidence=0.5)
        attr = data.get("attribute")
        fr = SpatialQueryFrame(
            is_spatial_value=True, attribute=attr,
            entity=data.get("entity"),
            scope={"level": data.get("level"), "zone": None},
            form=data.get("form") or "point",
            confidence=0.75, method="haiku",
            authority_sheet_types=ATTRIBUTE_SHEET_AUTHORITY.get(attr, []),
        )
        return fr
    except Exception as exc:  # noqa: BLE001
        logger.warning("[svr-p1] haiku fallback failed (%s)", exc)
        return None


def parse(query: str) -> SpatialQueryFrame:
    """Return the spatial-value frame. Never raises; returns a not-spatial frame
    on any error so callers can safely treat None-ish frames as 'use normal path'."""
    if not _parser_enabled() or not query or not isinstance(query, str):
        return SpatialQueryFrame()
    q = query.strip()
    attr = _detect_attribute(q)
    ent = _detect_entity(q)
    scope = _detect_scope(q)
    form = _detect_form(q)

    # [SVR-P7] infer area for superlative/aggregate questions about units/apartments
    # that lack an explicit attribute word (e.g. "which is the smallest apartment").
    if not attr and re.search(r"\b(largest|smallest|biggest|most|least)\b", q, re.I) \
            and re.search(r"\b(unit|units|apartment|apartments|apt)\b", q, re.I):
        attr = "area"

    if attr:
        # confidence: attribute + (entity OR explicit scope) = strong; attribute only = medium
        conf = 0.95 if (ent or scope.get("level") is not None) else 0.6
        return SpatialQueryFrame(
            is_spatial_value=True, attribute=attr, entity=ent, scope=scope,
            form=form, confidence=conf, method="regex",
            authority_sheet_types=ATTRIBUTE_SHEET_AUTHORITY.get(attr, []),
        )

    # No attribute via regex, but an entity + value-ish phrasing → try Haiku
    if ent and re.search(r"\bwhat|how|size|value|spec|tell me\b", q, re.I):
        hf = _haiku_fill(q)
        if hf is not None:
            return hf

    return SpatialQueryFrame()
