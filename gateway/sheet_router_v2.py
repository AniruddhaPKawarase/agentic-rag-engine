"""gateway/sheet_router.py — Explicit deterministic sheet-routing.

Goal: bypass Mongo $text non-determinism for queries where we KNOW exactly which
drawing class should hold the answer. Use anchored regex against `sheetNumber` +
`drawingTitle` (which use the existing btree+prefix indexes) — NOT $text scoring.

Inputs:
  - user query (raw, "what is the ceiling height in first floor lobby?")
  - TADR-routed (trade, role)  e.g. ('A', 'RCP')

Outputs:
  - {"drawing_title_pattern": "<regex>", "sheet_number_pattern": "<regex>", "ranked_score": "high|med|low"}
  - Or None when no specific routing is inferred (fall back to $text)

The router emits a Mongo filter the caller can use directly:
    coll.find({**project_filter(pid),
               "sheetNumber": {"$regex": trade_pat, "$options": "i"},
               "drawingTitle": {"$regex": floor_pat + ".*" + role_pat, "$options": "i"}})

Pure-Python. No I/O. ~5ms per call. Deterministic.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Floor ordinals — multi-form recognition
# Maps query phrase -> standardized regex pieces for drawingTitle matching
_FLOOR_PHRASES: List[Tuple[str, str]] = [
    # (regex matching the user phrase, mongo regex piece for drawingTitle)
    (r"\b(?:first|1st|ground)\s*(?:floor|fl)\b",       r"(?:1ST|FIRST|GROUND)\s*(?:FLOOR|FL)"),
    (r"\b(?:second|2nd)\s*(?:floor|fl)\b",             r"(?:2ND|SECOND)\s*(?:FLOOR|FL)"),
    (r"\b(?:third|3rd)\s*(?:floor|fl)\b",              r"(?:3RD|THIRD)\s*(?:FLOOR|FL)"),
    (r"\b(?:fourth|4th)\s*(?:floor|fl)\b",             r"(?:4TH|FOURTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:fifth|5th)\s*(?:floor|fl)\b",              r"(?:5TH|FIFTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:sixth|6th)\s*(?:floor|fl)\b",              r"(?:6TH|SIXTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:seventh|7th)\s*(?:floor|fl)\b",            r"(?:7TH|SEVENTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:eighth|8th)\s*(?:floor|fl)\b",             r"(?:8TH|EIGHTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:ninth|9th)\s*(?:floor|fl)\b",              r"(?:9TH|NINTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:tenth|10th)\s*(?:floor|fl)\b",             r"(?:10TH|TENTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:eleventh|11th)\s*(?:floor|fl)\b",          r"(?:11TH|ELEVENTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:twelfth|12th)\s*(?:floor|fl)\b",           r"(?:12TH|TWELFTH)\s*(?:FLOOR|FL)"),
    (r"\b(?:thirteenth|13th)\s*(?:floor|fl)\b",        r"(?:13TH|THIRTEENTH)\s*(?:FLOOR|FL)"),
    (r"\bcellar\b",                                    r"CELLAR"),
    (r"\bbasement\b",                                  r"BASEMENT|CELLAR"),
    (r"\broof\b",                                      r"ROOF"),
    (r"\bmezzanine\b",                                 r"MEZZANINE"),
    (r"\blevel\s*0?(\d+)\b",                           r"(?:LEVEL\s*0?\g<1>|\g<1>(?:ST|ND|RD|TH)?\s*FLOOR)"),  # special — see fn
    (r"\bl\s*0?(\d+)\b(?!\w)",                         r"(?:L\s*0?\g<1>|\g<1>(?:ST|ND|RD|TH)?\s*FLOOR)"),
]
_FLOOR_COMPILED = [(re.compile(p, re.I), tt) for p, tt in _FLOOR_PHRASES]


# Role -> drawingTitle regex (camelCase Mongo field "drawingTitle")
# Anchored at start-or-middle; non-anchored on end (titles often have variants like "(OVERALL)")
ROLE_TITLE_REGEX: Dict[str, str] = {
    "RCP":       r"REFLECT(?:ED)?\s*CEILING\s*PLAN|\bRCP\b",
    "Schedule":  r"\bSCHEDULE(?:S)?\b",
    "Section":   r"\bSECTION(?:S)?\b",
    "Elevation": r"\bELEVATION(?:S)?\b",
    "Detail":    r"\bDETAIL(?:S)?\b|\bTYPICAL\b",
    "Diagram":   r"\b(?:DIAGRAM|RISER|SINGLE[-\s]?LINE|ONE[-\s]?LINE|SCHEMATIC)\b",
    "Notes":     r"\b(?:NOTES|LEGEND|ABBREVIATION)",
    "Plan":      r"\bPLAN\b",
}

# Trade -> sheetNumber regex (camelCase Mongo field "sheetNumber")
TRADE_SHEET_REGEX: Dict[str, str] = {
    "A":  r"^(?:A|AD|AI)[-\s]?\d",
    "S":  r"^(?:S|SD)[-\s]?\d",
    "M":  r"^(?:M|ME|MP|MH)[-\s]?\d",
    "E":  r"^(?:E|EE|EL)[-\s]?\d",
    "P":  r"^(?:P|PL)[-\s]?\d",
    "C":  r"^(?:C|CV)[-\s]?\d",
    "L":  r"^(?:L|LP)[-\s]?\d",
    "T":  r"^(?:T|TC|LV|IT|AV)[-\s]?\d",
    "V":  r"^(?:V|VT)[-\s]?\d",
    "G":  r"^(?:G|GP|CS)[-\s]?\d",
    "K":  r"^(?:K|FS)[-\s]?\d",
    "ID": r"^(?:ID|I)[-\s]?\d",
    "PE": r"^(?:PE|EQ)[-\s]?\d",
    "AS": r"^AS[-\s]?\d",
    "FP": r"^FP[-\s]?\d",
    "FA": r"^FA[-\s]?\d",
}

# Topic detection — additional context-narrowing keywords ("slab edge plan" vs generic "plan")
# (topic_pattern, trade_must_be, role_must_be, additional_title_regex)
# These are CONJUNCTIVE — all three (topic, trade, role) must match before adding the title-anchor
_TOPIC_TITLE_REFINERS: List[Tuple[str, Optional[str], Optional[str], str]] = [
    (r"\bslab\b", "A", "Plan", r"SLAB\s*EDGE"),
    (r"\bslab\b", "S", "Plan", r"SLAB"),
    (r"\bceiling\b|\bsoffit\b", "A", "RCP", r"REFLECT(?:ED)?\s*CEILING|\bRCP\b"),
    (r"\bdoor\b", "A", "Schedule", r"DOOR\s*SCHEDULE"),
    (r"\bwindow\b", "A", "Schedule", r"WINDOW\s*SCHEDULE"),
    (r"\bfinish(?:es)?\b", "A", "Schedule", r"FINISH(?:\s*ROOM)?\s*SCHEDULE|ROOM\s*FINISH"),
    (r"\b(?:room|partition)\b", "A", "Schedule", r"(?:ROOM|PARTITION)\s*SCHEDULE"),
    (r"\bpanel\b", "E", "Schedule", r"PANEL\s*SCHEDULE"),
    (r"\b(?:fcu|fan\s*coil)\b", "M", "Schedule", r"(?:FCU|FAN\s*COIL)\s*SCHEDULE"),
    (r"\b(?:ahu|air\s*handler|air\s*handling)\b", "M", "Schedule", r"(?:AHU|AIR\s*HANDL\w*)\s*SCHEDULE"),
    (r"\b(?:vav|variable)\b", "M", "Schedule", r"(?:VAV|VARIABLE)\s*SCHEDULE"),
    (r"\bdiffuser\b", "M", "Schedule", r"DIFFUSER\s*SCHEDULE"),
    (r"\bplumbing\s*fixture\b|\bfixture(?:s)?\b(?:.*plumbing|.*plumb)", "P", "Schedule", r"(?:PLUMBING\s*)?FIXTURE\s*SCHEDULE"),
    (r"\briser\b", "P", "Diagram", r"RISER\s*DIAGRAM"),
    (r"\briser\b", "FP", "Diagram", r"RISER\s*DIAGRAM"),
    (r"\briser\b", "E", "Diagram", r"RISER\s*DIAGRAM"),
    (r"\bsprinkler\b", "FP", "Plan", r"SPRINKLER"),
]
_TOPIC_REFINERS_COMPILED = [
    (re.compile(p, re.I), t, r, tr) for p, t, r, tr in _TOPIC_TITLE_REFINERS
]


def extract_floor_title_pattern(query: str) -> Optional[str]:
    """Find a floor ordinal in the query. Return the regex piece to AND into drawingTitle, or None."""
    q = (query or "")
    for rx, title_piece in _FLOOR_COMPILED:
        m = rx.search(q)
        if m:
            # Handle level/L numeric back-reference
            if r"\g<1>" in title_piece or "\\g<1>" in title_piece:
                num = m.group(1) if m.groups() else "1"
                try:
                    n = int(num)
                except ValueError:
                    n = 1
                # Map to ordinal suffix
                suffix = {1:"ST",2:"ND",3:"RD"}.get(n, "TH")
                return f"(?:LEVEL\\s*0?{n}|{n}{suffix}\\s*FLOOR)"
            return title_piece
    return None


def topic_title_refinement(query: str, trade: Optional[str], role: Optional[str]) -> Optional[str]:
    """If query topic matches a known refinement for this (trade, role), return tighter title regex."""
    q = (query or "")
    for rx, t_req, r_req, tr in _TOPIC_REFINERS_COMPILED:
        if (t_req is None or t_req == trade) and (r_req is None or r_req == role):
            if rx.search(q):
                return tr
    return None


def route(
    query: str,
    trade: Optional[str],
    role: Optional[str],
) -> Optional[Dict[str, str]]:
    """Build a deterministic Mongo $match clause for the (query, trade, role).

    Returns dict with keys:
        sheet_number_regex  — regex for `sheetNumber` (e.g. r"^A[-\\s]?\\d" for arch)
        drawing_title_regex — regex for `drawingTitle` combining floor + role/topic
        confidence          — "high" (floor + role/topic both matched)
                              "medium" (role/topic matched, no floor)
                              "low" (only trade matched — fall through to $text)
        rationale           — short human-readable description

    Returns None if no specific routing is inferred (trade unknown).
    """
    if not trade or trade in ("unknown", "G"):
        return None
    sheet_re = TRADE_SHEET_REGEX.get(trade)
    if not sheet_re:
        return None

    role_re = ROLE_TITLE_REGEX.get(role) if role and role != "unknown" else None
    floor_re = extract_floor_title_pattern(query)
    topic_re = topic_title_refinement(query, trade, role)

    # Build drawingTitle pattern: combine floor + role/topic (AND via .* — lookahead-free)
    pieces: List[str] = []
    parts_log: List[str] = []

    if floor_re:
        pieces.append(floor_re)
        parts_log.append(f"floor={floor_re[:30]}")
    if topic_re:
        pieces.append(topic_re)
        parts_log.append(f"topic={topic_re[:30]}")
    elif role_re:
        pieces.append(role_re)
        parts_log.append(f"role={role_re[:30]}")

    if not pieces:
        # Only the trade matched — too broad to route deterministically
        return None

    # AND-of-patterns via $and-style approach: emit a pattern that requires ALL of them
    # We can't use $and inside a single regex without lookahead, so just emit a multi-anchor pattern.
    # The caller will use $and: [{"drawingTitle":{"$regex":p1}},{"drawingTitle":{"$regex":p2}}]
    # for the strict case. For convenience return both forms.
    if len(pieces) >= 2:
        confidence = "high"
    else:
        confidence = "medium"

    return {
        "sheet_number_regex": sheet_re,
        "drawing_title_pieces": pieces,  # list — caller AND's them together
        "drawing_title_regex": "|".join(pieces),  # OR form for soft-match fallback
        "confidence": confidence,
        "rationale": ", ".join(parts_log),
    }


def build_mongo_filter(
    project_id: int,
    query: str,
    trade: Optional[str],
    role: Optional[str],
    *,
    strict: bool = True,
) -> Optional[Dict]:
    """Build a complete Mongo find() filter using the route + project_id.

    `strict=True` -> require ALL drawingTitle pieces (AND'd) AND sheet prefix.
    `strict=False` -> require sheet prefix + ANY drawingTitle piece (OR).

    Returns None if route() yields nothing.
    """
    r = route(query, trade, role)
    if r is None:
        return None

    base: Dict = {
        "projectId": int(project_id),
        "sheetNumber": {"$regex": r["sheet_number_regex"], "$options": "i"},
    }
    pieces = r["drawing_title_pieces"]
    if strict and len(pieces) >= 2:
        # All-of: drawingTitle must match each piece independently
        base["$and"] = [
            {"drawingTitle": {"$regex": p, "$options": "i"}} for p in pieces
        ]
    else:
        # Any-of: drawingTitle matches OR of pieces
        base["drawingTitle"] = {"$regex": r["drawing_title_regex"], "$options": "i"}
    return base


__all__ = [
    "route",
    "build_mongo_filter",
    "extract_floor_title_pattern",
    "topic_title_refinement",
    "ROLE_TITLE_REGEX",
    "TRADE_SHEET_REGEX",
]
