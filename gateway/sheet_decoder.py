"""
sheet_decoder.py
================

Pure-Python decoder that maps an existing `drawings_v3` document to its
(trade, role) cell — WITHOUT any new Mongo schema fields.

Inputs (existing fields):
    sheetNumber    e.g. "A-100", "A-601.00", "M-201A", "FP-301", "ME-201"
    drawingTitle   e.g. "REFLECTED CEILING PLAN", "FRAMING PLAN", ...

Outputs (bounded vocabulary):
    trade : one of TRADES (16 values)
    role  : one of ROLES (8 values)

Pure function, no I/O, no LLM call, ~5ms per call. Safe to call inside a
Mongo $match stage as a $regex test as well — the patterns are anchored.

Design constraints (locked in design):
    - No new schema fields on drawings_v3
    - Regex MUST be anchored to use sheetNumber's btree index efficiently
    - Sheet prefixes follow AIA Layer Guidelines + Uniform Drawing System
    - Polysemy stays out of this layer (drawings only have one trade per sheet)
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Bounded vocabularies
# ---------------------------------------------------------------------------
TRADES: Tuple[str, ...] = (
    "G",   # General / Cover / Code / Life Safety
    "C",   # Civil
    "L",   # Landscape
    "AS",  # Architectural Site (overlap with Civil)
    "A",   # Architectural
    "ID",  # Interiors
    "S",   # Structural
    "M",   # Mechanical (HVAC)
    "P",   # Plumbing
    "E",   # Electrical
    "FA",  # Fire Alarm
    "FP",  # Fire Protection (Sprinkler)
    "T",   # Telecom / Low-Voltage
    "V",   # Vertical Transportation
    "K",   # Kitchen / Food Service
    "PE",  # Process / Specialty Equipment
    "unknown",  # bucket for unparseable
)

ROLES: Tuple[str, ...] = (
    "Plan",
    "Section",
    "Elevation",
    "Detail",
    "Schedule",
    "Diagram",
    "Notes",
    "RCP",   # Reflected Ceiling Plan — subtype of Plan, broken out
    "unknown",
)


# ---------------------------------------------------------------------------
# TRADE patterns — anchored on sheetNumber prefix
# ---------------------------------------------------------------------------
# Order matters: more-specific patterns first (FP before F, AS before A, etc.)
TRADE_TO_SHEET_REGEX: Tuple[Tuple[str, re.Pattern], ...] = (
    # Multi-letter prefixes (must come BEFORE single-letter ones)
    ("FP", re.compile(r"^FP[-\s]?\d", re.IGNORECASE)),
    ("FA", re.compile(r"^FA[-\s]?\d", re.IGNORECASE)),
    ("AS", re.compile(r"^AS[-\s]?\d", re.IGNORECASE)),
    ("ID", re.compile(r"^ID[-\s]?\d|^I-?\d", re.IGNORECASE)),
    ("PE", re.compile(r"^PE[-\s]?\d|^EQ[-\s]?\d", re.IGNORECASE)),
    ("AD", re.compile(r"^AD[-\s]?\d", re.IGNORECASE)),  # arch demo — aliased to A
    ("AI", re.compile(r"^AI[-\s]?\d", re.IGNORECASE)),  # arch interior — aliased to A
    ("CS", re.compile(r"^CS[-\s]?\d", re.IGNORECASE)),  # cover sheet — aliased to G
    # Telecom variants
    ("LV", re.compile(r"^LV[-\s]?\d", re.IGNORECASE)),  # low voltage — aliased to T
    ("IT", re.compile(r"^IT[-\s]?\d", re.IGNORECASE)),  # IT — aliased to T
    ("TC", re.compile(r"^TC[-\s]?\d", re.IGNORECASE)),  # telecom — aliased to T
    ("AV", re.compile(r"^AV[-\s]?\d", re.IGNORECASE)),  # AV — aliased to T
    # Mechanical variants
    ("ME", re.compile(r"^ME[-\s]?\d", re.IGNORECASE)),  # mech-elec? aliased to M
    ("MP", re.compile(r"^MP[-\s]?\d", re.IGNORECASE)),  # mech-plumb — aliased to M
    ("MH", re.compile(r"^MH[-\s]?\d", re.IGNORECASE)),  # mech-HVAC — aliased to M
    # Electrical variants
    ("EE", re.compile(r"^EE[-\s]?\d", re.IGNORECASE)),  # elec - aliased to E
    ("EL", re.compile(r"^EL[-\s]?\d", re.IGNORECASE)),  # elec lighting — aliased to E
    # Plumbing variants
    ("PL", re.compile(r"^PL[-\s]?\d", re.IGNORECASE)),  # plumbing — aliased to P
    # Structural variants
    ("SD", re.compile(r"^SD[-\s]?\d", re.IGNORECASE)),  # struct details — aliased to S
    # Civil variants
    ("CV", re.compile(r"^CV[-\s]?\d", re.IGNORECASE)),  # civil — aliased to C
    # Landscape variants
    ("LP", re.compile(r"^LP[-\s]?\d", re.IGNORECASE)),  # landscape plan — aliased to L
    # Vertical transport
    ("VT", re.compile(r"^VT[-\s]?\d", re.IGNORECASE)),  # vert. transport — aliased to V
    # Kitchen / Food service
    ("FS", re.compile(r"^FS[-\s]?\d", re.IGNORECASE)),  # food service — aliased to K
    # General
    ("GP", re.compile(r"^GP[-\s]?\d", re.IGNORECASE)),  # general — aliased to G
    # Single-letter prefixes — placed LAST so multi-letters win
    ("A",  re.compile(r"^A[-\s]?\d", re.IGNORECASE)),
    ("S",  re.compile(r"^S[-\s]?\d", re.IGNORECASE)),
    ("M",  re.compile(r"^M[-\s]?\d", re.IGNORECASE)),
    ("E",  re.compile(r"^E[-\s]?\d", re.IGNORECASE)),
    ("P",  re.compile(r"^P[-\s]?\d", re.IGNORECASE)),
    ("C",  re.compile(r"^C[-\s]?\d", re.IGNORECASE)),
    ("L",  re.compile(r"^L[-\s]?\d", re.IGNORECASE)),
    ("T",  re.compile(r"^T[-\s]?\d", re.IGNORECASE)),
    ("V",  re.compile(r"^V[-\s]?\d", re.IGNORECASE)),
    ("K",  re.compile(r"^K[-\s]?\d", re.IGNORECASE)),
    ("G",  re.compile(r"^G[-\s]?\d", re.IGNORECASE)),
)

# Alias map — sub-trade prefixes collapse to their parent trade
_ALIAS_TO_PARENT: Dict[str, str] = {
    "AD": "A", "AI": "A",
    "CS": "G", "GP": "G",
    "LV": "T", "IT": "T", "TC": "T", "AV": "T",
    "ME": "M", "MP": "M", "MH": "M",
    "EE": "E", "EL": "E",
    "PL": "P",
    "SD": "S",
    "CV": "C",
    "LP": "L",
    "VT": "V",
    "FS": "K",
}


# ---------------------------------------------------------------------------
# ROLE patterns — anchored on drawingTitle keywords
# ---------------------------------------------------------------------------
# Order matters: most-specific roles first; "Plan" is the default fallback.
ROLE_TO_TITLE_REGEX: Tuple[Tuple[str, re.Pattern], ...] = (
    # RCP — must match BEFORE Plan
    ("RCP", re.compile(r"REFLECT(ED)?\s+CEILING|\bRCP\b", re.IGNORECASE)),
    # Diagram — riser, single-line, one-line
    ("Diagram", re.compile(
        r"\b(DIAGRAM|RISER|SINGLE[-\s]?LINE|ONE[-\s]?LINE|SCHEMATIC)\b",
        re.IGNORECASE,
    )),
    # Schedule — tabular
    ("Schedule", re.compile(r"\bSCHEDULE\b", re.IGNORECASE)),
    # Detail — must NOT also be Section
    ("Detail", re.compile(
        r"\bDETAIL(S)?\b|\bTYPICAL\b(?!.*SECTION)",
        re.IGNORECASE,
    )),
    # Section — vertical cut (not Section Plan)
    ("Section", re.compile(
        r"\bSECTION(S)?\b(?!.*\bPLAN\b)",
        re.IGNORECASE,
    )),
    # Elevation — side view (not "site elevation plan" type)
    ("Elevation", re.compile(
        r"\bELEVATION(S)?\b(?!.*\bPLAN\b)",
        re.IGNORECASE,
    )),
    # Notes — general notes, legend, abbreviations
    ("Notes", re.compile(
        r"\b(NOTES|LEGEND|ABBREVIATION|GENERAL\s+INFORMATION)\b",
        re.IGNORECASE,
    )),
    # Plan — default for the rest (floor plan, roof plan, site plan, etc.)
    ("Plan", re.compile(r"\bPLAN\b", re.IGNORECASE)),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def decode_trade(sheet_number: str | None) -> str:
    """Return one of TRADES (incl. 'unknown') for a given sheet number.

    Examples
    --------
    >>> decode_trade("A-100")
    'A'
    >>> decode_trade("AD-201")
    'A'
    >>> decode_trade("FP-100")
    'FP'
    >>> decode_trade("ME-201")
    'M'
    >>> decode_trade("X-99")
    'unknown'
    >>> decode_trade(None)
    'unknown'
    >>> decode_trade("")
    'unknown'
    """
    if not sheet_number or not isinstance(sheet_number, str):
        return "unknown"
    s = sheet_number.strip()
    if not s:
        return "unknown"
    for trade_code, pattern in TRADE_TO_SHEET_REGEX:
        if pattern.match(s):
            return _ALIAS_TO_PARENT.get(trade_code, trade_code)
    return "unknown"


def decode_role(drawing_title: str | None) -> str:
    """Return one of ROLES (incl. 'unknown') for a given drawing title.

    Examples
    --------
    >>> decode_role("REFLECTED CEILING PLAN — 1ST FLOOR")
    'RCP'
    >>> decode_role("FOUNDATION PLAN")
    'Plan'
    >>> decode_role("DOOR SCHEDULE")
    'Schedule'
    >>> decode_role("WALL SECTION A")
    'Section'
    >>> decode_role("TYPICAL DETAILS")
    'Detail'
    >>> decode_role("PLUMBING RISER DIAGRAM")
    'Diagram'
    >>> decode_role("GENERAL NOTES")
    'Notes'
    >>> decode_role(None)
    'unknown'
    """
    if not drawing_title or not isinstance(drawing_title, str):
        return "unknown"
    t = drawing_title.strip()
    if not t:
        return "unknown"
    for role, pattern in ROLE_TO_TITLE_REGEX:
        if pattern.search(t):
            return role
    return "unknown"


def decode_trade_role(doc: dict) -> Tuple[str, str]:
    """Convenience: extract (trade, role) for a drawings_v3 document.

    Accepts the existing Mongo doc shape and tolerates field aliases.
    """
    if not isinstance(doc, dict):
        return ("unknown", "unknown")
    sheet = (
        doc.get("sheetNumber")
        or doc.get("sheet_number")
        or doc.get("drawingName")
        or doc.get("drawing_name")
        or ""
    )
    title = (
        doc.get("drawingTitle")
        or doc.get("drawing_title")
        or doc.get("display_title")
        or doc.get("displayTitle")
        or doc.get("pdfName")
        or doc.get("pdf_name")
        or ""
    )
    return (decode_trade(sheet), decode_role(title))


# ---------------------------------------------------------------------------
# Mongo $match helpers — used by trade_aware_retrieval.py
# ---------------------------------------------------------------------------
# Per-trade regex string (without /.../ delimiters) — for $regex operator
TRADE_TO_MONGO_REGEX: Dict[str, str] = {
    "A":  r"^(A|AD|AI)[-\s]?\d",
    "S":  r"^(S|SD)[-\s]?\d",
    "M":  r"^(M|ME|MP|MH)[-\s]?\d",
    "E":  r"^(E|EE|EL)[-\s]?\d",
    "P":  r"^(P|PL)[-\s]?\d",
    "C":  r"^(C|CV)[-\s]?\d",
    "L":  r"^(L|LP)[-\s]?\d",
    "T":  r"^(T|TC|LV|IT|AV)[-\s]?\d",
    "V":  r"^(V|VT)[-\s]?\d",
    "G":  r"^(G|GP|CS)[-\s]?\d",
    "K":  r"^(K|FS)[-\s]?\d",
    "ID": r"^(ID|I)[-\s]?\d",
    "PE": r"^(PE|EQ)[-\s]?\d",
    "AS": r"^AS[-\s]?\d",
    "FP": r"^FP[-\s]?\d",
    "FA": r"^FA[-\s]?\d",
}

# Per-role regex string for $regex on drawingTitle
ROLE_TO_MONGO_REGEX: Dict[str, str] = {
    "RCP":       r"REFLECT(ED)?\s+CEILING|\bRCP\b",
    "Schedule":  r"\bSCHEDULE\b",
    "Detail":    r"\bDETAIL(S)?\b|\bTYPICAL\b",
    "Section":   r"\bSECTION(S)?\b",
    "Elevation": r"\bELEVATION(S)?\b",
    "Diagram":   r"\b(DIAGRAM|RISER|SINGLE[-\s]?LINE|ONE[-\s]?LINE|SCHEMATIC)\b",
    "Notes":     r"\b(NOTES|LEGEND|ABBREVIATION)\b",
    "Plan":      r"\bPLAN\b",
}


def mongo_match_stage(trade: str | None, role: str | None) -> dict | None:
    """Build a Mongo $match stage that filters by (trade, role).

    Returns None when both inputs are missing/unknown. The returned dict is
    safe to insert as the FIRST stage of an aggregate pipeline — the
    sheetNumber prefix regex will use the existing btree index.

    Either argument can be None / 'unknown' to filter on only one dimension.
    """
    conds = []
    if trade and trade != "unknown" and trade in TRADE_TO_MONGO_REGEX:
        conds.append({
            "sheetNumber": {"$regex": TRADE_TO_MONGO_REGEX[trade], "$options": "i"}
        })
    if role and role != "unknown" and role in ROLE_TO_MONGO_REGEX:
        conds.append({
            "drawingTitle": {"$regex": ROLE_TO_MONGO_REGEX[role], "$options": "i"}
        })
    if not conds:
        return None
    if len(conds) == 1:
        return {"$match": conds[0]}
    return {"$match": {"$and": conds}}
