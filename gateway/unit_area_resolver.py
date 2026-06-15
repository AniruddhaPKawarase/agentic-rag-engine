"""[UAR-V1] Unit / room area resolver — deterministic square-footage answers.

Problem (2026-06-15): "square foot of room U 218" on project 7224 returned
"not found / manual calculation required" from the regular pipeline (it
retrieved the wrong sheet — A-210 Level 01 — and matched copyright boilerplate),
while Deep Dive hallucinated "413 SF". The true answer is 1,153 SF, labeled on
A-320 (Overall Reflected Ceiling Plan – Level 02).

The data is already extracted and structured on drawings_v3:
  - ``unit_tags_mined``  : [{unit_id: "U-218", bbox_pt: [...], page}]
  - ``sf_callouts``      : [{area_sf: 1153, raw: "1153 SF", bbox_pt: [...], page}]
No existing tool queries these, and none does the spatial pairing needed to
answer "what is the area of unit X". This resolver does:

  detect_unit_area_query(query) -> normalized unit_id (e.g. "U-218") or None
  resolve_unit_area(project_id, unit_id) -> {area_sf, confidence, source...} or None

Pairing: for each drawing carrying the unit tag, find the nearest sf_callout by
bbox-center distance (same page), then vote across all variants. High agreement
+ a small tag→callout distance ⇒ high confidence. Anything ambiguous returns a
low-confidence / None result so the caller falls through to the normal pipeline.

Flag: UNIT_AREA_RESOLVER_ENABLED (default true). Strictly additive.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_rag.unit_area_resolver")

RESOLVER_VERSION = "uar-v1"

# A tag→callout center distance (in PDF points) at or below this is treated as a
# confident pairing (the tag sits on its own area label). Empirically ~11pt for
# clean pairs on 7224 A-320; loose pairs land ~150-195pt.
_NEAR_PT = 60.0

# A real sheet number: A-320, M-301, ID-120, RCP-01, S-101a (NOT "BIMe",
# "Overlays_...", "100", or a VLM sentence like "Not explicitly visible...").
_CLEAN_SHEET = re.compile(r"^[A-Z]{1,4}-?\d{1,4}[A-Za-z]?$", re.I)


def _citation_score(doc: Dict[str, Any]) -> int:
    """Higher = better drawing to cite for an area. Prefers a real plan sheet
    number, a reflected-ceiling / overall plan, UI-openable metadata; rejects
    composite 'overlay' / BIMe model sheets that are not real published sheets."""
    sheet = (doc.get("sheetNumber") or doc.get("drawingName") or "").strip()
    title = (doc.get("drawingTitle") or "").upper()
    name_u = (doc.get("drawingName") or "").upper()
    dtype = (doc.get("drawingType") or "").lower()
    disc = (doc.get("discipline") or "").lower()
    # Reject non-published composite sheets outright (still count for area voting,
    # just never chosen as the citation).
    if ("OVERLAY" in title or "OVERLAY" in name_u or "BIME" in title or "BIME" in name_u
            or "OVERLAYS" in name_u):
        return -100
    score = 0
    if _CLEAN_SHEET.match(sheet):
        score += 10              # real sheet number — the big differentiator
    if doc.get("s3BucketPath") and doc.get("pdfName"):
        score += 4               # UI-openable
    if "reflected ceiling" in title or "rcp" in dtype or dtype == "reflected_ceiling_plan":
        score += 5               # RCP is where unit area labels live on this project
    if "overall" in title and "plan" in title:
        score += 3
    if "finish plan" in title or "floor plan" in title:
        score += 2
    if disc.startswith("arch") or "architect" in disc or "interior" in disc:
        score += 1
    return score

# ── area-intent detection ────────────────────────────────────────────────
_AREA_INTENT = re.compile(
    r"(square\s*f(oot|eet|ootage)?|sq\.?\s*ft|\bsf\b|\bs\.f\.?\b|"
    r"\barea\b|how\s+(big|large)|floor\s*area|how\s+many\s+(square|sq)|size\s+of)",
    re.I,
)

# explicit unit code: U-218 / U 218 / U218  (letter U only — that's the data convention)
_UNIT_CODE = re.compile(r"\bU\s*-?\s*(\d{2,4})\b", re.I)
# "unit 218" / "apartment 218" / "apt 218" / "unit no 218"
_UNIT_WORD = re.compile(
    r"\b(?:unit|apartment|apt)\s*(?:no\.?|number|#)?\s*-?\s*U?-?\s*(\d{2,4})\b", re.I
)


def normalize_unit_id(raw_digits: str) -> str:
    return f"U-{raw_digits}"


def detect_unit_area_query(query: str) -> Optional[str]:
    """Return a normalized unit id (``U-218``) when the query is asking for the
    area / square footage of a specific unit, else None.

    Requires BOTH an area intent AND a unit token. Conservative by design — a
    miss just means the normal pipeline runs.
    """
    if not query or not isinstance(query, str):
        return None
    if not _AREA_INTENT.search(query):
        return None
    m = _UNIT_CODE.search(query) or _UNIT_WORD.search(query)
    if not m:
        return None
    return normalize_unit_id(m.group(1))


# ── geometry ───────────────────────────────────────────────────────────────
def _center(bbox: Any) -> Optional[tuple]:
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return ((float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0)
    except (TypeError, ValueError):
        return None


def _dist(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def pair_unit_to_area(unit_tags: List[Dict], sf_callouts: List[Dict],
                      unit_id: str) -> Optional[Dict[str, Any]]:
    """Within one drawing, find the area_sf whose callout is nearest the unit
    tag's center on the same page. Returns {area_sf, distance_pt} or None."""
    targets = [u for u in (unit_tags or [])
               if isinstance(u, dict) and str(u.get("unit_id", "")).upper().replace(" ", "")
               == unit_id.upper().replace(" ", "")]
    if not targets:
        return None
    best_area, best_d = None, 1e9
    for u in targets:
        uc = _center(u.get("bbox_pt"))
        if not uc:
            continue
        u_page = u.get("page")
        for s in (sf_callouts or []):
            if not isinstance(s, dict):
                continue
            if u_page is not None and s.get("page") is not None and s.get("page") != u_page:
                continue
            sc = _center(s.get("bbox_pt"))
            if not sc:
                continue
            d = _dist(uc, sc)
            if d < best_d:
                best_d, best_area = d, s.get("area_sf")
    if best_area is None:
        return None
    return {"area_sf": best_area, "distance_pt": round(best_d, 1)}


def _get_collection(name: str):
    try:
        from core.db import get_collection
        return get_collection(name)
    except Exception:  # pragma: no cover - fallback for non-package contexts
        from agentic.core.db import get_collection  # type: ignore
        return get_collection(name)


def _sign_url(s3_path: Optional[str], pdf_name: Optional[str]) -> Optional[str]:
    """Best-effort presigned URL so the cited drawing is UI-openable."""
    if not s3_path or not pdf_name:
        return None
    key = f"{s3_path.rstrip('/')}/{pdf_name}.pdf"
    try:
        from shared.s3_utils.operations import generate_presigned_url
        return generate_presigned_url(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[uar] presign failed for %s: %s", key, exc)
        return None


def _enabled() -> bool:
    return os.getenv("UNIT_AREA_RESOLVER_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )


def resolve_unit_area(project_id: int, unit_id: str) -> Optional[Dict[str, Any]]:
    """Resolve the area (SF) of a unit by spatially pairing its tag to the
    nearest SF callout, voting across every drawing/variant that carries it.

    Returns None (caller falls through to the normal pipeline) when the data is
    absent or the pairing is ambiguous.
    """
    if not _enabled():
        return None
    try:
        coll = _get_collection("drawings_v3")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[uar] cannot open drawings_v3: %s", exc)
        return None

    # Only scan drawings that actually carry this unit tag. Exact match first
    # (fast, index-friendly); fall back to case-insensitive regex if none.
    proj = {"_id": 0, "drawingId": 1, "sheetNumber": 1, "drawingName": 1,
            "drawingTitle": 1, "drawingType": 1, "discipline": 1, "page": 1,
            "pdfName": 1, "s3BucketPath": 1, "trade": 1,
            "unit_tags_mined": 1, "sf_callouts": 1}
    # Require sf_callouts present (only sheets with area labels can pair) — this
    # also filters to the plan sheets that carry the canonical citation.
    base = {"projectId": int(project_id), "sf_callouts.0": {"$exists": True}}
    docs = []
    try:
        # bounded scan: nested-array match may be unindexed; cap docs + server
        # time so a miss can't stall the request (it falls through to pipeline).
        docs = list(coll.find({**base, "unit_tags_mined.unit_id": unit_id}, proj)
                    .limit(60).max_time_ms(9000))
        if not docs:
            docs = list(coll.find(
                {**base, "unit_tags_mined.unit_id":
                 {"$regex": f"^{re.escape(unit_id)}$", "$options": "i"}}, proj)
                .limit(60).max_time_ms(9000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[uar] query failed: %s", exc)
        return None
    if not docs:
        return None

    # Vote: collect (area, distance) pairings; prefer near pairings. For each
    # winning area, remember the BEST drawing to cite (clean sheet number etc.).
    near_areas: List[int] = []
    all_pairs: List[Dict] = []
    best_doc_for_area: Dict[int, Dict] = {}
    for d in docs:
        pr = pair_unit_to_area(d.get("unit_tags_mined"), d.get("sf_callouts"), unit_id)
        if not pr:
            continue
        all_pairs.append({"area_sf": pr["area_sf"], "distance_pt": pr["distance_pt"],
                          "sheet": d.get("sheetNumber") or d.get("drawingName")})
        if pr["distance_pt"] <= _NEAR_PT:
            near_areas.append(pr["area_sf"])
            cur = best_doc_for_area.get(pr["area_sf"])
            if cur is None or _citation_score(d) > _citation_score(cur):
                best_doc_for_area[pr["area_sf"]] = d

    if not near_areas:
        logger.info("[uar] %s: no confident pairing (pairs=%s)", unit_id, all_pairs[:6])
        return None

    counts = Counter(near_areas)
    area_sf, votes = counts.most_common(1)[0]
    agreement = votes / len(near_areas)
    confidence = "high" if (agreement >= 0.6 and votes >= 2) else "medium"
    src = best_doc_for_area.get(area_sf) or {}

    s3_path = src.get("s3BucketPath")
    pdf_name = src.get("pdfName")
    source_doc = {
        "s3_path": s3_path or "",
        "pdf_name": pdf_name or "",
        "file_name": pdf_name or "",
        "drawing_name": src.get("sheetNumber") or src.get("drawingName") or "",
        "drawing_title": src.get("drawingTitle") or "",
        "display_title": src.get("drawingTitle") or "",
        "sheet_number": src.get("sheetNumber") or "",
        "page": src.get("page") or 1,
        "drawing_id": src.get("drawingId"),
        "trade": src.get("trade") or "",
        "source_document_type": "drawing",
        "download_url": _sign_url(s3_path, pdf_name),
    }
    result = {
        "unit_id": unit_id,
        "area_sf": area_sf,
        "confidence": confidence,
        "agreement": round(agreement, 2),
        "votes": votes,
        "n_pairings": len(near_areas),
        "source_sheet": source_doc["drawing_name"],
        "source_title": source_doc["drawing_title"],
        "source_doc": source_doc,
        "all_pairs": all_pairs[:8],
        "resolver": RESOLVER_VERSION,
    }
    logger.info("[uar] %s -> %s SF (conf=%s votes=%d/%d sheet=%s)",
                unit_id, area_sf, confidence, votes, len(near_areas), source_doc["drawing_name"])
    return result


def build_answer_text(res: Dict[str, Any]) -> str:
    """Human answer with a grounded citation to the source sheet."""
    area = res["area_sf"]
    area_str = f"{int(area):,}" if isinstance(area, (int, float)) else str(area)
    sheet = res.get("source_sheet") or "the floor/ceiling plan"
    title = res.get("source_title") or ""
    title_clean = " ".join(str(title).split())
    where = f"{title_clean} (Ref: {sheet})" if title_clean else f"Ref: {sheet}"
    return (f"The square footage of unit {res['unit_id']} is {area_str} SF, "
            f"as labeled on {where}.")
