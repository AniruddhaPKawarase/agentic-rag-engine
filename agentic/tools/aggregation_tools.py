"""Fix #3 — Deterministic MongoDB aggregation tools.

Three tools the agent MUST use for counting or "typical-level" questions
instead of eyeballing counts from prose:

- :func:`count_equipment_tags`: Extract distinct equipment tags (e.g.
  ``DOAS-1``, ``AHU-3``, ``VAV-201``) from drawing OCR text and return
  ``{total, unique_tags, by_drawing, by_level}``. No LLM guessing.

- :func:`find_typical_levels`: Cluster drawings by normalised title so floors
  that share a plan group fall together. Returns
  ``{typical_groups, unique_levels, by_level_summary}``.

- :func:`list_schedule_entries`: Extract structured rows from VisionOCR's
  ``pages.vision_elements`` when available, falling back to regex over
  ``fullText`` / drawing text. Used for schedule-driven questions
  ("what rooftop units are on the roof").

The tools return structured dicts the agent can reason about directly; they
also include short ``samples`` so the agent can sanity-check the extraction
before committing to a number.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.db import get_collection
from tools.validation import validate_project_id, validate_search_text

logger = logging.getLogger("agentic_rag.tools.aggregation")

DRAWING_COLLECTION = "drawing"
VISION_COLLECTION = "drawingVision"


# ---------------------------------------------------------------------------
# count_equipment_tags
# ---------------------------------------------------------------------------

# Matches equipment tags like DOAS-1, AHU-3A, VAV-201, EF-B2, RTU-10
# Prefix 1-6 uppercase letters, optional dash, 1-4 digits, optional alpha suffix
_TAG_PATTERN = re.compile(r"\b([A-Z]{1,6})[-\s]?(\d{1,4})([A-Za-z]?)\b")

# Matches level/floor hints in a drawing title so we can bucket counts by level
_LEVEL_PATTERNS = [
    re.compile(r"\b(LEVEL|LVL|L|FLOOR|FL)\s*-?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*(?:ST|ND|RD|TH)?\s*FLOOR\b", re.IGNORECASE),
    re.compile(r"\bL(\d{1,3})\b"),
    re.compile(r"\b(\d{1,3})[ABCD]?\b"),  # weak fallback
]


def _infer_level(text: str) -> Optional[int]:
    """Best-effort level extraction from a drawing title or name."""
    if not text:
        return None
    for pat in _LEVEL_PATTERNS[:3]:  # skip weak fallback unless desperate
        m = pat.search(text)
        if m:
            try:
                return int(m.group(2) if m.lastindex == 2 else m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _keyword_regex(keywords: Iterable[str]) -> re.Pattern[str]:
    parts = [re.escape(k.strip()) for k in keywords if k and k.strip()]
    if not parts:
        raise ValueError("keywords must not be empty")
    return re.compile(r"(?:" + "|".join(parts) + r")", re.IGNORECASE)


def count_equipment_tags(
    project_id: int,
    keywords: List[str],
    level_filter: Optional[int] = None,
    drawing_title_filter: Optional[str] = None,
    max_fragments: int = 2000,
) -> Dict[str, Any]:
    """Count distinct equipment tags matching *keywords* on this project's drawings.

    The count is deterministic: we pull every drawing fragment whose text
    contains one of the keywords, extract tag patterns like ``DOAS-1`` with a
    single regex, and deduplicate. No LLM is involved. The agent receives the
    unique tag list plus breakdowns so it can report faithfully.

    Parameters
    ----------
    project_id : int
    keywords : list[str]
        Keywords identifying the equipment family. Must include variants the
        agent expects to see, e.g. ``["DOAS", "Dedicated Outdoor Air"]``.
    level_filter : int, optional
        Only count tags that appear on drawings whose title/name implies this
        level number.
    drawing_title_filter : str, optional
        Optional regex substring on ``drawingTitle`` to narrow scope (e.g.
        ``"PLUMBING"``).
    max_fragments : int
        Safety cap on fragments scanned.
    """
    project_id = validate_project_id(project_id)
    if not keywords:
        raise ValueError("keywords required")
    coll = get_collection(DRAWING_COLLECTION)

    kw_re = _keyword_regex(keywords)
    query: Dict[str, Any] = {"projectId": project_id, "text": kw_re}
    if drawing_title_filter:
        query["drawingTitle"] = re.compile(re.escape(drawing_title_filter), re.IGNORECASE)

    projection = {
        "text": 1, "drawingTitle": 1, "drawingName": 1, "pdfName": 1,
        "page": 1, "s3BucketPath": 1, "_id": 0,
    }
    fragments = list(coll.find(query, projection).limit(max_fragments))

    # Extract tags and organise
    unique_tags: dict[str, dict[str, Any]] = {}
    by_drawing: dict[str, int] = defaultdict(int)
    by_level: dict[int, set[str]] = defaultdict(set)
    samples_by_tag: dict[str, list[str]] = defaultdict(list)
    source_records: dict[tuple, dict[str, Any]] = {}  # for source extraction

    for frag in fragments:
        text = (frag.get("text") or "").strip()
        if not text:
            continue
        level = _infer_level(frag.get("drawingTitle") or "") or _infer_level(
            frag.get("drawingName") or ""
        )
        if level_filter is not None and level != level_filter:
            continue

        matched_tag = False
        # Only count tag matches whose PREFIX was in the keyword list.
        for match in _TAG_PATTERN.finditer(text):
            prefix, number, suffix = match.group(1), match.group(2), match.group(3)
            full_tag = f"{prefix}-{number}{suffix}".upper()
            # Require the PREFIX itself to be present (case-insensitive) in keywords
            if not any(prefix.upper() == kw.upper() or kw.upper() in prefix.upper() for kw in keywords):
                continue
            matched_tag = True
            unique_tags.setdefault(full_tag, {
                "tag": full_tag,
                "pdfName": frag.get("pdfName", ""),
                "drawingName": frag.get("drawingName", ""),
                "drawingTitle": frag.get("drawingTitle", ""),
                "first_page": frag.get("page"),
                "s3BucketPath": frag.get("s3BucketPath", ""),
            })
            drawing_key = frag.get("drawingTitle") or frag.get("drawingName") or frag.get("pdfName") or "unknown"
            by_drawing[drawing_key] += 1
            if level is not None:
                by_level[level].add(full_tag)
            if len(samples_by_tag[full_tag]) < 3:
                samples_by_tag[full_tag].append(text[:200])

        # Record the source fragment so _extract_sources picks it up
        if matched_tag:
            key = (
                frag.get("pdfName") or "",
                frag.get("drawingName") or "",
                frag.get("drawingTitle") or "",
            )
            source_records.setdefault(key, {
                "pdfName": frag.get("pdfName", ""),
                "drawingName": frag.get("drawingName", ""),
                "drawingTitle": frag.get("drawingTitle", ""),
                "s3BucketPath": frag.get("s3BucketPath", ""),
                "page": frag.get("page"),
            })

    # 'results' key is recursed by _extract_sources so tool output flows into
    # the final response's source_documents naturally.
    results_for_extraction = list(source_records.values())

    result = {
        "total_unique_tags": len(unique_tags),
        "unique_tags": sorted(unique_tags.keys()),
        "tag_details": list(unique_tags.values()),
        "by_drawing": [
            {"drawing": k, "mentions": v} for k, v in sorted(by_drawing.items(), key=lambda x: -x[1])[:20]
        ],
        "by_level": [
            {"level": lvl, "tags": sorted(list(tags))} for lvl, tags in sorted(by_level.items())
        ],
        "samples": {
            tag: samples for tag, samples in list(samples_by_tag.items())[:8]
        },
        "results": results_for_extraction,
        "fragments_scanned": len(fragments),
        "level_filter_applied": level_filter,
        "keywords_used": keywords,
    }
    logger.info(
        "count_equipment_tags: project=%d kw=%s level=%s tags=%d fragments=%d",
        project_id, keywords, level_filter, len(unique_tags), len(fragments),
    )
    return result


# ---------------------------------------------------------------------------
# find_typical_levels
# ---------------------------------------------------------------------------

_TYPICAL_HINT_PATTERNS = [
    # "3 THRU 6"
    re.compile(r"\b(\d{1,3})\s*THRU\s*(\d{1,3})\b", re.IGNORECASE),
    # "TYPICAL LEVELS 3-6" / "TYPICAL FLOORS 3 to 6"
    re.compile(r"\bTYPICAL\s+(?:LEVELS?|FLOORS?)\s+(\d{1,3})\s*(?:-|to|thru)\s*(\d{1,3})\b", re.IGNORECASE),
    # "FLOORS 3-6 TYPICAL"
    re.compile(r"\bFLOORS?\s+(\d{1,3})\s*-\s*(\d{1,3})\s+TYPICAL\b", re.IGNORECASE),
]


def _normalise_title_for_clustering(title: str) -> str:
    """Strip level numbers from a drawing title so 3RD-FLOOR-PLAN and
    5TH-FLOOR-PLAN cluster together."""
    if not title:
        return ""
    t = title.upper()
    # Drop ordinal-level phrases
    t = re.sub(r"\b\d+\s*(ST|ND|RD|TH)?\s*FLOOR\b", "FLOOR", t)
    t = re.sub(r"\bLEVEL\s*\d+\b", "LEVEL", t)
    t = re.sub(r"\bL\d+\b", "L", t)
    # Strip punctuation + extra whitespace
    t = re.sub(r"[\W_]+", " ", t).strip()
    # Collapse multi-space
    t = re.sub(r"\s+", " ", t)
    return t


def find_typical_levels(
    project_id: int,
    set_id: Optional[int] = None,
    min_cluster_size: int = 2,
) -> Dict[str, Any]:
    """Group the project's drawings by normalised title and return which
    levels share the same plan group ("typical" floors) vs which stand alone.

    Also inspects text fragments for explicit "3 THRU 6" typical hints.

    Parameters
    ----------
    min_cluster_size : int
        A plan group must contain at least this many distinct levels before we
        call it typical. Default 2.
    """
    project_id = validate_project_id(project_id)
    coll = get_collection(DRAWING_COLLECTION)

    match: Dict[str, Any] = {"projectId": project_id}
    if set_id is not None:
        match["setId"] = int(set_id)

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": {"title": "$drawingTitle", "name": "$drawingName"},
            "fragment_count": {"$sum": 1},
            "pdfName": {"$first": "$pdfName"},
            "s3BucketPath": {"$first": "$s3BucketPath"},
        }},
    ]
    unique_drawings = list(coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=20000))

    # Cluster by normalised title
    cluster_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    level_to_cluster: dict[int, str] = {}
    all_observed_levels: set[int] = set()
    for row in unique_drawings:
        title = row["_id"].get("title") or ""
        name = row["_id"].get("name") or ""
        level = _infer_level(title) or _infer_level(name)
        if level is None:
            continue
        all_observed_levels.add(level)
        norm = _normalise_title_for_clustering(title) or _normalise_title_for_clustering(name)
        if not norm:
            continue
        cluster_members[norm].append({
            "level": level,
            "drawing_title": title,
            "drawing_name": name,
            "pdf_name": row.get("pdfName"),
            "s3BucketPath": row.get("s3BucketPath", ""),
            "fragments": row.get("fragment_count", 0),
        })

    typical_groups: list[dict[str, Any]] = []
    unique_singletons: list[dict[str, Any]] = []
    for norm, members in cluster_members.items():
        levels_in_cluster = sorted({m["level"] for m in members})
        if len(levels_in_cluster) >= min_cluster_size:
            typical_groups.append({
                "title_pattern": norm,
                "levels": levels_in_cluster,
                "sample_drawings": [m["drawing_title"] for m in members[:3]],
            })
            for lvl in levels_in_cluster:
                level_to_cluster[lvl] = norm
        else:
            for m in members:
                unique_singletons.append(m)

    # Scavenge text fragments for explicit "THRU" hints
    hinted_groups: list[Tuple[int, int]] = []
    for pat in _TYPICAL_HINT_PATTERNS:
        hits = coll.find(
            {"projectId": project_id, "text": pat},
            {"text": 1, "_id": 0},
        ).limit(200)
        for h in hits:
            text = h.get("text", "")
            for m in pat.finditer(text):
                try:
                    a, b = int(m.group(1)), int(m.group(2))
                    if 0 < a <= b <= 200:
                        hinted_groups.append((a, b))
                except (ValueError, IndexError):
                    pass

    # Dedupe hinted ranges
    hinted_groups = sorted(set(hinted_groups))

    # Levels that never appear in any typical group
    standalone_levels = sorted(
        lvl for lvl in all_observed_levels if lvl not in level_to_cluster
    )

    # Build a flat list of contributing drawings so _extract_sources picks
    # them up via the standard 'results' recursion path.
    source_records: list[dict[str, Any]] = []
    for members in cluster_members.values():
        for m in members:
            source_records.append({
                "pdfName": m.get("pdf_name", ""),
                "drawingName": m.get("drawing_name", ""),
                "drawingTitle": m.get("drawing_title", ""),
                "s3BucketPath": m.get("s3BucketPath", ""),
            })

    result = {
        "typical_groups": sorted(typical_groups, key=lambda g: g["levels"]),
        "standalone_levels": standalone_levels,
        "explicit_typical_hints": [
            {"range_start": a, "range_end": b} for a, b in hinted_groups
        ],
        "observed_levels": sorted(all_observed_levels),
        "total_unique_drawings": len(unique_drawings),
        "results": source_records[:20],
    }
    logger.info(
        "find_typical_levels: project=%d typical_groups=%d singletons=%d hints=%d",
        project_id, len(typical_groups), len(unique_singletons), len(hinted_groups),
    )
    return result


# ---------------------------------------------------------------------------
# list_schedule_entries
# ---------------------------------------------------------------------------

_SCHEDULE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "doas": ["DOAS", "Dedicated Outdoor Air"],
    "ahu": ["AHU", "Air Handling Unit", "Air Handler"],
    "vav": ["VAV", "Variable Air Volume"],
    "rtu": ["RTU", "Rooftop Unit"],
    "fan": ["FAN", "EF-", "SF-", "RF-", "TF-"],
    "pump": ["PUMP", "P-", "CP-"],
    "valve": ["VALVE", "V-", "BV-", "GV-", "CV-"],
    "plumbing_fixture": ["FIXTURE", "PLUMBING FIXTURE", "WC-", "LAV-"],
    "panelboard": ["PANELBOARD", "PANEL"],
    "chiller": ["CHILLER", "CH-"],
}


def list_schedule_entries(
    project_id: int,
    schedule_type: str,
    level_filter: Optional[int] = None,
    max_rows: int = 50,
) -> Dict[str, Any]:
    """Pull schedule entries matching *schedule_type* from VisionOCR pages.

    Strategy:
    1. Try ``drawingVision.pages.vision_elements`` first — those are
       structurally extracted schedule rows when VisionOCR had a good page.
    2. Fall back to ``drawing`` collection text fragments, reusing the
       ``count_equipment_tags`` pattern to pull unique tags of the requested
       schedule family.
    """
    project_id = validate_project_id(project_id)
    schedule_key = (schedule_type or "").strip().lower()
    keywords = _SCHEDULE_TYPE_KEYWORDS.get(schedule_key)
    if not keywords:
        # Allow raw user keyword too
        validate_search_text(schedule_type)
        keywords = [schedule_type]

    # Path 1: VisionOCR
    vision_rows: List[Dict[str, Any]] = []
    try:
        vision = get_collection(VISION_COLLECTION)
        kw_re = _keyword_regex(keywords)
        docs = vision.find(
            {
                "projectId": project_id,
                "$or": [
                    {"pages.page_summary": kw_re},
                    {"pages.key_notes": kw_re},
                    {"pages.vision_elements.label": kw_re},
                    {"pages.vision_elements.text": kw_re},
                ],
            },
            {
                "sourceFile": 1, "pages.sheet_number": 1,
                "pages.page_summary": 1, "pages.vision_elements": 1, "_id": 0,
            },
        ).limit(max_rows)
        for d in docs:
            pages = d.get("pages") or []
            if not pages:
                continue
            page = pages[0]
            for elem in (page.get("vision_elements") or [])[:20]:
                label = str(elem.get("label") or elem.get("text") or "")
                if kw_re.search(label):
                    vision_rows.append({
                        "source_file": d.get("sourceFile", ""),
                        "sheet_number": page.get("sheet_number", ""),
                        "label": label[:120],
                        "attributes": {
                            k: v for k, v in elem.items()
                            if k not in ("label", "text") and isinstance(v, (str, int, float, bool))
                        },
                    })
    except Exception as exc:
        logger.debug("list_schedule_entries: vision path failed %s", exc)

    # Path 2: drawing fallback (re-uses tag extraction)
    fallback = count_equipment_tags(
        project_id=project_id,
        keywords=keywords,
        level_filter=level_filter,
    ) if not vision_rows else None

    result = {
        "schedule_type": schedule_type,
        "keywords_used": keywords,
        "vision_rows": vision_rows[:max_rows],
        "fallback_tag_extraction": fallback,
        "source": "vision" if vision_rows else "drawing_ocr_fallback",
    }
    logger.info(
        "list_schedule_entries: project=%d type=%s vision_rows=%d fallback=%s",
        project_id, schedule_type, len(vision_rows),
        (fallback or {}).get("total_unique_tags"),
    )
    return result
