"""
Drawing retrieval tools for the v3 collection (``drawings_v3``).

These tools are dual-purpose:
  1. **Semantic search** via the Atlas Vector Search index
     ``drawings_v3_vec_idx`` (text-embedding-3-small, cosine).
  2. **Structured field lookup** exploiting v3's per-page rich schema:
     ``schedules``, ``symbols``, ``pageSummary``, ``fullText``,
     ``discipline``, ``drawingType``, ``csiDivisions``, ``titleBlock``.

All tools return dicts with the legacy field names the agent's
``_extract_sources`` expects (``drawingName``, ``drawingTitle``, ``pdfName``,
``s3BucketPath``, ``page``, ``sheet_number``) so source-doc normalization
downstream "just works" without touching ``agent.py`` further. They also
surface v3-only fields (``pageSummary``, ``schedules``, ``symbols``) so the
agent can short-circuit reasoning when structured data is present.

NO Mongo writes from these tools — read-only.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id, validate_search_text
from tools.hybrid_retrieval import maybe_fuse_with_keyword
from tools.block_promotion import fold_blocks_into  # Layer 2 (env-gated)

logger = logging.getLogger("agentic_rag.tools.drawings_v3")

COLLECTION = "drawings_v3"
VECTOR_INDEX = "drawings_v3_vec_idx"
EMBEDDING_MODEL = "text-embedding-3-small"

# Projection — fields we return to the agent on every retrieval. Keep small
# enough that even a top-20 hit list stays under the agent's tool-result
# truncation threshold (~14KB).
_LIGHT_PROJECTION: Dict[str, Any] = {
    "_id": 0,
    "drawingName": 1, "drawingTitle": 1, "sheetNumber": 1, "sheet_number": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "discipline": 1, "drawingType": 1, "csiDivisions": 1,
    "trade": 1, "tradeId": 1, "setId": 1, "setTrade": 1,
    "drawingId": 1,
    # v3-specific quick hints (do NOT include full textBlocks here — too big)
    "pageSummary": 1,
    "schedulesCount": 1, "symbolsCount": 1, "notesCount": 1, "textLength": 1,
    "scale": 1, "level": 1, "revision": 1,
}

# Heavy projection — used when caller wants the full content of one drawing
_FULL_PROJECTION: Dict[str, Any] = {
    "_id": 0, "embedding": 0,  # vector is huge and not useful to the agent
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _openai_client():
    """Lazy OpenAI client init — kept module-level cached for reuse."""
    from openai import OpenAI
    return OpenAI()


def _embed_query(query: str) -> List[float]:
    """Embed a single user query → 1536-dim vector."""
    resp = _openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return resp.data[0].embedding


def _flatten_for_agent(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the dict has BOTH camelCase and snake_case keys the agent's
    ``_extract_sources`` looks at. v3 stores both ``sheetNumber`` and
    ``sheet_number`` already, but be defensive."""
    if not isinstance(doc, dict):
        return doc
    if "sheet_number" not in doc and doc.get("sheetNumber"):
        doc["sheet_number"] = doc["sheetNumber"]
    if "drawingName" not in doc and doc.get("sheet_number"):
        doc["drawingName"] = doc["sheet_number"]
    return doc


# ---------------------------------------------------------------------------
# Public tools (the agent calls these)
# ---------------------------------------------------------------------------


def search_drawings_v3(
    project_id: int,
    search_text: str,
    limit: int = 10,
    discipline: Optional[str] = None,
    drawing_type: Optional[str] = None,
    set_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Semantic search over drawings_v3 via vector index.

    Parameters
    ----------
    project_id : int
        Project scope (required).
    search_text : str
        Natural-language query. Embedded then matched against the
        ``embedding`` field via $vectorSearch.
    limit : int
        Top-K hits to return (default 10, max 25 enforced).
    discipline : str, optional
        Pre-filter — e.g. ``"Architecture"``, ``"Mechanical"``, ``"Civil"``.
    drawing_type : str, optional
        Pre-filter — e.g. ``"floor_plan"``, ``"schedule"``,
        ``"reflected_ceiling_plan"``, ``"site_plan"``.
    set_id : int, optional
        Pre-filter — drawing set ID.

    Returns
    -------
    list of dict
        Top-K drawings ordered by cosine similarity. Each dict includes
        ``drawingName``, ``drawingTitle``, ``pdfName``, ``s3BucketPath``,
        ``pageSummary``, ``schedulesCount``, ``symbolsCount``, plus a
        ``relevance`` score in [0,1].
    """
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit, max_limit=25)
    coll = get_collection(COLLECTION)

    try:
        q_vec = _embed_query(search_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_drawings_v3: embed failed (%s); falling back to text search", exc)
        return _text_fallback(coll, project_id, search_text, limit, discipline, drawing_type)

    filt: Dict[str, Any] = {"projectId": project_id}
    if discipline:
        filt["discipline"] = discipline
    if drawing_type:
        filt["drawingType"] = drawing_type
    if set_id is not None:
        filt["setId"] = set_id

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": q_vec,
                "filter": filt,
                "limit": limit,
                "numCandidates": min(200, limit * 20),
            }
        },
        {
            "$project": {
                **_LIGHT_PROJECTION,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(coll.aggregate(pipeline, maxTimeMS=15000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_drawings_v3 vectorSearch failed (%s); falling back", exc)
        return _text_fallback(coll, project_id, search_text, limit, discipline, drawing_type)

    out = []
    for r in results:
        r["relevance"] = round(float(r.pop("score", 0.0)), 4)
        out.append(_flatten_for_agent(r))
    # === HYBRID RETRIEVAL (additive, env-gated; no-op when OFF) ===
    out = maybe_fuse_with_keyword(
        vector_hits=out,
        coll=coll,
        project_id=project_id,
        search_text=search_text,
        limit=limit,
        extra_filter=filt,
        projection=_LIGHT_PROJECTION,
    )
    # === LAYER 2 BLOCK PROMOTION (additive, env-gated) ===
    out = fold_blocks_into(
        page_hits=out,
        project_id=project_id,
        search_text=search_text,
        limit=limit,
        discipline=discipline,
        drawing_type=drawing_type,
        set_id=set_id,
    )
    logger.info(
        "search_drawings_v3: project=%d query=%r filters=%r hits=%d",
        project_id, search_text[:40], filt, len(out),
    )
    return out


def _text_fallback(coll, project_id: int, search_text: str, limit: int,
                   discipline: Optional[str], drawing_type: Optional[str]) -> List[Dict[str, Any]]:
    """Regex fallback when vector path fails (rare — embedding API outage)."""
    import re
    pat = re.compile(re.escape(search_text), re.IGNORECASE)
    query: Dict[str, Any] = {
        "projectId": project_id,
        "$or": [
            {"drawingTitle": pat},
            {"pageSummary": pat},
            {"reconstructedText": pat},
        ],
    }
    if discipline:
        query["discipline"] = discipline
    if drawing_type:
        query["drawingType"] = drawing_type
    results = list(coll.find(query, _LIGHT_PROJECTION).limit(limit))
    return [_flatten_for_agent(r | {"relevance": 0.5}) for r in results]


def get_drawing_by_sheet(
    project_id: int,
    sheet_number: str,
) -> Optional[Dict[str, Any]]:
    """Exact sheet lookup. Returns the WHOLE page doc (fullText, textBlocks,
    schedules, symbols, titleBlock) — use this when the user names a specific
    sheet like "A-101" or "M-301".

    Sheet matching is case-insensitive and accepts both ``sheetNumber`` and
    ``drawingName`` storage paths. Hyphens / spaces are normalized.
    """
    project_id = validate_project_id(project_id)
    if not sheet_number or not isinstance(sheet_number, str):
        return None
    coll = get_collection(COLLECTION)

    norm = sheet_number.strip().upper()
    norm_alt = norm.replace(" ", "-").replace("--", "-")
    candidates = {norm, norm_alt, norm.replace("-", ""), norm.replace("-", " ")}

    query = {
        "projectId": project_id,
        "$or": [
            {"sheetNumber": {"$in": list(candidates)}},
            {"sheet_number": {"$in": list(candidates)}},
            {"drawingName": {"$in": list(candidates)}},
        ],
    }
    doc = coll.find_one(query, _FULL_PROJECTION)
    if not doc:
        return None
    return _flatten_for_agent(doc)


def list_drawings_v3(
    project_id: int,
    discipline: Optional[str] = None,
    drawing_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Enumerate drawings in the project, optionally filtered by discipline
    or drawing type. Useful for the agent to discover what sheets exist.

    Returns lightweight per-drawing metadata sorted by sheetNumber.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=200)
    coll = get_collection(COLLECTION)

    query: Dict[str, Any] = {"projectId": project_id}
    if discipline:
        query["discipline"] = discipline
    if drawing_type:
        query["drawingType"] = drawing_type
    results = list(
        coll.find(query, _LIGHT_PROJECTION).sort("sheetNumber", 1).limit(limit)
    )
    return [_flatten_for_agent(r) for r in results]


def get_drawing_schedules(
    project_id: int,
    sheet_number: str,
) -> List[Dict[str, Any]]:
    """Return the parsed ``schedules[]`` array from a specific drawing.

    THIS IS THE KEY V3 WIN for schedule questions — instead of OCR-parsing
    tables out of text fragments, return the structured schedule data
    directly. Each schedule is a dict with type + rows.
    """
    doc = get_drawing_by_sheet(project_id, sheet_number)
    if not doc:
        return []
    schedules = doc.get("schedules") or []
    if not isinstance(schedules, list):
        return []
    logger.info(
        "get_drawing_schedules: project=%d sheet=%s found=%d schedules",
        project_id, sheet_number, len(schedules),
    )
    return schedules


def get_drawing_symbols(
    project_id: int,
    sheet_number: str,
    kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return parsed ``symbols[]`` from a drawing, optionally filtered by
    symbol kind (e.g. ``"door"``, ``"window"``, ``"diffuser"``,
    ``"receptacle"``).
    """
    doc = get_drawing_by_sheet(project_id, sheet_number)
    if not doc:
        return []
    symbols = doc.get("symbols") or []
    if not isinstance(symbols, list):
        return []
    if kind:
        kind_lc = kind.lower()
        symbols = [
            s for s in symbols
            if isinstance(s, dict) and kind_lc in (s.get("kind") or "").lower()
        ]
    logger.info(
        "get_drawing_symbols: project=%d sheet=%s kind=%s found=%d",
        project_id, sheet_number, kind, len(symbols),
    )
    return symbols


def get_drawing_full_text(
    project_id: int,
    sheet_number: str,
) -> Dict[str, Any]:
    """Return ``fullText`` + ``reconstructedText`` + ``pageSummary`` for
    deep content extraction. Used when the agent needs the entire page
    body to reason over.
    """
    doc = get_drawing_by_sheet(project_id, sheet_number)
    if not doc:
        return {}
    return {
        "drawingName": doc.get("drawingName") or doc.get("sheetNumber"),
        "drawingTitle": doc.get("drawingTitle"),
        "pageSummary": doc.get("pageSummary"),
        "fullText": doc.get("fullText"),
        "reconstructedText": doc.get("reconstructedText"),
        "textBlocks": (doc.get("textBlocks") or [])[:50],  # cap for token budget
        "titleBlock": doc.get("titleBlock"),
        "scale": doc.get("scale"),
        "revision": doc.get("revision"),
        "pdfName": doc.get("pdfName"),
        "s3BucketPath": doc.get("s3BucketPath"),
    }


def find_drawing_with_most_symbols(
    project_id: int,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Direct structured query — answers questions like "which drawing
    has the most symbols?" deterministically. v1 had to guess via OCR text;
    v3 just sorts by ``symbolsCount``.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=20)
    coll = get_collection(COLLECTION)
    results = list(
        coll.find(
            {"projectId": project_id, "symbolsCount": {"$gt": 0}},
            _LIGHT_PROJECTION,
        ).sort("symbolsCount", -1).limit(limit)
    )
    return [_flatten_for_agent(r) for r in results]


def find_drawings_with_schedules(
    project_id: int,
    schedule_type_hint: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Find drawings that have schedules. Optional fuzzy filter by
    schedule type hint (e.g. "door", "window", "panel", "duct") — matched
    against ``drawingTitle`` since schedule type isn't always stored
    structured.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=50)
    coll = get_collection(COLLECTION)
    query: Dict[str, Any] = {
        "projectId": project_id,
        "schedulesCount": {"$gt": 0},
    }
    if schedule_type_hint:
        import re
        pat = re.compile(re.escape(schedule_type_hint), re.IGNORECASE)
        query["$or"] = [
            {"drawingTitle": pat},
            {"pageSummary": pat},
        ]
    results = list(
        coll.find(query, _LIGHT_PROJECTION).sort("schedulesCount", -1).limit(limit)
    )
    return [_flatten_for_agent(r) for r in results]
