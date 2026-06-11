"""
Specification retrieval tools for the v3 collection (``specifications_v3``).

Same dual-purpose pattern as drawing_tools_v3:
  1. **Semantic search** via ``specifications_v3_vec_idx`` (Atlas Vector
     Search, cosine similarity, text-embedding-3-small).
  2. **Structured lookup** by CSI number / section title / division.

KEY V3 WIN OVER V1:
  - v1 ``specification`` fragments a single spec section into 6-12 separate
    Mongo docs that share ``parentId``. Retrieval surfaces all of them and
    the agent stitches the text together — error-prone.
  - v3 ``specifications_v3`` has **ONE doc per logical section** with the
    full consolidated ``fullText`` (mean ~13K chars) and structured
    ``submittalsStructured`` / ``warrantiesStructured`` fields. The agent
    gets the whole section in one tool call. Goodbye fragmentation.

Read-only — no Mongo writes.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id, validate_search_text
from tools.hybrid_retrieval import maybe_fuse_with_keyword

logger = logging.getLogger("agentic_rag.tools.specs_v3")

COLLECTION = "specifications_v3"
VECTOR_INDEX = "specifications_v3_vec_idx"
EMBEDDING_MODEL = "text-embedding-3-small"

# Light projection — for list/search results (full text excluded)
_LIGHT_PROJECTION: Dict[str, Any] = {
    "_id": 0,
    "sectionTitle": 1, "specificationNumber": 1,
    "csi": 1, "csiDivision": 1, "drawingName": 1, "drawingTitle": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1, "pageCount": 1,
    "parentId": 1, "docNumber": 1,
    "setId": 1, "specSetId": 1, "specSetName": 1, "tradeId": 1,
    "textLength": 1,
    "sectionText": 1,  # short preview (1-3 lines)
}

# Heavy projection — caller wants the whole section
_FULL_PROJECTION: Dict[str, Any] = {
    "_id": 0, "embedding": 0,  # vector is huge, not for agent
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI()


def _embed_query(query: str) -> List[float]:
    resp = _openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return resp.data[0].embedding


def _normalize_csi(csi: str) -> List[str]:
    """Generate plausible variants of a CSI number: '23 07 19', '230719',
    '23-07-19' all refer to the same section.
    """
    if not csi:
        return []
    raw = csi.strip().upper()
    no_space = raw.replace(" ", "").replace("-", "").replace(".", "")
    return list({raw, no_space, raw.replace(" ", "-")})


def _flatten_for_agent(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure agent's ``_extract_sources`` finds the expected keys. v3 specs
    don't have ``drawingName`` on every doc — populate from ``csi`` if
    missing so display chains downstream don't break.
    """
    if not isinstance(doc, dict):
        return doc
    if not doc.get("drawingName") and doc.get("csi"):
        doc["drawingName"] = doc["csi"]
    if not doc.get("drawingTitle") and doc.get("sectionTitle"):
        doc["drawingTitle"] = doc["sectionTitle"]
    return doc


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def search_specifications_v3(
    project_id: int,
    search_text: str,
    limit: int = 5,
    csi_division: Optional[str] = None,
    set_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Semantic search across spec sections.

    Parameters
    ----------
    project_id : int
        Project scope.
    search_text : str
        Natural-language query — embedded and matched against
        the consolidated ``fullText`` of each spec section.
    limit : int
        Top-K (default 5, max 15). Each hit IS a whole section, so 5
        is usually enough — no need for 20+ fragments.
    csi_division : str, optional
        Pre-filter — e.g. ``"23"`` for HVAC, ``"26"`` for Electrical,
        ``"22"`` for Plumbing.
    set_id : int, optional
        Pre-filter — spec set ID.

    Returns
    -------
    list of dict
        Each entry has ``sectionTitle``, ``csi``, ``csiDivision``,
        ``pdfName``, ``s3BucketPath``, ``sectionText`` (1-3 line preview),
        plus ``relevance`` in [0, 1].
    """
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit, max_limit=15)
    coll = get_collection(COLLECTION)

    try:
        q_vec = _embed_query(search_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_specifications_v3: embed failed (%s); regex fallback", exc)
        return _text_fallback(coll, project_id, search_text, limit, csi_division)

    filt: Dict[str, Any] = {"projectId": project_id}
    if csi_division:
        filt["csiDivision"] = str(csi_division).strip()
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
                "numCandidates": min(150, limit * 30),
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
        logger.warning("search_specifications_v3 vectorSearch failed (%s)", exc)
        return _text_fallback(coll, project_id, search_text, limit, csi_division)

    out = []
    for r in results:
        r["relevance"] = round(float(r.pop("score", 0.0)), 4)
        # Trim sectionText preview to first ~300 chars for the result-list view
        if r.get("sectionText"):
            r["sectionText"] = (r["sectionText"] or "")[:300]
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
    logger.info(
        "search_specifications_v3: project=%d query=%r filter=%r hits=%d",
        project_id, search_text[:40], filt, len(out),
    )
    return out


def _text_fallback(coll, project_id: int, search_text: str, limit: int,
                   csi_division: Optional[str]) -> List[Dict[str, Any]]:
    """Regex fallback used only when embedding API is unreachable."""
    pat = re.compile(re.escape(search_text), re.IGNORECASE)
    query: Dict[str, Any] = {
        "projectId": project_id,
        "$or": [
            {"sectionTitle": pat},
            {"sectionText": pat},
            {"fullText": pat},
        ],
    }
    if csi_division:
        query["csiDivision"] = str(csi_division).strip()
    results = list(coll.find(query, _LIGHT_PROJECTION).limit(limit))
    return [_flatten_for_agent(r | {"relevance": 0.5}) for r in results]


def get_spec_by_csi(
    project_id: int,
    csi_number: str,
) -> Optional[Dict[str, Any]]:
    """Exact lookup by CSI number — accepts ``"23 07 19"``, ``"230719"``,
    or ``"23-07-19"`` formats. Returns the FULL section doc (with
    consolidated ``fullText``).
    """
    project_id = validate_project_id(project_id)
    if not csi_number:
        return None
    coll = get_collection(COLLECTION)

    variants = _normalize_csi(csi_number)
    query = {
        "projectId": project_id,
        "$or": [
            {"csi": {"$in": variants}},
            {"specificationNumber": {"$in": variants}},
        ],
    }
    doc = coll.find_one(query, _FULL_PROJECTION)
    if not doc:
        return None
    return _flatten_for_agent(doc)


def get_full_spec_section(
    project_id: int,
    section_title: Optional[str] = None,
    csi: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the FULL consolidated section text (no fragmentation).

    Prefer csi when both are given. Falls back to fuzzy section_title.
    """
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)

    if csi:
        return get_spec_by_csi(project_id, csi)
    if not section_title:
        return None
    pat = re.compile(re.escape(section_title.strip()), re.IGNORECASE)
    doc = coll.find_one(
        {"projectId": project_id, "sectionTitle": pat},
        _FULL_PROJECTION,
    )
    return _flatten_for_agent(doc) if doc else None


def list_specifications_v3(
    project_id: int,
    csi_division: Optional[str] = None,
    set_id: Optional[int] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Enumerate spec sections, optionally filtered by CSI division.
    Useful for the agent to discover what specs exist.
    """
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=200)
    coll = get_collection(COLLECTION)
    query: Dict[str, Any] = {"projectId": project_id}
    if csi_division:
        query["csiDivision"] = str(csi_division).strip()
    if set_id is not None:
        query["setId"] = set_id
    results = list(
        coll.find(query, _LIGHT_PROJECTION).sort("csi", 1).limit(limit)
    )
    return [_flatten_for_agent(r) for r in results]


def get_spec_submittals(
    project_id: int,
    csi: str,
) -> Any:
    """Return the structured submittals data for a spec section.
    v3 has ``submittalsStructured`` parsed during ingestion — direct
    structured access instead of text-parsing.
    """
    doc = get_spec_by_csi(project_id, csi)
    if not doc:
        return None
    return doc.get("submittalsStructured") or doc.get("submittals")


def get_spec_warranties(
    project_id: int,
    csi: str,
) -> Any:
    """Return the structured warranties data for a spec section."""
    doc = get_spec_by_csi(project_id, csi)
    if not doc:
        return None
    return doc.get("warrantiesStructured") or doc.get("warranties")
