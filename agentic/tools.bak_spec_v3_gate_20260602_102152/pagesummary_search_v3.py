"""
Tier B3 — pageSummary semantic search channel.

Embeds the query, runs Atlas Vector Search against
``drawings_v3_pagesummary_vec_idx`` (built over ``pageSummary_embedding``).

The pageSummary field is an LLM-generated dense semantic description of the
page (96-100% populated per 2026-05-27 audit). Searching it directly often
beats searching the textBlocks-derived ``embedding`` for high-level / topic
queries (e.g. "which drawing shows the demolition plan", "find architectural
floor plans").

Returns drawings with the SAME light shape as ``search_drawings_v3`` so
downstream source-doc extraction handles them without changes.

NO Mongo writes — read-only.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id, validate_search_text

logger = logging.getLogger("agentic_rag.tools.pagesummary_v3")

COLLECTION = "drawings_v3"
VECTOR_INDEX = "drawings_v3_pagesummary_vec_idx"
EMBEDDING_MODEL = "text-embedding-3-small"

_LIGHT_PROJECTION: Dict[str, Any] = {
    "_id": 0,
    "drawingName": 1, "drawingTitle": 1, "sheetNumber": 1, "sheet_number": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "discipline": 1, "drawingType": 1, "csiDivisions": 1,
    "trade": 1, "tradeId": 1, "setId": 1,
    "drawingId": 1,
    "pageSummary": 1,
    "symbolsCount": 1, "schedulesCount": 1,
    "titleBlock": 1,
}


@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI()


def _embed_query(text: str) -> List[float]:
    resp = _openai().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def search_drawings_by_summary_v3(
    project_id: int,
    search_text: str,
    limit: int = 10,
    discipline: Optional[str] = None,
    num_candidates: int = 200,
) -> List[Dict[str, Any]]:
    """Semantic search drawings via pageSummary embedding (LLM-generated
    page descriptions). Complements ``search_drawings_v3`` (which embeds
    a different text source).

    Use for HIGH-LEVEL / TOPIC queries:
      - "which drawings show the demolition plan"
      - "find roof structural framing drawings"
      - "drawings about HVAC system layout"

    For specific NEEDLE-IN-HAYSTACK text queries, prefer ``search_drawings_v3``
    or ``search_drawing_blocks_v3``.
    """
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit, max_limit=50)
    num_candidates = max(limit * 5, min(num_candidates, 1000))

    try:
        qvec = _embed_query(search_text)
    except Exception as exc:
        logger.warning("search_drawings_by_summary_v3: embed failed (%s)", exc)
        return []

    pre_filter: Dict[str, Any] = {"projectId": {"$eq": project_id}}
    if discipline:
        pre_filter["discipline"] = {"$eq": discipline.strip()}

    coll = get_collection(COLLECTION)
    try:
        pipeline: List[Dict[str, Any]] = [
            {"$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "pageSummary_embedding",
                "queryVector": qvec,
                "numCandidates": num_candidates,
                "limit": limit,
                "filter": pre_filter,
            }},
            {"$project": {**_LIGHT_PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
        ]
        results = list(coll.aggregate(pipeline))
    except Exception as exc:
        logger.warning("search_drawings_by_summary_v3 vectorSearch failed (%s)", exc)
        return []

    logger.info(
        "search_drawings_by_summary_v3: project=%d query=%r discipline=%r hits=%d",
        project_id, search_text, discipline, len(results),
    )
    return results
