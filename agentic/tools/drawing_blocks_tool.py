"""Block-level retrieval tool for the drawings_v3_blocks collection (Layer 2).

This tool is ADDITIVE. It does NOT replace any existing tool. It is only
registered with the agent when the env flag ``ENABLE_BLOCK_RETRIEVAL=true``.

Schema produced is intentionally compatible with the existing agent's
``_extract_sources`` plumbing: each result includes ``drawingName``,
``drawingTitle``, ``sheetNumber``, ``sheet_number``, ``pdfName``,
``s3BucketPath`` plus the new ``blockText`` field so the LLM can see the
specific note that matched.

NO Mongo writes from this tool — read-only.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id, validate_search_text
from tools.hybrid_retrieval import maybe_fuse_with_keyword

logger = logging.getLogger("agentic_rag.tools.blocks_v3")

COLLECTION = "drawings_v3_blocks"
VECTOR_INDEX = "drawings_v3_blocks_vec_idx"
EMBEDDING_MODEL = "text-embedding-3-small"

_LIGHT_PROJECTION: Dict[str, Any] = {
    "_id": 0,
    "blockId": 1,
    "blockKind": 1, "blockIdx": 1,
    "blockText": 1, "blockTextLength": 1,
    "drawingName": 1, "drawingTitle": 1,
    "sheetNumber": 1, "sheet_number": 1,
    "pdfName": 1, "s3BucketPath": 1, "page": 1,
    "discipline": 1, "drawingType": 1, "csiDivisions": 1,
    "trade": 1, "tradeId": 1, "setId": 1, "setTrade": 1,
    "drawingId": 1, "level": 1, "scale": 1, "revision": 1,
}


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI()


def _embed_query(query: str) -> List[float]:
    resp = _openai_client().embeddings.create(model=EMBEDDING_MODEL, input=query)
    return resp.data[0].embedding


def _flatten_for_agent(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        return doc
    if "sheet_number" not in doc and doc.get("sheetNumber"):
        doc["sheet_number"] = doc["sheetNumber"]
    if "drawingName" not in doc and doc.get("sheet_number"):
        doc["drawingName"] = doc["sheet_number"]
    # Truncate blockText preview so candidate lists stay small
    bt = doc.get("blockText")
    if isinstance(bt, str) and len(bt) > 400:
        doc["blockText"] = bt[:400]
    return doc


def search_drawing_blocks_v3(
    project_id: int,
    search_text: str,
    limit: int = 10,
    discipline: Optional[str] = None,
    sheet_number: Optional[str] = None,
    block_kind: Optional[str] = None,
    set_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Semantic search at the text-block level (one note or one annotation
    per indexed document). Useful for queries where the answer is a single
    callout buried inside a busy drawing.

    Parameters
    ----------
    project_id : int
        Project scope (required).
    search_text : str
        Natural-language query, embedded for vector search.
    limit : int
        Max hits to return (default 10, capped at 25).
    discipline : str, optional
        Pre-filter — e.g. ``"Plumbing"``, ``"Sprinkler"``, ``"Electrical"``.
    sheet_number : str, optional
        Pre-filter — exact sheet match, e.g. ``"P-100A"``, ``"FP-100"``.
    block_kind : str, optional
        Pre-filter — ``"textBlock"`` or ``"note"``.
    set_id : int, optional
        Pre-filter — drawing set ID.

    Returns
    -------
    list of dict
        Top-K block hits with ``blockText`` so the LLM can read the exact
        note that matched. Each dict also carries the parent drawing's
        metadata (``drawingName``, ``sheetNumber``, ``s3BucketPath``).
    """
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit, max_limit=25)
    coll = get_collection(COLLECTION)

    try:
        q_vec = _embed_query(search_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_drawing_blocks_v3: embed failed (%s)", exc)
        return []

    filt: Dict[str, Any] = {"projectId": project_id}
    if discipline:
        filt["discipline"] = discipline
    if sheet_number:
        filt["sheetNumber"] = sheet_number
    if block_kind in ("textBlock", "note"):
        filt["blockKind"] = block_kind
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
                "numCandidates": min(400, limit * 25),
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
        logger.warning("search_drawing_blocks_v3 vectorSearch failed (%s)", exc)
        return []

    out: List[Dict[str, Any]] = []
    for r in results:
        r["relevance"] = round(float(r.pop("score", 0.0)), 4)
        out.append(_flatten_for_agent(r))

    # Optional dedup: keep top-1 per sheetNumber so the LLM sees diverse
    # drawings instead of multiple blocks from the same sheet flooding the
    # candidate list. Controlled by env flag (default ON for blocks).
    if os.getenv("BLOCK_DEDUPE_BY_SHEET", "true").strip().lower() in ("1", "true", "yes", "on"):
        seen, dedup = set(), []
        for r in out:
            sn = r.get("sheetNumber") or r.get("drawingName") or r.get("blockId")
            if sn in seen:
                continue
            seen.add(sn)
            dedup.append(r)
        out = dedup

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
        "search_drawing_blocks_v3: project=%d query=%r filters=%r hits=%d",
        project_id, search_text[:50], filt, len(out),
    )
    return out
