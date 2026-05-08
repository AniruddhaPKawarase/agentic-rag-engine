"""
MongoDB tools for the legacy drawing collection (2.8M OCR fragments).

Uses aggregation pipelines to reconstruct page-level context from
bounding-box fragments. Never returns raw fragments to the agent.

CRITICAL: The drawing collection has NO indexes beyond _id.
All queries MUST include projectId to limit scan scope.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from core.db import get_collection
from core.text_reconstruction import reconstruct_drawing_text
from tools.validation import validate_drawing_id, validate_limit, validate_project_id, validate_search_text

logger = logging.getLogger("agentic_rag.tools.drawing")

COLLECTION = "drawing"


def list_project_drawings(
    project_id: int,
    set_id: int = None,
    drawing_title: str = None,
    drawing_name: str = None,
) -> List[Dict]:
    """List unique drawings for a project with metadata."""
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)
    match = {"projectId": project_id}
    if set_id:
        match["setId"] = set_id
    if drawing_title:
        match["drawingTitle"] = re.compile(re.escape(drawing_title), re.IGNORECASE)
    if drawing_name:
        match["drawingName"] = re.compile(re.escape(drawing_name), re.IGNORECASE)

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": "$drawingId",
            "drawingTitle": {"$first": "$drawingTitle"},
            "drawingName": {"$first": "$drawingName"},
            "setTrade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
            "pdfName": {"$first": "$pdfName"},
            "s3BucketPath": {"$first": "$s3BucketPath"},
            "page_count": {"$addToSet": "$page"},
            "fragment_count": {"$sum": 1},
            "trades": {"$first": "$trades"},
            "csi_division": {"$first": "$csi_division"},
        }},
        {"$project": {
            "drawingId": "$_id",
            "drawingTitle": 1,
            "drawingName": 1,
            "setTrade": 1,
            "pdfName": 1,
            "s3BucketPath": 1,
            "page_count": {"$size": "$page_count"},
            "fragment_count": 1,
            "trades": 1,
            "csi_division": 1,
        }},
        {"$sort": {"drawingName": 1}},
        {"$limit": 200},
    ]

    results = list(coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=30000))
    logger.info(f"list_project_drawings: project={project_id}, found={len(results)}")
    return results


def get_drawing_text(
    project_id: int,
    drawing_id: int,
    page: int = None,
) -> Dict:
    """Get reconstructed text for a specific drawing.

    Also returns ``matching_fragments[]`` — top-K per-fragment data with
    x/y/width/height/page so downstream citation extractors can build
    bbox-precise highlights. Without these, source_documents end up with
    bbox_pt=null even when the underlying Mongo data has the coords.
    """
    project_id = validate_project_id(project_id)
    drawing_id = validate_drawing_id(drawing_id)
    coll = get_collection(COLLECTION)
    query: Dict[str, Any] = {"projectId": project_id, "drawingId": drawing_id}
    if page:
        query["page"] = page

    # Fetch fragments sorted server-side (avoids Python re-sort)
    # NOTE: width + height now projected so we can build full bbox in
    # _extract_sources → _build_source_doc.
    fragments = list(coll.find(
        query,
        {"text": 1, "x": 1, "y": 1, "width": 1, "height": 1, "page": 1,
         "drawingTitle": 1, "drawingName": 1, "trade": 1,
         "pdfName": 1, "s3BucketPath": 1, "_id": 0},
    ).sort([("page", 1), ("y", 1), ("x", 1)]).limit(5000))

    if not fragments:
        return {"error": f"No fragments found for drawingId={drawing_id}"}

    text = reconstruct_drawing_text(fragments, pre_sorted=True)
    title = fragments[0].get("drawingTitle", "Unknown")
    name = fragments[0].get("drawingName", "Unknown")
    trade = fragments[0].get("trade", "Unknown")

    # Surface a capped set of fragments with bbox+text. The downstream
    # citation extractor (_emit_fragment_docs in agentic/core/agent.py)
    # walks this list and emits one source_doc per fragment, each with
    # bbox_px / bbox_pt populated. Cap at 50 so the LLM context isn't
    # flooded.
    top_fragments = []
    for f in fragments[:50]:
        if not f.get("text"):
            continue
        # Skip ultra-short fragments (page numbers, single chars) that
        # would clutter highlights without adding citation value.
        if len(f.get("text", "").strip()) < 4:
            continue
        top_fragments.append({
            "text":   f.get("text"),
            "x":      f.get("x"),
            "y":      f.get("y"),
            "width":  f.get("width"),
            "height": f.get("height"),
            "page":   f.get("page"),
        })

    return {
        "drawingId": drawing_id,
        "drawingTitle": title,
        "drawingName": name,
        "trade": trade,
        "pdfName": fragments[0].get("pdfName", ""),
        "s3BucketPath": fragments[0].get("s3BucketPath", ""),
        "fragment_count": len(fragments),
        "reconstructed_text": text[:30000],
        "text_length": len(text),
        # Citation highlight feed — consumed by _extract_sources
        "matching_fragments": top_fragments,
        "page": fragments[0].get("page"),
    }


def search_drawing_text(
    project_id: int,
    search_text: str,
    limit: int = 10,
    drawing_title: str = None,
    drawing_name: str = None,
) -> List[Dict]:
    """Search drawing fragments for specific text content."""
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit)
    coll = get_collection(COLLECTION)
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)

    match_stage = {"projectId": project_id, "text": pattern}
    if drawing_title:
        match_stage["drawingTitle"] = re.compile(re.escape(drawing_title), re.IGNORECASE)
    if drawing_name:
        match_stage["drawingName"] = re.compile(re.escape(drawing_name), re.IGNORECASE)

    # Aggregation: group by drawing for the LLM's narrative summary, but ALSO
    # keep the top fragment-level matches (with x/y/width/height) so the
    # downstream citation extractor has bbox data for highlight rendering.
    pipeline = [
        {"$match": match_stage},
        {"$group": {
            "_id": "$drawingId",
            "drawingTitle": {"$first": "$drawingTitle"},
            "drawingName": {"$first": "$drawingName"},
            "setTrade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
            "pdfName": {"$first": "$pdfName"},
            "s3BucketPath": {"$first": "$s3BucketPath"},
            "page_first": {"$first": "$page"},
            "matching_texts": {"$push": "$text"},
            # Carry per-fragment bbox+page; capped to first 8 to keep payload small.
            "matching_fragments": {"$push": {
                "text": "$text", "x": "$x", "y": "$y",
                "width": "$width", "height": "$height", "page": "$page",
            }},
            "match_count": {"$sum": 1},
        }},
        {"$sort": {"match_count": -1}},
        {"$limit": limit},
    ]

    results = []
    for doc in coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=30000):
        previews = doc.get("matching_texts", [])[:5]
        # Trim fragments to top-8 (longer texts first as they're richer hits)
        frags = sorted(
            (f for f in (doc.get("matching_fragments") or [])
             if f and f.get("text")),
            key=lambda f: -len(f.get("text") or ""),
        )[:8]
        results.append({
            "drawingId": doc["_id"],
            "drawingTitle": doc.get("drawingTitle", ""),
            "drawingName": doc.get("drawingName", ""),
            "setTrade": doc.get("setTrade", ""),
            "pdfName": doc.get("pdfName", ""),
            "s3BucketPath": doc.get("s3BucketPath", ""),
            "page": doc.get("page_first"),
            "match_count": doc.get("match_count", 0),
            "sample_matches": previews,
            # Per-fragment data for citation/highlight (consumed by
            # _extract_sources → _build_source_doc in agentic.core.agent).
            "matching_fragments": frags,
        })

    logger.info(f"search_drawing_text: project={project_id}, query='{search_text[:30]}', found={len(results)}")
    return results


def search_drawings_by_trade(
    project_id: int,
    trade: str,
    limit: int = 20,
    drawing_title: str = None,
    drawing_name: str = None,
) -> List[Dict]:
    """Search drawings by trade name (e.g., 'Electrical', 'Mechanical')."""
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)
    coll = get_collection(COLLECTION)
    trade_regex = re.compile(re.escape(trade), re.IGNORECASE)

    match_stage = {
        "projectId": project_id,
        "$or": [
            {"setTrade": trade_regex},
            {"trade": trade_regex},
            {"trades.0": trade_regex},
        ],
    }
    if drawing_title:
        match_stage["drawingTitle"] = re.compile(re.escape(drawing_title), re.IGNORECASE)
    if drawing_name:
        match_stage["drawingName"] = re.compile(re.escape(drawing_name), re.IGNORECASE)

    pipeline = [
        {"$match": match_stage},
        {"$group": {
            "_id": "$drawingId",
            "drawingTitle": {"$first": "$drawingTitle"},
            "drawingName": {"$first": "$drawingName"},
            "setTrade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
            "pdfName": {"$first": "$pdfName"},
            "s3BucketPath": {"$first": "$s3BucketPath"},
            "fragment_count": {"$sum": 1},
        }},
        {"$sort": {"drawingName": 1}},
        {"$limit": limit},
    ]

    results = list(coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=30000))
    logger.info(f"search_drawings_by_trade: project={project_id}, trade={trade}, found={len(results)}")
    return results


def list_unique_drawing_titles(
    project_id: int,
    set_id: int = None,
) -> List[Dict]:
    """Get unique drawingTitle + drawingName pairs for a project.

    Used by the orchestrator for document discovery when the agent cannot
    answer a question — presents available document groups to the user.

    Returns a deduplicated list sorted by drawingTitle, with:
    - drawingTitle: human-readable title (e.g. "Mechanical Lower Level Plan")
    - drawingName: drawing identifier (e.g. "M-101")
    - trade: trade name (e.g. "Mechanical")
    - pdfName: associated PDF filename
    - fragment_count: number of OCR fragments for this drawing
    """
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)
    match: Dict[str, Any] = {"projectId": project_id}
    if set_id:
        match["setId"] = int(set_id)

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": {"drawingTitle": "$drawingTitle", "drawingName": "$drawingName"},
            "trade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
            "pdfName": {"$first": "$pdfName"},
            "fragment_count": {"$sum": 1},
        }},
        {"$project": {
            "_id": 0,
            "drawingTitle": "$_id.drawingTitle",
            "drawingName": "$_id.drawingName",
            "trade": 1,
            "pdfName": 1,
            "fragment_count": 1,
        }},
        {"$sort": {"drawingTitle": 1}},
    ]

    results = list(coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=15000))

    # Handle null/empty titles — fall back to drawingName or pdfName
    for r in results:
        if not r.get("drawingTitle"):
            r["drawingTitle"] = r.get("drawingName") or r.get("pdfName") or "Untitled"

    logger.info(
        "list_unique_drawing_titles: project=%d, found=%d unique titles",
        project_id, len(results),
    )
    return results
