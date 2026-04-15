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


def list_project_drawings(project_id: int, set_id: int = None) -> List[Dict]:
    """List unique drawings for a project with metadata."""
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)
    match = {"projectId": project_id}
    if set_id:
        match["setId"] = set_id

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
    """Get reconstructed text for a specific drawing."""
    project_id = validate_project_id(project_id)
    drawing_id = validate_drawing_id(drawing_id)
    coll = get_collection(COLLECTION)
    query: Dict[str, Any] = {"projectId": project_id, "drawingId": drawing_id}
    if page:
        query["page"] = page

    # Fetch fragments sorted server-side (avoids Python re-sort)
    fragments = list(coll.find(
        query,
        {"text": 1, "x": 1, "y": 1, "page": 1,
         "drawingTitle": 1, "drawingName": 1, "trade": 1, "_id": 0},
    ).sort([("page", 1), ("y", 1), ("x", 1)]).limit(5000))

    if not fragments:
        return {"error": f"No fragments found for drawingId={drawing_id}"}

    text = reconstruct_drawing_text(fragments, pre_sorted=True)
    title = fragments[0].get("drawingTitle", "Unknown")
    name = fragments[0].get("drawingName", "Unknown")
    trade = fragments[0].get("trade", "Unknown")

    return {
        "drawingId": drawing_id,
        "drawingTitle": title,
        "drawingName": name,
        "trade": trade,
        "fragment_count": len(fragments),
        "reconstructed_text": text[:30000],
        "text_length": len(text),
    }


def search_drawing_text(
    project_id: int,
    search_text: str,
    limit: int = 10,
) -> List[Dict]:
    """Search drawing fragments for specific text content."""
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit)
    coll = get_collection(COLLECTION)
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)

    pipeline = [
        {"$match": {"projectId": project_id, "text": pattern}},
        {"$group": {
            "_id": "$drawingId",
            "drawingTitle": {"$first": "$drawingTitle"},
            "drawingName": {"$first": "$drawingName"},
            "setTrade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
            "matching_texts": {"$push": "$text"},
            "match_count": {"$sum": 1},
        }},
        {"$sort": {"match_count": -1}},
        {"$limit": limit},
    ]

    results = []
    for doc in coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=30000):
        # Show first 5 matching fragments as preview
        previews = doc.get("matching_texts", [])[:5]
        results.append({
            "drawingId": doc["_id"],
            "drawingTitle": doc.get("drawingTitle", ""),
            "drawingName": doc.get("drawingName", ""),
            "setTrade": doc.get("setTrade", ""),
            "match_count": doc.get("match_count", 0),
            "sample_matches": previews,
        })

    logger.info(f"search_drawing_text: project={project_id}, query='{search_text[:30]}', found={len(results)}")
    return results


def search_drawings_by_trade(
    project_id: int,
    trade: str,
    limit: int = 20,
) -> List[Dict]:
    """Search drawings by trade name (e.g., 'Electrical', 'Mechanical')."""
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)
    coll = get_collection(COLLECTION)
    trade_regex = re.compile(re.escape(trade), re.IGNORECASE)

    pipeline = [
        {"$match": {
            "projectId": project_id,
            "$or": [
                {"setTrade": trade_regex},
                {"trade": trade_regex},
                {"trades.0": trade_regex},
            ],
        }},
        {"$group": {
            "_id": "$drawingId",
            "drawingTitle": {"$first": "$drawingTitle"},
            "drawingName": {"$first": "$drawingName"},
            "setTrade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
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
