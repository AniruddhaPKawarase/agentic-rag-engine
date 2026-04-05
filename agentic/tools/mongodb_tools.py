"""
MongoDB query tools for the drawingVision collection.
Uses the shared client from core.db — no duplicate connections.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from config import VISION_COLLECTION
from core.db import get_collection
from tools.validation import validate_limit, validate_project_id, validate_search_text, validate_source_file

logger = logging.getLogger("agentic_rag.tools.vision")


def _get_vision_collection():
    return get_collection(VISION_COLLECTION)


# ═══════════════════════════════════════════════════════════════════════
# TOOL 1: List all drawings for a project
# ═══════════════════════════════════════════════════════════════════════

def list_drawings(project_id: int, set_id: int = None) -> List[Dict]:
    """List all drawings for a project with summary info."""
    project_id = validate_project_id(project_id)
    coll = _get_vision_collection()
    query: Dict[str, Any] = {"projectId": project_id}
    if set_id:
        query["setId"] = int(set_id)

    results = []
    for doc in coll.find(query, {
        "sourceFile": 1, "trades": 1, "csi_division": 1,
        "pages.drawing_type": 1, "pages.page_summary": 1,
        "pages.sheet_number": 1, "pages.scale": 1,
    }):
        page = doc.get("pages", [{}])[0] if doc.get("pages") else {}
        results.append({
            "sourceFile": doc.get("sourceFile", ""),
            "drawing_type": page.get("drawing_type", "unknown"),
            "sheet_number": page.get("sheet_number", ""),
            "scale": page.get("scale", ""),
            "page_summary": page.get("page_summary", "")[:200],
            "trades": doc.get("trades", {}),
        })

    logger.info(f"list_drawings: project={project_id}, found={len(results)}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# TOOL 2: Search by text content (full-text search)
# ═══════════════════════════════════════════════════════════════════════

def search_by_text(
    project_id: int,
    search_text: str,
    limit: int = 10,
) -> List[Dict]:
    """Full-text search across page summaries, notes, and blocks."""
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit)
    coll = _get_vision_collection()

    results = []
    try:
        cursor = coll.find(
            {"projectId": project_id, "$text": {"$search": search_text}},
            {
                "score": {"$meta": "textScore"},
                "sourceFile": 1,
                "trades": 1,
                "pages.drawing_type": 1,
                "pages.page_summary": 1,
                "pages.key_notes": 1,
                "pages.general_notes": 1,
                "pages.sheet_number": 1,
            },
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)

        for doc in cursor:
            page = doc.get("pages", [{}])[0] if doc.get("pages") else {}
            results.append({
                "sourceFile": doc.get("sourceFile", ""),
                "relevance_score": round(doc.get("score", 0), 2),
                "drawing_type": page.get("drawing_type", "unknown"),
                "sheet_number": page.get("sheet_number", ""),
                "page_summary": page.get("page_summary", ""),
                "key_notes": page.get("key_notes", "")[:500],
                "general_notes": page.get("general_notes", "")[:500],
                "trades": doc.get("trades", {}),
            })
    except Exception as e:
        logger.warning(f"Text search failed, falling back to regex: {type(e).__name__}")
        results = _regex_search(coll, project_id, search_text, limit)

    logger.info(f"search_by_text: query='{search_text[:50]}', found={len(results)}")
    return results


def _regex_search(coll, project_id: int, text: str, limit: int) -> List[Dict]:
    """Fallback regex search when $text index isn't available."""
    pattern = re.compile(re.escape(text), re.IGNORECASE)
    results = []

    for doc in coll.find(
        {
            "projectId": project_id,
            "$or": [
                {"pages.page_summary": pattern},
                {"pages.key_notes": pattern},
                {"pages.general_notes": pattern},
                {"pages.blocks.text": pattern},
            ],
        },
    ).limit(limit):
        page = doc.get("pages", [{}])[0] if doc.get("pages") else {}
        results.append({
            "sourceFile": doc.get("sourceFile", ""),
            "relevance_score": 1.0,
            "drawing_type": page.get("drawing_type", "unknown"),
            "sheet_number": page.get("sheet_number", ""),
            "page_summary": page.get("page_summary", ""),
            "key_notes": page.get("key_notes", "")[:500],
            "general_notes": page.get("general_notes", "")[:500],
            "trades": doc.get("trades", {}),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
# TOOL 3: Search by trade/drawing type
# ═══════════════════════════════════════════════════════════════════════

def search_by_filters(
    project_id: int,
    trade: str = None,
    drawing_type: str = None,
    limit: int = 10,
) -> List[Dict]:
    """Search drawings by trade name and/or drawing type."""
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit)
    coll = _get_vision_collection()
    query: Dict[str, Any] = {"projectId": project_id}

    if trade:
        trade_regex = re.compile(re.escape(trade), re.IGNORECASE)
        # Search trades array directly
        query["trades"] = trade_regex

    if drawing_type:
        query["pages.drawing_type"] = re.compile(re.escape(drawing_type), re.IGNORECASE)

    results = []
    for doc in coll.find(query).limit(limit):
        page = doc.get("pages", [{}])[0] if doc.get("pages") else {}
        results.append({
            "sourceFile": doc.get("sourceFile", ""),
            "drawing_type": page.get("drawing_type", "unknown"),
            "sheet_number": page.get("sheet_number", ""),
            "page_summary": page.get("page_summary", ""),
            "key_notes": page.get("key_notes", "")[:500],
            "general_notes": page.get("general_notes", "")[:500],
            "trades": doc.get("trades", {}),
        })

    logger.info(f"search_by_filters: trade={trade}, type={drawing_type}, found={len(results)}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# TOOL 4: Get specific content from a drawing
# ═══════════════════════════════════════════════════════════════════════

def get_drawing_content(
    project_id: int,
    source_file: str,
    content_type: str = "all",
) -> Optional[Dict]:
    """Get specific content from a drawing by sourceFile OR sheet_number."""
    project_id = validate_project_id(project_id)
    coll = _get_vision_collection()

    # Try by sourceFile first, then by sheet_number (e.g. "M-101A")
    doc = None
    if source_file:
        # Try exact match
        doc = coll.find_one({"projectId": project_id, "sourceFile": source_file})
        # If not found, try partial match (sheet number in filename)
        if not doc:
            import re
            pattern = re.compile(re.escape(source_file), re.IGNORECASE)
            doc = coll.find_one({"projectId": project_id, "sourceFile": pattern})
        # Try as sheet_number
        if not doc:
            doc = coll.find_one({"projectId": project_id, "pages.sheet_number": source_file})

    if not doc:
        return None

    page = doc.get("pages", [{}])[0] if doc.get("pages") else {}

    if content_type == "notes":
        return {
            "sourceFile": doc.get("sourceFile"),
            "key_notes": page.get("key_notes", ""),
            "general_notes": page.get("general_notes", ""),
        }
    elif content_type == "elements":
        return {
            "sourceFile": doc.get("sourceFile"),
            "vision_elements": page.get("vision_elements", []),
        }
    elif content_type == "summary":
        return {
            "sourceFile": doc.get("sourceFile"),
            "page_summary": page.get("page_summary", ""),
            "drawing_type": page.get("drawing_type", ""),
            "trades": doc.get("trades", {}),
            "csi_division": doc.get("csi_division", {}),
        }
    else:
        return {
            "sourceFile": doc.get("sourceFile"),
            "page_summary": page.get("page_summary", ""),
            "key_notes": page.get("key_notes", ""),
            "general_notes": page.get("general_notes", ""),
            "vision_elements": page.get("vision_elements", []),
            "title_block": page.get("title_block", {}),
            "trades": doc.get("trades", {}),
        }
