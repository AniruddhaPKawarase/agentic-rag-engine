"""
MongoDB tools for the specification collection (80K docs).

Specifications have richer text content (mean 2,218 chars) with
sectionTitle, sectionText, fullText, submittals, and warranties.
"""

import logging
import re
from typing import Any, Dict, List

from core.db import get_collection
from tools.validation import validate_limit, validate_project_id, validate_search_text

logger = logging.getLogger("agentic_rag.tools.spec")

COLLECTION = "specification"


def list_specifications(project_id: int, limit: int = 50) -> List[Dict]:
    """List available specifications for a project."""
    project_id = validate_project_id(project_id)
    limit = validate_limit(limit, max_limit=100)
    coll = get_collection(COLLECTION)

    pipeline = [
        {"$match": {"projectId": project_id}},
        {"$group": {
            "_id": {"pdfName": "$pdfName", "sectionTitle": "$sectionTitle"},
            "specificationNumber": {"$first": "$specificationNumber"},
            "page_count": {"$addToSet": "$page"},
            "fragment_count": {"$sum": 1},
            "sample_text": {"$first": "$text"},
        }},
        {"$project": {
            "pdfName": "$_id.pdfName",
            "sectionTitle": "$_id.sectionTitle",
            "specificationNumber": 1,
            "pages": {"$size": "$page_count"},
            "fragments": "$fragment_count",
            "preview": {"$substrCP": [{"$ifNull": ["$sample_text", ""]}, 0, 200]},
        }},
        {"$sort": {"specificationNumber": 1}},
        {"$limit": limit},
    ]

    results = list(coll.aggregate(pipeline, maxTimeMS=20000))
    logger.info(f"list_specifications: project={project_id}, found={len(results)}")
    return results


def search_specification_text(
    project_id: int,
    search_text: str,
    limit: int = 10,
) -> List[Dict]:
    """Search specification content by keywords."""
    project_id = validate_project_id(project_id)
    search_text = validate_search_text(search_text)
    limit = validate_limit(limit)
    coll = get_collection(COLLECTION)

    # Try text search first (uses fullText_text index)
    try:
        results = list(coll.find(
            {"projectId": project_id, "$text": {"$search": search_text}},
            {
                "score": {"$meta": "textScore"},
                "pdfName": 1, "sectionTitle": 1, "specificationNumber": 1,
                "text": 1, "page": 1, "_id": 0,
            },
        ).sort([("score", {"$meta": "textScore"})]).limit(limit))

        if results:
            logger.info(f"search_specification_text (text index): query='{search_text[:30]}', found={len(results)}")
            return [{
                "pdfName": r.get("pdfName", ""),
                "sectionTitle": r.get("sectionTitle", ""),
                "specificationNumber": r.get("specificationNumber", ""),
                "text": (r.get("text") or "")[:500],
                "page": r.get("page"),
                "relevance": round(r.get("score", 0), 2),
            } for r in results]
    except Exception:
        pass

    # Fallback: regex search
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)
    results = list(coll.find(
        {"projectId": project_id, "$or": [
            {"text": pattern},
            {"sectionTitle": pattern},
            {"sectionText": pattern},
        ]},
        {"pdfName": 1, "sectionTitle": 1, "specificationNumber": 1,
         "text": 1, "page": 1, "_id": 0},
    ).limit(limit))

    logger.info(f"search_specification_text (regex): query='{search_text[:30]}', found={len(results)}")
    return [{
        "pdfName": r.get("pdfName", ""),
        "sectionTitle": r.get("sectionTitle", ""),
        "specificationNumber": r.get("specificationNumber", ""),
        "text": (r.get("text") or "")[:500],
        "page": r.get("page"),
        "relevance": 1.0,
    } for r in results]


def get_specification_section(
    project_id: int,
    section_title: str = None,
    pdf_name: str = None,
) -> List[Dict]:
    """Get full text of a specific specification section."""
    project_id = validate_project_id(project_id)
    coll = get_collection(COLLECTION)
    query: Dict[str, Any] = {"projectId": project_id}

    if section_title:
        query["sectionTitle"] = re.compile(re.escape(section_title), re.IGNORECASE)
    if pdf_name:
        query["pdfName"] = re.compile(re.escape(pdf_name), re.IGNORECASE)

    results = list(coll.find(
        query,
        {"pdfName": 1, "sectionTitle": 1, "specificationNumber": 1,
         "text": 1, "sectionText": 1, "page": 1,
         "submittalsStructured": 1, "warrantiesStructured": 1, "_id": 0},
    ).sort("page", 1).limit(50))

    # Combine text from fragments
    combined = []
    for r in results:
        text = r.get("sectionText") or r.get("text") or ""
        combined.append({
            "pdfName": r.get("pdfName", ""),
            "sectionTitle": r.get("sectionTitle", ""),
            "specificationNumber": r.get("specificationNumber", ""),
            "text": text[:2000],
            "page": r.get("page"),
            "submittals": r.get("submittalsStructured"),
            "warranties": r.get("warrantiesStructured"),
        })

    logger.info(f"get_specification_section: project={project_id}, found={len(combined)}")
    return combined
