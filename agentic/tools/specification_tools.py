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
                "text": 1, "page": 1, "s3BucketPath": 1, "_id": 0,
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
                "s3BucketPath": r.get("s3BucketPath", ""),
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
         "text": 1, "page": 1, "s3BucketPath": 1, "_id": 0},
    ).limit(limit))

    logger.info(f"search_specification_text (regex): query='{search_text[:30]}', found={len(results)}")
    return [{
        "pdfName": r.get("pdfName", ""),
        "sectionTitle": r.get("sectionTitle", ""),
        "specificationNumber": r.get("specificationNumber", ""),
        "text": (r.get("text") or "")[:500],
        "page": r.get("page"),
        "s3BucketPath": r.get("s3BucketPath", ""),
        "relevance": 1.0,
    } for r in results]


def get_specification_section(
    project_id: int,
    section_title: str = None,
    pdf_name: str = None,
) -> List[Dict]:
    """Get fragments of a specific specification section.

    Returns one dict per fragment (up to 50, sorted by page). Each fragment's
    text is capped at 2000 chars to stay token-friendly for the agent. For a
    consolidated whole-section view, use :func:`get_full_specification_text`.
    """
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
         "text": 1, "sectionText": 1, "page": 1, "s3BucketPath": 1,
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
            "s3BucketPath": r.get("s3BucketPath", ""),
            "submittals": r.get("submittalsStructured"),
            "warranties": r.get("warrantiesStructured"),
        })

    logger.info(f"get_specification_section: project={project_id}, found={len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Fix #5 — Parent-document retrieval for specs
#
# Instead of feeding the agent a list of truncated fragments, this tool
# concatenates every fragment that belongs to the same section into a single
# readable "parent document" and returns up to a few of those parents.
#
# Matching strategy:
#   * If pdf_name is supplied, anchor on pdfName (exact-ish).
#   * Else if section_title is supplied, match sectionTitle / sectionText /
#     specificationNumber / text with a case-insensitive regex so keywords like
#     "plumbing insulation" resolve the right section.
#
# Return shape is intentionally distinct from ``get_specification_section`` so
# the agent can clearly choose "I need the whole section" vs "I need fragments
# in context". The orchestrator and UI do not depend on this shape directly —
# it only flows into the model's context window.
# ---------------------------------------------------------------------------


# Soft budget per parent section; stays inside GPT-4.1 single-tool-result limit
_PARENT_SECTION_CHAR_CAP = 18000
# Tool-result hard cap (matches agent.py's 14k truncation threshold)
_PARENT_RESULT_CHAR_CAP = 13000


def get_full_specification_text(
    project_id: int,
    section_title: str = None,
    pdf_name: str = None,
    specification_number: str = None,
    max_sections: int = 3,
) -> List[Dict]:
    """Return full consolidated text for one or more specification sections.

    Works across the two spec schemas we observed in the wild:

    * **Schema A** (e.g. project 7166): ``sectionTitle``, ``sectionText``,
      ``text``, ``specificationNumber`` are populated per fragment.
    * **Schema B** (e.g. projects 7222 / 7223): ``sectionTitle`` is ``None``,
      ``sectionText``/``text`` are ``None``, content lives in ``fullText``, and
      the human-readable name sits on ``drawingTitle`` / ``drawingName`` with
      the CSI number often baked into ``pdfName``.

    Strategy:
    1. Find candidate fragments matching any anchor (pdf_name, section_title,
       specification_number) across a broad set of fields (``sectionTitle``,
       ``drawingTitle``, ``specificationNumber``, ``pdfName``,
       ``sectionText``, ``text``, ``fullText``).
    2. Group by ``(pdfName, coalesce(sectionTitle, drawingTitle))`` — that
       tuple defines a "section parent" in either schema.
    3. For each parent, fetch all fragments, prefer ``fullText`` then
       ``sectionText`` then ``text``, concat in page order, cap at
       :data:`_PARENT_SECTION_CHAR_CAP`.
    4. Return up to ``max_sections`` parents ordered by match density.

    Each parent dict carries ``s3BucketPath`` + ``pdfName`` so the orchestrator
    can build a signed download link — no UI contract changes.
    """
    project_id = validate_project_id(project_id)
    max_sections = max(1, min(int(max_sections or 3), 5))
    coll = get_collection(COLLECTION)

    match: Dict[str, Any] = {"projectId": project_id}

    anchor_filters: List[Dict[str, Any]] = []
    if pdf_name:
        anchor_filters.append({"pdfName": re.compile(re.escape(pdf_name), re.IGNORECASE)})
    if specification_number:
        pat_num = re.compile(re.escape(specification_number), re.IGNORECASE)
        anchor_filters += [
            {"specificationNumber": pat_num},
            {"drawingName": pat_num},
            {"pdfName": pat_num},
        ]
    if section_title:
        pat = re.compile(re.escape(section_title), re.IGNORECASE)
        anchor_filters += [
            {"sectionTitle": pat},
            {"drawingTitle": pat},
            {"specificationNumber": pat},
            {"pdfName": pat},
            {"sectionText": pat},
            {"text": pat},
            {"fullText": pat},
        ]

    if anchor_filters:
        match["$or"] = anchor_filters

    # Rank parents by number of matching fragments (match density).
    # Use $ifNull so Schema-B (sectionTitle=None) still groups by drawingTitle.
    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": {
                "pdfName": "$pdfName",
                "title": {"$ifNull": ["$sectionTitle", "$drawingTitle"]},
            },
            "pdfName": {"$first": "$pdfName"},
            "sectionTitle": {"$first": "$sectionTitle"},
            "drawingTitle": {"$first": "$drawingTitle"},
            "drawingName": {"$first": "$drawingName"},
            "specificationNumber": {
                "$first": {"$ifNull": ["$specificationNumber", "$drawingName"]},
            },
            "s3BucketPath": {"$first": "$s3BucketPath"},
            "page_min": {"$min": "$page"},
            "fragment_count": {"$sum": 1},
        }},
        {"$sort": {"fragment_count": -1, "page_min": 1}},
        {"$limit": max_sections},
    ]

    try:
        parents = list(coll.aggregate(pipeline, allowDiskUse=True, maxTimeMS=20000))
    except Exception as exc:
        logger.warning("get_full_specification_text aggregation failed: %s", exc)
        return []

    results: List[Dict] = []
    for p in parents:
        frag_query: Dict[str, Any] = {
            "projectId": project_id,
            "pdfName": p["pdfName"],
        }
        # Constrain by section/drawing title if available to avoid sibling
        # sections inside the same PDF bleeding in.
        title = p.get("sectionTitle") or p.get("drawingTitle")
        if p.get("sectionTitle"):
            frag_query["sectionTitle"] = p["sectionTitle"]
        elif p.get("drawingTitle"):
            frag_query["drawingTitle"] = p["drawingTitle"]

        fragments = list(coll.find(
            frag_query,
            {
                "fullText": 1, "sectionText": 1, "text": 1, "page": 1,
                "submittalsStructured": 1, "warrantiesStructured": 1, "_id": 0,
            },
        ).sort("page", 1).limit(200))

        parts: List[str] = []
        submittals: Any = None
        warranties: Any = None
        for frag in fragments:
            body = (frag.get("fullText") or frag.get("sectionText") or frag.get("text") or "").strip()
            if not body:
                continue
            page = frag.get("page")
            parts.append(f"[page {page}]\n{body}" if page is not None else body)
            if submittals is None and frag.get("submittalsStructured"):
                submittals = frag["submittalsStructured"]
            if warranties is None and frag.get("warrantiesStructured"):
                warranties = frag["warrantiesStructured"]

        full_text = "\n\n".join(parts)[:_PARENT_SECTION_CHAR_CAP]
        results.append({
            "pdfName": p.get("pdfName", ""),
            "sectionTitle": title or "",
            "specificationNumber": p.get("specificationNumber", ""),
            "s3BucketPath": p.get("s3BucketPath", ""),
            "fragment_count": p.get("fragment_count", len(fragments)),
            "full_text": full_text,
            "truncated": len(full_text) >= _PARENT_SECTION_CHAR_CAP,
            "submittals": submittals,
            "warranties": warranties,
        })

    # Enforce overall result budget so the agent's tool-result log stays small
    budget = _PARENT_RESULT_CHAR_CAP
    trimmed: List[Dict] = []
    for r in results:
        cost = len(r["full_text"])
        if cost > budget and trimmed:
            break
        if cost > budget:
            r["full_text"] = r["full_text"][:budget]
            r["truncated"] = True
            trimmed.append(r)
            break
        budget -= cost
        trimmed.append(r)

    logger.info(
        "get_full_specification_text: project=%d title=%r pdf=%r parents=%d",
        project_id, section_title, pdf_name, len(trimmed),
    )
    return trimmed
