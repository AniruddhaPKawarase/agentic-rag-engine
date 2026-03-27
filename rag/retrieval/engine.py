"""Core semantic retrieval functions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import faiss

from . import state
from .embeddings import convert_l2_to_similarity, get_embedding
from .loaders import _get_project_config
from .metadata import (
    build_pdf_download_url,
    extract_field,
    get_display_title,
    get_document_name,
    get_drawing_title,
    get_page_number,
    get_s3_path,
)

def retrieve_context(
    query: str,
    top_k: int = 5,
    min_score: float = 0.1,
    filter_source_type: Optional[str] = None,
    filter_project_id: Optional[int] = None,
    filter_drawing_name: Optional[str] = None,
    filter_drawing_title: Optional[str] = None,
    filter_pdf_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search over a project's FAISS index.

    Args:
        query:              Natural language question.
        top_k:              Maximum results to return.
        min_score:          Minimum similarity score (0–1).
        filter_source_type: "drawing", "specification", or None for all.
        filter_project_id:  Project to search. Falls back to last-initialized
                            project, then 7212 for backward compatibility.

    Returns:
        List of result dicts, sorted by similarity (descending). Each dict
        includes: index, text, source_type, similarity, distance, token_count,
        created_at, pdf_name, page, s3_path, drawing_id, drawing_name,
        drawing_title, display_title, download_url, section_title, trade_name,
        material, quantity, unit, set_id, trade_id, project_id.
    """
    
    # Resolve project
    if filter_project_id is None:
        filter_project_id = state._current_project_id or 7212

    config = _get_project_config(filter_project_id)

    if config.index is None:
        raise RuntimeError(f"FAISS index not initialized for project {filter_project_id}.")

    # Embed + normalize
    query_vec = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # Over-fetch to absorb post-filter losses
    k_search = min(top_k * 3, config.index.ntotal)
    distances, indices = config.index.search(query_vec, k=k_search)

    results:     List[Dict[str, Any]] = []
    seen_texts:  set                  = set()

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(config.metadata):
            continue

        similarity = convert_l2_to_similarity(float(dist))
        if similarity < min_score:
            continue

        meta = config.metadata_dict.get(int(idx))
        if not meta:
            continue

        # Source-type filter
        if filter_source_type and meta.get("source_type") != filter_source_type:
            continue

        # Project filter (defensive — index should already be project-scoped)
        meta_pid = meta.get("project_id")
        if meta_pid is not None and int(meta_pid) != filter_project_id:
            continue

        # Hard filter by specific PDF names (for document-scoped chat)
        if filter_pdf_names:
            chunk_pdf = (meta.get("pdfName") or meta.get("pdf_name") or "").strip()
            chunk_drawing = (meta.get("drawingName") or meta.get("drawing_name") or "").strip()
            matched = False
            for pin_name in filter_pdf_names:
                pin_lower = pin_name.lower()
                if (chunk_pdf.lower() == pin_lower or
                    chunk_drawing.lower() == pin_lower or
                    pin_lower in chunk_pdf.lower() or
                    pin_lower in chunk_drawing.lower()):
                    matched = True
                    break
            if not matched:
                continue

        # Exact-text deduplication
        text = (meta.get("text") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        # Field extraction
        s3_path      = get_s3_path(meta)
        pdf_name     = get_document_name(meta)
        drawing_title = get_drawing_title(meta)
        display_title = get_display_title(meta)
        download_url  = build_pdf_download_url(
            s3_path or "",
            meta.get("pdfName") or meta.get("pdf_name") or "",
        )

        results.append({
            "index":         int(idx),
            "text":          text,
            "source_type":   meta.get("source_type", "unknown"),
            "similarity":    float(similarity),
            "distance":      float(dist),
            "token_count":   meta.get("token_count", 0),
            "created_at":    meta.get("created_at", ""),

            # Document identity
            "pdf_name":      pdf_name,
            "page":          get_page_number(meta),
            "s3_path":       s3_path,

            # Human-readable display fields (NEW)
            "drawing_title": drawing_title,   # raw field value from metadata
            "display_title": display_title,   # best available human-readable name
            "download_url":  download_url,    # clickable HTTPS link to the PDF

            # Extended metadata
            "drawing_id":    extract_field(meta, "drawing_id",   "drawingId"),
            "drawing_name":  extract_field(meta, "drawing_name", "drawingName"),
            "section_title": extract_field(meta, "section_title","sectionTitle"),
            "trade_name":    extract_field(meta, "trade_name",   "TradeName", "tradeName", "trade"),
            "material":      extract_field(meta, "material",     "Material"),
            "quantity":      extract_field(meta, "quantity",     "Quantity"),
            "unit":          extract_field(meta, "unit",         "Unit"),
            "set_id":        extract_field(meta, "set_id",       "setId"),
            "trade_id":      extract_field(meta, "trade_id",     "tradeId"),
            "set_name":      extract_field(meta, "set_name",     "setName"),
            "set_trade":     extract_field(meta, "set_trade",    "setTrade"),
            "project_id":    filter_project_id,
        })

        if len(results) >= top_k:
            break

    # ── Drawing name/title soft boost (1.5x similarity for matches) ──────
    if filter_drawing_name or filter_drawing_title:
        fname = (filter_drawing_name or "").upper()
        ftitle = (filter_drawing_title or "").lower()
        for r in results:
            meta_name = (r.get("drawing_name") or "").upper()
            meta_title = (r.get("drawing_title") or "").lower()
            if fname and fname in meta_name:
                r["similarity"] = min(1.0, r["similarity"] * 1.5)
            if ftitle and ftitle in meta_title:
                r["similarity"] = min(1.0, r["similarity"] * 1.5)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def retrieve_context_with_session(
    query: str,
    session_context: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    min_score: float = 0.1,
    filter_source_type: Optional[str] = None,
    filter_project_id: Optional[int] = None,
    filter_drawing_name: Optional[str] = None,
    filter_drawing_title: Optional[str] = None,
    filter_pdf_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve context with optional session-topic enhancement.

    When session_context contains recent_topics, the query is lightly enriched
    with the last three topics to improve result relevance for follow-up questions.
    """
    enhanced_query = query

    if session_context:
        recent_topics = session_context.get("recent_topics", [])
        if recent_topics:
            topics_str    = ", ".join(recent_topics[-3:])
            enhanced_query = f"{query} [Related to: {topics_str}]"

    return retrieve_context(
        query                = enhanced_query,
        top_k                = top_k,
        min_score            = min_score,
        filter_source_type   = filter_source_type,
        filter_project_id    = filter_project_id,
        filter_drawing_name  = filter_drawing_name,
        filter_drawing_title = filter_drawing_title,
        filter_pdf_names     = filter_pdf_names,
    )
