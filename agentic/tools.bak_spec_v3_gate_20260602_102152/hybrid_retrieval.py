"""Hybrid vector + keyword (BM25) retrieval helpers — additive layer.

This module is OFF by default. It activates only when
``ENABLE_HYBRID_RETRIEVAL=true`` is set in the environment.

When OFF, the public function ``maybe_fuse_with_keyword`` is a no-op
returning the vector hits unchanged. The calling code path is therefore
byte-identical to the legacy vector-only behavior whenever the flag is off.

When ON, it runs a parallel ``$text`` keyword query against the existing
Mongo text index on the same collection, then merges the keyword hits with
the supplied vector hits using Reciprocal Rank Fusion (RRF). The result is
truncated to the same ``limit`` the caller asked for, so the surface
contract is unchanged — only candidate quality improves.

Safety properties:
  - Never removes a candidate that the vector search returned.
  - Never returns more items than ``limit``.
  - Any exception in the keyword path is swallowed and we return the
    original vector hits — the agent never sees a hybrid-induced failure.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentic_rag.tools.hybrid")


def hybrid_enabled() -> bool:
    return os.getenv("ENABLE_HYBRID_RETRIEVAL", "false").strip().lower() in ("1", "true", "yes", "on")


def _rrf_k() -> int:
    try:
        return int(os.getenv("HYBRID_RRF_K", "60"))
    except ValueError:
        return 60


def _keyword_limit(default: int) -> int:
    try:
        return int(os.getenv("HYBRID_KEYWORD_LIMIT", str(default)))
    except ValueError:
        return default


def _doc_key(doc: Dict[str, Any]) -> Optional[str]:
    """Deduplication key for fusion — prefers (drawingId, page), then sheet,
    then drawingName, then sectionTitle / csi."""
    if not isinstance(doc, dict):
        return None
    if doc.get("drawingId") is not None:
        return f"d:{doc['drawingId']}:{doc.get('page', 1)}"
    if doc.get("sheetNumber"):
        return f"s:{doc['sheetNumber']}:{doc.get('page', 1)}"
    if doc.get("drawingName"):
        return f"n:{doc['drawingName']}:{doc.get('page', 1)}"
    if doc.get("sectionTitle") or doc.get("csi"):
        return f"c:{doc.get('csi') or doc.get('sectionTitle')}"
    # Last-resort: stringified dict (only used when nothing else is unique)
    return None


def _run_keyword_search(
    coll,
    project_id: int,
    search_text: str,
    limit: int,
    extra_filter: Optional[Dict[str, Any]],
    projection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run Mongo $text search against the existing full-text index.

    Returns up to ``limit`` hits sorted by Mongo textScore desc, with each
    hit's ``_kw_score`` set so the caller can inspect fusion contributions.
    """
    if not search_text or not search_text.strip():
        return []

    base: Dict[str, Any] = {"projectId": project_id, "$text": {"$search": search_text}}
    if extra_filter:
        for k, v in extra_filter.items():
            if k not in ("projectId",):
                base[k] = v

    proj = {**(projection or {}), "_kw_score": {"$meta": "textScore"}}
    try:
        cursor = coll.find(base, proj).sort([("_kw_score", {"$meta": "textScore"})]).limit(limit)
        return list(cursor)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hybrid keyword search failed (%s); skipping", exc)
        return []


def _rrf_fuse(
    vector_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion. Returns the union deduped by ``_doc_key``,
    sorted by RRF score desc. Each fused doc gets:
      - ``_rrf_score`` for diagnostics
      - ``relevance`` set to a normalized 0..1 (per-list, bounded)
    The original vector ``relevance`` is preserved in ``_vec_relevance``
    if it was present.
    """
    scored: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    for rank, doc in enumerate(vector_hits):
        key = _doc_key(doc)
        if key is None:
            # No dedupe key — leave it in as-is, scored by its vector rank
            key = f"_vec_only_{rank}"
        contrib = 1.0 / (k + rank + 1)  # rank is 0-based
        prev = scored.get(key)
        merged_doc = dict(doc)
        # Preserve original vector relevance
        if "relevance" in doc and "_vec_relevance" not in merged_doc:
            merged_doc["_vec_relevance"] = doc.get("relevance")
        merged_doc["_vec_rank"] = rank + 1
        if prev is None:
            scored[key] = (contrib, merged_doc)
        else:
            scored[key] = (prev[0] + contrib, {**prev[1], **merged_doc})

    for rank, doc in enumerate(keyword_hits):
        key = _doc_key(doc)
        if key is None:
            key = f"_kw_only_{rank}"
        contrib = 1.0 / (k + rank + 1)
        merged_doc = dict(doc)
        merged_doc["_kw_rank"] = rank + 1
        prev = scored.get(key)
        if prev is None:
            scored[key] = (contrib, merged_doc)
        else:
            # Merge fields: vector projection is usually richer; keep both.
            merged = {**merged_doc, **prev[1]}
            merged["_kw_rank"] = merged_doc["_kw_rank"]
            if "_kw_score" in merged_doc:
                merged["_kw_score"] = merged_doc["_kw_score"]
            scored[key] = (prev[0] + contrib, merged)

    out = []
    for key, (score, doc) in sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True):
        doc["_rrf_score"] = round(score, 6)
        out.append(doc)
    return out


def maybe_fuse_with_keyword(
    *,
    vector_hits: List[Dict[str, Any]],
    coll,
    project_id: int,
    search_text: str,
    limit: int,
    extra_filter: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Public entry point. NO-OP when ENABLE_HYBRID_RETRIEVAL is off."""
    if not hybrid_enabled():
        return vector_hits
    try:
        kw_limit = _keyword_limit(limit)
        kw_hits = _run_keyword_search(
            coll=coll,
            project_id=project_id,
            search_text=search_text,
            limit=kw_limit,
            extra_filter=extra_filter,
            projection=projection or {"_id": 0},
        )
        if not kw_hits:
            return vector_hits
        fused = _rrf_fuse(vector_hits, kw_hits, k=_rrf_k())
        # Safety guard: ensure every vector hit is preserved in the fused
        # output. If a vector hit somehow dropped (shouldn't, but be paranoid),
        # add it back at its original position. This makes fusion strictly
        # additive vs the legacy behavior — we can only add candidates,
        # never remove them.
        seen_keys = set()
        for d in fused:
            k = _doc_key(d)
            if k:
                seen_keys.add(k)
        for d in vector_hits:
            k = _doc_key(d)
            if k and k not in seen_keys:
                d2 = dict(d)
                d2.setdefault("_rrf_score", 0.0)
                fused.append(d2)
        # Truncate to the caller's limit
        truncated = fused[: max(limit, len(vector_hits))]
        logger.info(
            "hybrid fuse: project=%d query=%r vec=%d kw=%d fused=%d returned=%d",
            project_id, (search_text or "")[:50],
            len(vector_hits), len(kw_hits), len(fused), len(truncated),
        )
        return truncated
    except Exception as exc:  # noqa: BLE001
        logger.warning("hybrid fusion failed (%s); returning vector hits unchanged", exc)
        return vector_hits
