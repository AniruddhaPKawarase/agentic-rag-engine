"""
Tier A1 — Cross-encoder reranker cascade (BGE-reranker-v2-m3, local CPU).

Sits BEFORE the existing LLM-as-judge reranker in the pipeline:

  hybrid retrieval (top-N) -> cross_encoder_rerank (top-K) -> _maybe_rerank (LLM, final)

Cross-encoders are purpose-built for (query, doc) scoring — 10x cheaper /
faster than an LLM rerank while delivering comparable quality, per
multiple production reports (Cohere v4: +17.2pp MRR@3; Legal RAG: 70% -> 95%;
Financial RAG: +15.5pp on hybrid+rerank).

Model: BAAI/bge-reranker-v2-m3 (568MB, multilingual, MIT-licensed).
First invocation downloads + loads into RAM (~5-10s). Subsequent calls
~50-100ms per query for ~30 candidate docs on CPU.

Configuration via .env:
  CROSS_ENCODER_RERANK_ENABLED=true|false   (default: false — opt-in)
  CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3
  CROSS_ENCODER_CANDIDATE_CAP=30            (max docs to rerank — anything more is wasted)
  CROSS_ENCODER_KEEP_TOP=15                  (top-K to pass to next stage)
  CROSS_ENCODER_MIN_KEEP=5                   (never drop below this)

Safe to call repeatedly; safe to disable via flag at runtime.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentic_rag.cross_encoder_rerank")

DEFAULT_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_CANDIDATE_CAP = int(os.environ.get("CROSS_ENCODER_CANDIDATE_CAP", "30"))
DEFAULT_KEEP_TOP = int(os.environ.get("CROSS_ENCODER_KEEP_TOP", "15"))
DEFAULT_MIN_KEEP = int(os.environ.get("CROSS_ENCODER_MIN_KEEP", "5"))

# Lazy model singleton — loaded on first use, kept in memory.
_model = None
_model_lock = threading.Lock()
_load_failed = False


def _get_model():
    """Lazy-load the cross-encoder model. Returns None if load fails
    (cascade gracefully degrades — caller treats absent model as no-op)."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder
            logger.info("cross_encoder_rerank: loading model %r ...", DEFAULT_MODEL)
            # max_length=512 keeps inference fast; truncates over-long docs
            _model = CrossEncoder(DEFAULT_MODEL, max_length=512, automodel_args={"torch_dtype": "float32"})
            logger.info("cross_encoder_rerank: model loaded.")
            return _model
        except Exception as exc:
            logger.warning("cross_encoder_rerank: model load failed (%s) — disabling for this session", exc)
            _load_failed = True
            return None


def _doc_text_for_rerank(doc: Dict[str, Any]) -> str:
    """Compose a representative text from a source_document for (query, doc)
    scoring. Prefers higher-signal fields: pageSummary -> text_excerpt ->
    drawingTitle -> sectionTitle -> display_title."""
    if not isinstance(doc, dict):
        return str(doc)
    pieces: List[str] = []
    for key in ("pageSummary", "text_excerpt", "drawingTitle",
                "sheet_title", "sectionTitle", "display_title",
                "drawing_title", "file_name"):
        v = doc.get(key)
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())
            if sum(len(p) for p in pieces) > 600:
                break
    if not pieces:
        # last resort: full source doc string repr (capped)
        return str(doc)[:600]
    return " | ".join(pieces)[:1200]


def cross_encoder_rerank(
    query: str,
    source_documents: List[Dict[str, Any]],
    candidate_cap: Optional[int] = None,
    keep_top: Optional[int] = None,
    min_keep: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Score (query, doc) pairs with BGE cross-encoder, return docs sorted
    best-first, capped at keep_top.

    Graceful degradation: if model fails to load, returns the input list
    unchanged. Never raises.
    """
    if not isinstance(source_documents, list) or len(source_documents) <= 1:
        return source_documents

    candidate_cap = candidate_cap or DEFAULT_CANDIDATE_CAP
    keep_top = keep_top or DEFAULT_KEEP_TOP
    min_keep = min_keep or DEFAULT_MIN_KEEP

    candidates = source_documents[:candidate_cap]
    model = _get_model()
    if model is None:
        return source_documents  # silently degrade

    pairs: List[Tuple[str, str]] = [(query, _doc_text_for_rerank(d)) for d in candidates]
    try:
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as exc:
        logger.warning("cross_encoder_rerank: predict failed (%s) — returning input order", exc)
        return source_documents

    scored: List[Tuple[float, Dict[str, Any]]] = list(zip([float(s) for s in scores], candidates))
    scored.sort(key=lambda t: t[0], reverse=True)

    # Keep top-K but never below min_keep
    target = max(min_keep, min(keep_top, len(scored)))
    kept = [d for (_, d) in scored[:target]]

    # Attach reranker score for debug / downstream LLM rerank to use
    for (s, d) in scored[:target]:
        try:
            d["_cross_encoder_score"] = round(s, 4)
        except Exception:
            pass

    # Preserve any source docs beyond candidate_cap at the tail (we don't drop them)
    tail = source_documents[candidate_cap:]
    logger.info(
        "cross_encoder_rerank: query=%r in=%d candidates=%d kept=%d tail=%d  best=%.3f worst=%.3f",
        (query or "")[:60], len(source_documents), len(candidates), len(kept), len(tail),
        scored[0][0] if scored else 0.0, scored[-1][0] if scored else 0.0,
    )
    return kept + tail
