"""Block-level promotion helper (Layer 2 orchestrator-side fusion).

Strictly additive. When ENABLE_BLOCK_RETRIEVAL is off, this is a no-op.
When on, it runs the block tool internally and folds the matched block
text into the parent drawing's candidate entry — so the LLM sees the
specific note that matched without having to call a separate tool.

Safety:
  - Never removes a candidate. Only adds/boosts.
  - Caps additions at 5 promotions per call.
  - Block scores below MIN_PROMOTE_SCORE are ignored to keep noise low.
  - Any exception is swallowed; caller gets back the original list.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_rag.tools.block_promote")

MIN_PROMOTE_SCORE = 0.85   # Tuned 2026-05-21 from 0.78 to reduce noise on schedule/spec queries
MAX_PROMOTIONS    = 5      # Hard cap to keep candidate lists small
BLOCK_FETCH_LIMIT = 10     # How many blocks we ask for internally


def _enabled() -> bool:
    return os.getenv("ENABLE_BLOCK_RETRIEVAL", "false").strip().lower() in ("1", "true", "yes", "on")


def _drawing_key(d: Dict[str, Any]) -> Optional[str]:
    """Stable key matching the page-level results from search_drawings_v3."""
    if not isinstance(d, dict):
        return None
    did = d.get("drawingId")
    pg = d.get("page", 1)
    if did is not None:
        return f"d:{did}:{pg}"
    sn = d.get("sheetNumber") or d.get("sheet_number") or d.get("drawingName")
    return f"s:{sn}:{pg}" if sn else None


def fold_blocks_into(
    *,
    page_hits: List[Dict[str, Any]],
    project_id: int,
    search_text: str,
    limit: int,
    discipline: Optional[str] = None,
    drawing_type: Optional[str] = None,
    set_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Augment page-level hits with block-level matches.

    Strategy:
      1. Call search_drawing_blocks_v3 with the same query.
      2. Group matched blocks by parent drawing (drawingId+page).
      3. For each parent in the page-hit list, attach a list of matched
         block texts under ``matched_blocks`` and boost ``relevance`` by
         a small amount per high-scoring block.
      4. For parents NOT in the page-hit list, insert a synthetic entry
         carrying the block text and parent meta (so the LLM still has a
         drawing-level citation handle).
      5. Re-sort by relevance desc, truncate to ``limit + MAX_PROMOTIONS``
         so we never drop original page hits.

    Returns a new list (does not mutate the input).
    """
    if not _enabled():
        return page_hits
    try:
        from tools.drawing_blocks_tool import search_drawing_blocks_v3

        block_hits = search_drawing_blocks_v3(
            project_id=project_id,
            search_text=search_text,
            limit=BLOCK_FETCH_LIMIT,
            discipline=discipline,
            block_kind=None,
            set_id=set_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fold_blocks_into: block search failed (%s); skipping", exc)
        return page_hits

    if not block_hits:
        return page_hits

    # Filter by score threshold; keep top-N
    promotable = []
    for b in block_hits:
        score = b.get("relevance")
        try:
            score = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        if score >= MIN_PROMOTE_SCORE:
            promotable.append((score, b))
    promotable.sort(key=lambda x: -x[0])
    promotable = promotable[:MAX_PROMOTIONS]

    if not promotable:
        return page_hits

    # Map page hits by key for quick attach
    out = [dict(p) for p in page_hits]   # shallow-copy to avoid mutating caller's
    by_key: Dict[str, Dict[str, Any]] = {}
    for d in out:
        k = _drawing_key(d)
        if k:
            by_key[k] = d

    promotions_inserted = 0
    for score, b in promotable:
        k = _drawing_key(b)
        block_text = (b.get("blockText") or "").strip()
        block_kind = b.get("blockKind", "block")
        attach_payload = {
            "text": block_text[:300],
            "kind": block_kind,
            "score": round(score, 4),
        }

        existing = by_key.get(k)
        if existing is not None:
            # Attach to existing parent entry
            mb = existing.setdefault("matched_blocks", [])
            if len(mb) < 3:           # keep payload bounded
                mb.append(attach_payload)
            # Boost relevance by a small bump per matched block (capped)
            try:
                cur = float(existing.get("relevance") or 0.0)
                bump = min(0.10, 0.05 + 0.05 * (score - MIN_PROMOTE_SCORE))
                existing["relevance"] = round(min(1.0, cur + bump), 4)
            except (TypeError, ValueError):
                pass
            existing["block_promoted"] = True
        else:
            # Insert synthetic page candidate carrying the block context.
            # We preserve the parent drawing's identifying fields from the
            # block doc itself (which already has drawingName/sheetNumber/
            # pdfName/s3BucketPath set during ingestion).
            synthetic = {
                "drawingId":   b.get("drawingId"),
                "page":        b.get("page", 1),
                "sheetNumber": b.get("sheetNumber"),
                "sheet_number": b.get("sheet_number") or b.get("sheetNumber"),
                "drawingName": b.get("drawingName"),
                "drawingTitle": b.get("drawingTitle"),
                "drawingType": b.get("drawingType"),
                "discipline":  b.get("discipline"),
                "trade":       b.get("trade"),
                "tradeId":     b.get("tradeId"),
                "setId":       b.get("setId"),
                "pdfName":     b.get("pdfName"),
                "s3BucketPath": b.get("s3BucketPath"),
                "level":       b.get("level"),
                "scale":       b.get("scale"),
                "revision":    b.get("revision"),
                "relevance":   round(score, 4),
                "matched_blocks": [attach_payload],
                "block_promoted": True,
                "promoted_from_blocks_only": True,
            }
            out.append(synthetic)
            promotions_inserted += 1

    # Re-sort by relevance desc, but keep stable order for ties
    out.sort(key=lambda d: float(d.get("relevance") or 0.0), reverse=True)

    # Cap result list — never drop original page hits, but bound total
    cap = max(limit, len(page_hits)) + MAX_PROMOTIONS
    out = out[:cap]

    logger.info(
        "fold_blocks_into: project=%d query=%r blocks_considered=%d promoted=%d new_inserts=%d returned=%d",
        project_id, (search_text or "")[:50],
        len(block_hits), len(promotable), promotions_inserted, len(out),
    )
    return out
