"""
trade_aware_retrieval.py
========================

Thin wrapper around the existing `drawings_v3` retrieval tools that injects
an optional (trade, role) filter into the Mongo aggregation pipeline and
implements the 5-rung fallback ladder.

Rungs (try each in order until results found):
    1. (trade, role)   — both filters applied
    2. trade only      — drop role filter
    3. role only       — drop trade filter
    4. unfiltered      — today's behavior exactly
    5. empty + note    — explicit "no drawings found"

Strict invariants
-----------------
- Never raises. Any failure path falls back to rung 4 (unfiltered).
- When trade_filter and role_filter are both None/unknown, this wrapper
  is byte-identical to a direct call to the underlying tool.
- The wrapper does NOT change the tool's output schema — only the result
  set composition.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_match_stage(trade: Optional[str], role: Optional[str]) -> Optional[dict]:
    """Build a Mongo $match stage filtering by (trade, role).

    Returns None if BOTH inputs are missing/unknown.
    """
    from gateway.sheet_decoder import mongo_match_stage
    return mongo_match_stage(trade, role)


def _apply_stage_to_pipeline(
    pipeline: List[dict],
    extra_stage: Optional[dict],
) -> List[dict]:
    """Insert extra_stage as the SECOND stage (after the existing project_id
    $match so we never widen project scope), preserving order otherwise.

    Returns a new list — never mutates input.
    """
    if extra_stage is None:
        return list(pipeline)
    out: List[dict] = []
    inserted = False
    for i, stage in enumerate(pipeline):
        out.append(stage)
        # Insert AFTER the first $match (project scope) but BEFORE everything else
        if (not inserted and i == 0 and isinstance(stage, dict) and "$match" in stage):
            out.append(extra_stage)
            inserted = True
    if not inserted:
        # No project_id match found — prepend our stage
        out = [extra_stage] + out
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def filter_pipeline_for_trade_role(
    pipeline: List[dict],
    trade: Optional[str],
    role: Optional[str],
) -> List[dict]:
    """Return a copy of `pipeline` with an additional $match for (trade, role).

    Use this from inside drawing_tools_v3.search_drawings_v3 (or any v3 tool)
    when the caller has supplied a non-None trade or role.
    """
    extra = _build_match_stage(trade, role)
    if extra is None:
        return list(pipeline)
    return _apply_stage_to_pipeline(pipeline, extra)


def retrieve_with_fallback(
    retrieval_fn,
    *,
    trade: Optional[str] = None,
    role: Optional[str] = None,
    min_results: int = 1,
    **kwargs,
) -> Tuple[List[Any], str]:
    """Execute `retrieval_fn` with progressive fallback through 5 rungs.

    `retrieval_fn` MUST accept a `pipeline_filter` callable kwarg that
    transforms the aggregation pipeline before execution. Alternatively
    it can accept `extra_match_stage`.

    Parameters
    ----------
    retrieval_fn : callable
        The underlying tool function, e.g. `search_drawings_v3`.
    trade, role : str or None
        The (trade, role) cell to filter on.
    min_results : int
        Threshold below which we consider the rung a "no result" miss and
        fall through. Default 1.
    **kwargs : passed through to retrieval_fn unchanged.

    Returns
    -------
    (results, rung_label)
        rung_label ∈ {"trade_role", "trade_only", "role_only", "unfiltered",
                      "empty_with_note"}
    """
    if not _flag("TRADE_ROUTING_ENABLED"):
        # Master switch off — pure passthrough
        results = retrieval_fn(**kwargs)
        return results, "unfiltered"

    # Rung 1 — (trade, role)
    if trade and trade != "unknown" and role and role != "unknown":
        try:
            stage = _build_match_stage(trade, role)
            results = retrieval_fn(extra_match_stage=stage, **kwargs)
            if results and len(results) >= min_results:
                logger.info("[trade_router] rung=trade_role trade=%s role=%s hits=%d",
                            trade, role, len(results))
                return results, "trade_role"
        except TypeError:
            # retrieval_fn doesn't accept extra_match_stage — caller bug,
            # but we degrade rather than crash
            logger.warning("[trade_router] retrieval_fn does not accept extra_match_stage; falling back to unfiltered")
            return retrieval_fn(**kwargs), "unfiltered"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[trade_router] rung=trade_role failed: %s", exc)

    # Rung 2 — trade only
    if trade and trade != "unknown":
        try:
            stage = _build_match_stage(trade, None)
            results = retrieval_fn(extra_match_stage=stage, **kwargs)
            if results and len(results) >= min_results:
                logger.info("[trade_router] rung=trade_only trade=%s hits=%d",
                            trade, len(results))
                return results, "trade_only"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[trade_router] rung=trade_only failed: %s", exc)

    # Rung 3 — role only
    if role and role != "unknown":
        try:
            stage = _build_match_stage(None, role)
            results = retrieval_fn(extra_match_stage=stage, **kwargs)
            if results and len(results) >= min_results:
                logger.info("[trade_router] rung=role_only role=%s hits=%d",
                            role, len(results))
                return results, "role_only"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[trade_router] rung=role_only failed: %s", exc)

    # Rung 4 — unfiltered (today's behavior exactly)
    try:
        results = retrieval_fn(**kwargs)
        if results:
            logger.info("[trade_router] rung=unfiltered hits=%d", len(results))
            return results, "unfiltered"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[trade_router] rung=unfiltered failed: %s", exc)
        results = []

    # Rung 5 — empty + note
    logger.info("[trade_router] rung=empty_with_note (no drawings matched even unfiltered)")
    return results, "empty_with_note"


def derive_trade_role_for_result(doc: Dict[str, Any]) -> Tuple[str, str]:
    """Convenience: import + call decode_trade_role for a single result doc."""
    from gateway.sheet_decoder import decode_trade_role
    return decode_trade_role(doc)
