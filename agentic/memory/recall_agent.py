"""
agentic.memory.recall_agent
===========================

Agent 1 of the v3.1 generation chain — **Memory Recall**.

Sits at the very front of the chain. Given a session ID and the user's
fresh question, it pulls together everything we already know about the
conversation so the downstream Query Rewriter can produce a
self-contained query, and so the retrieval agent can be primed with
session context if needed.

Three signal sources are merged into the return value:

1. **rolling_summary** — session-level summary maintained by the memory
   writer (Agent 6). Falls back to the legacy ``custom_instructions``
   field on older session objects.
2. **recent_turns** — the last N messages by chronological order. We
   read them straight off ``ConversationSession.messages`` so we can
   include the true ``turn_index`` (needed for dedup against semantic
   results).
3. **semantic_turns** — top-K hits from
   ``SessionVectorStore.search(...)`` on the embedded query. Already-seen
   turns (by ``turn_index``) are dropped so the rewriter doesn't waste
   its context budget on duplicates.

Plus a lightweight ``topic_tags`` extraction (drawing identifiers like
``S-101`` or ``M101``, plus a fixed set of construction trades) — the
rewriter doesn't strictly need this, but it's cheap and useful for
logging / future routing.

Hard rules
----------
* This function NEVER raises. On any failure it returns a safe-empty
  payload with ``had_context: False`` so the chain can fall through to
  the no-context path without special-casing.
* The kill switch ``MEMORY_RECALL_ENABLED`` (default: enabled) lets ops
  disable the whole step in one knob.
* Imports of the heavy ``MemoryManager`` are deferred to call time —
  the module-level import would open a DB / S3 connection, and we
  don't want that on test collection.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_rag.memory.recall")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Construction trades we tag inside recent turns. Lower-cased comparison.
_TRADE_KEYWORDS: tuple[str, ...] = (
    "Mechanical",
    "Electrical",
    "Plumbing",
    "Structural",
    "Architectural",
    "Civil",
    "Fire",
    "Concrete",
    "Steel",
    "HVAC",
)

# Drawing identifiers like "S-101", "M101", "E604a", "MEP-12".
_DRAWING_REGEX = re.compile(r"\b[A-Z]{1,4}-?\d+[a-zA-Z]?\b")

_MAX_TOPIC_TAGS = 10
_TOPIC_SCAN_TURNS = 30


def _flag_enabled(name: str, default: str = "true") -> bool:
    """Truthy env-var check. Defaults to enabled when unset."""
    return os.getenv(name, default).strip().lower() == "true"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recall(
    *,
    session_id: Optional[str],
    user_query: str,
    top_k_recent: int = 6,
    top_k_semantic: int = 3,
) -> Dict[str, Any]:
    """Pull conversational context for ``user_query`` from session memory.

    Parameters
    ----------
    session_id:
        Session key. ``None`` short-circuits to the empty payload.
    user_query:
        The user's fresh question — used as the embedding seed for
        semantic recall.
    top_k_recent:
        How many of the most recent messages to include verbatim.
    top_k_semantic:
        How many semantic matches to fetch from the vector store.

    Returns
    -------
    dict
        Always contains the keys ``rolling_summary``, ``recent_turns``,
        ``semantic_turns``, ``topic_tags``, ``had_context``. When
        ``had_context`` is ``False`` all collections are empty and
        ``rolling_summary`` is ``None`` — callers can branch on that
        single flag.
    """
    empty = _empty_payload()

    # --- Skip rule: kill switch ---
    if not _flag_enabled("MEMORY_RECALL_ENABLED"):
        logger.debug("recall skipped: MEMORY_RECALL_ENABLED is off")
        return empty

    if not session_id:
        return empty

    try:
        # Lazy import — avoids touching the MemoryManager (which may
        # open a DB connection) at module-load time.
        from traditional.memory_manager import get_memory_manager

        mm = get_memory_manager()
        session = mm.get_session(session_id)
        if session is None:
            logger.debug("recall: session %s not found", session_id)
            return empty

        if not getattr(session, "messages", None):
            # Session exists but has no turns yet — same as no context.
            return empty

        rolling_summary = _extract_rolling_summary(session)
        recent_turns = _extract_recent_turns(session, top_k_recent)
        semantic_turns = _extract_semantic_turns(
            session_id=session_id,
            user_query=user_query,
            top_k=top_k_semantic,
            recent_turn_indices={t["turn_index"] for t in recent_turns},
        )
        topic_tags = _extract_topic_tags(session)

        return {
            "rolling_summary": rolling_summary,
            "recent_turns": recent_turns,
            "semantic_turns": semantic_turns,
            "topic_tags": topic_tags,
            "had_context": True,
        }
    except Exception as exc:  # noqa: BLE001 - swallowed by contract
        logger.warning("recall failed for %s: %s", session_id, exc)
        return empty


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _empty_payload() -> Dict[str, Any]:
    return {
        "rolling_summary": None,
        "recent_turns": [],
        "semantic_turns": [],
        "topic_tags": [],
        "had_context": False,
    }


def _extract_rolling_summary(session: Any) -> Optional[str]:
    """Prefer the explicit ``rolling_summary`` field; fall back to the
    legacy ``custom_instructions`` when running against an older
    ``ConversationContext`` that doesn't have the new field yet.
    """
    ctx = getattr(session, "context", None)
    if ctx is None:
        return None
    summary = getattr(ctx, "rolling_summary", None)
    if summary:
        return summary
    legacy = getattr(ctx, "custom_instructions", None)
    return legacy or None


def _extract_recent_turns(session: Any, top_k_recent: int) -> List[Dict[str, Any]]:
    """Return the last ``top_k_recent`` messages with their absolute
    turn_index. Content is capped at 500 chars to keep prompt budget
    in line with what the writer already stores in vectors.
    """
    messages = list(getattr(session, "messages", []) or [])
    if not messages:
        return []

    # Slice from the tail; preserve absolute turn_index from the full list.
    start = max(0, len(messages) - top_k_recent)
    out: List[Dict[str, Any]] = []
    for idx in range(start, len(messages)):
        msg = messages[idx]
        role = getattr(msg, "role", None) or "user"
        content = getattr(msg, "content", "") or ""
        out.append(
            {
                "role": role,
                "content": content[:500],
                "turn_index": idx,
            }
        )
    return out


def _extract_semantic_turns(
    *,
    session_id: str,
    user_query: str,
    top_k: int,
    recent_turn_indices: set,
) -> List[Dict[str, Any]]:
    """Embed the query and fetch top-K semantic matches from the vector
    store. Drop anything already in the recent-turns slice. Failures
    here degrade gracefully to an empty list — semantic recall is a
    nice-to-have, not a blocker.
    """
    if top_k <= 0 or not user_query or not user_query.strip():
        return []

    try:
        from agentic.memory.embeddings import embed_text
        from agentic.memory.vector_store import SessionVectorStore

        query_embedding = embed_text(user_query)
        if not query_embedding:
            return []

        timeout_ms = int(os.getenv("MEMORY_ATLAS_TIMEOUT_MS", "150"))
        store = SessionVectorStore()
        hits = store.search(
            session_id,
            query_embedding,
            top_k=top_k,
            atlas_timeout_ms=timeout_ms,
        ) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic recall failed for %s: %s", session_id, exc)
        return []

    deduped: List[Dict[str, Any]] = []
    for hit in hits:
        turn_index = hit.get("turn_index")
        if turn_index in recent_turn_indices:
            continue
        deduped.append(hit)
    return deduped


def _extract_topic_tags(session: Any) -> List[str]:
    """Cheap heuristic tagger over the last ``_TOPIC_SCAN_TURNS``
    messages. Returns up to ``_MAX_TOPIC_TAGS`` unique tags, drawing
    names preserved as-is and trade names returned in their canonical
    Title-case form from ``_TRADE_KEYWORDS``.
    """
    messages = list(getattr(session, "messages", []) or [])
    if not messages:
        return []

    window = messages[-_TOPIC_SCAN_TURNS:]
    blob = " ".join(getattr(m, "content", "") or "" for m in window)

    tags: List[str] = []
    seen: set = set()

    # Drawing identifiers — preserve original case for readability.
    for match in _DRAWING_REGEX.findall(blob):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            tags.append(match)
            if len(tags) >= _MAX_TOPIC_TAGS:
                return tags

    # Trades — case-insensitive presence check.
    blob_lower = blob.lower()
    for trade in _TRADE_KEYWORDS:
        key = trade.lower()
        if key in blob_lower and key not in seen:
            seen.add(key)
            tags.append(trade)
            if len(tags) >= _MAX_TOPIC_TAGS:
                break

    return tags
