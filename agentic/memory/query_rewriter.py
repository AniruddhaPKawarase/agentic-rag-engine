"""
agentic.memory.query_rewriter
=============================

Agent 2 of the v3.1 generation chain — **Query Rewriter**.

Sits immediately after Memory Recall and immediately before the
ReAct retrieval agent. Job: turn a follow-up like "what about the
second one?" into a self-contained query the retrieval agent can
answer without prior conversation history.

Latency-first design
--------------------
The rewriter is on the user's hot path. We DO NOT call the LLM unless
we have positive evidence the question actually depends on prior
context. The skip rules, in order:

1. ``QUERY_REWRITER_ENABLED != "true"`` — global kill switch.
2. ``memory_context.had_context`` is False — no session, no prior
   turns. Pass-through.
3. **Anaphora heuristic** (``QUERY_REWRITER_SKIP_HEURISTIC == "true"``,
   default on): if the lower-cased query contains none of the known
   coreference markers AND the query is already substantive
   (>30 chars), assume it is self-contained and skip the LLM. This
   catches e.g. "What are the fire safety requirements for atrium
   smoke control?" — long, no pronouns, definitely standalone.

If we do call the LLM we run a tight, low-temperature prompt and
treat the LLM's output with suspicion: oversized output or output
that contains preamble (``"Here is..."``, ``"Rewritten:..."``) is
discarded in favour of the original query.

Note on streaming
-----------------
The ``stream`` parameter is accepted for signature consistency with
the rest of the chain but is ignored — the rewriter's output is a
single short string and there's nothing useful to stream.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("agentic_rag.memory.rewriter")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Coreference / continuation markers. Spaces are intentional — they
# stop us matching e.g. "italic" because "it" appears as a substring.
_ANAPHORA_MARKERS: tuple[str, ...] = (
    "it ",
    "its ",
    "that ",
    "those ",
    "these ",
    "this ",
    "second one",
    "first one",
    "other one",
    "same",
    "also",
    "again",
    "what about",
    "how about",
    "and ",
    " them ",
    "they ",
    " he ",
    " she ",
)

# When the LLM's output starts with these patterns it has produced
# preamble instead of the rewritten query — discard.
_FORBIDDEN_OUTPUT_PREFIXES: tuple[str, ...] = (
    "here is",
    "here's",
    "rewritten:",
    "rewritten query:",
    "the rewritten",
    "sure,",
    "sure!",
    "i've rewritten",
    "i have rewritten",
)

_DEFAULT_REWRITER_MODEL = "gpt-4o-mini"
_DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"
_REWRITER_MAX_TOKENS = 200
_REWRITER_TEMPERATURE = 0.1
_REWRITER_OUTPUT_MAX_RATIO = 3.0
_HEURISTIC_MIN_QUERY_LEN = 30


_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a self-contained "
    "query that an independent search agent can answer without prior "
    "conversation history. Resolve pronouns and references using the "
    "provided context. If the question is already self-contained, "
    "return it unchanged. Output ONLY the rewritten query — no "
    "preamble, no explanation."
)


def _flag_enabled(name: str, default: str = "true") -> bool:
    """Truthy env-var check. Defaults to enabled when unset."""
    return os.getenv(name, default).strip().lower() == "true"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rewrite(
    *,
    user_query: str,
    memory_context: Dict[str, Any],
    stream: bool = False,  # accepted for chain consistency; ignored
) -> Dict[str, Any]:
    """Resolve coreference in ``user_query`` using memory context.

    Parameters
    ----------
    user_query:
        The user's fresh question.
    memory_context:
        Output of ``recall_agent.recall(...)``. The function reads
        ``had_context``, ``rolling_summary``, ``recent_turns``, and
        ``semantic_turns``.
    stream:
        Ignored — kept on the signature so callers can wire the same
        flag through every chain step. The rewriter's response is a
        single short string; streaming gives no UX benefit.

    Returns
    -------
    dict
        ``contextualized_query`` (str), ``was_rewritten`` (bool),
        ``skip_reason`` (str | None). The query field is always
        populated and safe to feed into retrieval.
    """
    # ---- Skip rule 1: kill switch ----
    if not _flag_enabled("QUERY_REWRITER_ENABLED"):
        return _passthrough(user_query, "flag_off")

    # ---- Skip rule 2: no session context to resolve against ----
    if not memory_context or not memory_context.get("had_context"):
        return _passthrough(user_query, "no_session_context")

    # ---- Skip rule 3: anaphora heuristic ----
    if _flag_enabled("QUERY_REWRITER_SKIP_HEURISTIC"):
        if _looks_self_contained(user_query):
            return _passthrough(user_query, "no_anaphora_markers")

    # ---- Build prompt and call LLM ----
    user_prompt = _build_user_prompt(user_query, memory_context)

    try:
        from agentic.generation.llm_client import generate

        rewritten_raw = generate(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=os.getenv("REWRITER_MODEL", _DEFAULT_REWRITER_MODEL),
            fallback_model=_DEFAULT_FALLBACK_MODEL,
            max_tokens=_REWRITER_MAX_TOKENS,
            temperature=_REWRITER_TEMPERATURE,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("rewriter LLM call failed: %s", exc)
        return _passthrough(user_query, "rewriter_llm_error")

    # The generate() contract returns str when stream=False, but be
    # defensive in case a mock returns something odd.
    if not isinstance(rewritten_raw, str):
        return _passthrough(user_query, "rewriter_output_invalid")

    rewritten = _clean_output(rewritten_raw)

    # ---- Sanity guards on the output ----
    if not rewritten:
        return _passthrough(user_query, "rewriter_output_invalid")

    if len(rewritten) > _REWRITER_OUTPUT_MAX_RATIO * max(len(user_query), 1):
        logger.debug(
            "rewriter output rejected (too long): %d > %.1f * %d",
            len(rewritten), _REWRITER_OUTPUT_MAX_RATIO, len(user_query),
        )
        return _passthrough(user_query, "rewriter_output_invalid")

    rewritten_lower = rewritten.lower().lstrip()
    for bad in _FORBIDDEN_OUTPUT_PREFIXES:
        if rewritten_lower.startswith(bad):
            logger.debug("rewriter output rejected (forbidden prefix): %r", bad)
            return _passthrough(user_query, "rewriter_output_invalid")

    # ---- Detect pass-through where LLM correctly returned the query unchanged ----
    if _normalise(rewritten) == _normalise(user_query):
        return {
            "contextualized_query": user_query,
            "was_rewritten": False,
            "skip_reason": "already_self_contained",
        }

    return {
        "contextualized_query": rewritten,
        "was_rewritten": True,
        "skip_reason": None,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _passthrough(user_query: str, reason: str) -> Dict[str, Any]:
    return {
        "contextualized_query": user_query,
        "was_rewritten": False,
        "skip_reason": reason,
    }


def _looks_self_contained(user_query: str) -> bool:
    """True when the heuristic says we should NOT call the rewriter.

    Conditions: query has no anaphora markers AND is long enough
    (>30 chars) to be a substantive standalone question.
    """
    q = " " + (user_query or "").lower() + " "
    has_marker = any(marker in q for marker in _ANAPHORA_MARKERS)
    if has_marker:
        return False
    return len(user_query or "") > _HEURISTIC_MIN_QUERY_LEN


def _build_user_prompt(user_query: str, mem: Dict[str, Any]) -> str:
    """Pack rolling summary + last 4 recent + top 2 semantic turns +
    the fresh question into a compact prompt body.
    """
    parts: list[str] = []

    summary = mem.get("rolling_summary")
    if summary:
        parts.append(f"Conversation summary so far:\n{summary}")

    recent = mem.get("recent_turns") or []
    if recent:
        last_four = recent[-4:]
        lines = [
            f"[{t.get('role', 'user')}] {t.get('content', '')}"
            for t in last_four
        ]
        parts.append("Recent turns:\n" + "\n".join(lines))

    semantic = mem.get("semantic_turns") or []
    if semantic:
        top_two = semantic[:2]
        lines = [
            f"[{t.get('role', 'user')}] {t.get('text_excerpt', t.get('content', ''))}"
            for t in top_two
        ]
        parts.append("Relevant earlier turns:\n" + "\n".join(lines))

    parts.append(f"Current question: {user_query}")
    return "\n\n".join(parts)


def _clean_output(raw: str) -> str:
    """Trim whitespace and strip a single layer of surrounding quotes."""
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def _normalise(s: str) -> str:
    """Whitespace-collapsed, lower-cased form for equality compares."""
    return " ".join((s or "").lower().split())
