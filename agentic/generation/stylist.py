"""Agent 5: Style/Tone Rewriter (`stylize`) and Agent 5b: Cache
Re-Expression (`reexpress_cached`).

Both wrap a single LLM call pattern and share a model + fallback. They differ
only in their system prompts and skip rules, because the goal is the same:
make the answer *feel* like a fresh, contextual ChatGPT/Claude reply.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterator, Optional, Union

from agentic.generation.llm_client import generate
from agentic.generation.text_normalizer import normalize_chunks, normalize_output

logger = logging.getLogger("agentic_rag.generation.stylist")

# ── Tunables ───────────────────────────────────────────────────────────
_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_FALLBACK = "gpt-4o-mini"

# A "jargon-heavy" run = 2+ ALL-CAPS technical tokens (>=3 chars) within
# a small window. This is a cheap proxy for "this still reads like a dump."
_ALLCAPS_TOKEN_RE = re.compile(r"\b[A-Z]{3,}(?:[-/][A-Z]{2,})*\b")
_JARGON_PROXIMITY_WINDOW = 60  # chars between consecutive ALL-CAPS tokens


# ── Public API: stylize ────────────────────────────────────────────────

def stylize(
    *,
    draft_answer: str,
    user_query: str,
    tone_preference: str = "concise_professional",
    last_assistant_turn: Optional[str] = None,
    stream: bool = False,
    answer_shape: Optional[dict] = None,
) -> Union[str, Iterator[str]]:
    """Polish a draft into a natural, ChatGPT-style reply.

    Args:
        draft_answer: The post-synthesizer answer.
        user_query: Original user query.
        tone_preference: Tone hint (free-form). Default ``concise_professional``.
        last_assistant_turn: Previous assistant message, if any. When present
            the polished answer can reference it for continuity.
        stream: Yield chunks when True; return full string when False.
        answer_shape: Optional shape classifier output (Agent 0.5). When
            ``None`` or ``shape='default'`` the prompts and skip threshold
            are unchanged from legacy behaviour.

    Returns:
        Either a final string or, when ``stream=True``, an iterator of chunks.
    """
    target_chars = _shape_target_chars(answer_shape)
    if _should_skip_stylize(draft_answer, target_chars=target_chars):
        logger.debug(
            "stylist skip draft_len=%d (short + low jargon density)",
            len(draft_answer),
        )
        if stream:
            return _yield_once(normalize_output(draft_answer))
        return normalize_output(draft_answer)

    model = os.getenv("STYLIST_MODEL", _DEFAULT_MODEL)
    fallback_model = os.getenv("STYLIST_MODEL_FALLBACK", _DEFAULT_FALLBACK)

    system_prompt = _STYLIZE_SYSTEM_PROMPT
    shape_block = _shape_block(answer_shape)
    if shape_block:
        system_prompt = f"{shape_block}\n\n{system_prompt}"

    user_prompt = _build_stylize_user_prompt(
        draft_answer=draft_answer,
        user_query=user_query,
        tone_preference=tone_preference,
        last_assistant_turn=last_assistant_turn,
    )

    max_tokens = _shape_max_tokens(answer_shape, default=400)

    logger.info(
        "stylist.invoke model=%s fallback=%s draft_len=%d followup=%s stream=%s shape=%s",
        model, fallback_model, len(draft_answer),
        last_assistant_turn is not None, stream,
        (answer_shape or {}).get("shape"),
    )

    result = generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        fallback_model=fallback_model,
        max_tokens=max_tokens,
        temperature=0.4,
        stream=stream,
    )
    if stream:
        return normalize_chunks(result)
    return normalize_output(result) if isinstance(result, str) else result


# ── Public API: reexpress_cached ───────────────────────────────────────

def reexpress_cached(
    *,
    cached_answer: str,
    user_query: str,
    rolling_summary: Optional[str] = None,
    last_assistant_turn: Optional[str] = None,
    is_followup: bool = False,
    stream: bool = False,
    answer_shape: Optional[dict] = None,
) -> Union[str, Iterator[str]]:
    """Re-express a cached answer to feel fresh and contextual.

    Skips entirely when there's no session context to weave in — replaying
    the exact cached string is cheaper and indistinguishable from a re-write
    when no prior turns exist.

    Args:
        cached_answer: The cached answer to be re-expressed.
        user_query: Current user question.
        rolling_summary: Optional 1-line conversation summary.
        last_assistant_turn: Previous assistant turn, if any.
        is_followup: Caller-determined hint that this query is a follow-up.
        stream: Yield chunks when True; return full string when False.

    Returns:
        Either a final string or, when ``stream=True``, an iterator of chunks.
    """
    if last_assistant_turn is None and rolling_summary is None and not is_followup:
        logger.debug(
            "reexpress_cached skip (no session context to re-express against)"
        )
        if stream:
            return _yield_once(normalize_output(cached_answer))
        return normalize_output(cached_answer)

    model = os.getenv("STYLIST_MODEL", _DEFAULT_MODEL)
    fallback_model = os.getenv("STYLIST_MODEL_FALLBACK", _DEFAULT_FALLBACK)

    system_prompt = _REEXPRESS_SYSTEM_PROMPT
    shape_block = _shape_block(answer_shape)
    if shape_block:
        system_prompt = f"{shape_block}\n\n{system_prompt}"

    user_prompt = _build_reexpress_user_prompt(
        cached_answer=cached_answer,
        user_query=user_query,
        rolling_summary=rolling_summary,
        last_assistant_turn=last_assistant_turn,
        is_followup=is_followup,
    )

    max_tokens = _shape_max_tokens(answer_shape, default=400)

    logger.info(
        "reexpress_cached.invoke model=%s fallback=%s cached_len=%d followup=%s stream=%s shape=%s",
        model, fallback_model, len(cached_answer), is_followup, stream,
        (answer_shape or {}).get("shape"),
    )

    result = generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        fallback_model=fallback_model,
        max_tokens=max_tokens,
        temperature=0.5,
        stream=stream,
    )
    if stream:
        return normalize_chunks(result)
    return normalize_output(result) if isinstance(result, str) else result


# ── Skip rule helpers ──────────────────────────────────────────────────

def _should_skip_stylize(draft_answer: str, *, target_chars: int = 0) -> bool:
    """Skip when draft is short *and* not jargon-heavy.

    When a shape is set (target_chars > 0), the threshold is tightened to
    ``min(STYLE_SKIP_THRESHOLD_CHARS, target_chars * 0.6)`` so a 100-char
    factoid still gets polished even when below the legacy 300-char cap.
    """
    threshold = int(os.getenv("STYLE_SKIP_THRESHOLD_CHARS", "300"))
    if target_chars and target_chars > 0:
        threshold = min(threshold, int(target_chars * 0.6))
    if len(draft_answer) >= threshold:
        return False
    return not _is_jargon_heavy(draft_answer)


def _shape_block(answer_shape: Optional[dict]) -> Optional[str]:
    """Build the shape system-prompt prefix or None when shape is missing/default."""
    if not answer_shape:
        return None
    from agentic.generation.answer_shape import shape_prompt_block

    return shape_prompt_block(answer_shape)


def _shape_target_chars(answer_shape: Optional[dict]) -> int:
    if not answer_shape:
        return 0
    shape = answer_shape.get("shape")
    if not shape or shape == "default":
        return 0
    try:
        return int(answer_shape.get("target_length_chars") or 0)
    except (TypeError, ValueError):
        return 0


def _shape_max_tokens(answer_shape: Optional[dict], *, default: int) -> int:
    """Reduce ``max_tokens`` proportionally to the shape's word budget."""
    if not answer_shape:
        return default
    shape = answer_shape.get("shape")
    if not shape or shape == "default":
        return default
    try:
        target_words = int(answer_shape.get("target_word_count") or 0)
    except (TypeError, ValueError):
        return default
    if target_words <= 0:
        return default
    return max(80, target_words * 2)


def _is_jargon_heavy(text: str) -> bool:
    """True if >=2 ALL-CAPS tech tokens cluster within a short window.

    This avoids mis-flagging single acronyms like "API" or "RFI" while
    catching dumps like "RCP HVAC VAV CFM AHU" that need polishing.
    """
    matches = list(_ALLCAPS_TOKEN_RE.finditer(text))
    if len(matches) < 2:
        return False
    for i in range(len(matches) - 1):
        gap = matches[i + 1].start() - matches[i].end()
        if 0 <= gap <= _JARGON_PROXIMITY_WINDOW:
            return True
    return False


def _yield_once(text: str) -> Iterator[str]:
    yield text


# ── Prompts ────────────────────────────────────────────────────────────

_STYLIZE_SYSTEM_PROMPT = (
    "You polish technical answers into natural, ChatGPT-style replies.\n"
    "\n"
    "Rules:\n"
    "- Keep all citations like [S-101 p1] exactly as they appear.\n"
    "- Vary sentence rhythm — no robotic, mechanical phrasing.\n"
    "- No greetings ('Hi', 'Hello'), no sign-offs ('Hope this helps').\n"
    "- No headers ('Answer:', 'Summary:'). No closing wrap-ups.\n"
    "- Preserve every fact and number — never invent or omit information.\n"
    "- If `last_assistant_turn` is present, the user is following up — "
    "briefly reference the prior context to show continuity (one short clause, "
    "not a full recap).\n"
    "- Output the polished answer text only — nothing else."
)

_REEXPRESS_SYSTEM_PROMPT = (
    "You're re-expressing a previously-cached answer so it feels fresh and "
    "contextual to the current conversation. Preserve every fact and citation "
    "(e.g. [S-101 p1]) — only the phrasing should change.\n"
    "\n"
    "Rules:\n"
    "- If the user is mid-conversation, weave in light continuity, e.g. "
    "'Building on what we found earlier…' or address `last_assistant_turn` "
    "directly. Keep it brief — one clause, not a recap.\n"
    "- Never reveal that the answer is cached or pre-computed.\n"
    "- Never invent new facts or change numbers/citations.\n"
    "- No greetings, sign-offs, or section headers.\n"
    "- Output the re-expressed answer text only — nothing else."
)


def _build_stylize_user_prompt(
    *,
    draft_answer: str,
    user_query: str,
    tone_preference: str,
    last_assistant_turn: Optional[str],
) -> str:
    parts: list[str] = []
    parts.append(f"USER QUERY:\n{user_query}\n")
    parts.append(f"TONE PREFERENCE: {tone_preference}\n")
    if last_assistant_turn:
        parts.append(
            f"PREVIOUS ASSISTANT TURN (for continuity):\n{last_assistant_turn.strip()}\n"
        )
    parts.append(f"DRAFT ANSWER (polish this):\n{draft_answer}\n")
    parts.append("Now output the polished version.")
    return "\n".join(parts)


def _build_reexpress_user_prompt(
    *,
    cached_answer: str,
    user_query: str,
    rolling_summary: Optional[str],
    last_assistant_turn: Optional[str],
    is_followup: bool,
) -> str:
    parts: list[str] = []
    parts.append(f"USER QUERY:\n{user_query}\n")
    parts.append(f"FOLLOW-UP: {'yes' if is_followup else 'no'}\n")
    if rolling_summary:
        parts.append(f"CONVERSATION SUMMARY (1-line):\n{rolling_summary.strip()}\n")
    if last_assistant_turn:
        parts.append(
            f"PREVIOUS ASSISTANT TURN:\n{last_assistant_turn.strip()}\n"
        )
    parts.append(f"CACHED ANSWER (re-express this, preserving facts):\n{cached_answer}\n")
    parts.append("Now output the re-expressed version.")
    return "\n".join(parts)
