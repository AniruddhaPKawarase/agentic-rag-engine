"""Agent 4: Answer Synthesizer.

Compresses verbose ReAct agent output into the *shortest* response that fully
answers the user's question, while preserving every inline citation found in
the raw answer (e.g. ``[S-101 p1]``).

Skip rules (latency-first):
    - raw_answer shorter than ``MIN_LEN_FOR_SYNTH`` chars AND
      no OCR-garble markers → pass through unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterator, Optional, Union

from agentic.generation.llm_client import generate
from agentic.generation.text_normalizer import normalize_chunks, normalize_output

logger = logging.getLogger("agentic_rag.generation.synthesizer")

# ── Tunables ───────────────────────────────────────────────────────────
MIN_LEN_FOR_SYNTH = 200          # below this → pass-through
MAX_RAW_ANSWER_CHARS = 16000     # truncate raw_answer in user prompt
TRUNC_HEAD_CHARS = 10000         # head slice when tail-preserving truncation kicks in
TRUNC_TAIL_CHARS = 5000          # tail slice when tail-preserving truncation kicks in
TOP_K_SOURCES_FOR_PROMPT = 5     # source_docs slice
SOURCE_EXCERPT_CHARS = 200       # per-source excerpt length

# Citation marker pattern — e.g. [S-101 p1], [E-604 p3], [ARCH-12a p10].
_CITATION_RE = re.compile(r"\[[A-Z]{1,4}-?\d+[a-zA-Z]?\s+p\d+\]")

# OCR garble markers — if any present we DO want to synthesize even on
# short answers, because the LLM can clean them up.
_OCR_GARBLE_MARKERS = ("▯", "□", "???")


# ── Public API ─────────────────────────────────────────────────────────

def synthesize(
    *,
    raw_answer: str,
    user_query: str,
    source_docs: list[dict],
    rolling_summary: Optional[str] = None,
    stream: bool = False,
    answer_shape: Optional[dict] = None,
) -> Union[str, Iterator[str]]:
    """Compress a raw ReAct answer into a short, citation-preserving reply.

    Args:
        raw_answer: The verbatim ``AgentResult.answer`` from the ReAct loop.
        user_query: The original user question (used for query-shape decisions).
        source_docs: Top-k retrieved source docs. Each dict should have
            ``drawing_name``, ``page``, ``text_excerpt`` (or ``text``).
        rolling_summary: Optional 1-line conversation summary for continuity.
        stream: Yield chunks when True; return full string when False.
        answer_shape: Optional shape classifier output (Agent 0.5). When
            ``None`` or ``shape='default'``, the system prompt and skip
            threshold are unchanged from the legacy behaviour. When a
            real shape is set, a length-budget prefix is injected and
            ``max_tokens`` is reduced proportionally.

    Returns:
        Either a final string or, when ``stream=True``, an iterator of chunks.
    """
    shape_block = _shape_block(answer_shape)
    target_chars = _shape_target_chars(answer_shape)

    if _should_skip(raw_answer, target_chars=target_chars):
        logger.debug(
            "synthesizer skip raw_len=%d (under threshold, no garble markers)",
            len(raw_answer),
        )
        if stream:
            return _yield_once(normalize_output(raw_answer))
        return normalize_output(raw_answer)

    model = os.getenv("SYNTHESIZER_MODEL", "claude-haiku-4-5")
    fallback_model = os.getenv("SYNTHESIZER_MODEL_FALLBACK", "gpt-4o-mini")

    system_prompt = _SYSTEM_PROMPT
    if shape_block:
        system_prompt = f"{shape_block}\n\n{system_prompt}"

    user_prompt = _build_user_prompt(
        raw_answer=raw_answer,
        user_query=user_query,
        source_docs=source_docs,
        rolling_summary=rolling_summary,
    )

    max_tokens = _shape_max_tokens(answer_shape, default=400)

    logger.info(
        "synthesizer.invoke model=%s fallback=%s raw_len=%d sources=%d stream=%s shape=%s",
        model, fallback_model, len(raw_answer), len(source_docs or []), stream,
        (answer_shape or {}).get("shape"),
    )

    result = generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        fallback_model=fallback_model,
        max_tokens=max_tokens,
        temperature=0.2,
        stream=stream,
    )
    if stream:
        return normalize_chunks(result)
    return normalize_output(result) if isinstance(result, str) else result


# ── Internals ──────────────────────────────────────────────────────────

def _should_skip(raw_answer: str, *, target_chars: int = 0) -> bool:
    """Pass-through rule: short *and* no OCR-garble markers.

    When a shape is set (target_chars > 0), the skip threshold is
    tightened to ``min(MIN_LEN_FOR_SYNTH, target_chars * 0.7)`` so we
    don't accidentally pass through a 100-char answer that exceeds a
    factoid's tighter budget.
    """
    threshold = MIN_LEN_FOR_SYNTH
    if target_chars and target_chars > 0:
        threshold = min(MIN_LEN_FOR_SYNTH, int(target_chars * 0.7))
    if len(raw_answer) >= threshold:
        return False
    return not any(marker in raw_answer for marker in _OCR_GARBLE_MARKERS)


def _shape_block(answer_shape: Optional[dict]) -> Optional[str]:
    """Lazy import so unit-testing answer_shape doesn't loop on synthesizer."""
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
    """Reduce ``max_tokens`` to roughly fit the shape's word budget.

    Formula: ``max(80, target_word_count * 2)`` — the 2x slack lets the
    LLM finish its sentence without blowing past the budget. Returns
    ``default`` when no shape is set, preserving legacy behaviour.
    """
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


def _yield_once(text: str) -> Iterator[str]:
    """Wrap a complete string into a single-chunk generator (stream contract)."""
    yield text


_SYSTEM_PROMPT = (
    "You're a construction-document analyst. Compress the raw answer into the "
    "shortest response that fully answers the user's question, preserving all "
    "factual content and every inline citation like [S-101 p1].\n"
    "\n"
    "Rules:\n"
    "- Factual queries: respond in 1-2 sentences.\n"
    "- Explanatory queries: respond in at most 6 sentences.\n"
    "- Use bullets only when listing 3 or more items.\n"
    "- Preserve every [drawing pX] citation that appears in the raw answer.\n"
    "- Never invent facts. If the raw answer doesn't contain something, do not add it.\n"
    "- No headers like 'Direct Answer:', 'Summary:', or 'Conclusion:'.\n"
    "- No closing wrap-ups or 'I hope this helps' lines.\n"
    "- No 'Based on the documents...' / 'According to the drawings...' preambles.\n"
    "- Output the answer text only — nothing else."
)


def _build_user_prompt(
    *,
    raw_answer: str,
    user_query: str,
    source_docs: list[dict],
    rolling_summary: Optional[str],
) -> str:
    """Pack the user-side context for the LLM."""
    truncated = _truncate_preserving_tail_citations(raw_answer)

    sources_summary = _summarize_sources(source_docs or [])

    parts: list[str] = []
    parts.append(f"USER QUERY:\n{user_query}\n")
    if rolling_summary:
        parts.append(f"CONVERSATION SO FAR (1-line):\n{rolling_summary.strip()}\n")
    parts.append(f"RAW ANSWER (from retrieval agent):\n{truncated}\n")
    parts.append(
        "TOP SOURCES (JSON, for citation grounding only — do NOT introduce "
        f"new facts):\n{sources_summary}\n"
    )
    parts.append(
        "Now produce the shortest faithful answer per the rules above."
    )
    return "\n".join(parts)


def _truncate_preserving_tail_citations(raw_answer: str) -> str:
    """Truncate ``raw_answer`` so the user prompt stays bounded.

    Default strategy is a head-only slice to ``MAX_RAW_ANSWER_CHARS``. If
    that would drop one or more citation markers from the tail, we
    switch to a head + tail strategy (``TRUNC_HEAD_CHARS`` from the
    front, ``TRUNC_TAIL_CHARS`` from the back, joined by an ellipsis
    notice) so tail citations like ``[S-101 p1]`` survive.
    """
    if len(raw_answer) <= MAX_RAW_ANSWER_CHARS:
        return raw_answer

    head_only = raw_answer[:MAX_RAW_ANSWER_CHARS]
    full_citations = _CITATION_RE.findall(raw_answer)
    head_citations = _CITATION_RE.findall(head_only)
    if len(full_citations) <= len(head_citations):
        # No tail citations would be dropped — head slice is fine.
        return head_only + "\n…[truncated]"

    head = raw_answer[:TRUNC_HEAD_CHARS]
    tail = raw_answer[-TRUNC_TAIL_CHARS:]
    return f"{head}\n\n[…earlier passages omitted…]\n\n{tail}"


def _summarize_sources(source_docs: list[dict]) -> str:
    """Compact JSON of top-k sources, citation-focused."""
    summary = []
    for doc in source_docs[:TOP_K_SOURCES_FOR_PROMPT]:
        excerpt_raw = doc.get("text_excerpt") or doc.get("text") or ""
        if not isinstance(excerpt_raw, str):
            excerpt_raw = str(excerpt_raw)
        summary.append({
            "drawing_name": doc.get("drawing_name") or doc.get("drawing"),
            "page": doc.get("page"),
            "text_excerpt": excerpt_raw[:SOURCE_EXCERPT_CHARS],
        })
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
