"""
Multi-source answer synthesizer.

Merges answers from RAG, email, and meeting sources into a single
final_answer. For single-source results, returns that answer directly.
For multi-source, calls the LLM with priority instructions:
  email / meeting  ->  HIGH  (project communications, recorded decisions)
  rag              ->  MEDIUM (drawings & specs)
  web              ->  LOW   (general knowledge)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

# Hallucination Guard v2 -- Pillar 6 anti-anchor (env-flag gated; no-op when off)
try:
    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
except Exception:  # pragma: no cover
    def _hg_p6_wrap(s: str) -> str:
        return s

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None


def _openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


_SYSTEM = (
    "You are a project assistant synthesizing answers from multiple data sources. "
    "Be concise and factual. "
    "For factual data (counts, dates, names, statuses, actions taken): "
    "prefer Email and Meeting sources over Drawings/Specs when they conflict."
)

_USER = """\
Question: {question}

{source_blocks}

Provide a single unified answer. If sources give different facts, cite which \
source supports each fact."""

_SOURCE_LABELS = {
    "email":   "Email - project communications (HIGH priority)",
    "meeting": "Meeting Notes - recorded decisions (HIGH priority)",
    "rfi":     "RFI - formal requests for information (HIGH priority)",
    "rag":     "Drawings & Specs - reference documents (MEDIUM priority)",
    "web":     "Web Search - general knowledge (LOW priority)",
}

# Preference order when synthesis fails or only one source needed
_PREFERENCE = ["email", "meeting", "rfi", "rag", "web"]


async def synthesize_answer(
    question: str,
    answers: dict[str, str],
    model: str = "gpt-4.1",
    timeout: int = 30,
) -> str:
    """Return synthesized answer from non-empty per-source answers.

    answers keys: "rag" | "email" | "meeting" | "web"
    Falls back to best single source on LLM error.
    """
    if not answers:
        return ""
    if len(answers) == 1:
        return next(iter(answers.values()))

    # Build ordered source blocks (high-priority first)
    ordered_keys = [k for k in _PREFERENCE if k in answers] + [
        k for k in answers if k not in _PREFERENCE
    ]
    blocks = "\n\n".join(
        f"--- {_SOURCE_LABELS.get(k, k.upper())} ---\n{answers[k]}"
        for k in ordered_keys
    )
    prompt = _USER.format(question=question, source_blocks=blocks)

    try:
        completion = await asyncio.wait_for(
            _openai().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _hg_p6_wrap(_SYSTEM)},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0,
            ),
            timeout=timeout,
        )
        return completion.choices[0].message.content.strip()

    except Exception as exc:
        logger.warning("Multi-source synthesis failed (%s) - using best single source", exc)
        for preferred in _PREFERENCE:
            if preferred in answers:
                return answers[preferred]
        return next(iter(answers.values()))
