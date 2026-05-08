"""LLM-as-judge scoring helpers for the v3.1 eval harness.

Uses gpt-4o (single call, temperature 0, max 50 tokens) to grade pairs
of (question, answer) and (turn1, turn2) interactions on a 1-5 rubric.

All scoring failures are non-fatal: a -1 is returned and logged so the
caller can drop the row from aggregates.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("agentic_rag.eval.llm_judge")

# Default judge model. Not configurable from CLI yet — change here if needed.
_JUDGE_MODEL = "gpt-4o"
_JUDGE_FALLBACK = "claude-haiku-4-5"
_JUDGE_MAX_TOKENS = 50
_JUDGE_TEMPERATURE = 0.0


_HUMAN_TONE_RUBRIC = """\
You are scoring how human / natural an AI assistant's answer reads on a 1-5 scale.

Rubric:
1 = Robotic database dump. Bullet-only, no transitions, reads like raw SQL output.
2 = Mostly mechanical with token attempts at human phrasing.
3 = Acceptable. Mix of mechanical and natural prose.
4 = Natural and conversational. Reads like a knowledgeable colleague explaining.
5 = Excellent. ChatGPT-style explanation: clear opener, organized body, helpful close.

Output ONLY a single integer 1-5. No words. No explanation. No punctuation.
"""

_MULTI_TURN_RUBRIC = """\
You are scoring whether a 2-turn conversation demonstrates that the assistant
correctly used context from turn 1 to answer turn 2.

Rubric:
1 = Ignored prior turn entirely. Answered turn 2 as if turn 1 never happened.
2 = Hint of prior context but mostly disconnected.
3 = Partial reference resolution. Got the topic but missed specifics.
4 = Clearly used prior turn. Resolved pronouns / implicit references correctly.
5 = Excellent. Seamlessly built on turn 1 — pronouns resolved, scope carried over,
    implicit references decoded perfectly.

Output ONLY a single integer 1-5. No words. No explanation. No punctuation.
"""


def _parse_score(text: str) -> int:
    """Pull the first integer 1-5 out of the model's response.

    Returns -1 when nothing parseable is found (caller treats as missing).
    """
    if not text:
        return -1
    match = re.search(r"[1-5]", text)
    if match:
        try:
            value = int(match.group(0))
            if 1 <= value <= 5:
                return value
        except ValueError:
            pass
    return -1


def _judge_call(system_prompt: str, user_prompt: str) -> int:
    """One-shot judge call. Returns -1 on any failure."""
    try:
        from agentic.generation.llm_client import generate
    except Exception as exc:
        logger.error("llm_judge: cannot import llm_client: %s", exc)
        return -1

    try:
        out = generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=_JUDGE_MODEL,
            fallback_model=_JUDGE_FALLBACK,
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=_JUDGE_TEMPERATURE,
            stream=False,
        )
    except Exception as exc:
        logger.warning("llm_judge call failed: %s", exc)
        return -1

    text = out if isinstance(out, str) else ""
    score = _parse_score(text)
    if score == -1:
        logger.warning("llm_judge: could not parse score from response: %r", text[:80])
    return score


def score_human_tone(question: str, answer: str) -> int:
    """Score 1-5 how human / natural an answer reads.

    Args:
        question: The user's question (for context).
        answer: The assistant's answer to grade.

    Returns:
        Integer 1-5 on success, -1 on parse / API failure.
    """
    if not answer:
        return -1
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Output a single integer 1-5."
    )
    return _judge_call(system_prompt=_HUMAN_TONE_RUBRIC, user_prompt=user_prompt)


def score_multi_turn_coherence(
    turn1_q: str,
    turn1_a: str,
    turn2_q: str,
    turn2_a: str,
) -> int:
    """Score 1-5 whether turn 2 used context from turn 1 correctly.

    Args:
        turn1_q: First-turn user question.
        turn1_a: First-turn assistant answer.
        turn2_q: Second-turn user question (often referential).
        turn2_a: Second-turn assistant answer to grade.

    Returns:
        Integer 1-5 on success, -1 on parse / API failure.
    """
    if not turn2_a:
        return -1
    user_prompt = (
        f"Turn 1 user: {turn1_q}\n"
        f"Turn 1 assistant: {turn1_a}\n\n"
        f"Turn 2 user: {turn2_q}\n"
        f"Turn 2 assistant: {turn2_a}\n\n"
        "Did the assistant correctly resolve the reference in turn 2 using turn 1?\n"
        "Output a single integer 1-5."
    )
    return _judge_call(system_prompt=_MULTI_TURN_RUBRIC, user_prompt=user_prompt)


def score_relative_tone(
    question: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
    label_a: str = "A",
    label_b: str = "B",
    label_c: str = "C",
) -> Optional[str]:
    """Pick which of three answers reads most human. Returns 'A'|'B'|'C'|None."""
    if not (answer_a and answer_b and answer_c):
        return None
    rubric = (
        "You are picking which of three AI answers reads MOST human and natural.\n"
        f"Output ONLY one letter: {label_a}, {label_b}, or {label_c}. No words.\n"
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Answer {label_a}:\n{answer_a}\n\n"
        f"Answer {label_b}:\n{answer_b}\n\n"
        f"Answer {label_c}:\n{answer_c}\n\n"
        f"Pick {label_a}, {label_b}, or {label_c}."
    )
    try:
        from agentic.generation.llm_client import generate

        out = generate(
            system_prompt=rubric,
            user_prompt=user_prompt,
            model=_JUDGE_MODEL,
            fallback_model=_JUDGE_FALLBACK,
            max_tokens=10,
            temperature=0.0,
            stream=False,
        )
    except Exception as exc:
        logger.warning("score_relative_tone failed: %s", exc)
        return None

    text = (out or "").strip().upper()
    for label in (label_a, label_b, label_c):
        if label.upper() in text:
            return label
    return None
