"""Agent 0.5: Answer Shape Classifier.

Classifies the user's query into one of 5 answer-length buckets so that
downstream Synthesizer + Stylist can produce appropriately-sized answers
(terse factoids vs full explanations).

Pipeline order: runs *before* the Synthesizer/Stylist, with the result
threaded through the chain. When the master flag
``ANSWER_SHAPE_CLASSIFIER_ENABLED=false`` it returns ``shape='default'``
and downstream agents preserve their pre-v3.1 behaviour byte-for-byte.

In v3.1 the default is ``true`` because the client explicitly asked for
adaptive answer length (terse factoids, rich explanations).

Strategy (latency-first):
    1. Skip rule    — flag-off → ``default``.
    2. Image override — has_image + short query + level/floor regex → ``factoid``.
    3. Regex fast-path — 5 ordered pattern groups, first match wins.
    4. LLM fallback — single 1-token call when query is non-trivial and
       no regex pattern matched. Errors → ``default`` (transparent).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("agentic_rag.generation.answer_shape")


# ── Public taxonomy ────────────────────────────────────────────────────

SHAPE_FACTOID = "factoid"
SHAPE_COUNT = "count"
SHAPE_LIST = "list"
SHAPE_COMPARISON = "comparison"
SHAPE_EXPLANATION = "explanation"
SHAPE_DEFAULT = "default"

_VALID_SHAPES = {
    SHAPE_FACTOID,
    SHAPE_COUNT,
    SHAPE_LIST,
    SHAPE_COMPARISON,
    SHAPE_EXPLANATION,
}


# ── Regex patterns (compiled once, in priority order) ──────────────────

_RE_FLAGS = re.IGNORECASE

_COUNT_PATTERNS = [
    re.compile(
        r"^\s*(how many|number of|total (number|count|amount) of|count of|count the)\b",
        _RE_FLAGS,
    ),
]

_FACTOID_PATTERNS = [
    # Allow an optional intervening noun (e.g. "the duct size", "the ceiling height")
    # so we catch both "what is the size?" and "what is the duct size?".
    re.compile(
        r"^\s*(what (is|are|was|were)|whats|what's) (the )?(?:\w+\s+)?"
        r"(size|height|width|depth|level|floor|thickness|diameter|capacity|"
        r"rating|voltage|amperage|wattage|GPM|CFM|BTU|HP|kW|temperature|"
        r"pressure|spacing|gauge|R-value|U-value|fire rating|model number|"
        r"manufacturer)\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"^\s*(how (thick|wide|tall|deep|high|big|long|heavy|much|fast))\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"^\s*(what (level|floor|sheet|drawing|trade|grade|class|type))\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"^\s*(which (level|floor|sheet|drawing|trade))\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"^\s*(size of|height of|width of|level for|floor for)\b",
        _RE_FLAGS,
    ),
]

_LIST_PATTERNS = [
    re.compile(
        r"^\s*(list|show me|give me|enumerate)\s+(all|every|each)\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"^\s*what (\w+ )?(drawings|fixtures|panels|valves|equipment|sheets|"
        r"trades|specifications|sections|materials|finishes|systems)\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"^\s*(name|identify|find) (all|the) \w+",
        _RE_FLAGS,
    ),
]

_COMPARISON_PATTERNS = [
    re.compile(r"^\s*compare\b", _RE_FLAGS),
    re.compile(
        r"\b(differences? between|discrepancies between|contrast)\b",
        _RE_FLAGS,
    ),
    re.compile(r"\bvs\.?\b", _RE_FLAGS),
]

_EXPLANATION_PATTERNS = [
    re.compile(
        r"^\s*(explain|describe|walk me through|tell me about|what does|how does)\b",
        _RE_FLAGS,
    ),
    re.compile(
        r"\b(what is the purpose|why is|why does)\b",
        _RE_FLAGS,
    ),
]

_IMAGE_FACTOID_RE = re.compile(
    r"^\s*(what|which) (level|floor|sheet|drawing|trade)",
    _RE_FLAGS,
)

# Ordered list of (shape, pattern_list) — first match wins.
_REGEX_GROUPS = [
    (SHAPE_COUNT, _COUNT_PATTERNS),
    (SHAPE_FACTOID, _FACTOID_PATTERNS),
    (SHAPE_LIST, _LIST_PATTERNS),
    (SHAPE_COMPARISON, _COMPARISON_PATTERNS),
    (SHAPE_EXPLANATION, _EXPLANATION_PATTERNS),
]


# ── Public API ─────────────────────────────────────────────────────────


def classify_shape(
    *,
    user_query: str,
    has_image: bool = False,
) -> Dict[str, Any]:
    """Classify the user query into an answer-shape bucket.

    Args:
        user_query: The user's question (post any rewrite/contextualization).
        has_image: Whether the request includes an image attachment.

    Returns:
        A dict with keys:

        * ``shape`` — one of ``factoid``, ``count``, ``list``, ``comparison``,
          ``explanation``, ``default``.
        * ``target_length_chars`` — desired max output length in chars.
        * ``target_word_count`` — convenient secondary cap (chars // 5).
        * ``was_llm_classified`` — ``False`` when regex fast-path hit.
        * ``confidence`` — float in [0.0, 1.0].
        * ``skip_reason`` — string when shape == ``default``, else ``None``.

    Master flag ``ANSWER_SHAPE_CLASSIFIER_ENABLED`` defaults to ``"true"``
    in v3.1. Set it to ``"false"`` in .env to revert to uniform-length
    answers (downstream agents will receive ``shape='default'`` and run
    their pre-v3.1 prompts unchanged).
    """
    # Default ON: client explicitly requested adaptive answer length.
    flag = os.getenv("ANSWER_SHAPE_CLASSIFIER_ENABLED", "true")
    if flag.strip().lower() != "true":
        return _build_default_result(skip_reason="flag_off")

    query = user_query or ""

    # 1. Image override — short query + level/floor → factoid.
    if has_image and len(query) < 100 and _IMAGE_FACTOID_RE.match(query):
        return _build_result(
            shape=SHAPE_FACTOID,
            confidence=0.95,
            was_llm_classified=False,
        )

    # 2. Regex fast-path.
    regex_shape = _match_regex(query)
    if regex_shape is not None:
        return _build_result(
            shape=regex_shape,
            confidence=0.9,
            was_llm_classified=False,
        )

    # 3. LLM fallback — only when query is non-trivial.
    if len(query.strip()) > 20:
        llm_shape = _llm_classify(query)
        if llm_shape is not None:
            return _build_result(
                shape=llm_shape,
                confidence=0.7,
                was_llm_classified=True,
            )

    # 4. Trivial / unclassified → default (zero behaviour change downstream).
    return _build_default_result(skip_reason="no_match")


# ── Internals ──────────────────────────────────────────────────────────


def _match_regex(query: str) -> Optional[str]:
    """Return the first matching shape label, or None if no group matched."""
    for shape, patterns in _REGEX_GROUPS:
        for pat in patterns:
            if pat.search(query):
                return shape
    return None


def _llm_classify(query: str) -> Optional[str]:
    """Call the configured LLM to classify the query into one of the 5 labels.

    Returns the parsed label on success, or ``None`` on any failure (in
    which case the caller falls back to ``shape='default'``).
    """
    model = os.getenv("SHAPE_CLASSIFIER_MODEL", "gpt-4o-mini")
    fallback_model = "gpt-4o-mini"

    system_prompt = (
        "You classify a user's question into one of: factoid, count, list, "
        "comparison, explanation. Reply with ONE word — the label. "
        "No punctuation, no explanation."
    )

    try:
        # Lazy import — keeps the module cheap to import in the regex-only path.
        from agentic.generation.llm_client import generate

        raw = generate(
            system_prompt=system_prompt,
            user_prompt=query,
            model=model,
            fallback_model=fallback_model,
            max_tokens=20,
            temperature=0.0,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001 — boundary, must not crash chain
        logger.warning("answer_shape.llm_classify failed: %s", exc)
        return None

    if not isinstance(raw, str):
        logger.warning(
            "answer_shape.llm_classify got non-str response: %r", type(raw)
        )
        return None

    label = re.sub(r"[^\w]+", "", raw.strip().lower())
    if label in _VALID_SHAPES:
        return label

    logger.info("answer_shape.llm_classify unrecognised label=%r", raw)
    return None


def _build_result(
    *,
    shape: str,
    confidence: float,
    was_llm_classified: bool,
) -> Dict[str, Any]:
    """Build a populated result dict for a real (non-default) shape."""
    target_chars = _target_chars_for(shape)
    return {
        "shape": shape,
        "target_length_chars": target_chars,
        "target_word_count": target_chars // 5,
        "was_llm_classified": was_llm_classified,
        "confidence": confidence,
        "skip_reason": None,
    }


def _build_default_result(*, skip_reason: str) -> Dict[str, Any]:
    """Build the canonical 'no behaviour change' result.

    ``shape='default'`` + ``target_length_chars=0`` is the contract that
    downstream agents read as "use existing prompts unchanged". This MUST
    be byte-identical when the flag is off so today's behaviour is
    perfectly preserved.
    """
    return {
        "shape": SHAPE_DEFAULT,
        "target_length_chars": 0,
        "target_word_count": 0,
        "was_llm_classified": False,
        "confidence": 0.0,
        "skip_reason": skip_reason,
    }


def _target_chars_for(shape: str) -> int:
    """Read env-tunable target length for the given shape."""
    env_map = {
        SHAPE_FACTOID: ("SHAPE_FACTOID_MAX_CHARS", "80"),
        SHAPE_COUNT: ("SHAPE_COUNT_MAX_CHARS", "60"),
        SHAPE_LIST: ("SHAPE_LIST_MAX_CHARS", "300"),
        SHAPE_COMPARISON: ("SHAPE_COMPARISON_MAX_CHARS", "400"),
        SHAPE_EXPLANATION: ("SHAPE_EXPLANATION_MAX_CHARS", "600"),
    }
    if shape not in env_map:
        return 0
    env_name, default = env_map[shape]
    raw = os.getenv(env_name, default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "answer_shape: bad env %s=%r, using default %s", env_name, raw, default
        )
        return int(default)


# ── Format-rule mapping (consumed by Synthesizer + Stylist) ────────────

FORMAT_RULES: Dict[str, str] = {
    SHAPE_FACTOID: (
        "Reply with ONE short sentence stating the value(s) only. "
        "No preamble, no 'According to the documents'. "
        "If multiple values, comma-separate."
    ),
    SHAPE_COUNT: (
        "Reply with the number followed by what's being counted. "
        "Example: '12 AHUs across the project.' One sentence."
    ),
    SHAPE_LIST: (
        "Reply with a tight bullet list, max 5 bullets, no headers, no intro line."
    ),
    SHAPE_COMPARISON: (
        "Reply in 2-3 sentences contrasting the items. No bullets unless necessary."
    ),
    SHAPE_EXPLANATION: (
        "Reply in <= 6 sentences. Plain paragraphs, no headers."
    ),
}


def shape_prompt_block(answer_shape: Optional[Dict[str, Any]]) -> Optional[str]:
    """Build the system-prompt prefix block for a given shape, or None.

    Returns ``None`` when ``answer_shape`` is missing or ``shape='default'``,
    so callers can do:

        block = shape_prompt_block(answer_shape)
        if block:
            system_prompt = block + "\\n\\n" + system_prompt

    and the no-shape path produces a byte-identical prompt to today's.
    """
    if not answer_shape:
        return None
    shape = answer_shape.get("shape")
    if not shape or shape == SHAPE_DEFAULT:
        return None
    rule = FORMAT_RULES.get(shape)
    if not rule:
        return None
    target_chars = int(answer_shape.get("target_length_chars") or 0)
    target_words = int(answer_shape.get("target_word_count") or (target_chars // 5))
    return (
        f"ANSWER_SHAPE: {shape}\n"
        f"LENGTH_BUDGET: respond in <= {target_words} words and "
        f"<= {target_chars} characters.\n"
        f"FORMAT_RULE: {rule}"
    )
