"""Fix #2 — LLM-as-judge reranker for source_documents.

Post-agentic, the orchestrator can have 20–40 source_documents accumulated
across the ReAct loop. Most of them touched the topic; only a few actually
answer the user's question. This reranker calls a cheap model
(``gpt-4.1-mini`` by default) once per query with the full candidate list and
asks it to score each 0–10.

Behaviour is intentionally additive:

- When the flag is off or the LLM call fails, the original ``source_documents``
  list is returned unchanged. UI contract is never broken.
- When the flag is on, the list is **reordered** best-first. A ``keep_top_k``
  option is available but the orchestrator wires it to ``None`` so the UI
  still sees every source — just with the best ones on top.
- Each reordered item gets an extra ``_rerank_score`` field so the frontend
  can decide to hide low-scoring entries later without backend changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_RERANK_MODEL = os.environ.get("RERANKER_MODEL", "gpt-4.1-mini")
# Most frontends render top 5-8 sources; cap the prompt at 30 to stay cheap.
DEFAULT_CANDIDATE_CAP = int(os.environ.get("RERANKER_CANDIDATE_CAP", "30"))
# Drop sources below this score (0 = never drop by score). Use together with
# RERANKER_MIN_KEEP so the list never falls below a minimum length.
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("RERANKER_SCORE_THRESHOLD", "0"))
# After dropping below threshold, guarantee at least this many items.
DEFAULT_MIN_KEEP = int(os.environ.get("RERANKER_MIN_KEEP", "3"))

# When false (default), we reorder source_documents but do NOT add the
# ``_rerank_score`` attribute — keeping the UI-visible shape of each source
# entry identical to the pre-fix schema. Flip to ``true`` if a future UI
# wants to surface the ranking score.
_INCLUDE_SCORE = os.environ.get("RERANKER_INCLUDE_SCORE", "false").strip().lower() in (
    "1", "true", "yes", "on"
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = (
    "You are scoring how directly each candidate construction document "
    "answers a user's question. You MUST return ONLY a JSON array of "
    "integers, nothing else."
)

_RERANK_USER_TEMPLATE = (
    "User question:\n{query}\n\n"
    "Below are {n} candidate documents retrieved by a RAG agent. For each "
    "one, score 0-10 on how likely it is to directly answer the question. "
    "10 = contains the exact answer. 7-9 = strong signal (right section or "
    "drawing, likely answers part of it). 4-6 = topically related. "
    "0-3 = tangential or unrelated.\n\n"
    "Output: a JSON array of exactly {n} integers, in the same order as the "
    "candidates below. Example for 3 candidates: [9, 2, 6]\n\n"
    "Candidates:\n{candidates}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_description(sd: dict[str, Any], index: int) -> str:
    """One-line description used in the scoring prompt."""
    title = (
        sd.get("display_title")
        or sd.get("drawing_title")
        or sd.get("drawingTitle")
        or sd.get("sectionTitle")
        or sd.get("pdf_name")
        or sd.get("pdfName")
        or sd.get("file_name")
        or "(untitled)"
    )
    kind_hint = ""
    s3 = sd.get("s3_path") or sd.get("s3BucketPath") or ""
    if "/Specification/" in s3:
        kind_hint = " [Specification]"
    elif "/Drawings/" in s3:
        kind_hint = " [Drawing]"
    page = sd.get("page")
    page_str = f" p.{page}" if page is not None else ""
    return f"[{index}]{kind_hint} {str(title)[:180]}{page_str}"


def _parse_scores(raw: str, n: int) -> Optional[list[float]]:
    """Parse the model's JSON-array response into a list of n floats.

    Returns None if we can't make sense of the output.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    # Coerce to floats, clamp to [0, 10], pad with 0.0 if model returned fewer
    scores: list[float] = []
    for i in range(n):
        v: float
        if i < len(parsed):
            try:
                v = float(parsed[i])
            except (TypeError, ValueError):
                v = 0.0
        else:
            v = 0.0
        scores.append(max(0.0, min(10.0, v)))
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_source_documents(
    query: str,
    source_documents: list[dict[str, Any]],
    model: str = DEFAULT_RERANK_MODEL,
    candidate_cap: int = DEFAULT_CANDIDATE_CAP,
    keep_top_k: Optional[int] = None,
    openai_client: Any = None,
    include_score: Optional[bool] = None,
    score_threshold: Optional[float] = None,
    min_keep: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return ``source_documents`` reordered best-first.

    Parameters
    ----------
    query : str
        The user's original question (not any rewritten form). Scoring must
        target the user's intent, not an internal sub-query.
    source_documents : list[dict]
        The exact list the orchestrator is about to return to the UI. We
        reorder in-place-ish (returning a new list, same dicts).
    keep_top_k : int | None
        If set, truncates to the top-K. Default ``None`` = keep all, just
        reorder — safest for UI since list length doesn't change.
    openai_client : Any
        Injectable for testing. Default: a fresh ``openai.OpenAI()`` instance.
    """
    if not source_documents:
        return source_documents
    # Guardrail: if <=1 item, nothing to rerank
    if len(source_documents) <= 1:
        return source_documents

    # Build the candidate block from up to candidate_cap items (preserving order)
    working = source_documents[: candidate_cap]
    descriptions = [_candidate_description(sd, i) for i, sd in enumerate(working)]
    user_prompt = _RERANK_USER_TEMPLATE.format(
        query=query.strip(),
        n=len(descriptions),
        candidates="\n".join(descriptions),
    )

    try:
        client = openai_client
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("rerank_source_documents LLM call failed: %s", exc)
        return source_documents  # fail open: return original order

    scores = _parse_scores(raw, len(working))
    if scores is None:
        logger.warning("rerank_source_documents: could not parse %r", raw[:200])
        return source_documents

    # Attach scores and sort stably by score desc (preserve original order for ties)
    indexed = [(i, sd, scores[i]) for i, sd in enumerate(working)]
    indexed.sort(key=lambda tup: (-tup[2], tup[0]))

    # UI-contract preservation: by default we DO NOT add the _rerank_score key.
    # The reorder is the visible effect; the score stays internal.
    emit_score = _INCLUDE_SCORE if include_score is None else bool(include_score)

    reordered: list[dict[str, Any]] = []
    for _, sd, score in indexed:
        new_sd = dict(sd)
        if emit_score:
            new_sd["_rerank_score"] = score
        reordered.append(new_sd)

    # Score-threshold drop: remove sources below the threshold while keeping
    # at least ``min_keep`` highest-scored entries even if all fail. This is
    # how we get rid of the "90% noise in references" behaviour — low-score
    # entries get filtered out rather than just reordered to the bottom.
    thresh = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
    floor = min_keep if min_keep is not None else DEFAULT_MIN_KEEP
    if thresh > 0 and reordered:
        # indexed[i] = (original_idx, sd, score); sorted best-first already
        kept = [(sd, s) for (_, sd, s) in indexed if s >= thresh]
        if len(kept) < floor:
            # Pad with the next-best entries so the caller always gets some.
            already = {id(sd) for sd, _ in kept}
            for _, sd, s in indexed:
                if len(kept) >= floor:
                    break
                if id(sd) not in already:
                    kept.append((sd, s))
        reordered = []
        for sd, score in kept:
            new_sd = dict(sd)
            if emit_score:
                new_sd["_rerank_score"] = score
            reordered.append(new_sd)

    # Append the tail we never scored (beyond candidate_cap) at the end — but
    # only if we DIDN'T apply a score threshold. Otherwise these unscored
    # items would dodge the filter.
    if len(source_documents) > candidate_cap and thresh <= 0:
        tail = source_documents[candidate_cap:]
        if emit_score:
            reordered.extend({**sd, "_rerank_score": None} for sd in tail)
        else:
            reordered.extend(dict(sd) for sd in tail)

    if keep_top_k is not None and keep_top_k > 0:
        reordered = reordered[: keep_top_k]
    return reordered
