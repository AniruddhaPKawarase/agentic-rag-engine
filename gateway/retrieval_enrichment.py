"""Fix #1 — Multi-Query decomposition + RAG-Fusion (RRF).

Runs BEFORE the agentic ReAct loop so the agent starts with a strong pool of
pre-retrieved candidates instead of having to discover them one tool call at a
time. The hint is injected into ``conversation_history`` as an additional
system message so the agent schema, orchestrator response envelope, and UI
contract remain completely unchanged.

Three stages:

1. :func:`decompose_query` asks a cheap model (default ``gpt-4.1-mini``) to
   rewrite the user's question into N semantically diverse sub-queries that
   together cover the full search intent (synonyms, trade names, CSI division
   numbers, drawing-type variants).
2. :func:`multi_query_retrieve` fans the sub-queries out across the existing
   MongoDB tools in parallel. No new data layer — we reuse the engine's own
   search tools so coverage matches what the agent would have retrieved
   anyway.
3. :func:`reciprocal_rank_fusion` merges all ranked lists with RRF
   (1 / (k + rank) scoring). Items that show up in multiple lists at
   shallow ranks bubble to the top.

Everything here is a pure-Python helper exposed via plain function signatures
so it's easy to unit-test without mocking the full agent stack. The
orchestrator is the only caller.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DECOMPOSITION_MODEL = os.environ.get(
    "MULTI_QUERY_MODEL", "gpt-4.1-mini"
)
DEFAULT_SUB_QUERIES = int(os.environ.get("MULTI_QUERY_COUNT", "4"))
DEFAULT_PER_TOOL_LIMIT = int(os.environ.get("MULTI_QUERY_PER_TOOL_LIMIT", "10"))
DEFAULT_FUSED_TOP_K = int(os.environ.get("MULTI_QUERY_FUSED_TOP_K", "15"))
DEFAULT_MAX_WORKERS = int(os.environ.get("MULTI_QUERY_MAX_WORKERS", "4"))


# ---------------------------------------------------------------------------
# Stage 1 — decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = (
    "You are a construction documentation librarian helping a RAG agent find "
    "relevant material in drawings, specifications, and OCR'd sheets."
)

_DECOMPOSE_USER_TEMPLATE = (
    "Rewrite the user question into exactly {n} semantically diverse search "
    "queries that together maximise recall against a construction project's "
    "documents. Use:\n"
    "- Synonyms and layman / technical variants of the key entities.\n"
    "- Trade names when relevant (mechanical, electrical, plumbing, "
    "structural, architectural, fire protection, civil).\n"
    "- CSI division numbers where obvious (e.g. 22 = plumbing, 23 = HVAC, "
    "26 = electrical, 27 = telecom, 28 = security, 33 = utilities).\n"
    "- Drawing-type variants (plan, schedule, riser, section, elevation, "
    "detail, legend, specification).\n"
    "Do NOT add explanations or commentary. Respond with ONLY a JSON array "
    "of {n} distinct strings.\n\n"
    'Example question: "What size are the stair pressurization fans on '
    'level 2?"\n'
    'Example output: ["stair pressurization fan size level 2", '
    '"stair pressurization fan schedule", "smoke control stair '
    'pressurization", "stair pressurization Division 23 HVAC"]\n\n'
    "User question: {query}"
)


def _sanitize_sub_queries(raw: list[Any], original: str, n: int) -> list[str]:
    """Dedupe, strip, cap length. Always returns at least one query."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().strip('"').strip("'")[:200]
        if not cleaned:
            continue
        key = re.sub(r"\s+", " ", cleaned.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= n:
            break
    if not out:
        out = [original.strip() or "construction documents"]
    return out


def decompose_query(
    query: str,
    n: int = DEFAULT_SUB_QUERIES,
    model: str = DEFAULT_DECOMPOSITION_MODEL,
    openai_client: Any = None,
) -> list[str]:
    """Return up to *n* sub-queries.

    Falls back to ``[query]`` when the LLM call fails or returns nonsense, so
    callers can always treat the result as at-least-length-1.
    """
    n = max(1, min(int(n or DEFAULT_SUB_QUERIES), 8))
    q = (query or "").strip()
    if not q:
        return []
    try:
        client = openai_client
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": _DECOMPOSE_USER_TEMPLATE.format(n=n, query=q)},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # network / auth / quota
        logger.warning("decompose_query LLM call failed: %s", exc)
        return [q]

    # Strip common markdown wrappers before json.loads
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw.strip()).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Last-resort: extract the first [...] block
        match = re.search(r"\[[^\[\]]*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("decompose_query: could not parse JSON from %r", raw[:120])
                return [q]
        else:
            logger.warning("decompose_query: no JSON array in response %r", raw[:120])
            return [q]

    if not isinstance(parsed, list):
        return [q]

    return _sanitize_sub_queries(parsed, q, n)


# ---------------------------------------------------------------------------
# Stage 2 — multi-query retrieval (reuses existing agent tools)
# ---------------------------------------------------------------------------

def _default_tools() -> list[tuple[str, Callable[..., list[dict]]]]:
    """Lazy-import the agent's own search tools so this module is importable
    even when the agentic package is unavailable (e.g. during unit tests)."""
    tools: list[tuple[str, Callable[..., list[dict]]]] = []
    try:
        from agentic.tools.specification_tools import search_specification_text  # type: ignore
        tools.append(("spec_search", search_specification_text))
    except Exception as exc:
        logger.debug("spec_search tool not importable: %s", exc)
    try:
        from agentic.tools.mongodb_tools import search_by_text as vision_search_text  # type: ignore
        tools.append(("vision_search", vision_search_text))
    except Exception as exc:
        logger.debug("vision_search tool not importable: %s", exc)
    try:
        from agentic.tools.drawing_tools import search_drawing_text as legacy_search_text  # type: ignore
        tools.append(("legacy_search", legacy_search_text))
    except Exception as exc:
        logger.debug("legacy_search tool not importable: %s", exc)
    return tools


def _safe_invoke(tool: Callable[..., list[dict]], project_id: int, sq: str, limit: int) -> list[dict]:
    try:
        results = tool(project_id=project_id, search_text=sq, limit=limit)
        return list(results or [])
    except Exception as exc:
        logger.debug("tool %s failed on %r: %s", tool.__name__, sq[:40], exc)
        return []


def multi_query_retrieve(
    project_id: int,
    sub_queries: list[str],
    tools: list[tuple[str, Callable[..., list[dict]]]] | None = None,
    per_tool_limit: int = DEFAULT_PER_TOOL_LIMIT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[list[dict]]:
    """Run each (sub_query × tool) combination in parallel, return ranked lists.

    Returns a flat list of ranked lists, one per (sub_query, tool) pair that
    produced any results. Ordering within each inner list is whatever the tool
    returned (typically best-first). The next stage fuses them.
    """
    resolved = tools if tools is not None else _default_tools()
    if not resolved or not sub_queries:
        return []

    jobs = [(sq, name, fn) for sq in sub_queries for (name, fn) in resolved]
    ranked_lists: list[list[dict]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_safe_invoke, fn, project_id, sq, per_tool_limit): (sq, name)
            for (sq, name, fn) in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            sq, name = futures[fut]
            try:
                results = fut.result()
            except Exception as exc:
                logger.debug("tool %s future error for %r: %s", name, sq[:40], exc)
                results = []
            if results:
                ranked_lists.append(results)
    return ranked_lists


# ---------------------------------------------------------------------------
# Stage 3 — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _default_id(item: dict[str, Any]) -> str:
    """Stable identity key used when merging ranked lists.

    Prefer the best available document anchor. Falls back to a repr of the
    dict, which at least dedupes exact structural duplicates.
    """
    if not isinstance(item, dict):
        return repr(item)[:200]
    for key in (
        "pdfName",
        "pdf_name",
        "sourceFile",
        "source_file",
        "drawingName",
        "drawing_name",
        "specificationNumber",
        "specification_number",
    ):
        val = item.get(key)
        if val:
            return f"{key}:{str(val)[:200]}"
    # Fallback to a compact hash-ish repr
    return repr(sorted(item.items()))[:200]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict]],
    k: int = 60,
    top_k: int | None = DEFAULT_FUSED_TOP_K,
    id_fn: Callable[[dict], str] | None = None,
) -> list[dict]:
    """Classic RRF: score[item] = Σ 1 / (k + rank_in_list_i + 1).

    Returns the merged list sorted by RRF score desc. If *top_k* is given,
    truncates to that many entries.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    id_of = id_fn or _default_id
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, dict] = {}
    contributing_lists: dict[str, int] = defaultdict(int)
    for lst in ranked_lists:
        if not lst:
            continue
        seen_in_this_list: set[str] = set()
        for rank, item in enumerate(lst):
            ident = id_of(item)
            if ident in seen_in_this_list:
                continue  # avoid double-counting duplicates inside one list
            seen_in_this_list.add(ident)
            scores[ident] += 1.0 / (k + rank + 1)
            contributing_lists[ident] += 1
            if ident not in first_seen:
                first_seen[ident] = item
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged: list[dict] = []
    for ident, score in ordered:
        item = dict(first_seen[ident])
        item["_rrf_score"] = round(score, 6)
        item["_rrf_support"] = contributing_lists[ident]
        merged.append(item)
    if top_k is not None and top_k > 0:
        merged = merged[: top_k]
    return merged


# ---------------------------------------------------------------------------
# Stage 4 — convenience bundle used by the orchestrator
# ---------------------------------------------------------------------------

def build_context_hint(
    query: str,
    project_id: int,
    n_sub_queries: int = DEFAULT_SUB_QUERIES,
    per_tool_limit: int = DEFAULT_PER_TOOL_LIMIT,
    top_k: int = DEFAULT_FUSED_TOP_K,
    decomposition_model: str = DEFAULT_DECOMPOSITION_MODEL,
    tools: list[tuple[str, Callable[..., list[dict]]]] | None = None,
    openai_client: Any = None,
) -> dict[str, Any]:
    """End-to-end: decompose → multi-query retrieve → RRF fuse.

    Returns a dict with:
      - ``sub_queries``: list[str]
      - ``fused``: list[dict]   (top_k items)
      - ``total_candidates_considered``: int
      - ``num_source_lists``: int
    """
    sub_queries = decompose_query(
        query=query, n=n_sub_queries, model=decomposition_model, openai_client=openai_client
    )
    ranked_lists = multi_query_retrieve(
        project_id=project_id,
        sub_queries=sub_queries,
        tools=tools,
        per_tool_limit=per_tool_limit,
    )
    fused = reciprocal_rank_fusion(ranked_lists, top_k=top_k)
    return {
        "sub_queries": sub_queries,
        "fused": fused,
        "total_candidates_considered": sum(len(lst) for lst in ranked_lists),
        "num_source_lists": len(ranked_lists),
    }


def format_hint_for_agent(hint: dict[str, Any], max_items: int = 10) -> str:
    """Render the fused candidates as a compact string to prepend to the
    agent's conversation history. The agent can choose to consult these
    candidates directly via its existing tools.
    """
    fused = hint.get("fused") or []
    sub = hint.get("sub_queries") or []
    if not fused:
        return ""
    items = fused[: max_items]
    lines = [
        "Pre-retrieved search candidates (from multi-query fusion):",
        f"Sub-queries used: {json.dumps(sub, ensure_ascii=False)}",
        "",
    ]
    for i, it in enumerate(items, 1):
        title = (
            it.get("drawingTitle")
            or it.get("drawing_title")
            or it.get("sectionTitle")
            or it.get("page_summary")
            or it.get("sourceFile")
            or it.get("pdfName")
            or ""
        )
        anchor = (
            it.get("pdfName")
            or it.get("sourceFile")
            or it.get("drawingName")
            or ""
        )
        lines.append(
            f"[{i}] {title[:110]}"
            f" | anchor={anchor[:60]}"
            f" | rrf={it.get('_rrf_score','?')}"
            f" | support={it.get('_rrf_support','?')}"
        )
    lines.append("")
    lines.append(
        "You MAY use these candidates as a starting point, but verify "
        "details with spec_get_full_text, legacy_get_text, vision_get_content, "
        "or spec_get_section before answering."
    )
    return "\n".join(lines)
