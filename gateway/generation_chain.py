"""
gateway.generation_chain
========================

Phase P4 — orchestrator that wires together all six v3.1 agents:

    1. Memory Recall      (agentic/memory/recall_agent.py)
    2. Query Rewriter     (agentic/memory/query_rewriter.py)
    3. ReAct Retrieval    (agentic/core/agent.py — UNCHANGED)
    4. Answer Synthesizer (agentic/generation/synthesizer.py)
    5. Style/Tone Rewriter (agentic/generation/stylist.py — `stylize`)
    5b. Cache Re-Expression (agentic/generation/stylist.py — `reexpress_cached`)
    6. Memory Writer       (agentic/memory/writer.py — fire-and-forget)

The single public entrypoint is :func:`run_generation_chain`. It works in
two modes:

* ``stream=False`` — coroutine returning a fully-assembled response dict.
* ``stream=True``  — async generator yielding dicts of the form
  ``{"event": "metadata"|"token"|"done", "delta": str|None, "metadata": dict|None}``.

Master kill switch
------------------
``V31_CHAIN_ENABLED=false`` (the default) prevents callers from routing
through this module at all — see ``gateway/orchestrator.py`` for the
gate. When the flag is on, individual agents still respect their own
sub-flags (``MEMORY_RECALL_ENABLED``, ``QUERY_REWRITER_ENABLED``, ...)
so ops can re-disable any single agent without redeploying.

Latency-correct serial design
-----------------------------
The original spec called for "Memory Recall in parallel with ReAct." But
the rewriter depends on recall output, and retrieval depends on the
rewritten query — so the front-end is *inherently* serial:

    Recall (~150ms) -> Rewriter (~200ms, often skipped) -> ReAct (~17s)
                       -> Synthesizer (~600ms, streamable)
                       -> Stylist     (~400ms, streamable)
                       -> Memory Writer (background, 0ms blocking)

The only true parallelism opportunity is between Recall and the
*cache-key generation* (both are <50ms and have no shared state), but
the saving is too small to justify the orchestration cost. We document
this here and proceed serially. See VERSION_NOTES.md for the design
trade-off.

Failure handling
----------------
Each sub-agent call is wrapped in try/except. On failure the chain falls
through to the previous stage's output, never crashing the whole
response. This keeps the chain "additive" with respect to today's
behaviour: even a total Recall + Rewriter + Synthesizer + Stylist
failure still returns the raw ReAct answer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Awaitable, Dict, List, Optional, Union
from gateway.answer_post_processor import post_process as _tier_e_post_process

logger = logging.getLogger("agentic_rag.chain")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _flag_enabled(name: str, default: str = "false") -> bool:
    """Truthy env-var check. Defaults to disabled when unset."""
    return os.getenv(name, default).strip().lower() == "true"


def chain_enabled() -> bool:
    """Master kill switch. Default OFF."""
    return _flag_enabled("V31_CHAIN_ENABLED", "false")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

# Type alias for the streaming yield shape.
StreamEvent = Dict[str, Any]


def run_generation_chain(
    *,
    user_query: str,
    session_id: Optional[str],
    project_id: int,
    set_id: Optional[int],
    scope: Optional[Dict[str, Any]],
    cached_result: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[list] = None,
    stream: bool = False,
) -> Union[Awaitable[Dict[str, Any]], AsyncIterator[StreamEvent]]:
    """Run the v3.1 generation chain end-to-end.

    The signature is the same in both modes; the *return type* differs:

    * ``stream=False`` returns an awaitable resolving to a response dict.
    * ``stream=True`` returns an async generator yielding events.

    Parameters
    ----------
    user_query:
        The user's fresh question (pre-rewrite).
    session_id:
        Conversation session ID. ``None`` disables the memory subsystem.
    project_id:
        Project ID for retrieval.
    set_id:
        Drawing-set ID for retrieval (may be ``None``).
    scope:
        Document-scope filter passed straight through to ReAct.
    cached_result:
        When not ``None``, signals a cache hit. Triggers Branch A
        (cache re-expression) instead of the full pipeline.
    conversation_history:
        Existing conversation history forwarded to ReAct. The chain does
        not modify this list.
    stream:
        Stream synthesizer + stylist tokens when ``True``.

    Returns
    -------
    Awaitable[dict] | AsyncIterator[dict]
        Either a coroutine resolving to the final response dict, or an
        async generator depending on ``stream``.
    """
    if stream:
        return _run_chain_stream(
            user_query=user_query,
            session_id=session_id,
            project_id=project_id,
            set_id=set_id,
            scope=scope,
            cached_result=cached_result,
            conversation_history=conversation_history,
        )
    return _run_chain_buffered(
        user_query=user_query,
        session_id=session_id,
        project_id=project_id,
        set_id=set_id,
        scope=scope,
        cached_result=cached_result,
        conversation_history=conversation_history,
    )


# ---------------------------------------------------------------------------
# Buffered (non-streaming) entrypoint
# ---------------------------------------------------------------------------


async def _run_chain_buffered(
    *,
    user_query: str,
    session_id: Optional[str],
    project_id: int,
    set_id: Optional[int],
    scope: Optional[Dict[str, Any]],
    cached_result: Optional[Dict[str, Any]],
    conversation_history: Optional[list],
) -> Dict[str, Any]:
    """Non-streaming chain run. Returns the full response dict."""
    start = time.monotonic()
    has_image = _scope_has_image(scope)

    if cached_result is not None:
        return await _branch_cache_hit(
            user_query=user_query,
            session_id=session_id,
            cached_result=cached_result,
            stream=False,
            start=start,
            has_image=has_image,
        )

    return await _branch_full_pipeline(
        user_query=user_query,
        session_id=session_id,
        project_id=project_id,
        set_id=set_id,
        scope=scope,
        conversation_history=conversation_history,
        stream=False,
        start=start,
        has_image=has_image,
    )


# ---------------------------------------------------------------------------
# Streaming entrypoint (async generator)
# ---------------------------------------------------------------------------


async def _run_chain_stream(
    *,
    user_query: str,
    session_id: Optional[str],
    project_id: int,
    set_id: Optional[int],
    scope: Optional[Dict[str, Any]],
    cached_result: Optional[Dict[str, Any]],
    conversation_history: Optional[list],
) -> AsyncIterator[StreamEvent]:
    """Streaming chain run. Yields metadata, tokens, and a final done event.

    Streaming protocol
    ------------------
    1. ``{"event": "metadata", "metadata": {...}}`` — emitted exactly once
       before any token. Carries the skip-rule decisions so the UI can
       render badges immediately.
    2. ``{"event": "token", "delta": "<chunk>"}`` — zero or more, in order.
    3. ``{"event": "done", "metadata": {final_answer: str, ...}}`` —
       emitted exactly once at the end, with the assembled response.
    """
    start = time.monotonic()
    has_image = _scope_has_image(scope)

    if cached_result is not None:
        async for evt in _branch_cache_hit_stream(
            user_query=user_query,
            session_id=session_id,
            cached_result=cached_result,
            start=start,
            has_image=has_image,
        ):
            yield evt
        return

    async for evt in _branch_full_pipeline_stream(
        user_query=user_query,
        session_id=session_id,
        project_id=project_id,
        set_id=set_id,
        scope=scope,
        conversation_history=conversation_history,
        start=start,
        has_image=has_image,
    ):
        yield evt


# ---------------------------------------------------------------------------
# Branch A — cache hit (buffered)
# ---------------------------------------------------------------------------


async def _branch_cache_hit(
    *,
    user_query: str,
    session_id: Optional[str],
    cached_result: Dict[str, Any],
    stream: bool,
    start: float,
    has_image: bool = False,
) -> Dict[str, Any]:
    """Re-express a cached answer to feel fresh.

    When ``CACHE_REEXPRESSION_ENABLED`` is off the cached result is
    returned unchanged — same shape, same fields, no LLM cost.
    """
    # Ensure caller has a session_id even when re-expression is off, so
    # multi-turn memory works on cache-hit follow-ups too.
    session_id = _ensure_session_id(
        session_id=session_id, user_query=user_query, project_id=None,
    )

    if not _flag_enabled("CACHE_REEXPRESSION_ENABLED", "false"):
        out = dict(cached_result)
        out["cache_reexpressed"] = False
        out["session_id"] = session_id
        out["processing_time_ms"] = int((time.monotonic() - start) * 1000)
        return out

    memory_context = await _safe_recall(session_id=session_id, user_query=user_query)
    last_assistant_turn, rolling_summary, is_followup = _derive_continuity_signals(memory_context)

    answer_shape = _safe_classify_shape(user_query=user_query, has_image=has_image)

    cached_answer = cached_result.get("answer", "") or ""
    new_answer = await _safe_reexpress_cached(
        cached_answer=cached_answer,
        user_query=user_query,
        rolling_summary=rolling_summary,
        last_assistant_turn=last_assistant_turn,
        is_followup=is_followup,
        stream=False,
        answer_shape=answer_shape,
    )

    out = dict(cached_result)
    out["answer"] = new_answer
    out["cache_reexpressed"] = new_answer != cached_answer
    out["answer_shape"] = answer_shape.get("shape")
    out["target_length_chars"] = answer_shape.get("target_length_chars")
    out["processing_time_ms"] = int((time.monotonic() - start) * 1000)
    return out


async def _branch_cache_hit_stream(
    *,
    user_query: str,
    session_id: Optional[str],
    cached_result: Dict[str, Any],
    start: float,
    has_image: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Streaming variant of Branch A."""
    enabled = _flag_enabled("CACHE_REEXPRESSION_ENABLED", "false")
    answer_shape = _safe_classify_shape(user_query=user_query, has_image=has_image)
    yield {
        "event": "metadata",
        "delta": None,
        "metadata": {
            "cache_hit": True,
            "cache_reexpressed_enabled": enabled,
            "memory_context_used": False,
            "query_rewritten": False,
            "synthesizer_used": False,
            "stylist_used": False,
            "answer_shape": answer_shape.get("shape"),
            "target_length_chars": answer_shape.get("target_length_chars"),
        },
    }

    if not enabled:
        full_text = cached_result.get("answer", "") or ""
        if full_text:
            yield {"event": "token", "delta": full_text, "metadata": None}
        out = dict(cached_result)
        out["cache_reexpressed"] = False
        out["answer_shape"] = answer_shape.get("shape")
        out["target_length_chars"] = answer_shape.get("target_length_chars")
        out["processing_time_ms"] = int((time.monotonic() - start) * 1000)
        yield {"event": "done", "delta": None, "metadata": out}
        return

    memory_context = await _safe_recall(session_id=session_id, user_query=user_query)
    last_assistant_turn, rolling_summary, is_followup = _derive_continuity_signals(memory_context)
    cached_answer = cached_result.get("answer", "") or ""

    chunks: List[str] = []
    async for chunk in _safe_reexpress_cached_stream(
        cached_answer=cached_answer,
        user_query=user_query,
        rolling_summary=rolling_summary,
        last_assistant_turn=last_assistant_turn,
        is_followup=is_followup,
        answer_shape=answer_shape,
    ):
        chunks.append(chunk)
        yield {"event": "token", "delta": chunk, "metadata": None}

    new_answer = "".join(chunks) or cached_answer
    out = dict(cached_result)
    out["answer"] = new_answer
    out["cache_reexpressed"] = new_answer != cached_answer
    out["answer_shape"] = answer_shape.get("shape")
    out["target_length_chars"] = answer_shape.get("target_length_chars")
    out["processing_time_ms"] = int((time.monotonic() - start) * 1000)
    yield {"event": "done", "delta": None, "metadata": out}


# ---------------------------------------------------------------------------
# Branch B — full pipeline (buffered)
# ---------------------------------------------------------------------------


async def _branch_full_pipeline(
    *,
    user_query: str,
    session_id: Optional[str],
    project_id: int,
    set_id: Optional[int],
    scope: Optional[Dict[str, Any]],
    conversation_history: Optional[list],
    stream: bool,
    start: float,
    has_image: bool = False,
) -> Dict[str, Any]:
    """Full Recall → Rewriter → ReAct → Synthesizer → Stylist → Writer."""
    # 0. Auto-create a session if the caller didn't pass one.
    # This is what makes multi-turn memory work out-of-the-box: every
    # response carries a session_id the caller can reuse on the next turn.
    session_id = _ensure_session_id(
        session_id=session_id, user_query=user_query, project_id=project_id,
    )

    # 1. Memory Recall
    memory_context = await _safe_recall(session_id=session_id, user_query=user_query)
    memory_used = bool(memory_context.get("had_context"))

    # 1.5. Answer Shape Classifier (Agent 0.5) — runs after recall so we
    # have the original user_query intact (the rewriter would lose intent).
    answer_shape = _safe_classify_shape(
        user_query=user_query, has_image=has_image
    )

    # 2. Query Rewriter
    rewrite_out = await _safe_rewrite(
        user_query=user_query, memory_context=memory_context
    )
    contextualized_query = rewrite_out.get("contextualized_query", user_query)
    query_rewritten = bool(rewrite_out.get("was_rewritten"))
    rewriter_skip_reason = rewrite_out.get("skip_reason")  # CONTEXT-FIX 2026-05-29

    # 2.5 Upgrade 1 — Multi-Query + RRF (flag-gated, no-op when off)
    # The orchestrator may have already injected the RRF hint into scope
    # (it runs multi-query *before* dispatching to the chain to keep the
    # work cache-safe). Detect that and skip re-running.
    upstream_hint = (scope or {}).get("rrf_hint") if isinstance(scope, dict) else None
    if upstream_hint:
        rrf_hint = upstream_hint
        chain_scope = scope  # already has the hint
        logger.info("chain.multi_query: hint provided upstream by orchestrator")
    else:
        rrf_hint = await _safe_multi_query_hint(
            user_query=contextualized_query,
            project_id=project_id,
            scope=scope,
        )
        chain_scope = scope
        if rrf_hint:
            chain_scope = dict(scope) if scope else {}
            chain_scope["rrf_hint"] = rrf_hint
    multi_query_used = rrf_hint is not None

    # 3. ReAct retrieval (the existing, unchanged engine)
    agent_result, agent_error = await _safe_run_agent(
        query=contextualized_query,
        project_id=project_id,
        set_id=set_id,
        conversation_history=conversation_history,
        scope=chain_scope,
    )
    raw_answer = _extract_answer(agent_result)
    source_docs = _extract_source_docs(agent_result)

    # 3.5 Upgrade 2 — LLM-as-Judge Reranker (flag-gated, no-op when off)
    pre_rerank_count = len(source_docs)
    source_docs = await _safe_rerank(
        user_query=user_query, source_docs=source_docs,
    )
    rerank_used = (
        _flag_enabled("RERANKER_ENABLED", "false") and pre_rerank_count > 1
    )

    # 4. Synthesizer
    synthesizer_used = False
    draft_answer = raw_answer
    if _flag_enabled("ANSWER_SYNTHESIZER_ENABLED", "false") and raw_answer:
        synth_text = await _safe_synthesize(
            raw_answer=raw_answer,
            user_query=user_query,
            source_docs=source_docs,
            rolling_summary=memory_context.get("rolling_summary"),
            answer_shape=answer_shape,
        )
        if synth_text and synth_text != raw_answer:
            draft_answer = synth_text
            synthesizer_used = True

    # 5. Stylist
    stylist_used = False
    final_answer = draft_answer
    if _flag_enabled("STYLE_REWRITER_ENABLED", "false") and draft_answer:
        last_assistant_turn = _last_assistant_turn(memory_context)
        stylized = await _safe_stylize(
            draft_answer=draft_answer,
            user_query=user_query,
            last_assistant_turn=last_assistant_turn,
            answer_shape=answer_shape,
        )
        if stylized and stylized != draft_answer:
            final_answer = stylized
            stylist_used = True

    # 5.5 Tier E post-processor (Method line + citation rewrite)
    final_answer = _tier_e_post_process(final_answer, agent_result)

    # 6. Memory writer (fire-and-forget — never await). Mints turn_id
    # synchronously so the HTTP response can echo it for Deep Dive trigger.
    # Persist source_documents alongside the turn so Deep Dive can find
    # the cited refs without re-hitting Mongo.
    turn_id = _dispatch_memory_writer(
        session_id=session_id,
        user_text=user_query,
        assistant_text=final_answer,
        project_id=project_id,
        set_id=set_id,
        source_documents=source_docs,
    )

    return _build_response_dict(
        user_query=user_query,
        contextualized_query=contextualized_query,
        agent_result=agent_result,
        agent_error=agent_error,
        final_answer=final_answer,
        memory_used=memory_used,
        query_rewritten=query_rewritten,
        synthesizer_used=synthesizer_used,
        stylist_used=stylist_used,
        cache_reexpressed=False,
        session_id=session_id,
        project_id=project_id,
        elapsed_ms=int((time.monotonic() - start) * 1000),
        answer_shape=answer_shape,
        multi_query_used=multi_query_used,
        rerank_used=rerank_used,
        source_docs_after=source_docs,
        turn_id=turn_id,
    )


async def _branch_full_pipeline_stream(
    *,
    user_query: str,
    session_id: Optional[str],
    project_id: int,
    set_id: Optional[int],
    scope: Optional[Dict[str, Any]],
    conversation_history: Optional[list],
    start: float,
    has_image: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Streaming variant of Branch B.

    Streams synthesizer tokens (or stylist tokens when synthesizer is
    skipped). The ReAct stage is monolithic — we cannot stream that.
    """
    session_id = _ensure_session_id(
        session_id=session_id, user_query=user_query, project_id=project_id,
    )
    memory_context = await _safe_recall(session_id=session_id, user_query=user_query)
    memory_used = bool(memory_context.get("had_context"))

    answer_shape = _safe_classify_shape(
        user_query=user_query, has_image=has_image
    )

    rewrite_out = await _safe_rewrite(
        user_query=user_query, memory_context=memory_context
    )
    contextualized_query = rewrite_out.get("contextualized_query", user_query)
    query_rewritten = bool(rewrite_out.get("was_rewritten"))
    rewriter_skip_reason = rewrite_out.get("skip_reason")  # CONTEXT-FIX 2026-05-29

    # Upgrade 1 — Multi-Query + RRF (flag-gated). Reuse upstream hint if
    # the orchestrator already produced one.
    upstream_hint = (scope or {}).get("rrf_hint") if isinstance(scope, dict) else None
    if upstream_hint:
        rrf_hint = upstream_hint
        chain_scope = scope
    else:
        rrf_hint = await _safe_multi_query_hint(
            user_query=contextualized_query, project_id=project_id, scope=scope,
        )
        chain_scope = scope
        if rrf_hint:
            chain_scope = dict(scope) if scope else {}
            chain_scope["rrf_hint"] = rrf_hint
    multi_query_used = rrf_hint is not None

    agent_result, agent_error = await _safe_run_agent(
        query=contextualized_query,
        project_id=project_id,
        set_id=set_id,
        conversation_history=conversation_history,
        scope=chain_scope,
    )
    raw_answer = _extract_answer(agent_result)
    source_docs = _extract_source_docs(agent_result)

    # Upgrade 2 — LLM-as-Judge Reranker (flag-gated)
    pre_rerank_count = len(source_docs)
    source_docs = await _safe_rerank(
        user_query=user_query, source_docs=source_docs,
    )
    rerank_used = (
        _flag_enabled("RERANKER_ENABLED", "false") and pre_rerank_count > 1
    )

    synthesizer_enabled = _flag_enabled("ANSWER_SYNTHESIZER_ENABLED", "false") and bool(raw_answer)
    stylist_enabled = _flag_enabled("STYLE_REWRITER_ENABLED", "false")

    yield {
        "event": "metadata",
        "delta": None,
        "metadata": {
            "cache_hit": False,
            "memory_context_used": memory_used,
            "query_rewritten": query_rewritten,
            "synthesizer_planned": synthesizer_enabled,
            "stylist_planned": stylist_enabled,
            "contextualized_query": contextualized_query,
            "answer_shape": answer_shape.get("shape"),
            "target_length_chars": answer_shape.get("target_length_chars"),
        },
    }

    # Stream the synthesizer output if enabled, otherwise stream raw_answer
    # straight through; then stream the stylist on top of the buffered draft.
    draft_answer = raw_answer
    synthesizer_used = False
    if synthesizer_enabled:
        chunks: List[str] = []
        async for chunk in _safe_synthesize_stream(
            raw_answer=raw_answer,
            user_query=user_query,
            source_docs=source_docs,
            rolling_summary=memory_context.get("rolling_summary"),
            answer_shape=answer_shape,
        ):
            chunks.append(chunk)
            # Only emit synth chunks when stylist is OFF — otherwise the
            # client would see two passes of tokens. With stylist ON we
            # buffer the draft and stream stylist tokens instead.
            if not stylist_enabled:
                yield {"event": "token", "delta": chunk, "metadata": None}
        synth_text = "".join(chunks)
        if synth_text and synth_text != raw_answer:
            draft_answer = synth_text
            synthesizer_used = True
    elif not stylist_enabled and raw_answer:
        # No LLM passes — emit the raw answer as a single token chunk so
        # the client sees a streaming UX even on the trivial path.
        yield {"event": "token", "delta": raw_answer, "metadata": None}

    final_answer = draft_answer
    stylist_used = False
    if stylist_enabled and draft_answer:
        last_assistant_turn = _last_assistant_turn(memory_context)
        chunks2: List[str] = []
        async for chunk in _safe_stylize_stream(
            draft_answer=draft_answer,
            user_query=user_query,
            last_assistant_turn=last_assistant_turn,
            answer_shape=answer_shape,
        ):
            chunks2.append(chunk)
            yield {"event": "token", "delta": chunk, "metadata": None}
        styled = "".join(chunks2)
        if styled and styled != draft_answer:
            final_answer = styled
            stylist_used = True

    turn_id = _dispatch_memory_writer(
        session_id=session_id,
        user_text=user_query,
        assistant_text=final_answer,
        project_id=project_id,
        set_id=set_id,
        source_documents=source_docs,
    )

    response = _build_response_dict(
        user_query=user_query,
        contextualized_query=contextualized_query,
        agent_result=agent_result,
        agent_error=agent_error,
        final_answer=final_answer,
        memory_used=memory_used,
        query_rewritten=query_rewritten,
        synthesizer_used=synthesizer_used,
        stylist_used=stylist_used,
        cache_reexpressed=False,
        session_id=session_id,
        project_id=project_id,
        elapsed_ms=int((time.monotonic() - start) * 1000),
        answer_shape=answer_shape,
        multi_query_used=multi_query_used,
        rerank_used=rerank_used,
        source_docs_after=source_docs,
        turn_id=turn_id,
    )
    yield {"event": "done", "delta": None, "metadata": response}


# ---------------------------------------------------------------------------
# Sub-agent wrappers (each isolates failures from the rest of the chain)
# ---------------------------------------------------------------------------


async def _safe_recall(*, session_id: Optional[str], user_query: str) -> Dict[str, Any]:
    """Run Memory Recall (Agent 1) without raising. Returns empty payload on any failure."""
    if not _flag_enabled("MEMORY_RECALL_ENABLED", "false") or not session_id:
        return _empty_memory_context()
    try:
        from agentic.memory.recall_agent import recall

        return await asyncio.to_thread(
            recall,
            session_id=session_id,
            user_query=user_query,
        )
    except Exception as exc:
        logger.warning("chain.recall failed: %s", exc)
        return _empty_memory_context()


async def _safe_rewrite(
    *,
    user_query: str,
    memory_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Query Rewriter (Agent 2) without raising. Falls back to passthrough."""
    if not _flag_enabled("QUERY_REWRITER_ENABLED", "false"):
        return {
            "contextualized_query": user_query,
            "was_rewritten": False,
            "skip_reason": "flag_off",
        }
    try:
        from agentic.memory.query_rewriter import rewrite

        return await asyncio.to_thread(
            rewrite,
            user_query=user_query,
            memory_context=memory_context,
            stream=False,
        )
    except Exception as exc:
        logger.warning("chain.rewrite failed, passing through original query: %s", exc)
        return {
            "contextualized_query": user_query,
            "was_rewritten": False,
            "skip_reason": "rewriter_chain_error",
        }


async def _safe_run_agent(
    *,
    query: str,
    project_id: int,
    set_id: Optional[int],
    conversation_history: Optional[list],
    scope: Optional[Dict[str, Any]],
) -> tuple[Any, Optional[str]]:
    """Run ReAct retrieval (Agent 3) without raising. Returns (result, error)."""
    try:
        from agentic.core.agent import run_agent

        result = await asyncio.to_thread(
            run_agent,
            query=query,
            project_id=project_id,
            set_id=set_id,
            conversation_history=conversation_history,
            scope=scope,
        )
        return result, None
    except Exception as exc:
        logger.error("chain.run_agent failed: %s", exc)
        return None, str(exc)


async def _safe_multi_query_hint(
    *,
    user_query: str,
    project_id: int,
    scope: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Upgrade 1 — Multi-Query + RRF (flag-gated by MULTI_QUERY_RRF_ENABLED).

    Decomposes the user's query into N paraphrased sub-queries, fans them
    out across the agent's retrieval tools in parallel, and fuses the top
    candidates via Reciprocal Rank Fusion. Returns a string hint to
    prepend to the agent's conversation history (it can choose to consult
    the candidates via its existing tools).

    Skip rules:
      * flag off → return None
      * scope active (user already pinned a doc) → return None
      * any failure → return None (logged, never raises)
    """
    if not _flag_enabled("MULTI_QUERY_RRF_ENABLED", "false"):
        logger.debug("chain.multi_query: flag off, skipping")
        return None
    # Skip ONLY when the user has pinned a specific document — not for any
    # scope dict. The orchestrator passes a non-empty scope dict for routing
    # purposes; we only want to skip when the user has already narrowed
    # retrieval to a specific drawing or spec section.
    if isinstance(scope, dict):
        pin_fields = ("drawing_title", "drawing_name", "section_title", "pdf_name")
        if any(scope.get(k) for k in pin_fields):
            logger.debug("chain.multi_query: user has pinned doc, skipping")
            return None
    logger.info("chain.multi_query: invoking build_context_hint")
    try:
        from gateway.retrieval_enrichment import (
            build_context_hint, format_hint_for_agent,
        )
        hint = await asyncio.to_thread(
            build_context_hint, query=user_query, project_id=project_id,
        )
        formatted = format_hint_for_agent(hint)
        if formatted:
            logger.info(
                "chain.multi_query.hit sub_queries=%d candidates=%d",
                len(hint.get("sub_queries") or []),
                len(hint.get("fused") or []),
            )
        else:
            logger.info(
                "chain.multi_query: build_context_hint returned no usable hint"
            )
        return formatted or None
    except Exception as exc:
        logger.warning("chain.multi_query failed (continuing): %s", exc)
        return None


async def _safe_rerank(
    *,
    user_query: str,
    source_docs: list,
) -> list:
    """Upgrade 2 — LLM-as-Judge Reranker (flag-gated by RERANKER_ENABLED).

    Reorders source_documents best-first using a small LLM that scores each
    candidate against the user's original question. Returns the reordered
    list; on any failure returns the input unchanged.
    """
    if not source_docs or len(source_docs) <= 1:
        return source_docs
    # Tier A1 cascade (2026-05-28) — cross-encoder pre-rerank (cheap CPU), runs before LLM rerank.
    if _flag_enabled("CROSS_ENCODER_RERANK_ENABLED", "false"):
        try:
            from gateway.cross_encoder_rerank import cross_encoder_rerank
            source_docs = await asyncio.to_thread(
                cross_encoder_rerank, query=user_query, source_documents=source_docs,
            )
            logger.info("chain.cross_encoder_rerank.applied output=%d", len(source_docs))
        except Exception as exc:
            logger.warning("chain.cross_encoder_rerank failed: %s", exc)
    if not _flag_enabled("RERANKER_ENABLED", "false"):
        return source_docs
    try:
        from gateway.reranker import rerank_source_documents
        reranked = await asyncio.to_thread(
            rerank_source_documents, query=user_query,
            source_documents=source_docs,
        )
        if isinstance(reranked, list) and reranked:
            logger.info(
                "chain.rerank.applied input=%d output=%d",
                len(source_docs), len(reranked),
            )
            return reranked
        return source_docs
    except Exception as exc:
        logger.warning("chain.rerank failed (continuing): %s", exc)
        return source_docs


async def _safe_synthesize(
    *,
    raw_answer: str,
    user_query: str,
    source_docs: list,
    rolling_summary: Optional[str],
    answer_shape: Optional[Dict[str, Any]] = None,
) -> str:
    """Run Synthesizer (Agent 4) buffered. Returns raw_answer on failure."""
    try:
        from agentic.generation.synthesizer import synthesize

        kwargs = dict(
            raw_answer=raw_answer,
            user_query=user_query,
            source_docs=source_docs,
            rolling_summary=rolling_summary,
            stream=False,
        )
        if _shape_is_active(answer_shape):
            kwargs["answer_shape"] = answer_shape
        out = await asyncio.to_thread(synthesize, **kwargs)
        return out if isinstance(out, str) else raw_answer
    except Exception as exc:
        logger.warning("chain.synthesize failed: %s", exc)
        return raw_answer


async def _safe_synthesize_stream(
    *,
    raw_answer: str,
    user_query: str,
    source_docs: list,
    rolling_summary: Optional[str],
    answer_shape: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Stream synthesizer chunks. Yields raw_answer as a single chunk on failure."""
    try:
        from agentic.generation.synthesizer import synthesize

        kwargs = dict(
            raw_answer=raw_answer,
            user_query=user_query,
            source_docs=source_docs,
            rolling_summary=rolling_summary,
            stream=True,
        )
        if _shape_is_active(answer_shape):
            kwargs["answer_shape"] = answer_shape
        gen = await asyncio.to_thread(synthesize, **kwargs)
    except Exception as exc:
        logger.warning("chain.synthesize_stream init failed: %s", exc)
        if raw_answer:
            yield raw_answer
        return

    async for chunk in _iter_to_async(gen):
        if isinstance(chunk, str) and chunk:
            yield chunk


async def _safe_stylize(
    *,
    draft_answer: str,
    user_query: str,
    last_assistant_turn: Optional[str],
    answer_shape: Optional[Dict[str, Any]] = None,
) -> str:
    """Run Stylist (Agent 5) buffered. Returns draft_answer on failure."""
    try:
        from agentic.generation.stylist import stylize

        kwargs = dict(
            draft_answer=draft_answer,
            user_query=user_query,
            last_assistant_turn=last_assistant_turn,
            stream=False,
        )
        if _shape_is_active(answer_shape):
            kwargs["answer_shape"] = answer_shape
        out = await asyncio.to_thread(stylize, **kwargs)
        return out if isinstance(out, str) else draft_answer
    except Exception as exc:
        logger.warning("chain.stylize failed: %s", exc)
        return draft_answer


async def _safe_stylize_stream(
    *,
    draft_answer: str,
    user_query: str,
    last_assistant_turn: Optional[str],
    answer_shape: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Stream stylist chunks. Yields draft_answer as a single chunk on failure."""
    try:
        from agentic.generation.stylist import stylize

        kwargs = dict(
            draft_answer=draft_answer,
            user_query=user_query,
            last_assistant_turn=last_assistant_turn,
            stream=True,
        )
        if _shape_is_active(answer_shape):
            kwargs["answer_shape"] = answer_shape
        gen = await asyncio.to_thread(stylize, **kwargs)
    except Exception as exc:
        logger.warning("chain.stylize_stream init failed: %s", exc)
        if draft_answer:
            yield draft_answer
        return

    async for chunk in _iter_to_async(gen):
        if isinstance(chunk, str) and chunk:
            yield chunk


async def _safe_reexpress_cached(
    *,
    cached_answer: str,
    user_query: str,
    rolling_summary: Optional[str],
    last_assistant_turn: Optional[str],
    is_followup: bool,
    stream: bool,
    answer_shape: Optional[Dict[str, Any]] = None,
) -> str:
    """Run Cache Re-Expression (Agent 5b) buffered."""
    try:
        from agentic.generation.stylist import reexpress_cached

        kwargs = dict(
            cached_answer=cached_answer,
            user_query=user_query,
            rolling_summary=rolling_summary,
            last_assistant_turn=last_assistant_turn,
            is_followup=is_followup,
            stream=False,
        )
        if _shape_is_active(answer_shape):
            kwargs["answer_shape"] = answer_shape
        out = await asyncio.to_thread(reexpress_cached, **kwargs)
        return out if isinstance(out, str) else cached_answer
    except Exception as exc:
        logger.warning("chain.reexpress_cached failed: %s", exc)
        return cached_answer


async def _safe_reexpress_cached_stream(
    *,
    cached_answer: str,
    user_query: str,
    rolling_summary: Optional[str],
    last_assistant_turn: Optional[str],
    is_followup: bool,
    answer_shape: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Stream cache re-expression chunks."""
    try:
        from agentic.generation.stylist import reexpress_cached

        kwargs = dict(
            cached_answer=cached_answer,
            user_query=user_query,
            rolling_summary=rolling_summary,
            last_assistant_turn=last_assistant_turn,
            is_followup=is_followup,
            stream=True,
        )
        if _shape_is_active(answer_shape):
            kwargs["answer_shape"] = answer_shape
        gen = await asyncio.to_thread(reexpress_cached, **kwargs)
    except Exception as exc:
        logger.warning("chain.reexpress_cached_stream init failed: %s", exc)
        if cached_answer:
            yield cached_answer
        return

    async for chunk in _iter_to_async(gen):
        if isinstance(chunk, str) and chunk:
            yield chunk


def _shape_is_active(answer_shape: Optional[Dict[str, Any]]) -> bool:
    """True when ``answer_shape`` is non-default (i.e. should reach generators).

    The default-shape result is the canonical 'no behaviour change'
    sentinel — we MUST NOT pass it through to ``synthesize/stylize`` as a
    kwarg, because a) it carries no signal and b) it would break test
    mocks whose function signatures predate this kwarg.
    """
    if not answer_shape:
        return False
    shape = answer_shape.get("shape")
    return bool(shape) and shape != "default"


def _safe_classify_shape(
    *,
    user_query: str,
    has_image: bool,
) -> Dict[str, Any]:
    """Run Answer Shape Classifier (Agent 0.5) without raising.

    Always returns a dict so downstream agents can rely on the shape
    contract. On any failure (including an import error) returns the
    canonical 'default' shape, which downstream agents treat as "use
    existing prompts unchanged" — preserving today's behaviour.
    """
    try:
        from agentic.generation.answer_shape import classify_shape

        return classify_shape(user_query=user_query, has_image=has_image)
    except Exception as exc:
        logger.warning("chain.classify_shape failed: %s", exc)
        return {
            "shape": "default",
            "target_length_chars": 0,
            "target_word_count": 0,
            "was_llm_classified": False,
            "confidence": 0.0,
            "skip_reason": "chain_error",
        }


def _scope_has_image(scope: Optional[Dict[str, Any]]) -> bool:
    """Defensively read ``scope.has_image`` from either dict or object scope."""
    if scope is None:
        return False
    if isinstance(scope, dict):
        return bool(scope.get("has_image", False))
    return bool(getattr(scope, "has_image", False))


def _ensure_session_id(
    *,
    session_id: Optional[str],
    user_query: str,
    project_id: Optional[int],
) -> Optional[str]:
    """Auto-create a session if the caller didn't pass one.

    Returns the existing or newly-created session_id. Returns None only if
    MemoryManager itself is unavailable (e.g. import failure) — in which
    case the chain still works, just without memory persistence.

    This matches v3.0 orchestrator behaviour: every query gets a session
    so multi-turn memory works out-of-the-box, and the user can pass that
    session_id back on the next request to maintain context.
    """
    if session_id:
        return session_id
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
        mm = get_memory_manager()
        new_id = mm.create_session(
            user_query=user_query or "Session created via chain",
            project_id=project_id,
        )
        logger.info("chain.session: auto-created session_id=%s", new_id)
        return new_id
    except Exception as exc:
        logger.warning("chain.session: auto-create failed: %s", exc)
        return None


def _dispatch_memory_writer(
    *,
    session_id: Optional[str],
    user_text: str,
    assistant_text: str,
    project_id: Optional[int],
    set_id: Optional[int],
    turn_id: Optional[str] = None,
    turn_type: str = "rag",
    parent_turn_id: Optional[str] = None,
    source_documents: Optional[list] = None,
) -> Optional[str]:
    """Fire-and-forget Memory Writer (Agent 6).

    Never raises. Never awaits. Returns immediately.

    Returns
    -------
    str | None
        The ``turn_id`` actually attached to this turn pair. When the
        caller passes ``turn_id=None`` we mint a fresh UUID4 here so the
        response builder can echo it back to the HTTP client without
        waiting for the background write to complete.
    """
    if not session_id or not assistant_text:
        return turn_id
    # Phase 0 — mint a stable turn_id synchronously. Generated here so the
    # value flows back into the response payload before the background
    # writer has finished persisting. Caller passes None on the normal
    # /query path; Deep Dive passes its own pre-allocated id.
    if turn_id is None:
        import uuid as _uuid_mod
        turn_id = str(_uuid_mod.uuid4())
    try:
        from agentic.memory.writer import MemoryWriter

        writer = MemoryWriter()
        # Writer's own contract is fire-and-forget — submit to its
        # internal pool. Do NOT await; do NOT block the response.
        writer.write_turn_async(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            project_id=project_id,
            set_id=str(set_id) if set_id is not None else None,
            turn_id=turn_id,
            turn_type=turn_type,
            parent_turn_id=parent_turn_id,
            source_documents=source_documents,
        )
    except Exception as exc:
        # Writer failure must NEVER affect the user response.
        logger.warning("chain.memory_writer dispatch failed: %s", exc)
    return turn_id


# ---------------------------------------------------------------------------
# Helpers — extraction, response shaping, async glue
# ---------------------------------------------------------------------------


def _empty_memory_context() -> Dict[str, Any]:
    return {
        "rolling_summary": None,
        "recent_turns": [],
        "semantic_turns": [],
        "topic_tags": [],
        "had_context": False,
    }


def _derive_continuity_signals(
    memory_context: Dict[str, Any],
) -> tuple[Optional[str], Optional[str], bool]:
    """Pull the inputs the Stylist re-expression prompt needs."""
    rolling_summary = memory_context.get("rolling_summary")
    last_assistant_turn = _last_assistant_turn(memory_context)
    recent = memory_context.get("recent_turns") or []
    is_followup = bool(recent) and bool(memory_context.get("had_context"))
    return last_assistant_turn, rolling_summary, is_followup


def _last_assistant_turn(memory_context: Dict[str, Any]) -> Optional[str]:
    """Find the most recent assistant message in recent_turns, if any."""
    recent = memory_context.get("recent_turns") or []
    for turn in reversed(recent):
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if content:
                return content
    return None


def _extract_answer(agent_result: Any) -> str:
    """Pull ``answer`` off an AgentResult-like object (or empty string)."""
    if agent_result is None:
        return ""
    return getattr(agent_result, "answer", "") or ""


def _extract_source_docs(agent_result: Any) -> list:
    """Pull ``source_docs`` (preferred) or ``sources`` off the result."""
    if agent_result is None:
        return []
    docs = getattr(agent_result, "source_docs", None)
    if docs:
        return docs
    return getattr(agent_result, "sources", []) or []


async def _iter_to_async(gen: Any) -> AsyncIterator[str]:
    """Adapt a sync iterator (from synthesize/stylize stream=True) to async.

    Each ``next()`` is dispatched to a worker thread to keep the event
    loop responsive while the underlying SDK streams chunks. Mocked
    iterators in tests run instantly so this stays cheap.
    """
    if gen is None:
        return
    if hasattr(gen, "__aiter__"):
        async for item in gen:
            yield item
        return
    iterator = iter(gen)
    sentinel = object()
    while True:
        try:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
        except Exception as exc:
            logger.warning("chain._iter_to_async: stream errored: %s", exc)
            return
        if chunk is sentinel:
            return
        yield chunk  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Citation-grounding filter (v3.2)
#
# Client feedback (2026-05-12): "list of reference drawings / specs / email /
# meeting should be precise or concise and should not give big list to go
# through to verify". The agent was returning the full deduped retrieval set
# (10-17 docs) even though the synthesized answer only cited 2-3 of them.
#
# This filter inspects the final answer text, extracts anchor tokens
# (sheet numbers like "A-101", CSI section numbers like "32 9300", capitalized
# section names like "PLUMBING FIXTURES", quoted phrases, page refs), then
# scores each source_doc against those anchors. Only docs that the answer
# actually grounded against are kept, capped at MAX_GROUNDED_SOURCES.
#
# Safety net: if zero docs survive (e.g. the LLM didn't cite any anchor
# cleanly), keep MIN_FALLBACK_SOURCES from the top of the original ranked list
# so the UI never shows an empty citation panel.
# ---------------------------------------------------------------------------

# Sheet-number pattern: A-101, CD-101, E-602, M-201a, S-101, RCP-01
_SHEET_NUM_RE = re.compile(r"\b[A-Z]{1,4}-\d{1,4}[A-Za-z]?\b")
# CSI section: "32 9300", "15 4000", or 6-digit "220500"
_CSI_NUM_RE = re.compile(r"\b\d{2}\s\d{2,4}\b|\b\d{6}\b")
# All-caps phrases of >=2 words (drawing/section titles): "PLUMBING FIXTURES"
_CAPS_PHRASE_RE = re.compile(r"\b[A-Z]{3,}(?:\s+[A-Z]{2,}){1,5}\b")
# Quoted phrases the LLM uses to cite
_QUOTED_RE = re.compile(r'"([^"]{4,80})"')
# Single all-caps word (single-word section names): "PLUMBING", "ELECTRICAL"
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{4,}\b")


def _extract_anchors(answer_text: str) -> Dict[str, set]:
    """Pull citation anchors from a final answer.

    Returns a dict with keys: sheet_nums, csi_nums, caps_phrases, quoted,
    caps_words — all lowercased/normalized sets for cheap membership tests.
    """
    if not answer_text:
        return {
            "sheet_nums": set(), "csi_nums": set(),
            "caps_phrases": set(), "quoted": set(), "caps_words": set(),
        }
    return {
        "sheet_nums": {m.upper() for m in _SHEET_NUM_RE.findall(answer_text)},
        "csi_nums": {m.strip() for m in _CSI_NUM_RE.findall(answer_text)},
        "caps_phrases": {m.upper() for m in _CAPS_PHRASE_RE.findall(answer_text)},
        "quoted": {m.lower() for m in _QUOTED_RE.findall(answer_text)},
        "caps_words": {m.upper() for m in _CAPS_WORD_RE.findall(answer_text)},
    }


def _score_doc_grounding(doc: dict, anchors: Dict[str, set]) -> float:
    """Compute how strongly a source_doc is cited by the answer."""
    if not isinstance(doc, dict):
        return 0.0
    score = 0.0

    # Defensive str-coercion: occasionally fields like csi_division come back
    # as dicts (e.g. structured CSI tagging) or None. Coerce everything to
    # a plain string before any .strip()/.upper() so the filter never crashes
    # on a single anomalous doc.
    def _s(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        return str(v)

    drawing_name = _s(doc.get("drawing_name")).upper().strip()
    pdf_name = _s(doc.get("pdf_name")).upper().strip()
    csi_div = _s(doc.get("csi_division")).strip()
    drawing_title = _s(doc.get("drawing_title")).upper().strip()
    display_title = _s(doc.get("display_title")).upper().strip()
    text_excerpt = _s(doc.get("text_excerpt")).lower()

    # Sheet number match (+3) — strongest signal
    for sheet in anchors["sheet_nums"]:
        if sheet and (sheet in drawing_name or sheet in pdf_name
                      or sheet in display_title):
            score += 3.0
            break

    # CSI section number match (+3)
    for csi in anchors["csi_nums"]:
        if csi and (csi in csi_div or csi in drawing_name
                    or csi in pdf_name):
            score += 3.0
            break

    # Drawing/section title match (+2) — full caps phrase
    for phrase in anchors["caps_phrases"]:
        if phrase and (phrase in drawing_title or phrase in display_title):
            score += 2.0
            break

    # Quoted phrase appears in text excerpt (+1)
    if text_excerpt:
        for quote in anchors["quoted"]:
            if quote and quote in text_excerpt:
                score += 1.0
                break

    # Single all-caps word match against drawing_title (+1)
    if drawing_title or display_title:
        for word in anchors["caps_words"]:
            # Skip generic words that match too aggressively
            if word in {"AND", "THE", "FOR", "WITH", "FROM", "INTO", "OVER",
                        "RAG", "PDF", "CSI"}:
                continue
            if (word in drawing_title) or (word in display_title):
                score += 1.0
                break

    return score


def _filter_grounded_sources(
    answer_text: str,
    source_docs: list,
    max_sources: int = 5,
    min_fallback: int = 3,
) -> list:
    """Filter source_docs to those actually cited by the answer.

    Returns the same dict shape (no field changes). Pure ranking + truncation.
    """
    if not source_docs:
        return source_docs
    if not answer_text or not answer_text.strip():
        # No answer to ground against — keep top-N from existing rank
        return source_docs[:max_sources]

    anchors = _extract_anchors(answer_text)
    # If the answer has no anchors at all (very short factoid like "8 feet"),
    # don't try to filter — keep top-N. Better to show too few than wrong ones.
    if not any(anchors.values()):
        return source_docs[:max_sources]

    scored: list[tuple[float, float, int, dict]] = []
    for idx, doc in enumerate(source_docs):
        ground_score = _score_doc_grounding(doc, anchors)
        retrieval_score = 0.0
        if isinstance(doc, dict):
            try:
                retrieval_score = float(doc.get("score") or 0.0)
            except (TypeError, ValueError):
                retrieval_score = 0.0
        # Sort key: grounding DESC, retrieval DESC, original index ASC
        scored.append((-ground_score, -retrieval_score, idx, doc))

    scored.sort()
    grounded = [t[3] for t in scored if -t[0] >= 1.0]

    # Safety net: if nothing scored, keep min_fallback from the top of the
    # original list (preserves prior behavior for queries the LLM answered
    # without citing anchors verbatim).
    if not grounded:
        return source_docs[:max(min_fallback, 1)]

    return grounded[:max_sources]


def _build_response_dict(
    *,
    user_query: str,
    contextualized_query: str,
    agent_result: Any,
    agent_error: Optional[str],
    final_answer: str,
    memory_used: bool,
    query_rewritten: bool,
    synthesizer_used: bool,
    stylist_used: bool,
    cache_reexpressed: bool,
    session_id: Optional[str],
    project_id: Optional[int],
    elapsed_ms: int,
    answer_shape: Optional[Dict[str, Any]] = None,
    multi_query_used: bool = False,
    rerank_used: bool = False,
    source_docs_after: Optional[list] = None,
    turn_id: Optional[str] = None,
    turn_type: str = "rag",
    parent_turn_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a response dict mirroring ``Orchestrator._build_response``.

    Adds five v3.1-specific flags so callers / dashboards can see which
    optional agents fired:

    * ``memory_context_used``
    * ``query_rewritten``
    * ``synthesizer_used``
    * ``stylist_used``
    * ``cache_reexpressed``
    """
    confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.2}

    answer = final_answer or ""
    sources = getattr(agent_result, "sources", []) or [] if agent_result else []
    confidence = getattr(agent_result, "confidence", "low") if agent_result else "low"
    cost = getattr(agent_result, "total_cost_usd", 0.0) if agent_result else 0.0
    steps = getattr(agent_result, "total_steps", 0) if agent_result else 0
    model = getattr(agent_result, "model", "") if agent_result else ""
    follow_ups = (
        getattr(agent_result, "follow_up_questions", []) or [] if agent_result else []
    )
    input_tokens = (
        getattr(agent_result, "total_input_tokens", 0) if agent_result else 0
    )
    output_tokens = (
        getattr(agent_result, "total_output_tokens", 0) if agent_result else 0
    )
    source_docs = (
        getattr(agent_result, "source_docs", []) or [] if agent_result else []
    )
    # If the chain reranked/filtered source_docs upstream, prefer that list
    # so downstream consumers (UI / eval CSV) see the post-rerank order.
    if source_docs_after is not None:
        source_docs = source_docs_after

    # Run the orchestrator's source-doc post-processor so the chain emits
    # the same normalized field names (s3_path, file_name, display_title,
    # download_url, pdf_name, drawing_name, drawing_title, page) that the
    # frontend / test pipeline / Postman expect.
    s3_paths_normalized: list[str] = []
    try:
        from gateway.orchestrator import _extract_source_documents
        # Prefer the agent's rich source_docs (built by _build_source_doc with
        # bbox_pt, text_excerpt, drawing_id, csi_division, score, trade) over
        # the basic ``sources`` string list. Fall back to ``sources`` only when
        # source_docs is empty (older agent paths or fallback responses).
        feed = source_docs if source_docs else sources
        normalized_docs, s3_paths_normalized = _extract_source_documents(feed)
        if normalized_docs:
            source_docs = normalized_docs
    except Exception as exc:
        logger.warning(
            "chain._build_response_dict: source-doc normalization failed (%s); "
            "falling back to raw chain source_docs", exc,
        )

    # Capture full pre-dedup retrieval set for transparency (Phase 2026-06-02 fix).
    # Returned to caller as `all_retrieved_sources` so client can see every drawing
    # that contributed to the answer — fixes the "5 dupes of same PDF" feedback.
    source_docs_pre_dedup = list(source_docs) if source_docs else []

    # Dedup source_documents by stable identity. Same proven logic as v3.0:
    # prefer entries that carry bbox/text excerpt (citation-ready). Without
    # this, the same PDF appears 5-20× because drawing-level + fragment-level
    # results both ranked through and the reranker keeps duplicates.
    if source_docs:
        before = len(source_docs)
        best_by_key: dict[str, dict] = {}
        no_key_docs: list[dict] = []  # string-only sources (no pdf identity)
        for doc in source_docs:
            if not isinstance(doc, dict):
                continue
            # Identity tuple: include BOTH pdf_name and drawing_name so different
            # drawings inside the same master PDF stay separate (Phase 2026-06-02 fix).
            # Page included as third axis for multi-page drawings.
            _pdf  = (doc.get("pdf_name") or doc.get("file_name") or doc.get("s3_path") or "").strip()
            _draw = (doc.get("drawing_name") or "").strip()
            _page = doc.get("page") or doc.get("page_count") or 0
            key = (_pdf, _draw, _page) if (_pdf or _draw) else ""

            if not key:
                no_key_docs.append(doc)
                continue
            prev = best_by_key.get(key)
            if prev is None:
                best_by_key[key] = doc
                continue
            # Tiebreak: prefer entry with bbox/text excerpt.
            has_bbox = bool(doc.get("bbox_pt") or doc.get("bbox_px"))
            prev_bbox = bool(prev.get("bbox_pt") or prev.get("bbox_px"))
            has_text = bool(doc.get("text_excerpt"))
            prev_text = bool(prev.get("text_excerpt"))
            new_rank = (1 if has_bbox else 0) * 2 + (1 if has_text else 0)
            prev_rank = (1 if prev_bbox else 0) * 2 + (1 if prev_text else 0)
            if new_rank > prev_rank:
                best_by_key[key] = doc
        source_docs = list(best_by_key.values()) + no_key_docs
        if before != len(source_docs):
            logger.info(
                "chain.dedup: source_documents %d -> %d", before, len(source_docs),
            )

    # ----- Citation-grounding filter (v3.2) -----
    # Trim source_documents to only those the final answer actually cites,
    # capped at MAX_GROUNDED_SOURCES (default 5). Preserves the full deduped
    # list under debug_info.source_documents_all for audit.
    source_docs_all_pre_filter: list = list(source_docs)
    grounded_filter_used = False
    if _flag_enabled("GROUNDED_CITATIONS_ENABLED", "true") and source_docs:
        try:
            max_cap = int(os.getenv("MAX_GROUNDED_SOURCES", "5"))
        except (TypeError, ValueError):
            max_cap = 5
        try:
            min_fallback = int(os.getenv("MIN_FALLBACK_SOURCES", "3"))
        except (TypeError, ValueError):
            min_fallback = 3
        before_filter = len(source_docs)
        try:
            source_docs = _filter_grounded_sources(
                answer_text=answer,
                source_docs=source_docs,
                max_sources=max_cap,
                min_fallback=min_fallback,
            )
            grounded_filter_used = True
            if before_filter != len(source_docs):
                logger.info(
                    "chain.grounded_filter: source_documents %d -> %d "
                    "(cap=%d, fallback=%d)",
                    before_filter, len(source_docs), max_cap, min_fallback,
                )
        except Exception as exc:
            logger.warning(
                "chain.grounded_filter failed (%s); keeping deduped list", exc,
            )

    # Resync s3_paths from the deduped source_documents AND filter out
    # display-title strings (e.g. "CD-101", "ELECTRICAL PANEL SCHEDULES")
    # that the agent may have emitted as bare strings — those aren't real
    # S3 keys. Real keys are long hashed strings. Heuristic: real S3 keys
    # are typically >20 chars with no spaces; display titles are shorter
    # or contain spaces / look like sheet numbers.
    seen_paths: set = set()
    resynced_s3: list[str] = []
    for sd in source_docs:
        if not isinstance(sd, dict):
            continue
        s = (sd.get("s3_path") or "").strip()
        if not s or s in seen_paths:
            continue
        # Filter out display-title-as-path: contains spaces, OR is short and
        # matches a sheet-number pattern. Real S3 keys are long hashed strings
        # without spaces.
        if " " in s:
            continue
        if len(s) < 20 and re.match(r"^[A-Z]{1,4}-?\d+[a-z]?$", s):
            # Looks like a sheet number (E-602, CD-101) — not an S3 key
            continue
        seen_paths.add(s)
        resynced_s3.append(s)
    s3_paths_normalized = resynced_s3

    # ----- Resync sources[] from filtered source_documents (v3.2) -----
    # Bug 1 (Reference Source Bugs): the public sources[] list was returning
    # the raw, unfiltered agent_result.sources — a mix of 55+ hashed pdf_names
    # + display titles + sheet numbers + "COVER SHEET"-style strings, all
    # contradicting the answer's actual citations. Derive sources[] from the
    # post-grounded-filter source_documents so there is ONE source of truth.
    def _label_of(sd: dict) -> str:
        for k in ("display_title", "drawing_name", "drawing_title",
                  "pdf_name", "file_name"):
            v = sd.get(k)
            if v is None:
                continue
            s = v if isinstance(v, str) else str(v)
            s = s.strip()
            if s:
                return s
        return ""

    if source_docs:
        resynced_sources: list[str] = []
        seen_keys: set = set()
        for sd in source_docs:
            if not isinstance(sd, dict):
                continue
            label = _label_of(sd)
            if not label or label in seen_keys:
                continue
            seen_keys.add(label)
            resynced_sources.append(label)
        sources = resynced_sources

    # v3 bbox enrichment — populates bbox_pt / bbox_px / page_width_pt /
    # page_height_pt / text_blocks on each source_doc from v3 Mongo
    # collections. Safe no-op for legacy projects.
    try:
        from gateway.orchestrator import _enrich_v3_with_bbox
        _enrich_v3_with_bbox(source_docs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("chain: v3 bbox enrich failed (%s); continuing", exc)

    return {
        # core answer
        "query": user_query,
        "contextualized_query": contextualized_query,
        "answer": answer,
        "rag_answer": answer,
        # retrieval metrics
        "retrieval_count": len(sources),
        "confidence": confidence,
        "confidence_score": confidence_map.get(confidence, 0.5),
        "average_score": confidence_map.get(confidence, 0.5),
        "is_clarification": confidence == "low",
        # follow-ups
        "follow_up_questions": follow_ups,
        # model + tokens
        "model_used": model,
        "token_usage": {
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
        # sources (normalized: s3_path, file_name, display_title, download_url,
        # pdf_name, drawing_name, drawing_title, page) — matches v3.0 frontend contract.
        "source_documents": source_docs,
        # Phase 2026-06-02 fix: full pre-dedup, pre-filter retrieval set so the
        # client can see every drawing that contributed to the answer (addresses
        # "5 copies of same PDF" / "missing related drawings" feedback).
        "all_retrieved_sources": source_docs_pre_dedup,
        "sources": sources,
        "s3_paths": s3_paths_normalized,
        "s3_path_count": len(s3_paths_normalized),
        # debug
        "debug_info": {
            "agentic_steps": steps,
            "agentic_cost_usd": cost,
            "agent_error": agent_error,
            # v3.2 grounded-citation audit fields
            "grounded_filter_used": grounded_filter_used,
            "source_documents_pre_filter_count": len(source_docs_all_pre_filter),
            "source_documents_post_filter_count": len(source_docs),
            # Phase 2026-06-02 fix: how many sources were retrieved before dedup
            "all_retrieved_sources_count": len(source_docs_pre_dedup),
        },
        # timing + identity
        "processing_time_ms": elapsed_ms,
        "project_id": project_id,
        "session_id": session_id,
        # engine metadata
        "search_mode": "agentic",
        "engine_used": "agentic",
        "fallback_used": False,
        "agentic_confidence": confidence,
        "success": agent_error is None,
        "error": agent_error,
        # v3.1-specific chain telemetry
        "memory_context_used": memory_used,
        "query_rewritten": query_rewritten,
        "synthesizer_used": synthesizer_used,
        "stylist_used": stylist_used,
        "cache_reexpressed": cache_reexpressed,
        # retrieval upgrades (Multi-Query/RRF + LLM-as-judge reranker)
        "multi_query_used": multi_query_used,
        "rerank_used": rerank_used,
        # Agent 0.5 telemetry (None when classifier flag off / errored)
        "answer_shape": (answer_shape or {}).get("shape"),
        "target_length_chars": (answer_shape or {}).get("target_length_chars"),
        # v3.3 — turn identity (Phase 0). Stable UUID per (user, assistant)
        # turn pair. UI passes turn_id back on Deep Dive trigger. ADDITIVE
        # only — old clients ignore. turn_type discriminator stays "rag" on
        # normal /query path; "deep_dive" on the Deep Dive endpoint.
        "turn_id": turn_id,
        "turn_type": turn_type,
        "parent_turn_id": parent_turn_id,
    }
