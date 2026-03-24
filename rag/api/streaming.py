"""Streaming answer generation for SSE responses — v2.

Reuses the same retrieval, intent detection, session management, and prompt
building as generation_unified.py. Only the final LLM call uses streaming.

v2 additions:
  - Follow-up questions generated via a lightweight post-stream LLM call
  - Confidence score + hallucination rollback (clarification mode)
  - Granular token tracking
  - Mode-aware conversation context
  - Context window budgeting
"""
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional

from . import state
from .helpers import (
    build_context_text,
    budget_context_window,
    compute_confidence_score,
    detect_trade_from_query,
    estimate_tokens,
    filter_chunks_by_trade,
    format_filename_for_display,
    parse_follow_up_questions,
)
from .intent import detect_intent
from .prompts import (
    build_clarification_prompt,
    build_hybrid_prompt,
    build_rag_prompt,
    build_web_prompt,
)

client = state.client
LLM_MODEL = state.LLM_MODEL
WEB_SEARCH_AVAILABLE = state.WEB_SEARCH_AVAILABLE
MEMORY_MANAGER = state.MEMORY_MANAGER
web_search = state.web_search
retrieve_context = state.retrieve_context

CONFIDENCE_THRESHOLD = 0.30

# ── Cost constants ────────────────────────────────────────────────────────────
_INPUT_COST_PER_1K = 0.0025
_OUTPUT_COST_PER_1K = 0.01


def _sse_event(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


async def stream_unified_answer(
    user_query: str,
    search_mode: str = "rag",
    top_k: int = 5,
    min_score: float = 0.1,
    temperature: float = 0.3,
    max_tokens: int = 500,
    include_citations: bool = False,
    include_s3_paths: bool = True,
    filter_source_type: Optional[str] = None,
    project_id: Optional[int] = None,
    session_id: Optional[str] = None,
    create_new_session: bool = False,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events for a streaming answer.

    Events:
      {"type": "start",    "search_mode": ..., "session_id": ...}
      {"type": "chunk",    "text": ...}
      {"type": "follow_up","questions": [...]}
      {"type": "done",     "session_id": ..., "confidence_score": ..., ...}
      {"type": "error",    "message": ...}
    """
    start_time = time.time()

    # ── 0. Intent detection ──────────────────────────────────────────────────
    intent_type, friendly_response = detect_intent(user_query)

    if intent_type in ("greeting", "small_talk", "thanks", "farewell"):
        current_session_id = _handle_intent_session(
            user_query, friendly_response, session_id, create_new_session,
            project_id, filter_source_type,
        )
        yield _sse_event({"type": "start", "search_mode": "intent", "session_id": current_session_id})
        yield _sse_event({"type": "chunk", "text": friendly_response})
        yield _sse_event({
            "type": "done",
            "session_id": current_session_id,
            "search_mode": "intent",
            "confidence_score": 1.0,
            "is_clarification": False,
            "follow_up_questions": [],
            "retrieval_count": 0,
            "web_source_count": 0,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        })
        return

    # ── 1. Session management ────────────────────────────────────────────────
    current_session_id = session_id
    if MEMORY_MANAGER:
        if create_new_session or not session_id:
            current_session_id = MEMORY_MANAGER.create_session(
                user_query=user_query, project_id=project_id,
                filter_source_type=filter_source_type,
            )
        elif session_id:
            sess = MEMORY_MANAGER.get_session(session_id)
            if sess:
                MEMORY_MANAGER.add_to_session(
                    session_id, "user", user_query, tokens=estimate_tokens(user_query),
                    metadata={"search_mode": search_mode},
                )
            else:
                current_session_id = MEMORY_MANAGER.create_session(
                    user_query=user_query, project_id=project_id,
                    filter_source_type=filter_source_type, session_id=session_id,
                )

    yield _sse_event({"type": "start", "search_mode": search_mode, "session_id": current_session_id})

    # ── 2. Retrieval with trade filtering + budgeting ────────────────────────
    rag_context_chunks: List[Dict] = []
    rag_context_text = ""
    web_context_text = ""
    web_sources: List[Dict] = []

    detected_trade = detect_trade_from_query(user_query)

    def _do_rag():
        chunks = retrieve_context(
            query=user_query, top_k=top_k + 4, min_score=min_score,
            filter_source_type=filter_source_type, filter_project_id=project_id,
        )
        valid = [c for c in chunks if c.get("text", "").strip() and len(c.get("text", "").strip()) > 10]
        if detected_trade:
            valid = filter_chunks_by_trade(valid, detected_trade)
        return budget_context_window(valid, max_context_tokens=4000, max_chunks=top_k)

    def _do_web():
        return web_search(user_query)

    def _web_text(ws_result):
        nonlocal web_sources
        wa = ws_result.get("answer", "")
        web_sources = ws_result.get("sources", [])
        if not web_sources:
            return ""
        txt = "WEB SEARCH RESULTS:\n\n"
        txt += f"Answer from web search: {wa}\n\nSources:\n"
        for i, s in enumerate(web_sources[:5]):
            txt += f"[{i+1}] {s.get('title', '')}\n    URL: {s.get('url', '')}\n\n"
        return txt

    try:
        if search_mode == "hybrid" and WEB_SEARCH_AVAILABLE:
            with ThreadPoolExecutor(max_workers=2) as pool:
                rf = pool.submit(_do_rag)
                wf = pool.submit(_do_web)
                rag_context_chunks = rf.result(timeout=15)
                rag_context_text = build_context_text(rag_context_chunks)
                web_context_text = _web_text(wf.result(timeout=15))
        elif search_mode in ("rag", "hybrid"):
            rag_context_chunks = _do_rag()
            rag_context_text = build_context_text(rag_context_chunks)
        elif search_mode == "web" and WEB_SEARCH_AVAILABLE:
            web_context_text = _web_text(_do_web())
    except Exception as exc:
        yield _sse_event({"type": "error", "message": f"Retrieval error: {exc}"})
        return

    # ── 3. Confidence check ──────────────────────────────────────────────────
    confidence = compute_confidence_score(rag_context_chunks, search_mode)
    is_clarification = (
        search_mode in ("rag", "hybrid")
        and confidence < CONFIDENCE_THRESHOLD
        and rag_context_chunks
    )

    # ── 4. Conversation context (mode-aware) ─────────────────────────────────
    conversation_messages: List[Dict] = []
    conversation_context = ""

    if MEMORY_MANAGER and current_session_id:
        sess = MEMORY_MANAGER.get_session(current_session_id)
        if sess:
            conversation_messages = sess.get_conversation_for_llm(max_tokens=2000, preserve_early_history=True)
            q_index = sess.get_conversation_index()
            recent = []
            for m in conversation_messages[-4:]:
                r, c = m.get("role", ""), m.get("content", "")
                m_mode = m.get("metadata", {}).get("search_mode", "") if isinstance(m.get("metadata"), dict) else ""
                if r == "user":
                    recent.append(f"User previously asked: {c[:150]}")
                elif r == "assistant":
                    if m_mode and m_mode != search_mode:
                        recent.append(f"[Previous answer from {m_mode.upper()} mode — now using {search_mode.upper()}]")
                    else:
                        recent.append(f"You previously answered: {c[:150]}")
            if recent:
                conversation_context = "\n".join(recent)
            if q_index:
                conversation_context += f"\n\nCOMPLETE LIST OF USER QUESTIONS IN THIS SESSION:\n{q_index}"

    # ── 5. Build system prompt ───────────────────────────────────────────────
    if is_clarification:
        summary = f"Retrieved {len(rag_context_chunks)} chunks, avg similarity: {confidence:.2f}"
        system_prompt = build_clarification_prompt(user_query, summary, conversation_context)
    elif search_mode == "rag":
        system_prompt = build_rag_prompt(user_query, rag_context_text, conversation_context, include_citations)
    elif search_mode == "web":
        system_prompt = build_web_prompt(user_query, web_context_text, conversation_context, include_citations)
    else:
        system_prompt = build_hybrid_prompt(user_query, rag_context_text, web_context_text, conversation_context, include_citations)

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_messages:
        for m in conversation_messages:
            if m.get("role") != "system":
                messages.append({"role": m["role"], "content": m.get("content", "")})

    # ── 6. Streaming LLM call ────────────────────────────────────────────────
    full_answer = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        stream = client.responses.create(
            model=LLM_MODEL,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            stream=True,
        )

        separator_buffer = ""
        in_followup_section = False

        for event in stream:
            if hasattr(event, "type"):
                if event.type == "response.output_text.delta":
                    delta = event.delta
                    if delta:
                        full_answer += delta
                        # Check if we've hit the follow-up separator
                        if not in_followup_section:
                            if "---FOLLOW_UP---" in full_answer:
                                in_followup_section = True
                                # Send only the answer part
                                parts = full_answer.split("---FOLLOW_UP---", 1)
                                # The remaining delta after separator is follow-up text
                                # Don't stream it to the user
                            else:
                                yield _sse_event({"type": "chunk", "text": delta})
                        # If in follow-up section, accumulate silently

                elif event.type == "response.completed":
                    if hasattr(event, "response") and hasattr(event.response, "usage"):
                        usage = event.response.usage
                        prompt_tokens = getattr(usage, "input_tokens", 0)
                        completion_tokens = getattr(usage, "output_tokens", 0)

    except Exception as exc:
        yield _sse_event({"type": "error", "message": f"LLM error: {exc}"})
        traceback.print_exc()
        return

    # ── 7. Parse follow-up questions ─────────────────────────────────────────
    clean_answer, follow_up_questions = parse_follow_up_questions(full_answer)

    if follow_up_questions:
        yield _sse_event({"type": "follow_up", "questions": follow_up_questions})

    # ── 8. Save to session ───────────────────────────────────────────────────
    if MEMORY_MANAGER and current_session_id and clean_answer:
        MEMORY_MANAGER.add_to_session(
            current_session_id, "assistant", clean_answer,
            tokens=estimate_tokens(clean_answer),
            metadata={
                "search_mode": search_mode,
                "retrieval_count": len(rag_context_chunks),
                "confidence": confidence,
            },
        )

    # ── 9. Build source documents ────────────────────────────────────────────
    source_documents = []
    unique_s3_paths = []
    if rag_context_chunks and include_s3_paths:
        seen = set()
        for chunk in rag_context_chunks:
            s3 = chunk.get("s3_path")
            if not s3:
                continue
            pn = chunk.get("pdf_name", "")
            fn = format_filename_for_display(pn) if pn else "Unknown"
            dk = f"{s3}::{fn}"
            if dk in seen:
                continue
            seen.add(dk)
            unique_s3_paths.append(s3)
            source_documents.append({
                "s3_path": s3,
                "file_name": fn,
                "display_title": chunk.get("drawing_title") or chunk.get("display_title") or fn,
                "download_url": chunk.get("download_url"),
            })
        unique_s3_paths = list(dict.fromkeys(unique_s3_paths))
        source_documents = source_documents[:10]

    avg_score = 0
    if rag_context_chunks:
        sims = [c.get("similarity", 0) for c in rag_context_chunks]
        avg_score = sum(sims) / len(sims) if sims else 0

    processing_time_ms = int((time.time() - start_time) * 1000)

    # ── 10. Token tracking ───────────────────────────────────────────────────
    cost = round(
        (prompt_tokens / 1000) * _INPUT_COST_PER_1K
        + (completion_tokens / 1000) * _OUTPUT_COST_PER_1K, 6,
    )

    yield _sse_event({
        "type": "done",
        "session_id": current_session_id,
        "search_mode": search_mode,
        "confidence_score": confidence,
        "is_clarification": is_clarification,
        "follow_up_questions": follow_up_questions,
        "retrieval_count": len(rag_context_chunks),
        "average_score": round(avg_score, 4),
        "web_source_count": len(web_sources),
        "s3_paths": unique_s3_paths,
        "s3_path_count": len(unique_s3_paths),
        "source_documents": source_documents,
        "web_sources": web_sources,
        "processing_time_ms": processing_time_ms,
        "model_used": LLM_MODEL,
        "project_id": project_id,
        "token_tracking": {
            "embedding_tokens": estimate_tokens(user_query),
            "context_tokens": estimate_tokens(rag_context_text) + estimate_tokens(web_context_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_estimate_usd": cost,
        },
    })


# ── Helper: intent session management ─────────────────────────────────────────

def _handle_intent_session(user_query, friendly_response, session_id,
                           create_new_session, project_id, filter_source_type):
    current_session_id = session_id
    if MEMORY_MANAGER:
        if create_new_session or not session_id:
            current_session_id = MEMORY_MANAGER.create_session(
                user_query=user_query, project_id=project_id,
                filter_source_type=filter_source_type,
            )
        elif session_id:
            sess = MEMORY_MANAGER.get_session(session_id)
            if sess:
                MEMORY_MANAGER.add_to_session(session_id, "user", user_query)
            else:
                current_session_id = MEMORY_MANAGER.create_session(
                    user_query=user_query, project_id=project_id,
                    filter_source_type=filter_source_type, session_id=session_id,
                )
        if current_session_id:
            MEMORY_MANAGER.add_to_session(current_session_id, "assistant", friendly_response)
    return current_session_id
