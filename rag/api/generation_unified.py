"""Unified RAG/Web/Hybrid answer generation flow — v2.

Changes from v1:
  - Follow-up questions (min 3) parsed from LLM output
  - Hallucination rollback: low-confidence retrieval → clarification prompt
  - Granular token tracking at every pipeline stage
  - Mode-aware conversation context (fixes cache/mode bug)
  - Context window budgeting for scalability (100k+ records)
  - Trade-based pre-filtering for relevance
  - Confidence scoring
"""
from typing import Any, Dict, List, Optional

import time
import traceback

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
from .greeting_agent import build_greeting_response, classify_query
from .intent import detect_intent, extract_drawing_reference, extract_document_reference
from .prompts import (
    build_clarification_prompt,
    build_hybrid_prompt,
    build_rag_prompt,
    build_web_prompt,
)

client = state.client
LLM_MODEL = state.LLM_MODEL
WEB_SEARCH_MODEL = state.WEB_SEARCH_MODEL
WEB_SEARCH_AVAILABLE = state.WEB_SEARCH_AVAILABLE
MEMORY_MANAGER = state.MEMORY_MANAGER
web_search = state.web_search
retrieve_context = state.retrieve_context

# ── Confidence threshold for hallucination rollback ───────────────────────────
CONFIDENCE_THRESHOLD = 0.30  # below this → ask clarification instead of answering

# ── Cost constants (GPT-4o pricing as of 2025-12) ────────────────────────────
_INPUT_COST_PER_1K = 0.0025   # $2.50 / 1M input tokens
_OUTPUT_COST_PER_1K = 0.01    # $10 / 1M output tokens


def _quick_rag_answer(
    user_query: str,
    rag_context_text: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate a RAG-only answer for the rag_answer field in hybrid mode.

    Uses a focused prompt that extracts whatever IS available in the docs.
    Only returns "not found" when context is genuinely empty/irrelevant.
    """
    _NO_CONTEXT_SENTINEL = "No relevant technical documentation found for this query."
    if not rag_context_text.strip() or rag_context_text.strip() == _NO_CONTEXT_SENTINEL:
        return "The requested information was not found in the project documents."

    prompt = (
        "You are a construction document reviewer. "
        "Your task is to extract and summarise whatever the project documents below "
        "say about the user's question — even if the documents only partially cover it.\n\n"
        "RULES:\n"
        "1. Use ONLY information that appears explicitly in the PROJECT DOCUMENTS below.\n"
        "2. If the documents contain relevant information, provide a clear, concise answer.\n"
        "3. If a part of the question is not covered by the documents, skip that part "
        "silently — do NOT say it is missing.\n"
        "4. Only if the documents contain absolutely NO relevant information at all, "
        "respond with: 'The requested information was not found in the project documents.'\n"
        "5. Keep the answer concise (3-6 sentences max). No follow-up questions.\n\n"
        f"PROJECT DOCUMENTS:\n{rag_context_text}\n\n"
        f"QUESTION: {user_query}\n\n"
        "Answer (from project documents only):"
    )
    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_output_tokens=min(max_tokens, 400),
        )
        raw = resp.output_text.strip()
        # Strip any accidental follow-up section
        return raw.split("---FOLLOW_UP---")[0].strip()
    except Exception:
        return "Unable to generate answer from project documents."


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens / 1000) * _INPUT_COST_PER_1K
        + (completion_tokens / 1000) * _OUTPUT_COST_PER_1K,
        6,
    )


def generate_unified_answer(
    user_query: str,
    search_mode: str = "rag",
    top_k: int = 5,
    min_score: float = 0.1,
    temperature: float = 0.3,
    max_tokens: int = 500,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    include_citations: bool = False,
    include_s3_paths: bool = True,
    filter_source_type: Optional[str] = None,
    project_id: Optional[int] = None,
    debug: bool = False,
    session_id: Optional[str] = None,
    create_new_session: bool = False,
    pin_documents: Optional[List[str]] = None,
    unpin: bool = False,
) -> Dict[str, Any]:
    """Unified generation with follow-ups, hallucination guard, and token tracking."""
    start_time = time.time()

    print(f"\n🎯 [v2] Starting unified generation for: '{user_query}'")
    print(f"   Mode: {search_mode} | Session: {session_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # 0. Intent Detection (zero-cost, regex only)
    # ══════════════════════════════════════════════════════════════════════════
    intent_type, friendly_response = detect_intent(user_query)

    if intent_type in ("greeting", "small_talk", "thanks", "farewell"):
        processing_time_ms = int((time.time() - start_time) * 1000)
        current_session_id = _manage_session_for_intent(
            user_query, friendly_response, session_id, create_new_session,
            project_id, filter_source_type,
        )
        return _intent_response(
            user_query, friendly_response, current_session_id,
            project_id, processing_time_ms,
        )

    # ── LLM-based greeting agent: catches sarcasm, off-topic, out-of-context ──
    # Only runs for queries that passed the regex check (not obvious greetings)
    # and are NOT meta_conversation (those need the full pipeline).
    if intent_type == "document_query":
        is_doc_query, greeting_resp = classify_query(user_query)
        if not is_doc_query:
            processing_time_ms = int((time.time() - start_time) * 1000)
            current_session_id = _manage_session_for_intent(
                user_query, greeting_resp, session_id, create_new_session,
                project_id, filter_source_type,
            )
            if MEMORY_MANAGER and current_session_id:
                MEMORY_MANAGER.add_to_session(
                    current_session_id, "assistant", greeting_resp,
                    tokens=estimate_tokens(greeting_resp),
                )
            return build_greeting_response(
                user_query, greeting_resp, current_session_id,
                project_id, processing_time_ms,
            )

    if intent_type == "meta_conversation":
        print("   📋 Meta-conversation question — will use conversation index")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Session Management
    # ══════════════════════════════════════════════════════════════════════════
    current_session_id = session_id
    session_stats = None

    if MEMORY_MANAGER:
        if create_new_session or not session_id:
            current_session_id = MEMORY_MANAGER.create_session(
                user_query=user_query,
                project_id=project_id,
                filter_source_type=filter_source_type,
            )
            # CRITICAL: Also add the first user message to the session
            # (create_session only stores it in metadata, not as a Message)
            MEMORY_MANAGER.add_to_session(
                session_id=current_session_id, role="user", content=user_query,
                tokens=estimate_tokens(user_query),
                metadata={"search_mode": search_mode, "project_id": project_id},
            )
        elif session_id:
            session = MEMORY_MANAGER.get_session(session_id)
            if session:
                MEMORY_MANAGER.add_to_session(
                    session_id=session_id, role="user", content=user_query,
                    tokens=estimate_tokens(user_query),
                    metadata={"search_mode": search_mode, "project_id": project_id},
                )
                session_stats = MEMORY_MANAGER.get_session_stats(session_id)
            else:
                current_session_id = MEMORY_MANAGER.create_session(
                    user_query=user_query, project_id=project_id,
                    filter_source_type=filter_source_type, session_id=session_id,
                )
                MEMORY_MANAGER.add_to_session(
                    session_id=current_session_id, role="user", content=user_query,
                    tokens=estimate_tokens(user_query),
                    metadata={"search_mode": search_mode, "project_id": project_id},
                )

    # ══════════════════════════════════════════════════════════════════════════
    # 1b. Document Pin State Management
    # ══════════════════════════════════════════════════════════════════════════
    active_pin_documents: Optional[List[str]] = None
    active_pin_titles: List[str] = []
    auto_unpinned = False

    if MEMORY_MANAGER and current_session_id:
        session = MEMORY_MANAGER.get_session(current_session_id)
        if session:
            ctx = session.context

            # Handle explicit unpin request
            if unpin or intent_type == "unpin_document":
                ctx.pinned_documents = []
                ctx.pinned_titles = []
                print("   📌 Documents unpinned — returning to full project scope")

            # Handle explicit pin request from API
            elif pin_documents:
                ctx.pinned_documents = pin_documents
                # Try to resolve titles from last source documents
                titles = []
                for pn in pin_documents:
                    matched_title = pn  # fallback
                    for doc in (ctx.last_source_documents or []):
                        if (doc.get("file_name", "").lower().startswith(pn.lower()) or
                                doc.get("pdf_name", "").lower() == pn.lower()):
                            matched_title = doc.get("display_title") or doc.get("file_name") or pn
                            break
                    titles.append(matched_title)
                ctx.pinned_titles = titles
                print(f"   📌 Documents pinned via API: {pin_documents}")

            # Handle natural language pin intent
            elif intent_type == "document_chat":
                doc_ref = extract_document_reference(user_query)
                if doc_ref and ctx.last_source_documents:
                    matched_pdfs = []
                    matched_titles = []
                    ref_lower = doc_ref.lower()
                    for doc in ctx.last_source_documents:
                        pdf = doc.get("file_name", doc.get("pdf_name", ""))
                        title = doc.get("display_title", "")
                        drawing = doc.get("drawing_name", "")
                        if (ref_lower in pdf.lower() or ref_lower in title.lower() or
                                ref_lower in drawing.lower() or
                                pdf.lower() in ref_lower or drawing.lower() == ref_lower):
                            matched_pdfs.append(pdf)
                            matched_titles.append(title or pdf)
                    if matched_pdfs:
                        ctx.pinned_documents = matched_pdfs
                        ctx.pinned_titles = matched_titles
                        print(f"   📌 Documents pinned via NL: {matched_pdfs}")
                    else:
                        print(f"   📌 Could not match '{doc_ref}' to source documents")

            # Read current pin state
            if ctx.pinned_documents:
                active_pin_documents = ctx.pinned_documents
                active_pin_titles = ctx.pinned_titles or []

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Retrieval (with trade filtering + context budgeting)
    # ══════════════════════════════════════════════════════════════════════════
    rag_context_chunks: List[Dict] = []
    rag_context_text = ""
    web_search_result = None
    web_context_text = ""
    web_sources: List[Dict] = []
    web_source_count = 0

    detected_trade = detect_trade_from_query(user_query)
    if detected_trade:
        print(f"   🏗️ Detected trade: {detected_trade}")

    # ── Query augmentation for follow-up and reference intents ────────────
    retrieval_query = user_query
    drawing_name_filter = None
    drawing_title_filter = None

    if MEMORY_MANAGER and current_session_id and intent_type in ("follow_up", "reference_previous"):
        session = MEMORY_MANAGER.get_session(current_session_id)
        if session:
            prev_q = session.get_last_user_message()
            prev_a = session.get_last_assistant_message()
            if intent_type == "follow_up" and prev_q:
                # Augment: "tell me more" → "tell me more about [previous topic]"
                retrieval_query = f"{user_query}. Context: Previously asked '{prev_q}'"
                if prev_a:
                    retrieval_query += f" and received: '{prev_a[:500]}'"
                print(f"   🔄 Follow-up augmented query (len={len(retrieval_query)})")
            elif intent_type == "reference_previous" and prev_a:
                # Augment: "what size" → include previous answer for reference
                retrieval_query = f"{user_query}. Reference from previous answer: '{prev_a[:500]}'"
                print(f"   🔗 Reference augmented query (len={len(retrieval_query)})")

    # ── Drawing name/title detection for filtered retrieval ───────────────
    d_name, d_title = extract_drawing_reference(user_query)
    if d_name:
        drawing_name_filter = d_name
        print(f"   📐 Drawing name detected: {d_name}")
    if d_title:
        drawing_title_filter = d_title
        print(f"   📐 Drawing title detected: {d_title}")

    def _do_rag(pdf_filter=None):
        chunks = retrieve_context(
            query=retrieval_query,
            top_k=top_k + 4,  # overfetch to allow filtering
            min_score=min_score,
            filter_source_type=filter_source_type,
            filter_project_id=project_id,
            filter_drawing_name=drawing_name_filter,
            filter_drawing_title=drawing_title_filter,
            filter_pdf_names=pdf_filter,
        )
        valid = [c for c in chunks if c.get("text", "").strip() and len(c.get("text", "").strip()) > 10]
        # Trade filter
        if detected_trade:
            valid = filter_chunks_by_trade(valid, detected_trade)
        # Budget context window for scalability
        return budget_context_window(valid, max_context_tokens=4000, max_chunks=top_k)

    def _do_web():
        return web_search(user_query)

    def _process_web(ws_result):
        nonlocal web_context_text, web_sources, web_source_count
        web_answer = ws_result.get("answer", "")
        web_sources = ws_result.get("sources", [])
        web_source_count = len(web_sources)
        if web_sources:
            web_context_text = "WEB SEARCH RESULTS:\n\n"
            web_context_text += f"Answer from web search: {web_answer}\n\n"
            web_context_text += "Sources:\n"
            for i, src in enumerate(web_sources[:5]):
                web_context_text += f"[{i+1}] {src.get('title', 'No title')}\n    URL: {src.get('url', 'No URL')}\n\n"

    # Run retrieval based on mode
    if search_mode == "hybrid" and WEB_SEARCH_AVAILABLE:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            rf = pool.submit(_do_rag)
            wf = pool.submit(_do_web)
            try:
                rag_context_chunks = rf.result(timeout=30)
                rag_context_text = build_context_text(rag_context_chunks, include_citations)
            except Exception as e:
                print(f"   ❌ RAG error: {e}")
            try:
                web_search_result = wf.result(timeout=60)
                _process_web(web_search_result)
            except Exception as e:
                print(f"   ❌ Web error: {e}")

    elif search_mode in ("rag", "hybrid"):
        try:
            rag_context_chunks = _do_rag(pdf_filter=active_pin_documents)

            # Auto-unpin: if pinned docs returned 0 results, retry with full project
            if active_pin_documents and len(rag_context_chunks) == 0:
                print(f"   📌 Auto-unpin: pinned docs returned 0 results, falling back to full project")
                rag_context_chunks = _do_rag(pdf_filter=None)
                auto_unpinned = True
                active_pin_documents = None
                active_pin_titles = []
                if MEMORY_MANAGER and current_session_id:
                    session = MEMORY_MANAGER.get_session(current_session_id)
                    if session:
                        session.context.pinned_documents = []
                        session.context.pinned_titles = []

            rag_context_text = build_context_text(rag_context_chunks, include_citations)
            pin_label = f" [PINNED: {active_pin_documents}]" if active_pin_documents else ""
            print(f"   📄 RAG retrieval: {len(rag_context_chunks)} chunks, context_len={len(rag_context_text)}{pin_label}")
        except Exception as e:
            print(f"   ❌ RAG error (retrieval failed, context will be empty): {e}")
            traceback.print_exc()

    elif search_mode == "web" and WEB_SEARCH_AVAILABLE:
        try:
            web_search_result = _do_web()
            _process_web(web_search_result)
        except Exception as e:
            print(f"   ❌ Web error: {e}")
            traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Confidence check → Hallucination rollback
    # ══════════════════════════════════════════════════════════════════════════
    confidence = compute_confidence_score(rag_context_chunks, search_mode)
    is_clarification = False

    if search_mode in ("rag", "hybrid") and confidence < CONFIDENCE_THRESHOLD and rag_context_chunks:
        print(f"   ⚠️ Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) → clarification mode")
        is_clarification = True

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Conversation context (MODE-AWARE — fixes the cache/mode bug)
    # ══════════════════════════════════════════════════════════════════════════
    conversation_messages: List[Dict] = []
    conversation_context = ""

    if MEMORY_MANAGER and current_session_id:
        session = MEMORY_MANAGER.get_session(current_session_id)
        if session:
            conversation_messages = session.get_conversation_for_llm(
                max_tokens=3000, preserve_early_history=True,
            )
            q_index = session.get_conversation_index()

            # ── Enhanced conversation context (v3) ────────────────────────
            # Last Q+A pair: FULL content (no truncation) for continuity
            # Older messages: 300 chars each for token efficiency
            context_parts = []

            # Get the last user question (before current) and last assistant answer
            last_user_q = session.get_last_user_message()
            last_assistant_a = session.get_last_assistant_message()

            if last_user_q:
                context_parts.append(f"[LAST QUESTION]: {last_user_q}")
            if last_assistant_a:
                context_parts.append(f"[LAST ANSWER]: {last_assistant_a}")

            # Older messages (beyond the last pair): 300 chars each
            older_msgs = conversation_messages[:-4] if len(conversation_messages) > 4 else []
            for msg in older_msgs[-6:]:  # max 6 older messages
                role = msg.get("role", "")
                content = msg.get("content", "")
                msg_mode = msg.get("metadata", {}).get("search_mode", "") if isinstance(msg.get("metadata"), dict) else ""
                if role == "user":
                    context_parts.append(f"User previously asked: {content[:300]}")
                elif role == "assistant":
                    if msg_mode and msg_mode != search_mode:
                        context_parts.append(f"[Previous answer was from {msg_mode.upper()} mode — user has now switched to {search_mode.upper()} mode]")
                    else:
                        context_parts.append(f"You previously answered: {content[:300]}")

            if context_parts:
                conversation_context = "\n".join(context_parts)
            if q_index:
                conversation_context += f"\n\nCOMPLETE LIST OF USER QUESTIONS IN THIS SESSION:\n{q_index}"

            # Trim conversation context to 3000 token budget
            while estimate_tokens(conversation_context) > 3000 and context_parts:
                # Remove oldest entries first (keep last Q+A pair)
                if len(context_parts) > 2:
                    context_parts.pop(2)  # Remove first older message
                    conversation_context = "\n".join(context_parts)
                    if q_index:
                        conversation_context += f"\n\nCOMPLETE LIST OF USER QUESTIONS IN THIS SESSION:\n{q_index}"
                else:
                    break
        elif conversation_history:
            conversation_messages = conversation_history

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Build prompt
    # ══════════════════════════════════════════════════════════════════════════
    context_tokens = estimate_tokens(rag_context_text) + estimate_tokens(web_context_text)

    if is_clarification:
        # Hallucination rollback prompt
        summary = f"Retrieved {len(rag_context_chunks)} chunks, avg similarity: {confidence:.2f}"
        system_prompt = build_clarification_prompt(user_query, summary, conversation_context)
    elif search_mode == "rag":
        system_prompt = build_rag_prompt(user_query, rag_context_text, conversation_context, include_citations)
    elif search_mode == "web":
        system_prompt = build_web_prompt(user_query, web_context_text, conversation_context, include_citations)
    else:
        system_prompt = build_hybrid_prompt(user_query, rag_context_text, web_context_text, conversation_context, include_citations)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Generate response
    # ══════════════════════════════════════════════════════════════════════════
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_messages:
        for msg in conversation_messages:
            if msg.get("role") != "system":
                messages.append({"role": msg["role"], "content": msg.get("content", "")})

    token_usage = None
    follow_up_questions: List[str] = []
    answer = ""

    try:
        response = client.responses.create(
            model=LLM_MODEL,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        raw_output = response.output_text.strip()
        answer, follow_up_questions = parse_follow_up_questions(raw_output)

        # Strip ALL citation markers when citations are disabled
        if not include_citations:
            import re as _re
            # (Excerpt [1]), (Excerpt 1), [Excerpt 1], (excerpt [2]), etc.
            answer = _re.sub(r'\s*[\(\[]\s*[Ee]xcerpt\s*\[?\d+\]?\s*[\)\]]', '', answer)
            # Bare trailing [1], [2], ... at end of a sentence/word
            answer = _re.sub(r'\s*\[\d+\]', '', answer)
            answer = answer.strip()

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0

        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        print(f"   ✅ LLM response: {len(answer)} chars, {len(follow_up_questions)} follow-ups")

    except Exception as e:
        answer = f"Error generating response: {str(e)}"
        print(f"   ❌ LLM error: {e}")
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Token tracking
    # ══════════════════════════════════════════════════════════════════════════
    embedding_tokens = estimate_tokens(user_query)  # embedding call
    p_tok = token_usage["prompt_tokens"] if token_usage else 0
    c_tok = token_usage["completion_tokens"] if token_usage else 0

    session_total = 0
    if MEMORY_MANAGER and current_session_id:
        stats = MEMORY_MANAGER.get_session_stats(current_session_id)
        session_total = stats.get("total_tokens", 0) if isinstance(stats, dict) else 0

    token_tracking = {
        "embedding_tokens": embedding_tokens,
        "context_tokens": context_tokens,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": p_tok + c_tok,
        "session_total_tokens": session_total + p_tok + c_tok,
        "cost_estimate_usd": _estimate_cost(p_tok, c_tok),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 8. Save assistant response to memory
    # ══════════════════════════════════════════════════════════════════════════
    if MEMORY_MANAGER and current_session_id and answer:
        MEMORY_MANAGER.add_to_session(
            session_id=current_session_id, role="assistant", content=answer,
            tokens=estimate_tokens(answer),
            metadata={
                "token_usage": token_usage,
                "search_mode": search_mode,
                "confidence": confidence,
                "is_clarification": is_clarification,
                "retrieval_count": len(rag_context_chunks),
                "web_source_count": web_source_count,
            },
        )
        session_stats = MEMORY_MANAGER.get_session_stats(current_session_id)

    # ══════════════════════════════════════════════════════════════════════════
    # 9. Build source documents
    # ══════════════════════════════════════════════════════════════════════════
    unique_s3_paths: List[str] = []
    source_documents: List[Dict] = []

    if rag_context_chunks and include_s3_paths:
        seen: set = set()
        for chunk in rag_context_chunks:
            s3_path = chunk.get("s3_path")
            if not s3_path:
                continue
            pdf_name = chunk.get("pdf_name", "")
            formatted_name = format_filename_for_display(pdf_name) if pdf_name else "Unknown"
            doc_key = f"{s3_path}::{formatted_name}"
            if doc_key in seen:
                continue
            seen.add(doc_key)
            display_title = chunk.get("drawing_title") or chunk.get("display_title") or formatted_name
            unique_s3_paths.append(s3_path)
            source_documents.append({
                "s3_path": s3_path,
                "file_name": formatted_name,
                "display_title": display_title,
                "download_url": chunk.get("download_url"),
            })
        unique_s3_paths = list(dict.fromkeys(unique_s3_paths))
        source_documents = source_documents[:10]

    # ══════════════════════════════════════════════════════════════════════════
    # 10. Build retrieved chunks detail
    # ══════════════════════════════════════════════════════════════════════════
    retrieved_chunks = []
    for chunk in rag_context_chunks:
        retrieved_chunks.append({
            "index": chunk.get("index", 0),
            "text": chunk.get("text", ""),
            "source_type": chunk.get("source_type", "unknown"),
            "similarity": chunk.get("similarity", 0),
            "distance": chunk.get("distance", 0),
            "token_count": estimate_tokens(chunk.get("text", "")),
            "drawing_id": str(chunk["drawing_id"]) if chunk.get("drawing_id") is not None else None,
            "pdf_name": chunk.get("pdf_name"),
            "s3_path": chunk.get("s3_path"),
            "page": chunk.get("page"),
            "set_id": str(chunk["set_id"]) if chunk.get("set_id") is not None else None,
            "trade_id": str(chunk["trade_id"]) if chunk.get("trade_id") is not None else None,
            "drawing_name": chunk.get("drawing_name"),
            "drawing_title": chunk.get("drawing_title"),
            "display_title": chunk.get("display_title"),
            "download_url": chunk.get("download_url"),
            "trade_name": chunk.get("trade_name"),
            "material": chunk.get("material"),
            "quantity": chunk.get("quantity"),
            "unit": chunk.get("unit"),
            "created_at": chunk.get("created_at"),
        })

    # ══════════════════════════════════════════════════════════════════════════
    # 10b. Separate RAG / Web answers for UI display
    # ══════════════════════════════════════════════════════════════════════════
    rag_answer: Optional[str] = None
    web_answer: Optional[str] = None

    if search_mode == "rag":
        # The main answer IS the RAG answer; no web answer
        rag_answer = answer
        web_answer = None

    elif search_mode == "web":
        # The main answer IS the web answer; no RAG answer
        rag_answer = None
        web_answer = answer

    elif search_mode == "hybrid":
        # rag_answer: quick RAG-only LLM call (tells UI what the docs say)
        # web_answer: raw answer returned directly by OpenAI web search tool
        #             (already available — no extra API call needed)
        rag_answer = _quick_rag_answer(user_query, rag_context_text, temperature, max_tokens)
        web_answer = (
            web_search_result.get("answer", "").strip()
            if web_search_result and web_search_result.get("answer", "").strip()
            else "No web search results were available for this query."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 11. Final response
    # ══════════════════════════════════════════════════════════════════════════
    processing_time_ms = int((time.time() - start_time) * 1000)
    avg_score = 0.0
    if rag_context_chunks:
        sims = [c.get("similarity", 0) for c in rag_context_chunks]
        avg_score = sum(sims) / len(sims) if sims else 0

    # In hybrid mode, rag_answer + web_answer replace the combined answer field.
    # Keep answer for rag/web modes; set None for hybrid so the UI uses the
    # separate fields exclusively.
    response_answer = None if search_mode == "hybrid" else answer

    final_result = {
        "query": user_query,
        "answer": response_answer,
        "rag_answer": rag_answer,
        "web_answer": web_answer,
        "retrieval_count": len(rag_context_chunks),
        "average_score": avg_score,
        "confidence_score": confidence,
        "is_clarification": is_clarification,
        "follow_up_questions": follow_up_questions,
        "model_used": LLM_MODEL,
        "token_usage": token_usage,
        "token_tracking": token_tracking,
        "s3_paths": unique_s3_paths,
        "s3_path_count": len(unique_s3_paths),
        "source_documents": source_documents,
        "retrieved_chunks": retrieved_chunks,
        "processing_time_ms": processing_time_ms,
        "project_id": project_id,
        "session_id": current_session_id,
        "session_stats": session_stats,
        "search_mode": search_mode,
        "web_sources": web_sources,
        "web_source_count": web_source_count,
        "pin_status": {
            "active": bool(active_pin_documents),
            "pinned_documents": active_pin_documents or [],
            "pinned_titles": active_pin_titles,
            "auto_unpinned": auto_unpinned,
        },
    }

    # Save source documents to session context for future document pinning
    if MEMORY_MANAGER and current_session_id and source_documents:
        session = MEMORY_MANAGER.get_session(current_session_id)
        if session:
            session.context.last_source_documents = source_documents

    if debug:
        final_result["debug_info"] = {
            "context_preview": rag_context_text[:500] + "..." if len(rag_context_text) > 500 else rag_context_text,
            "web_context_preview": web_context_text[:500] + "..." if len(web_context_text) > 500 else web_context_text,
            "has_rag_context": len(rag_context_chunks) > 0,
            "has_web_context": web_context_text != "",
            "detected_trade": detected_trade,
            "confidence": confidence,
            "is_clarification": is_clarification,
            "system_prompt_preview": system_prompt[:300] + "...",
            "conversation_messages_count": len(conversation_messages),
        }
    else:
        final_result["debug_info"] = None

    print(f"   ✅ Done in {processing_time_ms}ms | confidence={confidence:.3f} | follow_ups={len(follow_up_questions)}")
    return final_result


# ── Helper: session management for intent responses ──────────────────────────

def _manage_session_for_intent(
    user_query, friendly_response, session_id, create_new_session,
    project_id, filter_source_type,
):
    current_session_id = session_id
    if MEMORY_MANAGER:
        if create_new_session or not session_id:
            current_session_id = MEMORY_MANAGER.create_session(
                user_query=user_query, project_id=project_id,
                filter_source_type=filter_source_type,
            )
        elif session_id:
            session = MEMORY_MANAGER.get_session(session_id)
            if session:
                MEMORY_MANAGER.add_to_session(session_id, "user", user_query, tokens=estimate_tokens(user_query))
            else:
                current_session_id = MEMORY_MANAGER.create_session(
                    user_query=user_query, project_id=project_id,
                    filter_source_type=filter_source_type, session_id=session_id,
                )
        if current_session_id:
            MEMORY_MANAGER.add_to_session(
                current_session_id, "assistant", friendly_response,
                tokens=estimate_tokens(friendly_response),
            )
    return current_session_id


def _intent_response(user_query, friendly_response, session_id, project_id, processing_time_ms):
    return {
        "query": user_query,
        "answer": friendly_response,
        "rag_answer": None,
        "web_answer": None,
        "retrieval_count": 0,
        "average_score": 0,
        "confidence_score": 1.0,
        "is_clarification": False,
        "follow_up_questions": [],
        "model_used": "intent_detector",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "token_tracking": {
            "embedding_tokens": 0, "context_tokens": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "session_total_tokens": 0,
            "cost_estimate_usd": 0.0,
        },
        "s3_paths": [],
        "s3_path_count": 0,
        "source_documents": [],
        "retrieved_chunks": [],
        "processing_time_ms": processing_time_ms,
        "project_id": project_id,
        "session_id": session_id,
        "session_stats": MEMORY_MANAGER.get_session_stats(session_id) if MEMORY_MANAGER and session_id else None,
        "search_mode": "intent",
        "web_sources": [],
        "web_source_count": 0,
        "debug_info": None,
    }
