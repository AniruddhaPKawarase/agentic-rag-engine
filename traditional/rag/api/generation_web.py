"""Web-search answer generation flow."""
from typing import Any, Dict, List, Optional

import time
import traceback

from . import state
from .helpers import build_context_text, format_filename_for_display
from .intent import detect_intent
from .prompts import build_hybrid_prompt, build_rag_prompt, build_web_prompt

client = state.client
LLM_MODEL = state.LLM_MODEL
WEB_SEARCH_MODEL = state.WEB_SEARCH_MODEL
WEB_SEARCH_AVAILABLE = state.WEB_SEARCH_AVAILABLE
MEMORY_MANAGER = state.MEMORY_MANAGER
web_search = state.web_search
retrieve_context = state.retrieve_context
estimate_tokens = getattr(state, "estimate_tokens", lambda text: len(text) // 4)

def generate_web_search_answer(
    user_query: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    session_id: Optional[str] = None,
    create_new_session: bool = False,
    use_conversation_history: bool = True
) -> Dict[str, Any]:
    """
    Generate an answer using web search with memory management.
    """
    start_time = time.time()
    
    print(f"\n🌐 Starting web search generation for query: '{user_query}'")
    print(f"   Session ID: {session_id}, New session: {create_new_session}")
    print(f"   Use conversation history: {use_conversation_history}")

    # ==============================
    # 0. Intent Detection (zero-cost, regex only)
    # ==============================
    intent_type, friendly_response = detect_intent(user_query)

    if intent_type in ("greeting", "small_talk", "thanks", "farewell"):
        processing_time_ms = int((time.time() - start_time) * 1000)
        print(f"🎯 Intent: {intent_type} -- returning friendly response ({processing_time_ms}ms)")

        current_session_id = session_id
        if MEMORY_MANAGER:
            if create_new_session or not session_id:
                current_session_id = MEMORY_MANAGER.create_session(user_query=user_query)
            elif session_id:
                session = MEMORY_MANAGER.get_session(session_id)
                if session:
                    MEMORY_MANAGER.add_to_session(session_id, "user", user_query)
                else:
                    current_session_id = MEMORY_MANAGER.create_session(
                        user_query=user_query, session_id=session_id
                    )
            if current_session_id:
                MEMORY_MANAGER.add_to_session(current_session_id, "assistant", friendly_response)

        return {
            "query": user_query,
            "answer": friendly_response,
            "sources": [],
            "source_count": 0,
            "model_used": "intent_detector",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "processing_time_ms": processing_time_ms,
            "session_id": current_session_id,
            "session_stats": MEMORY_MANAGER.get_session_stats(current_session_id) if MEMORY_MANAGER and current_session_id else None,
            "web_search_available": WEB_SEARCH_AVAILABLE,
        }

    # ==============================
    # 1. Memory Management
    # ==============================
    current_session_id = session_id
    session_stats = None
    
    if MEMORY_MANAGER:
        if create_new_session or not session_id:
            # Create new session
            current_session_id = MEMORY_MANAGER.create_session(
                user_query=user_query,
                project_id=None,  # Web search doesn't use project_id
                filter_source_type=None
            )
            print(f"✅ Created new session for web search: {current_session_id}")
        elif session_id:
            # Use existing session
            session = MEMORY_MANAGER.get_session(session_id)
            if session:
                # Store user query in session
                query_tokens = estimate_tokens(user_query)
                MEMORY_MANAGER.add_to_session(
                    session_id=session_id,
                    role="user",
                    content=user_query,
                    tokens=query_tokens,
                    metadata={
                        "query_type": "web_search"
                    }
                )
                
                session_stats = MEMORY_MANAGER.get_session_stats(session_id)
                print(f"✅ Using existing session: {session_id}")
            else:
                # Session not found, create new one
                current_session_id = MEMORY_MANAGER.create_session(
                    user_query=user_query,
                    project_id=None,
                    filter_source_type=None,
                    session_id=session_id
                )
                print(f"⚠️ Session not found, created new: {current_session_id}")
    
    # ==============================
    # 2. Get conversation history from memory manager
    # ==============================
    conversation_messages = []
    conversation_question_index = ""

    if use_conversation_history and MEMORY_MANAGER and current_session_id:
        session = MEMORY_MANAGER.get_session(current_session_id)
        if session:
            conversation_messages = session.get_conversation_for_llm(
                max_tokens=2000,
                preserve_early_history=True
            )
            conversation_question_index = session.get_conversation_index()
            print(f"📝 Retrieved {len(conversation_messages)} messages from conversation history")
            if conversation_question_index:
                print(f"📋 Built conversation index with {conversation_question_index.count(chr(10)) + 1} questions")

    # ==============================
    # 3. Build conversation context for system prompt
    # ==============================
    conversation_context = ""
    if conversation_messages and use_conversation_history:
        recent_exchanges = []
        for msg in conversation_messages[-4:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                recent_exchanges.append(f"User previously asked: {content[:150]}...")
            elif role == "assistant":
                recent_exchanges.append(f"You previously answered: {content[:150]}...")

        if recent_exchanges:
            conversation_context = "\n".join(recent_exchanges)

    # Append the full question index for meta-question support
    if conversation_question_index:
        conversation_context += f"\n\nCOMPLETE LIST OF USER QUESTIONS IN THIS SESSION:\n{conversation_question_index}"
    
    # ==============================
    # 4. Build system prompt with conversation context
    # ==============================
    if conversation_context and use_conversation_history:
        system_prompt = f"""You are a helpful AI assistant with web search capabilities.
You are having a conversation with a user.

RECENT CONVERSATION CONTEXT:
{conversation_context}

INSTRUCTIONS:
1. You have access to real-time web search results for the user's question.
2. Use the web search results to provide accurate, up-to-date information.
3. If this is a follow-up question, refer back to what was discussed in the recent conversation.
4. Cite your sources using the provided web search results.
5. Be conversational and helpful - remember this is a dialogue.

CURRENT QUESTION FROM USER:
{user_query}

Provide a helpful, accurate answer based on web search results:"""
    else:
        system_prompt = f"""You are a helpful AI assistant with web search capabilities.

INSTRUCTIONS:
1. You have access to real-time web search results for the user's question.
2. Use the web search results to provide accurate, up-to-date information.
3. Cite your sources using the provided web search results.

QUESTION:
{user_query}

Provide a helpful, accurate answer based on web search results:"""
    
    # ==============================
    # 5. Prepare messages for LLM (with web search tool)
    # ==============================
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (but exclude the current query if it's already in session)
    if conversation_messages and use_conversation_history:
        for msg in conversation_messages:
            # Skip system messages and any that might duplicate the current query
            if msg.get("role") != "system":
                messages.append(msg)
    
    print(f"🤖 Total messages for LLM (web search): {len(messages)}")
    
    # ==============================
    # 6. Call web search API with conversation context
    # ==============================
    web_search_result = {}
    sources = []
    answer = ""
    token_usage = None
    
    try:
        # Prepare the full context for web search
        search_context = user_query
        if conversation_context and use_conversation_history:
            search_context = f"{conversation_context}\n\nCurrent question: {user_query}"
        
        print(f"🌐 Performing web search for: '{search_context[:100]}...'")
        
        # Call the web search function
        web_search_result = web_search(search_context)
        
        answer = web_search_result.get("answer", "")
        sources = web_search_result.get("sources", [])
        
        # If web search returns an answer, we'll use it directly
        # Otherwise, we can fall back to generating with the LLM
        if not answer or answer.strip() == "":
            print("⚠️ Web search returned empty answer, falling back to LLM generation")
            
            # Prepare messages for LLM with web search results
            if sources:
                sources_text = "\n\nWEB SEARCH RESULTS:\n"
                for i, source in enumerate(sources):
                    sources_text += f"[{i+1}] {source.get('title', 'No title')}\n"
                    sources_text += f"    URL: {source.get('url', 'No URL')}\n\n"
                
                # Update system prompt to include web search results
                messages[-1]["content"] += f"\n\n{sources_text}"
            
            # Generate response with LLM
            response = client.responses.create(
                model=WEB_SEARCH_MODEL,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            answer = response.output_text.strip()
            
            # Extract token usage if available
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                }
        else:
            print(f"✅ Web search returned answer: {len(answer)} characters")
            print(f"   Sources found: {len(sources)}")
            
            # For token estimation (since we don't have exact token count from web search)
            answer_tokens = estimate_tokens(answer) if 'estimate_tokens' in globals() else len(answer.split())
            token_usage = {
                "prompt_tokens": estimate_tokens(search_context) if 'estimate_tokens' in globals() else len(search_context.split()),
                "completion_tokens": answer_tokens,
                "total_tokens": (estimate_tokens(search_context) + answer_tokens) if 'estimate_tokens' in globals() else len(search_context.split()) + answer_tokens
            }
            
    except Exception as e:
        answer = f"Error performing web search: {str(e)}"
        print(f"❌ Web search error: {e}")
        traceback.print_exc()
    
    # ==============================
    # 7. Save assistant response to memory
    # ==============================
    if MEMORY_MANAGER and current_session_id:
        answer_tokens = estimate_tokens(answer) if 'estimate_tokens' in globals() else len(answer.split())
        MEMORY_MANAGER.add_to_session(
            session_id=current_session_id,
            role="assistant",
            content=answer,
            tokens=answer_tokens,
            metadata={
                "token_usage": token_usage,
                "query_type": "web_search",
                "sources_count": len(sources),
                "web_search_used": True
            }
        )
        
        # Update session stats
        session_stats = MEMORY_MANAGER.get_session_stats(current_session_id)
    
    # ==============================
    # 8. Calculate processing time
    # ==============================
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    # ==============================
    # 9. Prepare final response
    # ==============================
    result = {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "source_count": len(sources),
        "model_used": WEB_SEARCH_MODEL,
        "token_usage": token_usage,
        "processing_time_ms": processing_time_ms,
        # Session information
        "session_id": current_session_id,
        "session_stats": session_stats,
        "web_search_available": WEB_SEARCH_AVAILABLE
    }
    
    print(f"✅ Web search generation completed in {processing_time_ms}ms")
    return result
