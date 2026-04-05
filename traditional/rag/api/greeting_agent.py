"""Greeting Agent — LLM-based classifier for non-document queries.

Handles greetings, sarcasm, jokes, off-topic questions, and any query
that is NOT about construction documents. Uses a lightweight LLM call
(gpt-4o-mini) for classification when the regex intent detector misses.

Pipeline:
  1. Regex intent detector (0 ms, free) — catches obvious greetings
  2. THIS agent (LLM call) — catches everything else that isn't a document query
  3. RAG pipeline — only runs for genuine document queries

This agent NEVER references S3 documents, FAISS indices, or project data.
"""
import os
import time
from typing import Dict, Optional, Tuple

from . import state

client = state.client
GREETING_MODEL = os.getenv("GREETING_MODEL", "gpt-4o-mini")

# ── Classification prompt ─────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """You are a query classifier for a Construction Documentation Q&A system.

Your job: Determine if the user's message is a DOCUMENT QUERY or a NON-DOCUMENT message.

DOCUMENT QUERY — the user wants information from construction project documents, drawings,
specifications, building codes, materials, trades, scheduling, or any construction-related
technical question. Examples:
  - "What are the plumbing specs for floor 3?"
  - "Show me the electrical panel details"
  - "What materials are used for the HVAC ductwork?"
  - "Tell me about the fire safety requirements"
  - "What does drawing E-2 say about conduit sizes?"

NON-DOCUMENT — anything else: greetings, small talk, jokes, sarcasm, personal questions,
off-topic questions, philosophical questions, math problems, coding help, general knowledge,
weather, sports, food, entertainment, etc. Examples:
  - "You're useless" / "This is garbage"
  - "Tell me a joke" / "What's 2+2?"
  - "What's the weather like?" / "Who won the World Cup?"
  - "Can you write Python code?" / "What is AI?"
  - "I'm bored" / "Do you have feelings?"
  - "Hmm" / "ok" / "lol" / "interesting"
  - "What's the meaning of life?"

Respond with EXACTLY one word: DOCUMENT or NON_DOCUMENT"""

_RESPONSE_PROMPT = """You are a friendly Construction Documentation Assistant.
The user said something that is NOT a document query (greeting, joke, off-topic, sarcasm, etc.).

Respond naturally and briefly (1-3 sentences). Be warm and helpful.
- For greetings: greet back and remind them you can help with construction docs.
- For sarcasm/negativity: stay positive, acknowledge, redirect to how you can help.
- For off-topic questions: briefly answer if trivial, then redirect to construction docs.
- For "ok"/"hmm"/filler: acknowledge and ask if they have a construction question.

NEVER reference S3 documents, FAISS indices, project data, or any technical system internals.
NEVER make up construction information.
Keep it conversational and SHORT."""


def classify_query(user_query: str) -> Tuple[bool, Optional[str]]:
    """Classify whether a query is a document query or non-document.

    Args:
        user_query: The user's raw input.

    Returns:
        Tuple of (is_document_query, friendly_response).
        - is_document_query: True if the query should go to RAG pipeline.
        - friendly_response: None if document query, else the greeting agent's response.
    """
    start = time.time()

    # Step 1: Classify
    try:
        classification = client.responses.create(
            model=GREETING_MODEL,
            input=[
                {"role": "system", "content": _CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_output_tokens=16,
        )
        label = classification.output_text.strip().upper()
    except Exception as e:
        print(f"   ⚠️ Greeting agent classification error: {e}")
        # On error, assume document query (safe fallback — let RAG handle it)
        return (True, None)

    if "DOCUMENT" in label and "NON" not in label:
        elapsed = int((time.time() - start) * 1000)
        print(f"   ✅ Greeting agent: DOCUMENT query ({elapsed}ms)")
        return (True, None)

    # Step 2: Generate friendly response for non-document queries
    try:
        response = client.responses.create(
            model=GREETING_MODEL,
            input=[
                {"role": "system", "content": _RESPONSE_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.7,
            max_output_tokens=150,
        )
        friendly = response.output_text.strip()
    except Exception as e:
        print(f"   ⚠️ Greeting agent response error: {e}")
        friendly = (
            "I'm your Construction Documentation Assistant. "
            "I'm best at answering questions about project drawings, specifications, "
            "and construction documents. How can I help you with that?"
        )

    elapsed = int((time.time() - start) * 1000)
    print(f"   ✅ Greeting agent: NON_DOCUMENT ({elapsed}ms) → {friendly[:80]}...")
    return (False, friendly)


def build_greeting_response(
    user_query: str,
    friendly_response: str,
    session_id: Optional[str],
    project_id: Optional[int],
    processing_time_ms: int,
) -> Dict:
    """Build a complete API response dict for a non-document query.

    Returns the same shape as generation_unified._intent_response() so the
    caller can return it directly without any special handling.
    """
    MEMORY_MANAGER = state.MEMORY_MANAGER
    session_stats = None
    if MEMORY_MANAGER and session_id:
        session_stats = MEMORY_MANAGER.get_session_stats(session_id)

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
        "model_used": GREETING_MODEL,
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
        "session_stats": session_stats,
        "search_mode": "greeting",
        "web_sources": [],
        "web_source_count": 0,
        "debug_info": None,
    }
