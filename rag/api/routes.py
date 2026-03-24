"""API routes, exception handlers, and startup hooks — v2."""
import datetime
import traceback
from typing import Optional

from fastapi import Body, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from .generation import generate_unified_answer, generate_web_search_answer
from .streaming import stream_unified_answer
from .helpers import build_context_text
from .models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionCreateRequest,
    SessionUpdateRequest,
    WebSearchRequest,
    WebSearchResponse,
)
from .state import (
    LLM_MODEL,
    MEMORY_MANAGER,
    WEB_SEARCH_AVAILABLE,
    WEB_SEARCH_MODEL,
    app,
    client,
    initialize_index,
    retrieve_context,
    web_search,
)

API_VERSION = "2.0.0"


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Construction Documentation QA API v2",
        "version": API_VERSION,
        "model": LLM_MODEL,
        "memory": "Enabled" if MEMORY_MANAGER else "Disabled",
        "web_search": "Available" if WEB_SEARCH_AVAILABLE else "Unavailable",
        "search_modes": ["rag", "web", "hybrid"],
        "features": [
            "follow_up_questions",
            "hallucination_rollback",
            "token_tracking",
            "streaming",
            "session_memory",
            "trade_filtering",
            "context_budgeting",
        ],
        "endpoints": {
            "POST /query": "Generate answer (RAG/web/hybrid) with follow-up questions",
            "POST /query/stream": "Streaming answer with SSE events",
            "POST /quick-query": "Quick query with mode switching (for UI)",
            "POST /web-search": "Web search only",
            "POST /sessions/create": "Create a new conversation session",
            "GET /sessions": "List all active sessions",
            "GET /sessions/{session_id}/stats": "Get session statistics",
            "POST /sessions/{session_id}/update": "Update session context",
            "DELETE /sessions/{session_id}": "Delete a session",
            "GET /health": "Check API health",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(project_id: Optional[int] = Query(None, description="Project ID to check")):
    """Health check endpoint"""
    openai_available = False
    try:
        client.models.list(limit=1)
        openai_available = True
    except Exception as e:
        print(f"OpenAI health check failed: {e}")

    retrieval_available = False
    index_vectors = 0
    metadata_records = 0

    try:
        from retrieve import PROJECTS
        retrieval_available = True
        if project_id and project_id in PROJECTS:
            config = PROJECTS[project_id]
            if config.loaded and config.index is not None:
                index_vectors = config.index.ntotal
            if config.metadata:
                metadata_records = len(config.metadata)
    except Exception as e:
        print(f"Retrieval health check failed: {e}")

    memory_available = MEMORY_MANAGER is not None
    active_sessions = len(MEMORY_MANAGER.sessions) if MEMORY_MANAGER else 0

    return HealthResponse(
        status="healthy" if openai_available and retrieval_available else "degraded",
        model=LLM_MODEL,
        retrieval_available=retrieval_available,
        web_search_available=WEB_SEARCH_AVAILABLE,
        openai_available=openai_available,
        memory_available=memory_available,
        active_sessions=active_sessions,
        index_vectors=index_vectors,
        metadata_records=metadata_records,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        version=API_VERSION,
        project_id=project_id,
    )


@app.get("/config")
async def get_config():
    """Get current API configuration"""
    return {
        "version": API_VERSION,
        "model": LLM_MODEL,
        "web_search_model": WEB_SEARCH_MODEL,
        "web_search_available": WEB_SEARCH_AVAILABLE,
        "max_top_k": 20,
        "default_temperature": 0.0,
        "default_max_tokens": 500,
        "confidence_threshold": 0.30,
        "features": [
            "follow_up_questions", "hallucination_rollback", "token_tracking",
            "streaming", "session_memory", "trade_filtering", "context_budgeting",
        ],
        "supported_source_types": ["drawing", "specification", "sql", None],
        "supported_projects": sorted([int(pid) for pid in __import__("retrieve").PROJECTS.keys()]),
    }


# ── Main query endpoint ──────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Generate an answer with follow-up questions, token tracking, and confidence scoring."""
    try:
        print(f"\n📨 Received query: '{request.query}' | mode={request.search_mode}")

        result = generate_unified_answer(
            user_query=request.query,
            search_mode=request.search_mode,
            top_k=request.top_k,
            min_score=request.min_score,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_citations=request.include_citations,
            include_s3_paths=request.include_s3_paths,
            filter_source_type=request.filter_source_type,
            project_id=request.project_id,
            debug=request.debug,
            session_id=request.session_id,
            create_new_session=request.create_new_session,
        )

        return QueryResponse(**result)

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ── Quick query (simplified for UI) ──────────────────────────────────────────

@app.post("/quick-query")
async def quick_query_endpoint(
    query: str = Body(..., embed=True),
    search_mode: str = Body("rag", embed=True),
    session_id: Optional[str] = Body(None, embed=True),
    project_id: Optional[int] = Body(None, embed=True),
):
    """Quick query with simplified parameters for UI."""
    try:
        result = generate_unified_answer(
            user_query=query,
            search_mode=search_mode,
            top_k=5,
            min_score=0.1,
            temperature=0.0,
            max_tokens=500,
            include_citations=False,
            include_s3_paths=True,
            filter_source_type=None,
            project_id=project_id,
            debug=False,
            session_id=session_id,
            create_new_session=False,
        )

        return {
            "success": True,
            "query": query,
            "answer": result.get("answer", ""),
            "follow_up_questions": result.get("follow_up_questions", []),
            "confidence_score": result.get("confidence_score", 0),
            "is_clarification": result.get("is_clarification", False),
            "search_mode": search_mode,
            "rag_sources": result.get("retrieval_count", 0),
            "web_sources": result.get("web_source_count", 0),
            "session_id": result.get("session_id"),
            "processing_time_ms": result.get("processing_time_ms", 0),
            "token_tracking": result.get("token_tracking"),
        }

    except Exception as e:
        print(f"❌ Error in quick query: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "query": query}


# ── Streaming query endpoint (SSE) ───────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming answer via Server-Sent Events.

    Event types:
      - {"type": "start",     "search_mode": ..., "session_id": ...}
      - {"type": "chunk",     "text": ...}
      - {"type": "follow_up", "questions": [...]}
      - {"type": "done",      "session_id": ..., "confidence_score": ..., ...}
      - {"type": "error",     "message": ...}
    """
    return StreamingResponse(
        stream_unified_answer(
            user_query=request.query,
            search_mode=request.search_mode,
            top_k=request.top_k,
            min_score=request.min_score,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_citations=request.include_citations,
            include_s3_paths=request.include_s3_paths,
            filter_source_type=request.filter_source_type,
            project_id=request.project_id,
            session_id=request.session_id,
            create_new_session=request.create_new_session,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Web Search Endpoint ──────────────────────────────────────────────────────

@app.post("/web-search", response_model=WebSearchResponse)
async def web_search_endpoint(request: WebSearchRequest):
    """Generate an answer using web search."""
    try:
        result = generate_web_search_answer(
            user_query=request.query,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            session_id=request.session_id,
            create_new_session=request.create_new_session,
            use_conversation_history=request.use_conversation_history,
        )
        return WebSearchResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing web search: {str(e)}")


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.post("/sessions/create")
async def create_session_endpoint(request: SessionCreateRequest):
    if not MEMORY_MANAGER:
        raise HTTPException(status_code=501, detail="Memory manager not available")
    session_id = MEMORY_MANAGER.create_session(
        user_query=request.initial_query or "New conversation",
        project_id=request.project_id,
        filter_source_type=request.filter_source_type,
    )
    if request.custom_instructions:
        MEMORY_MANAGER.update_context(session_id=session_id, custom_instructions=request.custom_instructions)
    return {
        "session_id": session_id,
        "message": "Session created successfully",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.post("/sessions/{session_id}/update")
async def update_session_endpoint(session_id: str, request: SessionUpdateRequest):
    if not MEMORY_MANAGER:
        raise HTTPException(status_code=501, detail="Memory manager not available")
    session = MEMORY_MANAGER.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    MEMORY_MANAGER.update_context(
        session_id=session_id, project_id=request.project_id,
        filter_source_type=request.filter_source_type,
        custom_instructions=request.custom_instructions,
    )
    return {"session_id": session_id, "message": "Session updated"}


@app.get("/sessions/{session_id}/stats")
async def get_session_stats_endpoint(session_id: str):
    if not MEMORY_MANAGER:
        raise HTTPException(status_code=501, detail="Memory manager not available")
    stats = MEMORY_MANAGER.get_session_stats(session_id)
    if "error" in stats:
        raise HTTPException(status_code=404, detail=stats["error"])
    return stats


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    if not MEMORY_MANAGER:
        raise HTTPException(status_code=501, detail="Memory manager not available")
    success = MEMORY_MANAGER.clear_session(session_id)
    if success:
        return {"message": "Session deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    if not MEMORY_MANAGER:
        raise HTTPException(status_code=501, detail="Memory manager not available")
    sessions_info = []
    for sid, session in MEMORY_MANAGER.sessions.items():
        sessions_info.append({
            "session_id": sid,
            "created_at": datetime.datetime.fromtimestamp(session.created_at).isoformat(),
            "last_accessed": datetime.datetime.fromtimestamp(session.last_accessed).isoformat(),
            "message_count": len(session.messages),
            "total_tokens": session.total_tokens,
            "project_id": session.context.project_id,
            "filter_source_type": session.context.filter_source_type,
        })
    return {"total_sessions": len(sessions_info), "sessions": sessions_info}


@app.get("/sessions/{session_id}/conversation")
async def get_session_conversation(session_id: str):
    if not MEMORY_MANAGER:
        raise HTTPException(status_code=501, detail="Memory manager not available")
    session = MEMORY_MANAGER.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    all_messages = [
        {
            "role": msg.role, "content": msg.content,
            "timestamp": datetime.datetime.fromtimestamp(msg.timestamp).isoformat(),
            "tokens": msg.tokens,
        }
        for msg in session.messages
    ]
    summaries = [
        {
            "summary_text": s.summary_text, "message_count": s.message_count,
            "start_time": datetime.datetime.fromtimestamp(s.start_time).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(s.end_time).isoformat(),
            "key_points": s.key_points,
        }
        for s in session.summaries
    ]
    return {
        "session_id": session_id,
        "total_messages": len(session.messages),
        "total_summaries": len(session.summaries),
        "total_tokens": session.total_tokens,
        "created_at": datetime.datetime.fromtimestamp(session.created_at).isoformat(),
        "last_accessed": datetime.datetime.fromtimestamp(session.last_accessed).isoformat(),
        "messages": all_messages,
        "summaries": summaries,
        "context": {
            "project_id": session.context.project_id,
            "filter_source_type": session.context.filter_source_type,
            "custom_instructions": session.context.custom_instructions,
        },
    }


# ── Debug / test endpoints ────────────────────────────────────────────────────

@app.get("/test-retrieve")
async def test_retrieve(
    query: str = "What are the instructions provided for LOUVERS?",
    top_k: int = 5,
    min_score: float = 0.1,
    project_id: Optional[int] = None,
    filter_source_type: Optional[str] = None,
):
    """Test retrieval directly without LLM generation."""
    try:
        chunks = retrieve_context(query, top_k=top_k, min_score=min_score,
                                  filter_source_type=filter_source_type, filter_project_id=project_id)
        if not chunks:
            return {"success": False, "message": "No chunks retrieved", "query": query}
        return {
            "success": True,
            "query": query,
            "project_id": project_id,
            "chunks_count": len(chunks),
            "average_similarity": sum(c.get("similarity", 0) for c in chunks) / len(chunks),
            "chunks": [
                {
                    "index": c.get("index"),
                    "source_type": c.get("source_type"),
                    "similarity": c.get("similarity"),
                    "text_preview": c.get("text", "")[:100] + "..." if len(c.get("text", "")) > 100 else c.get("text", ""),
                    "pdf_name": c.get("pdf_name"),
                    "page": c.get("page"),
                }
                for c in chunks[:top_k]
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/debug-pipeline")
async def debug_pipeline(
    query: str = "What are the instructions provided for LOUVERS?",
    project_id: Optional[int] = None,
):
    """Debug the entire RAG pipeline."""
    try:
        chunks = retrieve_context(query, top_k=5, min_score=0.1, filter_project_id=project_id)
        context_text = build_context_text(chunks)

        from .helpers import compute_confidence_score, detect_trade_from_query
        confidence = compute_confidence_score(chunks, "rag")
        trade = detect_trade_from_query(query)

        return {
            "query": query,
            "project_id": project_id,
            "detected_trade": trade,
            "confidence_score": confidence,
            "would_clarify": confidence < 0.30,
            "chunks_retrieved": len(chunks),
            "context_text_length": len(context_text),
            "context_preview": context_text[:500] + "..." if len(context_text) > 500 else context_text,
            "chunks_details": [
                {
                    "source_type": c.get("source_type"),
                    "similarity": c.get("similarity"),
                    "pdf_name": c.get("pdf_name"),
                    "page": c.get("page"),
                    "text_preview": c.get("text", "")[:200],
                }
                for c in chunks[:5]
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/test-web-search")
async def test_web_search(
    query: str = "What are the latest developments in AI?",
    session_id: Optional[str] = None,
):
    """Test web search functionality."""
    try:
        result = generate_web_search_answer(
            user_query=query, temperature=0.0, max_tokens=500,
            session_id=session_id,
            create_new_session=False if session_id else True,
            use_conversation_history=True,
        )
        return {
            "success": True,
            "query": query,
            "session_id": result.get("session_id"),
            "answer_preview": result.get("answer", "")[:300],
            "source_count": result.get("source_count", 0),
            "sources": result.get("sources", []),
            "processing_time_ms": result.get("processing_time_ms", 0),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "success": False})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(exc)}", "success": False})


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print(f"🚀 Construction Documentation QA API v{API_VERSION} starting...")
    print(f"   Model: {LLM_MODEL} | Web Search: {WEB_SEARCH_AVAILABLE}")
    print(f"   Features: follow-ups, hallucination rollback, token tracking, streaming")

    if MEMORY_MANAGER:
        cleaned = MEMORY_MANAGER.cleanup_old_sessions(max_age_hours=24)
        if cleaned:
            print(f"   🧹 Cleaned {cleaned} old session(s)")

    try:
        initialize_index()
        # Verify which projects actually loaded
        from retrieve import PROJECTS
        loaded = [pid for pid, cfg in PROJECTS.items() if cfg.loaded and cfg.index is not None]
        failed = [pid for pid, cfg in PROJECTS.items() if not cfg.loaded or cfg.index is None]
        print(f"   ✅ Projects loaded: {loaded}")
        if failed:
            print(f"   ⚠️ Projects NOT loaded: {failed}")

        for pid in loaded[:3]:
            try:
                chunks = retrieve_context("test query", top_k=2, filter_project_id=pid)
                print(f"   ✅ Project {pid}: {len(chunks)} chunks retrieved (smoke test OK)")
            except Exception as e:
                print(f"   ⚠️ Project {pid} smoke test failed: {e}")
    except Exception as e:
        print(f"   ⚠️ Retrieval init failed: {e}")
        traceback.print_exc()

    if WEB_SEARCH_AVAILABLE:
        print("   ✅ Web search available")
