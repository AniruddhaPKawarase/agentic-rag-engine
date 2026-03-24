"""Backward-compatible API entrypoint.

The implementation lives in `rag.api.*` modules.
"""

from rag.api.helpers import (
    build_context_text,
    budget_context_window,
    compute_confidence_score,
    detect_trade_from_query,
    estimate_tokens,
    extract_source_name,
    filter_chunks_by_trade,
    format_filename_for_display,
    parse_follow_up_questions,
    validate_context_quality,
)
from rag.api.generation import generate_unified_answer, generate_web_search_answer
from rag.api.models import (
    BatchQueryRequest,
    BatchQueryResponse,
    BatchResponseItem,
    ConversationQuery,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievalChunk,
    SessionCreateRequest,
    SessionUpdateRequest,
    SourceDocument,
    TokenTracking,
    WebSearchRequest,
    WebSearchResponse,
)
from rag.api.prompts import (
    build_clarification_prompt,
    build_hybrid_prompt,
    build_rag_prompt,
    build_web_prompt,
)
from rag.api.routes import (
    create_session_endpoint,
    debug_pipeline,
    delete_session_endpoint,
    general_exception_handler,
    get_config,
    get_session_conversation,
    get_session_stats_endpoint,
    health_check,
    http_exception_handler,
    list_sessions,
    query_documents,
    quick_query_endpoint,
    root,
    startup_event,
    test_retrieve,
    test_web_search,
    update_session_endpoint,
    web_search_endpoint,
)
from rag.api.state import (
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


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "generate:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
