# Unified RAG Agent -- Sandbox API Reference

**Environment:** SANDBOX (non-production)
**Base URL:** `http://54.197.189.113:8001`
**Protocol:** HTTP (no TLS)
**Auth:** `X-API-Key` header on all endpoints except `/`, `/health`, `/metrics`

---

## Postman Setup

### Environment Variables

| Variable | Value |
|----------|-------|
| `{{base_url}}` | `http://54.197.189.113:8001` |
| `{{project_id}}` | `7222` |
| `{{session_id}}` | _(create one first via POST /sessions/create)_ |
| `{{api_key}}` | _(get from team -- omit if sandbox runs in dev mode)_ |

### Collection-Level Headers

Set these on the collection so every request inherits them:

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-API-Key` | `{{api_key}}` |

### Import Shortcut

Create a Postman Collection named **"Unified RAG Agent -- Sandbox"**. Add the environment above, then import each request below. Every `curl` is copy-paste ready.

---

## Endpoint Index (24 endpoints)

| # | Method | Endpoint | Section |
|---|--------|----------|---------|
| 1 | GET | `/` | [Root](#1-root) |
| 2 | GET | `/health` | [Health](#2-health) |
| 3 | GET | `/config` | [Config](#3-config) |
| 4 | POST | `/query` | [Query](#4-query) |
| 5 | POST | `/query/stream` | [Stream](#5-streaming-query) |
| 6 | POST | `/quick-query` | [Quick Query](#6-quick-query) |
| 7 | POST | `/web-search` | [Web Search](#7-web-search) |
| 8 | POST | `/sessions/create` | [Create Session](#8-create-session) |
| 9 | GET | `/sessions` | [List Sessions](#9-list-sessions) |
| 10 | GET | `/sessions/{id}/stats` | [Session Stats](#10-session-stats) |
| 11 | GET | `/sessions/{id}/conversation` | [Conversation](#11-session-conversation) |
| 12 | POST | `/sessions/{id}/update` | [Update Session](#12-update-session) |
| 13 | DELETE | `/sessions/{id}` | [Delete Session](#13-delete-session) |
| 14 | POST | `/sessions/{id}/pin-document` | [Pin Document](#14-pin-document) |
| 15 | DELETE | `/sessions/{id}/pin-document` | [Unpin Document](#15-unpin-document) |
| 16 | GET | `/projects/{project_id}/documents` | [Document Discovery](#16-document-discovery) |
| 17 | POST | `/sessions/{id}/scope` | [Set Scope](#17-set-scope) |
| 18 | DELETE | `/sessions/{id}/scope` | [Clear Scope](#18-clear-scope) |
| 19 | GET | `/sessions/{id}/scope` | [Get Scope](#19-get-scope) |
| 20 | GET | `/admin/sessions` | [Admin Sessions](#20-admin-list-sessions) |
| 21 | POST | `/admin/cache/refresh` | [Admin Cache Refresh](#21-admin-cache-refresh) |
| 22 | GET | `/test-retrieve` | [Test Retrieve](#22-test-retrieve) |
| 23 | GET | `/debug-pipeline` | [Debug Pipeline](#23-debug-pipeline) |
| 24 | GET | `/metrics` | [Metrics](#24-metrics) |

---

## Test Projects (Sandbox)

| Project ID | Name | Best Engine | Notes |
|-----------|------|-------------|-------|
| 2361 | AVE Horsham | AgenticRAG | drawingVision data (~50 docs) |
| 7222 | -- | Traditional | FAISS index |
| 7325 | HSB Potomac | Both | drawingVision + FAISS |
| 7212 | Manchester | Traditional | FAISS index (11k vectors) |
| 7166 | -- | Traditional | FAISS index |

---

## Engines

| Engine | Role | Data Source | Model |
|--------|------|-------------|-------|
| **AgenticRAG** | Primary | MongoDB (drawingVision, drawing 2.8M, specification) | GPT-4.1 |
| **Traditional RAG** | Fallback | FAISS vector indexes | GPT-4o |

Fallback triggers: low confidence, answer < 20 chars, no sources, needs_escalation, or exception.

---

## Endpoints

### 1. Root

**GET /**

Returns service info and available endpoint list. Public (no API key required).

```bash
curl http://54.197.189.113:8001/
```

**Response (200):**
```json
{
  "service": "Unified RAG Agent",
  "version": "1.0.0",
  "engines": ["agentic", "traditional"],
  "endpoints": {
    "query": "POST /query",
    "stream": "POST /query/stream",
    "quick_query": "POST /quick-query",
    "web_search": "POST /web-search",
    "health": "GET /health",
    "config": "GET /config",
    "sessions": "GET /sessions"
  }
}
```

---

### 2. Health

**GET /health**

Check initialization status of both engines. Public (no API key required).

```bash
curl http://54.197.189.113:8001/health
```

**Response (200):**
```json
{
  "status": "healthy",
  "engines": {
    "agentic": { "initialized": true },
    "traditional": { "faiss_loaded": true }
  },
  "fallback_enabled": true
}
```

`faiss_loaded: false` means FAISS is in standby and will lazy-load on first fallback query.

---

### 3. Config

**GET /config**

Current configuration (secrets redacted).

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/config
```

**Response (200):**
```json
{
  "host": "0.0.0.0",
  "port": 8001,
  "log_level": "INFO",
  "agentic_model": "gpt-4.1",
  "agentic_model_fallback": "gpt-4.1-mini",
  "agentic_max_steps": 8,
  "traditional_model": "gpt-4o",
  "traditional_embedding_model": "text-embedding-3-small",
  "fallback_enabled": true,
  "fallback_timeout_seconds": 30,
  "faiss_lazy_load": true,
  "storage_backend": "s3",
  "mongo_db": "iField"
}
```

---

### 4. Query

**POST /query**

Main query endpoint. Runs AgenticRAG first; auto-falls back to Traditional RAG on low confidence.

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{
    "query": "What XVENT models are specified in the mechanical drawings?",
    "project_id": 7222
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | -- | Natural language question (1--2000 chars) |
| `project_id` | int | Yes | -- | Project ID (1--999999) |
| `session_id` | string | No | null | Reuse existing session for conversation context |
| `engine` | string | No | null | Force engine: `"agentic"`, `"traditional"`, or null (auto) |
| `set_id` | int | No | null | MongoDB set filter (AgenticRAG only) |
| `search_mode` | string | No | null | `"rag"` (default), `"web"`, or `"hybrid"` |
| `conversation_history` | list | No | null | Previous messages for sessionless context |
| `generate_document` | bool | No | true | Generate document output |
| `filter_source_type` | string | No | null | `"drawing"` or `"specification"` |
| `filter_drawing_name` | string | No | null | Filter to a specific drawing name |

**Response (200):** Full response envelope with `answer`, `sources`, `confidence`, `engine_used`, `fallback_used`, `s3_paths`, `source_documents`, `debug_info`, `processing_time_ms`, and more. See [Response Schema](#query-response-schema) below.

#### Force Agentic Only

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What XVENT models are specified?", "project_id": 2361, "engine": "agentic"}'
```

#### Force Traditional Only

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What electrical panels are shown?", "project_id": 7325, "engine": "traditional"}'
```

#### With Session

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "Tell me more about the panel ratings", "project_id": 7325, "session_id": "{{session_id}}"}'
```

#### With Set ID Filter

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What HVAC units are on the first floor?", "project_id": 2361, "set_id": 101}'
```

#### With Conversation History (Sessionless)

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{
    "query": "What about the second floor?",
    "project_id": 2361,
    "conversation_history": [
      {"role": "user", "content": "What HVAC units are on the first floor?"},
      {"role": "assistant", "content": "The first floor has AHU-1 and AHU-2..."}
    ]
  }'
```

#### Web Search Mode

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "ASHRAE 90.1 energy code requirements", "project_id": 2361, "search_mode": "web"}'
```

Returns `web_answer` populated, `rag_answer` = null, `search_mode` = "web".

#### Hybrid Mode (RAG + Web in Parallel)

```bash
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What XVENT models and what are the latest industry standards?", "project_id": 2361, "search_mode": "hybrid"}'
```

Returns both `rag_answer` and `web_answer` with a combined `answer` field.

#### Query Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original query text |
| `answer` | string | Combined answer |
| `rag_answer` | string/null | RAG-only answer |
| `web_answer` | string/null | Web-only answer (web/hybrid modes) |
| `retrieval_count` | int | Number of sources found |
| `average_score` | float | Average retrieval similarity (0--1) |
| `confidence_score` | float | Confidence score (0--1) |
| `is_clarification` | bool | True if answer is a clarification request |
| `follow_up_questions` | list[str] | Suggested follow-up questions |
| `model_used` | string | LLM model used |
| `token_usage` | dict | Token counts: prompt, completion, total |
| `s3_paths` | list[str] | S3 paths of source documents |
| `s3_path_count` | int | Count of unique S3 paths |
| `source_documents` | list[dict] | Structured sources: s3_path, file_name, display_title, download_url |
| `retrieved_chunks` | list[dict] | Context chunks with similarity scores |
| `debug_info` | dict/null | Debug details: agentic_steps, agentic_cost_usd |
| `processing_time_ms` | int | Total processing time in ms |
| `project_id` | int/null | Project ID queried |
| `session_id` | string/null | Session ID used |
| `session_stats` | dict/null | Session statistics |
| `search_mode` | string | `"agentic"`, `"rag"`, `"web"`, `"hybrid"`, `"greeting"` |
| `web_sources` | list[dict] | Web search sources |
| `web_source_count` | int | Count of web sources |
| `pin_status` | dict/null | Document pin status (traditional only) |
| `engine_used` | string | `"agentic"`, `"traditional"`, `"traditional_fallback"` |
| `fallback_used` | bool | True if agentic failed and traditional answered |
| `agentic_confidence` | string/null | Original agentic confidence when fallback triggered |
| `error` | string/null | Error message, if any |

---

### 5. Streaming Query

**POST /query/stream**

SSE streaming. Tokens arrive as `text/event-stream` events as the model generates them.

```bash
curl -N -X POST http://54.197.189.113:8001/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What XVENT models are specified?", "project_id": 7222}'
```

**Request Body:** Same as [POST /query](#4-query).

**Response:** `text/event-stream` (SSE)

```
data: {"type": "token", "delta": "The XVENT"}
data: {"type": "token", "delta": " models specified"}
data: {"type": "done", "answer": "The XVENT models...", "sources": [...]}
data: [DONE]
```

If agentic streaming is unavailable, falls back to a single full-result SSE event.

---

### 6. Quick Query

**POST /quick-query**

Simplified response -- returns only `answer`, `sources`, `confidence`, and `engine_used`. Use this when you do not need the full response envelope.

```bash
curl -X POST http://54.197.189.113:8001/quick-query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What panels are in the electrical schedule?", "project_id": 7222}'
```

**Request Body:** Same as [POST /query](#4-query).

**Response (200):**
```json
{
  "answer": "The electrical schedule shows panels AP4, AP5, AP7...",
  "sources": [{"name": "E0.03"}],
  "confidence": "high",
  "engine_used": "agentic"
}
```

---

### 7. Web Search

**POST /web-search**

Dedicated web search endpoint. Uses the Traditional engine's web search capability (OpenAI web_search tool).

```bash
curl -X POST http://54.197.189.113:8001/web-search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "ASHRAE 90.1 energy code requirements for HVAC", "project_id": 7222}'
```

**Request Body:** Same as [POST /query](#4-query).

**Response (200):**
```json
{
  "success": true,
  "result": {
    "answer": "ASHRAE 90.1-2022 requires...",
    "sources": ["https://www.ashrae.org/..."]
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Traditional engine not available for web search"
}
```

---

### 8. Create Session

**POST /sessions/create**

Create a new conversation session. Returns a session ID you use in subsequent queries for multi-turn context.

```bash
curl -X POST http://54.197.189.113:8001/sessions/create \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"project_id": 7222}'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | int | No | Associate session with a project |

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

If MemoryManager is unavailable, returns a stub UUID with `"stub": true`.

---

### 9. List Sessions

**GET /sessions**

List all active sessions.

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions
```

**Response (200):**
```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "project_id": 7222,
      "created_at": "2026-04-15T10:30:00Z",
      "message_count": 4
    }
  ]
}
```

---

### 10. Session Stats

**GET /sessions/{session_id}/stats**

Session statistics including per-engine usage counts and cumulative cost.

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/{{session_id}}/stats
```

**Response (200):**
```json
{
  "success": true,
  "engine_usage": {
    "agentic": 5,
    "traditional": 1,
    "fallback": 1
  },
  "last_engine": "agentic",
  "total_cost_usd": 0.087
}
```

---

### 11. Session Conversation

**GET /sessions/{session_id}/conversation**

Full conversation history for a session.

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/{{session_id}}/conversation
```

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "conversation": [
    {"role": "user", "content": "What XVENT models?", "timestamp": 1712345678},
    {"role": "assistant", "content": "The XVENT models are...", "timestamp": 1712345680}
  ]
}
```

---

### 12. Update Session

**POST /sessions/{session_id}/update**

Update session context such as project, filters, or custom instructions.

```bash
curl -X POST http://54.197.189.113:8001/sessions/{{session_id}}/update \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{
    "project_id": 7325,
    "filter_source_type": "drawing",
    "custom_instructions": "Focus on electrical systems only"
  }'
```

**Request Body:** Arbitrary key-value pairs to update the session context.

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

### 13. Delete Session

**DELETE /sessions/{session_id}**

Delete a session and all its history.

```bash
curl -X DELETE -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/{{session_id}}
```

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "deleted": true
}
```

---

### 14. Pin Document

**POST /sessions/{session_id}/pin-document**

Pin documents to a session. When pinned, Traditional RAG scopes FAISS search to only these documents.

```bash
curl -X POST http://54.197.189.113:8001/sessions/{{session_id}}/pin-document \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"document_ids": ["M-101A", "M-401"]}'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | list[str] | Yes | Document identifiers to pin |

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "pinned": true
}
```

---

### 15. Unpin Document

**DELETE /sessions/{session_id}/pin-document**

Remove pinned documents from a session, returning to full project search.

```bash
curl -X DELETE http://54.197.189.113:8001/sessions/{{session_id}}/pin-document \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"document_ids": ["M-101A"]}'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | list[str] | Yes | Document identifiers to unpin |

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "unpinned": true
}
```

---

### 16. Document Discovery

**GET /projects/{project_id}/documents**

List available drawing titles and spec sections for a project. Used by the Angular frontend to show document groups when the agent cannot answer -- the user can select a document to scope queries.

```bash
curl -H "X-API-Key: {{api_key}}" \
  "http://54.197.189.113:8001/projects/7222/documents"
```

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `set_id` | int | No | Filter by set ID |

**With set_id:**

```bash
curl -H "X-API-Key: {{api_key}}" \
  "http://54.197.189.113:8001/projects/7222/documents?set_id=101"
```

**Response (200):**
```json
{
  "success": true,
  "project_id": 7222,
  "document_count": 12,
  "documents": [
    {
      "title": "M-101A MECHANICAL LOWER LEVEL PLAN",
      "type": "drawing",
      "drawing_name": "M-101A"
    }
  ]
}
```

---

### 17. Set Scope

**POST /sessions/{session_id}/scope**

Set document scope for a session. Subsequent queries in this session are filtered to the scoped document.

```bash
curl -X POST http://54.197.189.113:8001/sessions/{{session_id}}/scope \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{
    "drawing_title": "M-101A MECHANICAL LOWER LEVEL PLAN",
    "drawing_name": "M-101A",
    "document_type": "drawing"
  }'
```

**Request Body:**

| Field | Type | Required | Max Length | Description |
|-------|------|----------|-----------|-------------|
| `drawing_title` | string | No | 200 | Drawing title to scope to |
| `drawing_name` | string | No | 200 | Drawing name/number |
| `document_type` | string | No | 20 | `"drawing"` (default) or `"specification"` |
| `section_title` | string | No | 200 | Spec section title |
| `pdf_name` | string | No | 200 | PDF file name |

All inputs are sanitized (control characters stripped, length capped).

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {
    "drawing_title": "M-101A MECHANICAL LOWER LEVEL PLAN",
    "drawing_name": "M-101A",
    "document_type": "drawing"
  }
}
```

---

### 18. Clear Scope

**DELETE /sessions/{session_id}/scope**

Clear document scope, returning the session to full project search.

```bash
curl -X DELETE -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/{{session_id}}/scope
```

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {}
}
```

---

### 19. Get Scope

**GET /sessions/{session_id}/scope**

Get the current document scope state for a session.

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/{{session_id}}/scope
```

**Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {
    "drawing_title": "M-101A MECHANICAL LOWER LEVEL PLAN",
    "drawing_name": "M-101A",
    "document_type": "drawing"
  },
  "previously_scoped": true
}
```

`previously_scoped` indicates whether this session has ever had a scope set (even if currently cleared).

---

### 20. Admin List Sessions

**GET /admin/sessions**

Admin view of all active sessions with scope state, engine usage, and cost tracking.

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/admin/sessions
```

**Response (200):**
```json
{
  "success": true,
  "count": 2,
  "sessions": [
    {
      "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "last_engine": "agentic",
      "total_cost_usd": 0.087,
      "scope": {
        "drawing_title": "",
        "drawing_name": "",
        "document_type": "drawing"
      },
      "engine_usage": {
        "agentic": 5,
        "traditional": 1,
        "fallback": 1
      }
    }
  ]
}
```

---

### 21. Admin Cache Refresh

**POST /admin/cache/refresh**

Invalidate the title cache for a specific project or all projects. Use after document uploads or metadata changes.

#### Invalidate One Project

```bash
curl -X POST http://54.197.189.113:8001/admin/cache/refresh \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"project_id": 7222}'
```

**Response (200):**
```json
{
  "success": true,
  "action": "invalidate_project",
  "project_id": 7222,
  "existed": true
}
```

#### Invalidate All Projects

```bash
curl -X POST http://54.197.189.113:8001/admin/cache/refresh \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{}'
```

**Response (200):**
```json
{
  "success": true,
  "action": "invalidate_all",
  "cleared": 5
}
```

---

### 22. Test Retrieve

**GET /test-retrieve**

Test FAISS vector retrieval directly without running the full generation pipeline. Useful for debugging retrieval quality.

```bash
curl -H "X-API-Key: {{api_key}}" \
  "http://54.197.189.113:8001/test-retrieve?query=electrical+panels&project_id=7222&top_k=3"
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | `"test"` | Search query |
| `project_id` | int | `7166` | Project to search |
| `top_k` | int | `5` | Max results to return |

**Response (200):**
```json
{
  "success": true,
  "query": "electrical panels",
  "project_id": 7222,
  "results_count": 3,
  "results": [
    {
      "text": "Panel AP4 200A 3-phase...",
      "similarity": 0.87,
      "drawing_name": "E-101",
      "source_type": "drawing"
    }
  ]
}
```

---

### 23. Debug Pipeline

**GET /debug-pipeline**

Debug information for the orchestrator, both engines, and title cache.

```bash
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/debug-pipeline
```

**Response (200):**
```json
{
  "orchestrator": {
    "fallback_enabled": true,
    "fallback_timeout": 30
  },
  "agentic": {
    "initialized": true
  },
  "traditional": {
    "faiss_loaded": true
  },
  "title_cache": {
    "projects_cached": 3,
    "total_entries": 150
  }
}
```

---

### 24. Metrics

**GET /metrics**

Prometheus-format metrics for monitoring dashboards. Public (no API key required).

```bash
curl http://54.197.189.113:8001/metrics
```

**Response (200):** Prometheus text format containing:
- `http_requests_total` -- request count by endpoint
- `http_request_duration_seconds` -- latency histograms
- `http_request_size_bytes` -- request sizes
- `http_response_size_bytes` -- response sizes

---

## Error Responses

| Status | Meaning |
|--------|---------|
| 200 | Success (check `success` field for soft errors) |
| 403 | Forbidden -- missing or invalid API key |
| 422 | Validation error -- bad request body |
| 500 | Internal server error |

**Hard error (HTTP status):**
```json
{
  "detail": "Forbidden"
}
```

**Soft error (200 with success=false):**
```json
{
  "success": false,
  "error": "Traditional engine not available for web search"
}
```

**Validation error (422):**
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

---

## Suggested Test Workflow

Run these in order to verify the sandbox is working end-to-end:

```bash
# 1. Health check -- verify both engines
curl http://54.197.189.113:8001/health

# 2. Config -- check model and fallback settings
curl -H "X-API-Key: {{api_key}}" http://54.197.189.113:8001/config

# 3. Create a session
curl -X POST http://54.197.189.113:8001/sessions/create \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"project_id": 7222}'
# --> Save the session_id from the response

# 4. Query with session (agentic-first)
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What drawings are available?", "project_id": 7222, "session_id": "SESSION_ID_HERE"}'

# 5. Quick query
curl -X POST http://54.197.189.113:8001/quick-query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "Panel ratings?", "project_id": 7222}'

# 6. Force traditional engine
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What electrical panels?", "project_id": 7222, "engine": "traditional"}'

# 7. Discover documents for the project
curl -H "X-API-Key: {{api_key}}" \
  "http://54.197.189.113:8001/projects/7222/documents"

# 8. Set scope on a document
curl -X POST http://54.197.189.113:8001/sessions/SESSION_ID_HERE/scope \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"drawing_name": "M-101A", "document_type": "drawing"}'

# 9. Query within scope
curl -X POST http://54.197.189.113:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: {{api_key}}" \
  -d '{"query": "What equipment is on this drawing?", "project_id": 7222, "session_id": "SESSION_ID_HERE"}'

# 10. Check scope state
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/SESSION_ID_HERE/scope

# 11. Clear scope
curl -X DELETE -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/SESSION_ID_HERE/scope

# 12. Session stats
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/SESSION_ID_HERE/stats

# 13. Conversation history
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/SESSION_ID_HERE/conversation

# 14. Test FAISS retrieval directly
curl -H "X-API-Key: {{api_key}}" \
  "http://54.197.189.113:8001/test-retrieve?query=electrical+panels&project_id=7222&top_k=3"

# 15. Admin -- view all sessions
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/admin/sessions

# 16. Debug pipeline
curl -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/debug-pipeline

# 17. Cleanup -- delete session
curl -X DELETE -H "X-API-Key: {{api_key}}" \
  http://54.197.189.113:8001/sessions/SESSION_ID_HERE
```

---

## Notes

- **No TLS**: Sandbox uses plain HTTP. Do not send production API keys over this connection.
- **Stub mode**: If a backend module (MemoryManager, session manager) is not available, endpoints return `"stub": true` with default/empty data instead of failing.
- **Lazy FAISS**: FAISS indexes load on first Traditional engine query, not at startup. First fallback query may be slower.
- **Rate limits**: No rate limiting on sandbox. Do not run load tests without coordinating with the team.
