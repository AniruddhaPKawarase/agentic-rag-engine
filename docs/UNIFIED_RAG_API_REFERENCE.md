# Unified RAG Agent — API Reference

**Base URL (Sandbox):** `http://54.197.189.113:8001`
**Base URL (via Nginx):** `http://54.197.189.113:8000/rag`
**Port:** 8001

---

## Quick Reference

| # | Method | Endpoint | Purpose |
|---|--------|----------|---------|
| 1 | GET | `/` | API info |
| 2 | GET | `/health` | Combined engine health |
| 3 | GET | `/config` | Settings (no secrets) |
| 4 | POST | `/query` | Agentic-first → fallback query |
| 5 | POST | `/query/stream` | SSE streaming query |
| 6 | POST | `/quick-query` | Simplified query (answer + sources only) |
| 7 | POST | `/web-search` | Web search (traditional engine only) |
| 8 | POST | `/sessions/create` | Create session |
| 9 | GET | `/sessions` | List all sessions |
| 10 | GET | `/sessions/{id}/stats` | Session stats + engine usage |
| 11 | GET | `/sessions/{id}/conversation` | Message history |
| 12 | POST | `/sessions/{id}/update` | Update session context |
| 13 | DELETE | `/sessions/{id}` | Delete session |
| 14 | POST | `/sessions/{id}/pin-document` | Pin docs for FAISS scoped search |
| 15 | DELETE | `/sessions/{id}/pin-document` | Unpin docs |
| 16 | GET | `/test-retrieve` | Test FAISS retrieval directly |
| 17 | GET | `/debug-pipeline` | Debug both engines |
| 18 | GET | `/metrics` | Prometheus metrics |

---

## Engines

| Engine | When Used | Data Source | Model |
|--------|-----------|-------------|-------|
| **AgenticRAG** (primary) | All queries by default | MongoDB (drawingVision, drawing 2.8M, specification) | GPT-4.1 |
| **Traditional RAG** (fallback) | When agentic returns low confidence, empty, or errors | FAISS vector indexes (8 projects) | GPT-4o |

**Fallback triggers:** Low confidence OR answer < 20 chars OR no sources OR needs_escalation OR exception

---

## 1. API Info

### GET /

**Purpose:** Service info and available endpoints.

**Request:**
```
GET http://54.197.189.113:8001/
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

## 2. Health Check

### GET /health

**Purpose:** Check status of both engines.

**Request:**
```
GET http://54.197.189.113:8001/health
```

**Response (200):**
```json
{
    "status": "healthy",
    "engines": {
        "agentic": {
            "initialized": true
        },
        "traditional": {
            "faiss_loaded": true
        }
    },
    "fallback_enabled": true
}
```

**Key fields:**
- `agentic.initialized`: true if MongoDB connected and indexes created
- `traditional.faiss_loaded`: true if FAISS indexes loaded (false = standby, lazy-loads on first fallback)
- `fallback_enabled`: true if auto-fallback is active

---

## 3. Config

### GET /config

**Purpose:** View current config (no secrets exposed).

**Request:**
```
GET http://54.197.189.113:8001/config
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

## 4. Query (Main Endpoint)

### POST /query

**Purpose:** Run a query through the orchestrator. AgenticRAG runs first; if confidence is low, auto-falls back to Traditional RAG.

**Request:**
```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "What XVENT models are specified in the mechanical drawings?",
    "project_id": 2361
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language question (1-2000 chars) |
| `project_id` | int | Yes | iFieldSmart project ID (1-999999) |
| `session_id` | string | No | Reuse existing session |
| `engine` | string | No | Force engine: `"agentic"`, `"traditional"`, or `null` (auto) |
| `set_id` | int | No | MongoDB set filter (AgenticRAG only) |
| `search_mode` | string | No | `"rag"` (project data only, default), `"web"` (web search only), `"hybrid"` (both in parallel → rag_answer + web_answer) |
| `conversation_history` | list | No | Previous messages for sessionless mode |
| `generate_document` | bool | No | Generate document (default true) |
| `filter_source_type` | string | No | `"drawing"` or `"specification"` |
| `filter_drawing_name` | string | No | Filter by specific drawing name |

**Response (200) — matches old QueryResponse schema:**
```json
{
    "query": "",
    "answer": "The XVENT models specified in the project are:\n\n- XVENT Model OHEB-44-* ...",
    "rag_answer": "The XVENT models specified in the project are:\n\n- XVENT Model OHEB-44-* ...",
    "web_answer": null,
    "retrieval_count": 10,
    "average_score": 0.9,
    "confidence_score": 0.9,
    "is_clarification": false,
    "follow_up_questions": [],
    "model_used": "gpt-4.1",
    "token_usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
    "token_tracking": null,
    "s3_paths": ["0104202614084657M100MECHANICALLOWERLEVELPLAN1-1.pdf", "..."],
    "s3_path_count": 10,
    "source_documents": [
        {
            "s3_path": "0104202614084657M100MECHANICALLOWERLEVELPLAN1-1.pdf",
            "file_name": "0104202614084657M100MECHANICALLOWERLEVELPLAN1-1.pdf",
            "display_title": "0104202614084657M100MECHANICALLOWERLEVELPLAN1-1.pdf",
            "download_url": null
        }
    ],
    "retrieved_chunks": [],
    "debug_info": {"agentic_steps": 2, "agentic_cost_usd": 0.0},
    "processing_time_ms": 8104,
    "project_id": null,
    "session_id": null,
    "session_stats": null,
    "search_mode": "agentic",
    "web_sources": [],
    "web_source_count": 0,
    "pin_status": null,
    "success": true,
    "engine_used": "agentic",
    "fallback_used": false,
    "agentic_confidence": "high",
    "error": null
}
```

**Response schema — all fields (matches old QueryResponse):**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original query text |
| `answer` | string | Combined answer |
| `rag_answer` | string/null | RAG-only answer |
| `web_answer` | string/null | Web-only answer |
| `retrieval_count` | int | Number of sources found |
| `average_score` | float | Average retrieval similarity (0-1) |
| `confidence_score` | float | Confidence score (0-1) |
| `is_clarification` | bool | True if answer is a clarification request |
| `follow_up_questions` | list[str] | Suggested follow-up questions |
| `model_used` | string | LLM model used |
| `token_usage` | dict | Token counts (prompt, completion, total) |
| `token_tracking` | dict/null | Granular per-stage token tracking |
| `s3_paths` | list[str] | S3 paths of source documents |
| `s3_path_count` | int | Count of unique S3 paths |
| `source_documents` | list[dict] | Structured source docs (s3_path, file_name, display_title, download_url) |
| `retrieved_chunks` | list[dict] | Retrieved context chunks with similarity scores |
| `debug_info` | dict/null | Debug details (agentic_steps, agentic_cost_usd) |
| `processing_time_ms` | int | Total processing time |
| `project_id` | int/null | Project ID queried |
| `session_id` | string/null | Session ID |
| `session_stats` | dict/null | Session statistics |
| `search_mode` | string | `"agentic"`, `"rag"`, `"web"`, `"hybrid"`, `"greeting"` |
| `web_sources` | list[dict] | Web search sources |
| `web_source_count` | int | Count of web sources |
| `pin_status` | dict/null | Document pin status (traditional only) |
| `engine_used` | string | `"agentic"` / `"traditional"` / `"traditional_fallback"` (NEW) |
| `fallback_used` | bool | True if agentic failed and traditional answered (NEW) |
| `agentic_confidence` | string/null | Original agentic confidence when fallback used (NEW) |
| `error` | string/null | Error message if any (NEW) |

---

### Example: Force Agentic Only

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "What XVENT models are specified?",
    "project_id": 2361,
    "engine": "agentic"
}
```

---

### Example: Force Traditional Only

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "What electrical panels are shown in the drawings?",
    "project_id": 7325,
    "engine": "traditional"
}
```

---

### Example: With Session

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "Tell me more about the panel ratings",
    "project_id": 7325,
    "session_id": "my-session-123"
}
```

---

### Example: With Set ID Filter

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "What HVAC units are on the first floor?",
    "project_id": 2361,
    "set_id": 101
}
```

---

### Example: With Conversation History (Sessionless)

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "What about the second floor?",
    "project_id": 2361,
    "conversation_history": [
        {"role": "user", "content": "What HVAC units are on the first floor?"},
        {"role": "assistant", "content": "The first floor has AHU-1 and AHU-2..."}
    ]
}
```

---

### Example: Web Search Only

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "ASHRAE 90.1 energy code requirements for HVAC insulation",
    "project_id": 2361,
    "search_mode": "web"
}
```

**Response:** `web_answer` populated, `rag_answer` = null, `search_mode` = "web"

---

### Example: Hybrid (Project Data + Web Search in Parallel)

```json
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
    "query": "What XVENT exhaust termination models are recommended and what are the latest industry standards for exhaust terminations?",
    "project_id": 2361,
    "search_mode": "hybrid"
}
```

**Response:**
```json
{
    "query": "What XVENT exhaust termination models...",
    "answer": "**From Project Data:**\nThe XVENT models specified are OHEB-44-*, OHEB-46-*...\n\n---\n\n**From Web Search:**\nAccording to ASHRAE 62.1-2022, exhaust terminations must be...",
    "rag_answer": "The XVENT models specified are OHEB-44-*, OHEB-46-*...",
    "web_answer": "According to ASHRAE 62.1-2022, exhaust terminations must be...",
    "search_mode": "hybrid",
    "web_sources": [{"title": "ASHRAE 62.1", "url": "https://..."}],
    "web_source_count": 3,
    "source_documents": [{"s3_path": "...", "file_name": "M-401...", "display_title": "M-401"}],
    "engine_used": "agentic",
    "...": "..."
}
```

**Key:** In hybrid mode, `rag_answer` has the project-specific answer and `web_answer` has the web search answer. The combined `answer` field has both with clear headers.

---

## 5. Streaming Query

### POST /query/stream

**Purpose:** SSE streaming — get tokens as they arrive.

**Request:** Same body as `/query`

```json
POST http://54.197.189.113:8001/query/stream
Content-Type: application/json

{
    "query": "What XVENT models are specified?",
    "project_id": 2361
}
```

**Response:** `text/event-stream` (SSE)

```
data: {"type": "token", "delta": "The XVENT"}
data: {"type": "token", "delta": " models specified"}
data: {"type": "token", "delta": " in the mechanical..."}
data: {"type": "done", "answer": "The XVENT models...", "sources": [...]}
data: [DONE]
```

**Note:** Test with `curl -N` (no buffer):
```bash
curl -N -X POST http://54.197.189.113:8001/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What XVENT models?", "project_id": 2361}'
```

---

## 6. Quick Query

### POST /quick-query

**Purpose:** Simplified response — only answer, sources, confidence, engine.

**Request:** Same body as `/query`

```json
POST http://54.197.189.113:8001/quick-query
Content-Type: application/json

{
    "query": "What panels are in the electrical schedule?",
    "project_id": 2361
}
```

**Response (200):**
```json
{
    "answer": "The electrical schedule shows panels AP4, AP5, AP7, and AL7...",
    "sources": [{"name": "E0.03"}],
    "confidence": "high",
    "engine_used": "agentic"
}
```

---

## 7. Web Search

### POST /web-search

**Purpose:** Web search using OpenAI's web_search tool (Traditional engine only).

**Request:**
```json
POST http://54.197.189.113:8001/web-search
Content-Type: application/json

{
    "query": "ASHRAE 90.1 energy code requirements for HVAC",
    "project_id": 2361
}
```

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

---

## 8. Create Session

### POST /sessions/create

**Purpose:** Create a new conversation session.

**Request:**
```json
POST http://54.197.189.113:8001/sessions/create
Content-Type: application/json

{
    "project_id": 2361
}
```

**Response (200):**
```json
{
    "success": true,
    "session_id": "a1b2c3d4e5f6"
}
```

---

## 9. List Sessions

### GET /sessions

**Purpose:** List all active sessions.

**Request:**
```
GET http://54.197.189.113:8001/sessions
```

**Response (200):**
```json
{
    "success": true,
    "sessions": [
        {
            "session_id": "a1b2c3d4e5f6",
            "project_id": 2361,
            "created_at": "2026-04-05T18:30:00Z",
            "message_count": 4
        }
    ]
}
```

---

## 10. Session Stats

### GET /sessions/{session_id}/stats

**Purpose:** Get session stats including engine usage breakdown.

**Request:**
```
GET http://54.197.189.113:8001/sessions/a1b2c3d4e5f6/stats
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

## 11. Session Conversation

### GET /sessions/{session_id}/conversation

**Purpose:** Get full conversation history.

**Request:**
```
GET http://54.197.189.113:8001/sessions/a1b2c3d4e5f6/conversation
```

**Response (200):**
```json
{
    "success": true,
    "session_id": "a1b2c3d4e5f6",
    "conversation": [
        {"role": "user", "content": "What XVENT models?", "timestamp": 1712345678},
        {"role": "assistant", "content": "The XVENT models are...", "timestamp": 1712345680}
    ]
}
```

---

## 12. Update Session

### POST /sessions/{session_id}/update

**Purpose:** Update session context (project, filters, custom instructions).

**Request:**
```json
POST http://54.197.189.113:8001/sessions/a1b2c3d4e5f6/update
Content-Type: application/json

{
    "project_id": 7325,
    "filter_source_type": "drawing",
    "custom_instructions": "Focus on electrical systems only"
}
```

**Response (200):**
```json
{
    "success": true,
    "session_id": "a1b2c3d4e5f6"
}
```

---

## 13. Delete Session

### DELETE /sessions/{session_id}

**Purpose:** Delete a session and its history.

**Request:**
```
DELETE http://54.197.189.113:8001/sessions/a1b2c3d4e5f6
```

**Response (200):**
```json
{
    "success": true,
    "session_id": "a1b2c3d4e5f6",
    "deleted": true
}
```

---

## 14. Pin Document

### POST /sessions/{session_id}/pin-document

**Purpose:** Pin documents to a session for FAISS scoped search. When pinned, traditional RAG only searches within these documents.

**Request:**
```json
POST http://54.197.189.113:8001/sessions/a1b2c3d4e5f6/pin-document
Content-Type: application/json

{
    "document_ids": ["M-101A", "M-401"]
}
```

**Response (200):**
```json
{
    "success": true,
    "session_id": "a1b2c3d4e5f6",
    "pinned": true
}
```

---

## 15. Unpin Document

### DELETE /sessions/{session_id}/pin-document

**Purpose:** Remove pinned documents from session.

**Request:**
```json
DELETE http://54.197.189.113:8001/sessions/a1b2c3d4e5f6/pin-document
Content-Type: application/json

{
    "document_ids": ["M-101A"]
}
```

**Response (200):**
```json
{
    "success": true,
    "session_id": "a1b2c3d4e5f6",
    "unpinned": true
}
```

---

## 16. Test FAISS Retrieval

### GET /test-retrieve

**Purpose:** Test FAISS vector retrieval directly without running the full pipeline.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | "test" | Search query |
| `project_id` | int | 7166 | Project to search |
| `top_k` | int | 5 | Max results |

**Request:**
```
GET http://54.197.189.113:8001/test-retrieve?query=electrical+panels&project_id=7325&top_k=3
```

**Response (200):**
```json
{
    "success": true,
    "query": "electrical panels",
    "project_id": 7325,
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

## 17. Debug Pipeline

### GET /debug-pipeline

**Purpose:** Debug information for both engines.

**Request:**
```
GET http://54.197.189.113:8001/debug-pipeline
```

**Response (200):**
```json
{
    "orchestrator": {
        "fallback_enabled": true,
        "fallback_timeout": 30
    },
    "agentic": {
        "initialized": true,
        "module_path": "/home/ubuntu/.../agentic/__init__.py"
    },
    "traditional": {
        "faiss_loaded": true,
        "module_path": "/home/ubuntu/.../traditional/__init__.py"
    }
}
```

---

## 18. Prometheus Metrics

### GET /metrics

**Purpose:** Prometheus-format metrics for monitoring dashboards.

**Request:**
```
GET http://54.197.189.113:8001/metrics
```

**Response (200):** Prometheus text format with:
- `http_requests_total` — request count by endpoint
- `http_request_duration_seconds` — latency histograms
- `http_request_size_bytes` — request sizes
- `http_response_size_bytes` — response sizes

---

## Test Projects (Sandbox)

| Project ID | Name | Best Engine | Notes |
|-----------|------|-------------|-------|
| 2361 | AVE Horsham | AgenticRAG | drawingVision data available (~50 docs) |
| 7325 | HSB Potomac | Both | drawingVision + FAISS indexes |
| 7212 | Manchester | Traditional | FAISS index only (11k vectors) |
| 7166 | — | Traditional | FAISS index only |
| 7223 | — | Traditional | FAISS index only |
| 7277 | — | Traditional | FAISS index only |
| 7292 | — | Traditional | FAISS index only |

---

## Postman Collection Quick Setup

1. Create a new Collection: **"Unified RAG Agent"**
2. Set Collection variable: `base_url` = `http://54.197.189.113:8001`

### Import these requests:

**System**
- `GET {{base_url}}/` — API info
- `GET {{base_url}}/health` — Health check
- `GET {{base_url}}/config` — Config (no secrets)
- `GET {{base_url}}/metrics` — Prometheus metrics
- `GET {{base_url}}/debug-pipeline` — Debug info

**Queries — AgenticRAG (primary)**
- `POST {{base_url}}/query` — Body: `{"query": "What XVENT models are specified in the mechanical drawings?", "project_id": 2361}`
- `POST {{base_url}}/query` — Body: `{"query": "What specifications are for HVAC insulation?", "project_id": 2361}`
- `POST {{base_url}}/query` — Body: `{"query": "List all drawings available", "project_id": 2361}`

**Queries — Traditional (forced)**
- `POST {{base_url}}/query` — Body: `{"query": "What electrical panels are shown?", "project_id": 7325, "engine": "traditional"}`
- `POST {{base_url}}/query` — Body: `{"query": "What plumbing fixtures are on the first floor?", "project_id": 7212, "engine": "traditional"}`

**Queries — Fallback test**
- `POST {{base_url}}/query` — Body: `{"query": "Show me the electrical plans", "project_id": 7166}` (agentic may have no data → fallback)

**Queries — With options**
- `POST {{base_url}}/query` — Body: `{"query": "What HVAC units?", "project_id": 2361, "set_id": 101}`
- `POST {{base_url}}/quick-query` — Body: `{"query": "Panel ratings?", "project_id": 2361}`
- `POST {{base_url}}/web-search` — Body: `{"query": "ASHRAE 90.1 energy code requirements", "project_id": 2361}`

**Sessions**
- `POST {{base_url}}/sessions/create` — Body: `{"project_id": 2361}`
- `GET {{base_url}}/sessions` — List all
- `GET {{base_url}}/sessions/{{session_id}}/stats` — Stats
- `GET {{base_url}}/sessions/{{session_id}}/conversation` — History
- `POST {{base_url}}/sessions/{{session_id}}/update` — Body: `{"project_id": 7325}`
- `DELETE {{base_url}}/sessions/{{session_id}}` — Delete

**Document Pinning**
- `POST {{base_url}}/sessions/{{session_id}}/pin-document` — Body: `{"document_ids": ["M-101A", "M-401"]}`
- `DELETE {{base_url}}/sessions/{{session_id}}/pin-document` — Body: `{"document_ids": ["M-101A"]}`

**Debug**
- `GET {{base_url}}/test-retrieve?query=electrical+panels&project_id=7325&top_k=3`

### Suggested Test Workflow:

1. `GET /health` → verify both engines
2. `POST /query` with project 2361 → agentic answer
3. `POST /query` with project 7325 + `engine: traditional` → traditional answer
4. `POST /query` with project 7166 (no agentic data) → observe fallback
5. `POST /sessions/create` → get session_id
6. `POST /query` with session_id → verify session context
7. `GET /sessions/{id}/stats` → see engine_usage counts
8. `GET /sessions/{id}/conversation` → see message history
9. `POST /quick-query` → simplified response
10. `GET /debug-pipeline` → verify both engines status

---

## Error Responses

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 422 | Validation error (bad request body) |
| 500 | Internal server error |

Error format:
```json
{
    "detail": "Error description"
}
```

Or for soft errors (engine unavailable):
```json
{
    "success": false,
    "error": "Traditional engine not available for web search"
}
```
