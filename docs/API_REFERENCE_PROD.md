# Unified RAG Agent — Production API Reference

**Base URL:** `https://ai5.ifieldsmart.com/rag`
**Port:** 8001 (behind Nginx reverse proxy on HTTPS)
**Protocol:** HTTPS only (TLS terminated at Nginx)
**Content-Type:** `application/json` (all POST requests)

Last updated: 2026-04-15

---

## Authentication

All endpoints require an API key via the `X-API-Key` header, **except** the following public endpoints:

| Public Endpoint | Purpose |
|-----------------|---------|
| `GET /` | API info |
| `GET /health` | Health check (load-balancer probes) |
| `GET /metrics` | Prometheus scraping |

Authentication uses timing-safe comparison (`hmac.compare_digest`) to prevent side-channel attacks.

**Header format:**
```
X-API-Key: your-api-key-here
```

**Error on missing or invalid key (403):**
```json
{
  "detail": "Forbidden"
}
```

If no `API_KEY` is configured on the server, authentication is disabled (dev mode only).

---

## Engines

| Engine | Role | Data Source | Model |
|--------|------|-------------|-------|
| **Agentic RAG** (primary) | All queries by default | MongoDB (`drawingVision`, `drawing`, `specification`) | GPT-4.1 |
| **Traditional RAG** (fallback) | When agentic returns low confidence, empty answer, or errors | FAISS vector indexes | GPT-4o |

**Fallback triggers:** Low confidence, answer < 20 chars, no sources, `needs_escalation`, or exception.

---

## Quick Reference (24 Endpoints)

| # | Method | Endpoint | Auth | Purpose |
|---|--------|----------|------|---------|
| 1 | GET | `/` | No | API info |
| 2 | GET | `/health` | No | Health check |
| 3 | GET | `/config` | Yes | Configuration summary |
| 4 | POST | `/query` | Yes | Main query (agentic-first) |
| 5 | POST | `/query/stream` | Yes | SSE streaming query |
| 6 | POST | `/quick-query` | Yes | Simplified query |
| 7 | POST | `/web-search` | Yes | Web search only |
| 8 | POST | `/sessions/create` | Yes | Create session |
| 9 | GET | `/sessions` | Yes | List all sessions |
| 10 | GET | `/sessions/{id}/stats` | Yes | Session stats |
| 11 | GET | `/sessions/{id}/conversation` | Yes | Conversation history |
| 12 | POST | `/sessions/{id}/update` | Yes | Update session context |
| 13 | DELETE | `/sessions/{id}` | Yes | Delete session |
| 14 | GET | `/projects/{id}/documents` | Yes | Document discovery |
| 15 | POST | `/sessions/{id}/scope` | Yes | Set document scope |
| 16 | DELETE | `/sessions/{id}/scope` | Yes | Clear document scope |
| 17 | GET | `/sessions/{id}/scope` | Yes | Get scope state |
| 18 | GET | `/admin/sessions` | Yes | Admin: all sessions |
| 19 | POST | `/admin/cache/refresh` | Yes | Admin: invalidate title cache |
| 20 | POST | `/sessions/{id}/pin-document` | Yes | Legacy: pin documents |
| 21 | DELETE | `/sessions/{id}/pin-document` | Yes | Legacy: unpin documents |
| 22 | GET | `/test-retrieve` | Yes | Debug: FAISS retrieval test |
| 23 | GET | `/debug-pipeline` | Yes | Debug: pipeline info |
| 24 | GET | `/metrics` | No | Prometheus metrics |

---

## Error Responses

All endpoints return one of these error formats:

**HTTP 403 -- Forbidden (authentication failure):**
```json
{
  "detail": "Forbidden"
}
```

**HTTP 422 -- Validation Error (invalid request body):**
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

**HTTP 500 -- Internal Server Error:**
```json
{
  "detail": "Internal Server Error"
}
```

**Soft error (engine unavailable, returned as HTTP 200):**
```json
{
  "success": false,
  "error": "Traditional engine not available for web search"
}
```

---

## Core Endpoints

---

### 1. GET / -- API Info

Returns service metadata and a list of available endpoints.

**Authentication:** Not required

**Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service name |
| `version` | string | API version |
| `engines` | string[] | Available engine names |
| `endpoints` | object | Map of endpoint name to path |

**Example Response (200):**
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

### 2. GET /health -- Health Check

Returns engine initialization status and fallback configuration. Used by load-balancer health probes.

**Authentication:** Not required

**Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/health
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` when the service is operational |
| `engines.agentic.initialized` | boolean | `true` if MongoDB is connected and indexes are created |
| `engines.traditional.faiss_loaded` | boolean | `true` if FAISS indexes are loaded (`false` = standby, lazy-loads on first fallback) |
| `fallback_enabled` | boolean | `true` if auto-fallback from agentic to traditional is active |

**Example Response (200):**
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

---

### 3. GET /config -- Configuration Summary

Returns the current runtime configuration with all secrets redacted.

**Authentication:** Required (`X-API-Key`)

**Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/config \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `host` | string | Server bind address |
| `port` | integer | Server port |
| `log_level` | string | Logging level (`INFO`, `DEBUG`, etc.) |
| `agentic_model` | string | Primary LLM model for agentic engine |
| `agentic_model_fallback` | string | Fallback LLM model for agentic engine |
| `agentic_max_steps` | integer | Maximum reasoning steps per agentic query |
| `traditional_model` | string | LLM model for traditional engine |
| `traditional_embedding_model` | string | Embedding model for FAISS retrieval |
| `fallback_enabled` | boolean | Whether auto-fallback is active |
| `fallback_timeout_seconds` | integer | Timeout before fallback triggers |
| `faiss_lazy_load` | boolean | Whether FAISS indexes are lazy-loaded |
| `storage_backend` | string | Storage backend (`s3`, `local`) |
| `mongo_db` | string | MongoDB database name |

**Example Response (200):**
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

### 4. POST /query -- Main Query (Agentic-First)

The primary query endpoint. Runs through the orchestrator: agentic engine first, with automatic fallback to traditional RAG if confidence is low, the answer is empty, or an error occurs.

**Authentication:** Required (`X-API-Key`)

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | -- | Natural language question. Min 1, max 2000 characters. |
| `project_id` | integer | Yes | -- | iFieldSmart project ID. Range: 1--999999. |
| `session_id` | string | No | `null` | Existing session ID for conversation continuity. |
| `engine` | string | No | `null` | Force a specific engine: `"agentic"`, `"traditional"`, or `null` (auto). |
| `set_id` | integer | No | `null` | MongoDB set filter (agentic engine only). |
| `search_mode` | string | No | `null` | `"rag"` (project data), `"web"` (web search), `"hybrid"` (both in parallel). Default behavior is agentic-first. |
| `conversation_history` | ConversationMessage[] | No | `null` | Previous messages for sessionless multi-turn. Each message: `{"role": "user"|"assistant", "content": "..."}`. Content max 10000 chars. |
| `generate_document` | boolean | No | `true` | Whether to generate a document alongside the answer. |
| `filter_source_type` | string | No | `null` | Filter sources by type: `"drawing"` or `"specification"`. |
| `filter_drawing_name` | string | No | `null` | Filter by a specific drawing name. |

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What XVENT models are specified in the mechanical drawings?",
    "project_id": 2361
  }'
```

**Full Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if the query completed without error |
| `answer` | string | Combined final answer (in hybrid mode, includes both RAG and web sections) |
| `rag_answer` | string or null | RAG-only answer from project data |
| `web_answer` | string or null | Web-search-only answer |
| `confidence` | string | Confidence level: `"high"`, `"medium"`, or `"low"` |
| `confidence_score` | float | Numeric confidence (0.0--1.0) |
| `is_clarification` | boolean | `true` if the response is a clarification request rather than a direct answer |
| `follow_up_questions` | string[] | Suggested follow-up questions for the user |
| `improved_queries` | string[] | Reformulated queries the agent tried internally |
| `query_tips` | string[] | Tips for the user to get better results |
| `needs_document_selection` | boolean | `true` if the agent could not determine which document to search and needs user input |
| `available_documents` | object[] | Documents available for selection when `needs_document_selection` is `true`. Each: `{"type": "drawing", "drawing_title": "..."}` |
| `scoped_to` | string or null | The document title the session is currently scoped to, if any |
| `source_documents` | object[] | Structured source document references (see schema below) |
| `s3_paths` | string[] | Flat list of S3 paths for all source documents |
| `s3_path_count` | integer | Count of unique S3 paths |
| `web_sources` | object[] | Web search source references (title, URL) |
| `web_source_count` | integer | Count of web sources |
| `model_used` | string | LLM model used for generation (e.g., `"gpt-4.1"`) |
| `token_usage` | object | Token consumption: `{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}` |
| `processing_time_ms` | integer | Total processing time in milliseconds |
| `session_id` | string or null | Session ID (returned if session was used or auto-created) |
| `search_mode` | string | Effective search mode: `"rag"`, `"web"`, `"hybrid"`, or `"agentic"` |
| `engine_used` | string | Engine that produced the answer: `"agentic"`, `"traditional"`, or `"traditional_fallback"` |
| `fallback_used` | boolean | `true` if agentic failed and traditional answered |
| `agentic_confidence` | string or null | Original agentic confidence when fallback was triggered |
| `pin_status` | object or null | Document pin status (traditional engine sessions only) |
| `debug_info` | object or null | Debug details including `agentic_steps` and `agentic_cost_usd` |
| `error` | string or null | Error message if the query failed |

**Source Document Object:**

| Field | Type | Description |
|-------|------|-------------|
| `s3_path` | string | Full S3 object key for the source file |
| `file_name` | string | Original file name |
| `display_title` | string | Human-readable display title |
| `download_url` | string or null | Pre-signed S3 download URL (when available) |
| `pdf_name` | string | PDF file name |
| `drawing_name` | string | Drawing identifier (e.g., `"M-101A"`) |
| `drawing_title` | string | Full drawing title |
| `page` | integer or null | Page number within the PDF |

**Example Response (200):**
```json
{
  "success": true,
  "answer": "The XVENT models specified in the project are:\n\n- XVENT Model OHEB-44-*\n- XVENT Model OHEB-46-*\n\nThese are exhaust termination models found in the mechanical drawings M-401 and M-402.",
  "rag_answer": "The XVENT models specified in the project are:\n\n- XVENT Model OHEB-44-*\n- XVENT Model OHEB-46-*\n\nThese are exhaust termination models found in the mechanical drawings M-401 and M-402.",
  "web_answer": null,
  "confidence": "high",
  "confidence_score": 0.92,
  "is_clarification": false,
  "follow_up_questions": [
    "What are the CFM ratings for each XVENT model?",
    "Which floors have XVENT exhaust terminations?"
  ],
  "improved_queries": [
    "XVENT exhaust termination models mechanical drawings"
  ],
  "query_tips": [],
  "needs_document_selection": false,
  "available_documents": [],
  "scoped_to": null,
  "source_documents": [
    {
      "s3_path": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
      "file_name": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
      "display_title": "M-401 Mechanical Roof Plan",
      "download_url": "https://s3.amazonaws.com/bucket/0104202614084657M401MECHANICALROOFPLAN1-1.pdf?X-Amz-...",
      "pdf_name": "M401MECHANICALROOFPLAN1-1.pdf",
      "drawing_name": "M-401",
      "drawing_title": "Mechanical Roof Plan",
      "page": 1
    },
    {
      "s3_path": "0104202614084657M402MECHANICALROOFPLAN1-2.pdf",
      "file_name": "0104202614084657M402MECHANICALROOFPLAN1-2.pdf",
      "display_title": "M-402 Mechanical Roof Plan",
      "download_url": null,
      "pdf_name": "M402MECHANICALROOFPLAN1-2.pdf",
      "drawing_name": "M-402",
      "drawing_title": "Mechanical Roof Plan",
      "page": null
    }
  ],
  "s3_paths": [
    "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
    "0104202614084657M402MECHANICALROOFPLAN1-2.pdf"
  ],
  "s3_path_count": 2,
  "web_sources": [],
  "web_source_count": 0,
  "model_used": "gpt-4.1",
  "token_usage": {
    "prompt_tokens": 4521,
    "completion_tokens": 387,
    "total_tokens": 4908
  },
  "processing_time_ms": 8104,
  "session_id": null,
  "search_mode": "agentic",
  "engine_used": "agentic",
  "fallback_used": false,
  "agentic_confidence": "high",
  "pin_status": null,
  "debug_info": {
    "agentic_steps": 2,
    "agentic_cost_usd": 0.012
  },
  "error": null
}
```

**Example: Force Traditional Engine:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What electrical panels are shown in the drawings?",
    "project_id": 7325,
    "engine": "traditional"
  }'
```

**Example: With Session:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "Tell me more about the panel ratings",
    "project_id": 7325,
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }'
```

**Example: With Set ID Filter:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What HVAC units are on the first floor?",
    "project_id": 2361,
    "set_id": 101
  }'
```

**Example: With Conversation History (Sessionless):**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What about the second floor?",
    "project_id": 2361,
    "conversation_history": [
      {"role": "user", "content": "What HVAC units are on the first floor?"},
      {"role": "assistant", "content": "The first floor has AHU-1 and AHU-2."}
    ]
  }'
```

**Example: Hybrid Mode (RAG + Web in Parallel):**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What XVENT exhaust termination models are recommended and what are the latest industry standards?",
    "project_id": 2361,
    "search_mode": "hybrid"
  }'
```

In hybrid mode, `rag_answer` contains the project-specific answer, `web_answer` contains the web search answer, and `answer` merges both with clear headers.

**Example: Filter by Source Type:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What are the insulation requirements?",
    "project_id": 2361,
    "filter_source_type": "specification"
  }'
```

---

### 5. POST /query/stream -- SSE Streaming Query

Server-Sent Events streaming endpoint. Attempts agentic streaming first; falls back to delivering the full query result as a single SSE event if streaming is unavailable.

**Authentication:** Required (`X-API-Key`)

**Request Body:** Same schema as `POST /query` (see endpoint 4).

**Response:** `text/event-stream` (SSE)

**Response Headers:**
| Header | Value |
|--------|-------|
| `Content-Type` | `text/event-stream` |
| `Cache-Control` | `no-cache` |
| `Connection` | `keep-alive` |
| `X-Accel-Buffering` | `no` |

**SSE Event Format:**
```
data: {"type": "token", "delta": "The XVENT"}

data: {"type": "token", "delta": " models specified"}

data: {"type": "done", "answer": "The XVENT models...", "sources": [...]}

data: [DONE]
```

The stream always terminates with `data: [DONE]\n\n`.

If an error occurs mid-stream:
```
data: {"error": "An internal error occurred. Please try again."}

data: [DONE]
```

**Example Request:**
```bash
curl -N -X POST https://ai5.ifieldsmart.com/rag/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What XVENT models are specified?",
    "project_id": 2361
  }'
```

> **Note:** Use `curl -N` to disable output buffering so you see tokens as they arrive.

---

### 6. POST /quick-query -- Simplified Query

A lightweight wrapper around the main query that returns only the answer, sources, confidence, and engine used. Useful for UI widgets or quick lookups where the full response schema is not needed.

**Authentication:** Required (`X-API-Key`)

**Request Body:** Same schema as `POST /query` (see endpoint 4).

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | The generated answer |
| `sources` | object[] | Source documents |
| `confidence` | string | `"high"`, `"medium"`, or `"low"` |
| `engine_used` | string | `"agentic"`, `"traditional"`, or `"unknown"` |

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/quick-query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What panels are in the electrical schedule?",
    "project_id": 2361
  }'
```

**Example Response (200):**
```json
{
  "answer": "The electrical schedule shows panels AP4, AP5, AP7, and AL7 with ratings ranging from 100A to 400A.",
  "sources": [
    {
      "s3_path": "E003ELECTRICALSCHEDULE.pdf",
      "file_name": "E003ELECTRICALSCHEDULE.pdf",
      "display_title": "E-003 Electrical Schedule"
    }
  ],
  "confidence": "high",
  "engine_used": "agentic"
}
```

---

### 7. POST /web-search -- Web Search Only

Runs a web-only search using OpenAI's web search capability through the traditional engine. Does not query project data.

**Authentication:** Required (`X-API-Key`)

**Request Body:** Same schema as `POST /query` (see endpoint 4). Only `query` and `project_id` are used.

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if the search completed |
| `result` | object | Contains `answer` (string) and `sources` (string[]) |
| `error` | string | Present only when `success` is `false` |

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/web-search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "ASHRAE 90.1 energy code requirements for HVAC insulation",
    "project_id": 2361
  }'
```

**Example Response (200):**
```json
{
  "success": true,
  "result": {
    "answer": "ASHRAE 90.1-2022 requires minimum R-values for HVAC duct insulation based on climate zone. In Climate Zone 4, supply ducts require R-6 insulation and return ducts require R-4.",
    "sources": [
      "https://www.ashrae.org/technical-resources/standards-and-guidelines",
      "https://www.energy.gov/eere/buildings/ashrae-standard-901"
    ]
  }
}
```

**Error Response (engine unavailable):**
```json
{
  "success": false,
  "error": "Traditional engine not available for web search"
}
```

---

## Session Endpoints

---

### 8. POST /sessions/create -- Create Session

Creates a new conversation session. Sessions persist conversation history and context across multiple queries.

**Authentication:** Required (`X-API-Key`)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | integer | No | Project to associate with this session |

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/sessions/create \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "project_id": 2361
  }'
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if created |
| `session_id` | string | The new session identifier (UUID) |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable (fallback stub session) |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

### 9. GET /sessions -- List Sessions

Returns all active sessions.

**Authentication:** Required (`X-API-Key`)

**Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/sessions \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieved |
| `sessions` | object[] | Array of session objects |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "project_id": 2361,
      "created_at": "2026-04-15T10:30:00Z",
      "message_count": 8
    },
    {
      "session_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "project_id": 7325,
      "created_at": "2026-04-15T11:00:00Z",
      "message_count": 3
    }
  ]
}
```

---

### 10. GET /sessions/{session_id}/stats -- Session Stats

Returns session statistics including engine usage breakdown and cost.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Example Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/stats \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieved |
| `session_id` | string | Session identifier |
| `engine_usage` | object | Query counts by engine: `{"agentic": N, "traditional": N, "fallback": N}` |
| `last_engine` | string | Engine used for the most recent query |
| `total_cost_usd` | float | Cumulative cost for this session |
| `stub` | boolean | Present and `true` if session manager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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

### 11. GET /sessions/{session_id}/conversation -- Conversation History

Returns the full conversation history for a session.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Example Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/conversation \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieved |
| `session_id` | string | Session identifier |
| `conversation` | object[] | Array of messages with `role`, `content`, and `timestamp` |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "conversation": [
    {
      "role": "user",
      "content": "What XVENT models are specified?",
      "timestamp": 1713178200
    },
    {
      "role": "assistant",
      "content": "The XVENT models specified in the project are OHEB-44-* and OHEB-46-*...",
      "timestamp": 1713178208
    },
    {
      "role": "user",
      "content": "What are their CFM ratings?",
      "timestamp": 1713178250
    },
    {
      "role": "assistant",
      "content": "The CFM ratings for the XVENT models are: OHEB-44 at 250 CFM and OHEB-46 at 350 CFM.",
      "timestamp": 1713178258
    }
  ]
}
```

---

### 12. POST /sessions/{session_id}/update -- Update Session

Updates session context such as project association, filters, or custom instructions.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | integer | No | Change the associated project |
| `filter_source_type` | string | No | Set a default source type filter |
| `custom_instructions` | string | No | Custom instructions for the session |

The request body accepts any key-value pairs; the session manager stores them as session context.

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/update \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "project_id": 7325,
    "filter_source_type": "drawing",
    "custom_instructions": "Focus on electrical systems only"
  }'
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if updated |
| `session_id` | string | Session identifier |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

### 13. DELETE /sessions/{session_id} -- Delete Session

Permanently deletes a session and its conversation history.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Example Request:**
```bash
curl -X DELETE https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if deleted |
| `session_id` | string | Session identifier |
| `deleted` | boolean | `true` confirming deletion |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "deleted": true
}
```

---

## Document Scope Endpoints

These endpoints let the Angular frontend scope queries to specific documents. When a user selects a document, all subsequent queries in that session are filtered to that document until the scope is cleared.

---

### 14. GET /projects/{project_id}/documents -- Document Discovery

Lists available drawing titles and specification sections for a project. Used by the frontend to present document groups when the agent cannot determine which document to search.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_id` | integer | The iFieldSmart project ID |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `set_id` | integer | No | `null` | Filter by MongoDB set ID |

**Example Request:**
```bash
curl "https://ai5.ifieldsmart.com/rag/projects/2361/documents?set_id=101" \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieved |
| `project_id` | integer | Project ID queried |
| `document_count` | integer | Total number of available documents |
| `documents` | object[] | Array of document objects with `type`, `drawing_title`, and other metadata |

**Example Response (200):**
```json
{
  "success": true,
  "project_id": 2361,
  "document_count": 12,
  "documents": [
    {
      "type": "drawing",
      "drawing_title": "M-100 Mechanical Lower Level Plan",
      "drawing_name": "M-100",
      "page_count": 3
    },
    {
      "type": "drawing",
      "drawing_title": "M-401 Mechanical Roof Plan",
      "drawing_name": "M-401",
      "page_count": 2
    },
    {
      "type": "specification",
      "drawing_title": "Section 23 00 00 - HVAC General",
      "drawing_name": null,
      "page_count": 15
    }
  ]
}
```

---

### 15. POST /sessions/{session_id}/scope -- Set Document Scope

Scopes a session to a specific document. All subsequent queries in this session will be filtered to the selected document until the scope is cleared.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Request Body:**

| Field | Type | Required | Max Length | Description |
|-------|------|----------|------------|-------------|
| `drawing_title` | string | No | 200 | The drawing title to scope to |
| `drawing_name` | string | No | 200 | The drawing name (e.g., `"M-401"`) |
| `document_type` | string | No | 20 | `"drawing"` or `"specification"`. Default: `"drawing"` |
| `section_title` | string | No | 200 | Specification section title |
| `pdf_name` | string | No | 200 | Specific PDF file name |

All string inputs are sanitized: control characters are stripped and values are truncated to their max length.

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/scope \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "drawing_title": "M-401 Mechanical Roof Plan",
    "drawing_name": "M-401",
    "document_type": "drawing"
  }'
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if scope was set |
| `session_id` | string | Session identifier |
| `scope` | object | The active scope state |
| `stub` | boolean | Present and `true` if session manager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {
    "drawing_title": "M-401 Mechanical Roof Plan",
    "drawing_name": "M-401",
    "document_type": "drawing",
    "section_title": "",
    "pdf_name": ""
  }
}
```

---

### 16. DELETE /sessions/{session_id}/scope -- Clear Document Scope

Clears the document scope for a session, returning to full project-wide search.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Example Request:**
```bash
curl -X DELETE https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/scope \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if scope was cleared |
| `session_id` | string | Session identifier |
| `scope` | object | The scope state (empty after clearing) |
| `stub` | boolean | Present and `true` if session manager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {}
}
```

---

### 17. GET /sessions/{session_id}/scope -- Get Scope State

Returns the current document scope state for a session.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Example Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/scope \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieved |
| `session_id` | string | Session identifier |
| `scope` | object | Current scope state (empty object if no scope is active) |
| `previously_scoped` | boolean | `true` if this session had a scope set at any point in the past |
| `stub` | boolean | Present and `true` if session manager is unavailable |

**Example Response (200) -- active scope:**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {
    "drawing_title": "M-401 Mechanical Roof Plan",
    "drawing_name": "M-401",
    "document_type": "drawing",
    "section_title": "",
    "pdf_name": ""
  },
  "previously_scoped": true
}
```

**Example Response (200) -- no scope:**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "scope": {},
  "previously_scoped": false
}
```

---

## Admin Endpoints

---

### 18. GET /admin/sessions -- All Sessions (Admin)

Returns all active in-memory sessions with their scope state, engine usage, and cost. Intended for admin dashboards and debugging.

**Authentication:** Required (`X-API-Key`)

**Example Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/admin/sessions \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieved |
| `count` | integer | Total number of active sessions |
| `sessions` | object[] | Array of session detail objects |

**Session Object:**

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `last_engine` | string | Engine used for the last query |
| `total_cost_usd` | float | Cumulative session cost |
| `scope` | object | Current document scope state |
| `engine_usage` | object | Query counts by engine |

**Example Response (200):**
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
        "drawing_title": "M-401 Mechanical Roof Plan",
        "drawing_name": "M-401",
        "document_type": "drawing"
      },
      "engine_usage": {
        "agentic": 5,
        "traditional": 1,
        "fallback": 1
      }
    },
    {
      "session_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "last_engine": "traditional",
      "total_cost_usd": 0.023,
      "scope": {},
      "engine_usage": {
        "agentic": 0,
        "traditional": 3,
        "fallback": 0
      }
    }
  ]
}
```

---

### 19. POST /admin/cache/refresh -- Invalidate Title Cache

Invalidates the drawing title cache. You can clear the cache for a specific project or clear the entire cache.

**Authentication:** Required (`X-API-Key`)

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | integer | No | Clear cache for a specific project. Omit to clear all. |

**Example Request -- clear single project:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/admin/cache/refresh \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "project_id": 2361
  }'
```

**Example Response (200) -- single project:**
```json
{
  "success": true,
  "action": "invalidate_project",
  "project_id": 2361,
  "existed": true
}
```

**Example Request -- clear all:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/admin/cache/refresh \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{}'
```

**Example Response (200) -- all:**
```json
{
  "success": true,
  "action": "invalidate_all",
  "cleared": 5
}
```

---

## Legacy Endpoints (Backward Compatibility)

These endpoints are maintained for backward compatibility with older client integrations. For new implementations, use the Document Scope endpoints (15--17) instead.

---

### 20. POST /sessions/{session_id}/pin-document -- Pin Documents

Pins documents to a session for scoped FAISS search. When documents are pinned, the traditional RAG engine only searches within the pinned documents.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | string[] | Yes | Array of document identifiers to pin (e.g., drawing names) |

**Example Request:**
```bash
curl -X POST https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/pin-document \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "document_ids": ["M-101A", "M-401"]
  }'
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if pinned |
| `session_id` | string | Session identifier |
| `pinned` | boolean | `true` confirming documents were pinned |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "pinned": true
}
```

---

### 21. DELETE /sessions/{session_id}/pin-document -- Unpin Documents

Removes pinned documents from a session, returning to full-index search.

**Authentication:** Required (`X-API-Key`)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | The session identifier |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | string[] | Yes | Array of document identifiers to unpin |

**Example Request:**
```bash
curl -X DELETE https://ai5.ifieldsmart.com/rag/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890/pin-document \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "document_ids": ["M-101A"]
  }'
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if unpinned |
| `session_id` | string | Session identifier |
| `unpinned` | boolean | `true` confirming documents were unpinned |
| `stub` | boolean | Present and `true` if MemoryManager is unavailable |

**Example Response (200):**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "unpinned": true
}
```

---

## Debug Endpoints

---

### 22. GET /test-retrieve -- Test FAISS Retrieval

Tests FAISS vector retrieval directly without running the full generation pipeline. Useful for verifying that indexes are loaded and retrieval is working correctly.

**Authentication:** Required (`X-API-Key`)

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | No | `"test"` | Search query |
| `project_id` | integer | No | `7166` | Project to search |
| `top_k` | integer | No | `5` | Maximum number of results to return |

**Example Request:**
```bash
curl "https://ai5.ifieldsmart.com/rag/test-retrieve?query=electrical+panels&project_id=7325&top_k=3" \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if retrieval completed |
| `query` | string | The query that was searched |
| `project_id` | integer | The project that was searched |
| `results_count` | integer | Number of results returned |
| `results` | object[] | Array of retrieval results with `text`, `similarity`, `drawing_name`, `source_type` |
| `error` | string | Present only when `success` is `false` |

**Example Response (200):**
```json
{
  "success": true,
  "query": "electrical panels",
  "project_id": 7325,
  "results_count": 3,
  "results": [
    {
      "text": "Panel AP4 200A 3-phase, 208/120V with 42 circuits. Main breaker 200A...",
      "similarity": 0.87,
      "drawing_name": "E-101",
      "source_type": "drawing"
    },
    {
      "text": "Panel AL7 100A 1-phase, 120/240V, 24 circuits. Serves lighting circuits...",
      "similarity": 0.82,
      "drawing_name": "E-103",
      "source_type": "drawing"
    },
    {
      "text": "Electrical panel schedule summary. All panels to be Square D QO series...",
      "similarity": 0.78,
      "drawing_name": "E-003",
      "source_type": "drawing"
    }
  ]
}
```

**Error Response (engine unavailable):**
```json
{
  "success": false,
  "error": "Traditional engine not available"
}
```

---

### 23. GET /debug-pipeline -- Pipeline Debug Info

Returns debug information for both engines, the orchestrator, and the title cache.

**Authentication:** Required (`X-API-Key`)

**Example Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/debug-pipeline \
  -H "X-API-Key: your-api-key-here"
```

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `orchestrator.fallback_enabled` | boolean | Whether fallback is active |
| `orchestrator.fallback_timeout` | integer | Timeout in seconds |
| `agentic.initialized` | boolean | Whether the agentic engine is initialized |
| `traditional.faiss_loaded` | boolean | Whether FAISS indexes are loaded |
| `title_cache` | object | Cache statistics (size, hit rate, etc.) or `{"status": "not available"}` |

**Example Response (200):**
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
    "total_projects": 5,
    "total_titles": 142,
    "hit_rate": 0.89,
    "last_refresh": "2026-04-15T09:00:00Z"
  }
}
```

---

### 24. GET /metrics -- Prometheus Metrics

Exposes Prometheus-format metrics for monitoring dashboards. Powered by `prometheus-fastapi-instrumentator`.

**Authentication:** Not required (designed for Prometheus scraping)

**Example Request:**
```bash
curl https://ai5.ifieldsmart.com/rag/metrics
```

**Response:** `text/plain` (Prometheus exposition format)

```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",handler="/query",status="200"} 1523.0

# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="POST",handler="/query",le="1.0"} 120.0
http_request_duration_seconds_bucket{method="POST",handler="/query",le="5.0"} 1400.0
http_request_duration_seconds_bucket{method="POST",handler="/query",le="10.0"} 1510.0
http_request_duration_seconds_bucket{method="POST",handler="/query",le="+Inf"} 1523.0

# HELP http_request_size_bytes HTTP request size in bytes
# TYPE http_request_size_bytes summary
http_request_size_bytes{method="POST",handler="/query",quantile="0.5"} 256.0

# HELP http_response_size_bytes HTTP response size in bytes
# TYPE http_response_size_bytes summary
http_response_size_bytes{method="POST",handler="/query",quantile="0.5"} 4096.0
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | counter | Request count by endpoint, method, and status |
| `http_request_duration_seconds` | histogram | Latency distribution per endpoint |
| `http_request_size_bytes` | summary | Request body sizes |
| `http_response_size_bytes` | summary | Response body sizes |

> **Note:** If `prometheus-fastapi-instrumentator` is not installed, the `/metrics` endpoint returns a 404.

---

## Request Models Reference

### QueryRequest

Used by endpoints 4, 5, 6, and 7.

```json
{
  "query": "string (1-2000 chars, required)",
  "project_id": "integer (1-999999, required)",
  "session_id": "string (optional)",
  "engine": "string: 'agentic' | 'traditional' | null (optional)",
  "set_id": "integer (optional)",
  "search_mode": "string: 'rag' | 'web' | 'hybrid' | null (optional)",
  "conversation_history": [
    {
      "role": "string: 'user' | 'assistant'",
      "content": "string (max 10000 chars)"
    }
  ],
  "generate_document": "boolean (default: true)",
  "filter_source_type": "string: 'drawing' | 'specification' | null (optional)",
  "filter_drawing_name": "string (optional)"
}
```

### ConversationMessage

Used within `conversation_history` in QueryRequest.

```json
{
  "role": "string: 'user' | 'assistant' (required, validated by regex)",
  "content": "string (max 10000 chars, required)"
}
```

---

## Postman Collection Setup

### Variables

| Variable | Value |
|----------|-------|
| `base_url` | `https://ai5.ifieldsmart.com/rag` |
| `api_key` | Your API key |

### Collection-Level Headers

Set these at the collection level so they apply to all requests:

```
X-API-Key: {{api_key}}
Content-Type: application/json
```

### Suggested Requests

**System**
- `GET {{base_url}}/` -- API info (no auth needed)
- `GET {{base_url}}/health` -- Health check (no auth needed)
- `GET {{base_url}}/config` -- Config summary
- `GET {{base_url}}/metrics` -- Prometheus metrics (no auth needed)
- `GET {{base_url}}/debug-pipeline` -- Debug info

**Core Queries**
- `POST {{base_url}}/query` -- `{"query": "What XVENT models are specified in the mechanical drawings?", "project_id": 2361}`
- `POST {{base_url}}/query` -- `{"query": "What electrical panels are shown?", "project_id": 7325, "engine": "traditional"}`
- `POST {{base_url}}/query` -- `{"query": "ASHRAE requirements?", "project_id": 2361, "search_mode": "hybrid"}`
- `POST {{base_url}}/quick-query` -- `{"query": "Panel ratings?", "project_id": 2361}`
- `POST {{base_url}}/web-search` -- `{"query": "ASHRAE 90.1 energy code requirements", "project_id": 2361}`

**Sessions**
- `POST {{base_url}}/sessions/create` -- `{"project_id": 2361}`
- `GET {{base_url}}/sessions` -- List all
- `GET {{base_url}}/sessions/{{session_id}}/stats` -- Stats
- `GET {{base_url}}/sessions/{{session_id}}/conversation` -- History
- `POST {{base_url}}/sessions/{{session_id}}/update` -- `{"project_id": 7325}`
- `DELETE {{base_url}}/sessions/{{session_id}}` -- Delete

**Document Scope**
- `GET {{base_url}}/projects/2361/documents` -- Discover documents
- `POST {{base_url}}/sessions/{{session_id}}/scope` -- `{"drawing_title": "M-401 Mechanical Roof Plan", "drawing_name": "M-401", "document_type": "drawing"}`
- `GET {{base_url}}/sessions/{{session_id}}/scope` -- Get scope
- `DELETE {{base_url}}/sessions/{{session_id}}/scope` -- Clear scope

**Admin**
- `GET {{base_url}}/admin/sessions` -- All sessions
- `POST {{base_url}}/admin/cache/refresh` -- `{"project_id": 2361}` or `{}`

**Legacy**
- `POST {{base_url}}/sessions/{{session_id}}/pin-document` -- `{"document_ids": ["M-101A", "M-401"]}`
- `DELETE {{base_url}}/sessions/{{session_id}}/pin-document` -- `{"document_ids": ["M-101A"]}`

**Debug**
- `GET {{base_url}}/test-retrieve?query=electrical+panels&project_id=7325&top_k=3`

### Suggested Test Workflow

1. `GET /health` -- verify both engines are operational
2. `GET /config` -- confirm model and fallback settings
3. `POST /sessions/create` -- create a session for project 2361
4. `POST /query` with session_id -- agentic query
5. `GET /sessions/{id}/stats` -- verify engine usage
6. `GET /projects/2361/documents` -- discover available documents
7. `POST /sessions/{id}/scope` -- scope to a specific drawing
8. `POST /query` with session_id -- query now scoped to drawing
9. `DELETE /sessions/{id}/scope` -- clear scope
10. `POST /quick-query` -- test simplified response
11. `POST /web-search` -- test web-only search
12. `GET /sessions/{id}/conversation` -- review full conversation history
13. `GET /admin/sessions` -- admin view of all active sessions
14. `DELETE /sessions/{id}` -- cleanup

---

## Rate Limiting

Rate limiting is enforced at the Nginx reverse proxy level, not at the application level. Contact your infrastructure team for current rate limit configuration.

---

## CORS

The API allows cross-origin requests. CORS is configured at the FastAPI application level with permissive defaults for the Angular frontend.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-15 | Added Document Scope endpoints (14--17), Admin endpoints (18--19), full PROD reference |
| 2026-04-05 | Initial 18-endpoint reference (sandbox) |
