# Unified RAG Agent — Postman API reference for UI

Endpoints the **demo-ui.html** and any future UI will use, ordered by the user story.

**Base URL:**
- Local dev: `http://localhost:8001` (or 8002/8003 if you pick a different port)
- Sandbox: `http://54.197.189.113:8001`
- Prod: `https://ai5.ifieldsmart.com/rag` (Nginx reverse-proxies with `/rag/*` prefix)

**Auth:** None for now (user explicitly deferred).
**Content-Type:** `application/json` for all POSTs except `/query/stream` (SSE).

---

## 1. Health check (use first, before any query)

`GET /health`

**Response 200:**
```json
{
  "status": "healthy",
  "engines": {
    "agentic": { "initialized": true },
    "traditional": { "faiss_loaded": false }
  },
  "fallback_enabled": true
}
```

---

## 2. The main query endpoint — one call does everything

`POST /query` — RAG, DocQA, clarify — all routed through this single endpoint.

### 2.1 — Plain RAG query (user story step 1)

```json
{
  "query": "list HVAC drawings in the project",
  "project_id": 7222
}
```

**Response (success, RAG path):**
```json
{
  "success": true,
  "answer": "The HVAC drawings include …",
  "source_documents": [
    {
      "s3_path": "agentic-ai-production/ifieldsmart/…/pdf",
      "file_name": "2511202514162291JRpkwyHotelMechPlans32-1",
      "display_title": "MECHANICAL FIRST FLOOR PLAN",
      "download_url": "https://ifieldsmart.s3.amazonaws.com/…?X-Amz-Signature=…",
      "pdf_name": "2511202514162291JRpkwyHotelMechPlans32-1",
      "page": 3
    }
  ],
  "confidence": "high",
  "session_id": "rag_abc",
  "active_agent": "rag",
  "engine_used": "agentic",
  "follow_up_questions": ["What about electrical?", "…"],
  "needs_clarification": false
}
```

**What the UI does with this:**
- Deduplicate `source_documents[]` by `s3_path`
- Render primary label = `file_name` (basename), subtitle = `display_title`
- "Download" link uses `download_url`
- "Chat with Document" button on each source calls endpoint **2.2** below

---

### 2.2 — Chat with Document (user story step 3) — explicit DocQA handoff

```json
{
  "query": "Give me a brief overview of this document.",
  "project_id": 7222,
  "session_id": "rag_abc",
  "search_mode": "docqa",
  "docqa_document": {
    "s3_path": "agentic-ai-production/ifieldsmart/…/pdf",
    "file_name": "2511202514162291JRpkwyHotelMechPlans32-1",
    "download_url": "https://ifieldsmart.s3.amazonaws.com/…?X-Amz-Signature=…",
    "pdf_name": "2511202514162291JRpkwyHotelMechPlans32-1"
  }
}
```

**Response (success, DocQA path):**
```json
{
  "success": true,
  "answer": "This drawing covers level-32 mechanical plans …",
  "source_documents": [
    { "file_name": "…", "page": 3, "snippet": "…" }
  ],
  "active_agent": "docqa",
  "selected_document": { "file_name": "…", "s3_path": "…" },
  "docqa_session_id": "dq_xyz",
  "engine_used": "docqa",
  "confidence": "high",
  "session_id": "rag_abc",
  "groundedness_score": 0.92
}
```

**Response (graceful fallback — DocQA unreachable or file unindexable):**
```json
{
  "success": false,
  "answer": "Could not load document for deep-dive. Try selecting a different source or ask a general question to continue. (…)",
  "active_agent": "rag",
  "engine_used": "docqa_fallback",
  "fallback_used": true,
  "session_id": "rag_abc"
}
```

**UI behavior:** if `engine_used=="docqa_fallback"`, flip badge back to RAG and show the `answer` as a system message.

---

### 2.3 — DocQA follow-up (user story step 4) — continue inside same doc

Send next user question with the SAME session_id and docqa_document. The bridge reuses the existing `docqa_session_id` (no re-upload):

```json
{
  "query": "Where is fire damper mentioned?",
  "project_id": 7222,
  "session_id": "rag_abc",
  "search_mode": "docqa",
  "docqa_document": {
    "s3_path": "agentic-ai-production/ifieldsmart/…/pdf",
    "file_name": "2511202514162291JRpkwyHotelMechPlans32-1",
    "download_url": "…",
    "pdf_name": "2511202514162291JRpkwyHotelMechPlans32-1"
  }
}
```

Response shape: same as 2.2 success; `docqa_session_id` is the SAME value as the first call (reuse confirmed).

---

### 2.4 — Auto-switch back to RAG (user story step 5)

Just omit `search_mode`. Any of:
- "show all missing scope across project"
- "list all drawings"
- "how many DOAS units"
- "every floor"

…triggers the Phase 3 classifier's project-wide override (score +0.9) → returns to RAG path.

```json
{
  "query": "show all missing scope across project",
  "project_id": 7222,
  "session_id": "rag_abc"
}
```

**Response:** `active_agent: "rag"`, `engine_used: "agentic"` (NOT docqa).

---

### 2.5 — Clarify prompt (user story step 6) — ambiguous pronoun

If user says "is it missing?" or "tell me about fire damper" with a selected doc in session, classifier confidence falls in 0.3-0.7 → clarify envelope:

```json
{
  "query": "is it missing?",
  "project_id": 7222,
  "session_id": "rag_abc"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "",
  "needs_clarification": true,
  "clarification_prompt": "Should I answer from the selected document (HVAC.pdf) or search the whole project?",
  "active_agent": "rag",
  "selected_document": { "file_name": "HVAC.pdf" },
  "engine_used": "classifier",
  "session_id": "rag_abc"
}
```

**UI behavior:** render yellow box with the prompt + two buttons:
- "Answer from selected document" → resubmit with `"mode_hint": "docqa"` + `search_mode: "docqa"` + `docqa_document`
- "Search whole project" → resubmit with `"mode_hint": "rag"`

---

### 2.6 — Explicit mode override (power user)

```json
{
  "query": "anything",
  "project_id": 7222,
  "session_id": "rag_abc",
  "mode_hint": "docqa",
  "search_mode": "docqa",
  "docqa_document": { "…": "…" }
}
```

`mode_hint` with `"rag"` or `"docqa"` **bypasses the classifier entirely** (confidence 1.0). Useful when UI wants absolute control (e.g., user clicked "Chat with Document" button — don't let classifier route elsewhere).

---

## 3. Streaming query (if UI wants token-by-token)

`POST /query/stream` — Server-Sent Events (text/event-stream)

Same body shape as `/query`. Response is a stream of `data: {…}\n\n` chunks ending with `data: [DONE]\n\n`.

**Nginx must have `proxy_buffering off`** for this to work in prod.

---

## 4. Quick query (simplified response)

`POST /quick-query`

Same body as `/query`. Response trimmed to:
```json
{ "answer": "…", "sources": [...], "confidence": "high", "engine_used": "agentic" }
```

---

## 5. Session management endpoints

### 5.1 — Create session
`POST /sessions/create`
```json
{}
```
Response:
```json
{ "session_id": "rag_new", "created_at": "2026-04-23T16:20:00Z" }
```

> Optional: `/query` creates a session automatically if `session_id` is null. This endpoint lets you pre-allocate.

### 5.2 — List sessions
`GET /sessions`

Response:
```json
{ "sessions": [ { "session_id": "rag_abc", "last_accessed": "…", "message_count": 12 } ] }
```

### 5.3 — Session stats (message count + engine mix)
`GET /sessions/{session_id}/stats`

```json
{
  "session_id": "rag_abc",
  "total_messages": 8,
  "engine_used": { "agentic": 5, "docqa": 3 },
  "active_agent": "docqa",
  "docqa_session_id": "dq_xyz"
}
```

### 5.4 — Conversation history (to restore on page reload)
`GET /sessions/{session_id}/conversation`

```json
{
  "session_id": "rag_abc",
  "messages": [
    { "role": "user", "content": "list HVAC drawings", "timestamp": "…" },
    { "role": "assistant", "content": "…", "source_documents": [...] }
  ]
}
```

### 5.5 — Update session context (rarely used from UI)
`POST /sessions/{session_id}/update`
```json
{ "metadata": { "project_name": "JRParkwayHotel" } }
```

### 5.6 — Delete session
`DELETE /sessions/{session_id}`

Response `204` or `{ "deleted": true }`.

---

## 6. Document pinning (FAISS scoped search — optional, pre-DocQA feature)

Still works alongside the new DocQA bridge. Useful when you want RAG to SCOPE to a specific drawing rather than hand off the whole thing to DocQA.

### 6.1 — Pin document
`POST /sessions/{session_id}/pin-document`
```json
{
  "drawing_title": "MECHANICAL FIRST FLOOR PLAN",
  "drawing_name": "M-101",
  "pdf_name": "2511202514162291JRpkwyHotelMechPlans32-1"
}
```

Subsequent `/query` calls in this session only return sources matching the pinned filter.

### 6.2 — Unpin
`DELETE /sessions/{session_id}/pin-document`

---

## 7. Config endpoint (debugging / admin)
`GET /config`

Returns model names, storage backend, fallback settings — no secrets.

---

## 8. Root info
`GET /`

Returns `{service, version, engines, endpoints}` — useful to confirm you're hitting the right service.

---

## Postman collection quick-import

Copy-paste ready environment variables:
- `baseURL` = `http://localhost:8001` (or sandbox)
- `projectId` = `7222`
- `sessionId` = (captured from first /query response)
- `s3Path`, `fileName`, `downloadUrl` = (captured from source_documents[0])

Suggested request order to rehearse the user story end-to-end:

1. `GET {{baseURL}}/health` → expect 200
2. `POST {{baseURL}}/query` with `{query: "list HVAC drawings", project_id: {{projectId}}}` → save `session_id`, save `source_documents[0]`
3. `POST {{baseURL}}/query` with Chat-with-Document payload (§2.2) → expect `active_agent: "docqa"`, save `docqa_session_id`
4. `POST {{baseURL}}/query` with DocQA follow-up (§2.3) → expect SAME `docqa_session_id`
5. `POST {{baseURL}}/query` with `{query: "show all missing scope across project", session_id: {{sessionId}}}` → expect `active_agent: "rag"`
6. `POST {{baseURL}}/query` with `{query: "is it missing?", session_id: {{sessionId}}}` → expect `needs_clarification: true`
7. `POST {{baseURL}}/query` same query + `{mode_hint: "rag"}` → expect normal RAG answer
8. `GET {{baseURL}}/sessions/{{sessionId}}/stats` → confirm engine mix
9. `DELETE {{baseURL}}/sessions/{{sessionId}}` → cleanup

---

## Envelope summary (all /query responses)

Every response has this shape (optional fields default-null). Old UI clients that only read `answer`, `sources`, `session_id` still work.

| Field | Type | When set |
|---|---|---|
| `success` | bool | always |
| `answer` | string | always (may be empty on clarify) |
| `source_documents` | list[dict] or null | RAG path: non-empty; DocQA path: page-level snippets; clarify: empty |
| `sources` | list[dict] | legacy, usually `[]` now; use `source_documents` |
| `confidence` | "high" \| "medium" \| "low" | always |
| `session_id` | string | always |
| `active_agent` | "rag" \| "docqa" | always |
| `engine_used` | "agentic" \| "traditional" \| "docqa" \| "docqa_fallback" \| "classifier" | always |
| `needs_clarification` | bool | true only in classifier clarify path |
| `clarification_prompt` | string or null | set with `needs_clarification` |
| `selected_document` | dict or null | set when DocQA path or clarify with selected doc in session |
| `docqa_session_id` | string or null | set in DocQA path success |
| `follow_up_questions` | list[string] | always, may be empty |
| `fallback_used` | bool | true when DocQA fallback fires |
| `groundedness_score` | float or null | set when self-RAG runs |
| `cost_usd`, `elapsed_ms`, `total_steps`, `model` | metrics | always |

---

## Notes for the UI team

1. **Dedupe source_documents by `s3_path`** — same drawing can be returned with different `page` values; collapse to one card with `pages: [3, 14, 22]`.
2. **Use `file_name` as primary label**, `display_title` as subtitle. Show `file_name` exactly as returned (do not re-strip extensions).
3. **Presigned URLs expire in 1 hour.** If a user keeps a session open past that, the `download_url` may 403. The gateway already re-signs on each response — just trust the latest `download_url` from the most recent `/query` response.
4. **DocQA mode is sticky via session state.** Once a doc is loaded, subsequent queries in DocQA mode reuse the same `docqa_session_id` (no re-upload cost). The classifier may still route OUT of DocQA if the user asks a project-wide question.
5. **Graceful fallback**: if `engine_used` comes back `"docqa_fallback"`, show the `answer` as a system message ("couldn't load, try another doc or ask a general question") and flip the badge back to RAG mode.
6. **Streaming available** on `/query/stream` if you want progressive UI; otherwise the single JSON response from `/query` is the simpler path.
