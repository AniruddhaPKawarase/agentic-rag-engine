# Unified RAG Agent — API Flow Guide for Frontend (Angular)

**Base URL:** `https://ai5.ifieldsmart.com/rag`
**Auth Header:** `X-API-Key: <key>` (all endpoints except `/`, `/health`, `/metrics`)

Last updated: 2026-04-16

---

## API Classification

| Category | Endpoints | When to Use |
|----------|-----------|-------------|
| **Bootstrap** | `GET /`, `GET /health`, `GET /config` | App initialization |
| **Core Query** | `POST /query`, `POST /query/stream`, `POST /quick-query` | Ask questions on documents |
| **Web Search** | `POST /web-search` | Industry standards / external info |
| **Session Lifecycle** | `POST /sessions/create`, `GET /sessions`, `DELETE /sessions/{id}` | Chat session management |
| **Session Context** | `GET /sessions/{id}/stats`, `GET /sessions/{id}/conversation`, `POST /sessions/{id}/update` | Session details & history |
| **Document Scope** | `GET /projects/{id}/documents`, `POST /sessions/{id}/scope`, `GET /sessions/{id}/scope`, `DELETE /sessions/{id}/scope` | Narrow queries to a specific document |
| **Admin** | `GET /admin/sessions`, `POST /admin/cache/refresh` | Admin dashboard |
| **Legacy/Debug** | `POST /sessions/{id}/pin-document`, `DELETE /sessions/{id}/pin-document`, `GET /test-retrieve`, `GET /debug-pipeline`, `GET /metrics` | Do NOT use in production UI |

---

## User Stories — API Sequence

---

### Story 1: App Boot & Health Check

**When:** Angular app loads, before showing any UI.

```
PARALLEL (on app init):
  ├── GET /health          → Check backend is alive, engines ready
  └── GET /                → Get API version, available engines (cache locally)
```

| Step | API | Purpose | Blocking? |
|------|-----|---------|-----------|
| 1a | `GET /health` | Verify `status: "healthy"`, check `engines.agentic.initialized` | Yes — show error banner if unhealthy |
| 1b | `GET /` | Cache API version + available endpoints | No — informational |

**Frontend logic:**
- If `/health` returns `engines.agentic.initialized: false` → show "Service starting up" state
- If `/health` fails entirely → show "Service unavailable" error

---

### Story 2: User Selects a Project — Document Discovery

**When:** User picks a project from the project dropdown/selector.

```
SEQUENCE:
  1. GET /projects/{project_id}/documents       → Fetch available drawings & specs
  2. POST /sessions/create { project_id }       → Create a fresh chat session
```

| Step | API | Purpose | Notes |
|------|-----|---------|-------|
| 1 | `GET /projects/{project_id}/documents` | Populate the document sidebar/panel with available drawings and specifications | Pass `?set_id=X` if your project uses sets |
| 2 | `POST /sessions/create` | Create a session tied to this project — returns `session_id` | Store `session_id` in component state. All future queries use this. |

**Frontend logic:**
- Display documents grouped by `type`: `"drawing"` vs `"specification"`
- Show `drawing_name` + `drawing_title` (e.g., "M-401 — Mechanical Roof Plan")
- Store the `session_id` — it's the key for everything that follows

---

### Story 3: Ask Questions on Reference Documents (PDF Interaction) — THE CORE FLOW

**When:** User types a question in the chat input and hits send.

```
SEQUENCE:
  1. POST /query/stream                         → Stream the answer via SSE
     ├── IF response.needs_document_selection    → Show document picker (Story 5)
     ├── IF response.source_documents exists     → Render source cards with PDF links
     ├── IF response.follow_up_questions exists  → Show suggestion chips
     └── IF response.fallback_used              → (optional) show "used fallback" indicator
```

| Step | API | Purpose | Notes |
|------|-----|---------|-------|
| 1 | **`POST /query/stream`** | Primary query — streams answer token-by-token via SSE | **Use this for chat UI.** Request body: `{ query, project_id, session_id }` |

**Request body (minimal):**
```json
{
  "query": "What XVENT models are specified in the mechanical drawings?",
  "project_id": 2361,
  "session_id": "session_7b007fb191b2"
}
```

**SSE Event Handling:**
```
event: data: {"type": "token", "delta": "The"}     → Append to chat bubble
event: data: {"type": "token", "delta": " XVENT"}  → Append to chat bubble
event: data: {"type": "done", "answer": "...", "sources": [...]}  → Finalize answer, render sources
event: data: [DONE]                                  → Close SSE connection
```

**Key response fields to render in UI:**

| Field | UI Element |
|-------|------------|
| `answer` | Chat bubble content (markdown) |
| `confidence` / `confidence_score` | Confidence badge (high=green, medium=yellow, low=red) |
| `source_documents[]` | Source cards — show `display_title`, link to `download_url` |
| `s3_paths[]` | PDF viewer links |
| `follow_up_questions[]` | Clickable suggestion chips below the answer |
| `needs_document_selection` | Trigger document picker modal (see Story 5) |
| `available_documents[]` | Documents to show in the picker |
| `scoped_to` | Show "Scoped to: M-401" badge in chat header |
| `web_sources[]` | Web reference links (when `search_mode: "hybrid"`) |
| `engine_used` | (Optional) "Powered by: agentic" small label |
| `processing_time_ms` | (Optional) "Answered in 8.1s" |
| `is_clarification` | If `true`, style the response as a clarification request, not an answer |

**Alternative: Non-streaming query**

Use `POST /query` (non-streaming) when:
- You need the full response object at once (e.g., for document generation)
- SSE is not feasible in a particular context
- You need `debug_info`, `token_usage`, `pin_status` fields (not available in stream)

**Alternative: Quick query for widgets**

Use `POST /quick-query` when:
- Building a search preview / tooltip / sidebar widget
- Only need: `answer`, `sources`, `confidence`, `engine_used`
- Lighter response = faster rendering

---

### Story 4: Multi-Turn Conversation (Follow-Up Questions)

**When:** User asks a follow-up question in the same chat session.

```
SEQUENCE:
  1. POST /query/stream { query, project_id, session_id }   → Continue conversation
```

**This is the same API as Story 3** — the `session_id` carries the conversation context. The backend automatically:
- Loads conversation history from the session
- Resolves pronouns ("What about the second floor?" understands context)
- Maintains document scope if set

**No additional API calls needed** — just keep passing the same `session_id`.

**To resume a previous session (user returns to an old chat):**

```
SEQUENCE:
  1. GET /sessions/{id}/conversation    → Load full chat history into UI
  2. GET /sessions/{id}/scope           → Check if a document scope is active
  3. POST /query/stream { ... }         → Continue from where they left off
```

| Step | API | Purpose |
|------|-----|---------|
| 1 | `GET /sessions/{id}/conversation` | Populate chat bubbles with previous messages |
| 2 | `GET /sessions/{id}/scope` | Show scope badge if a document is scoped |
| 3 | `POST /query/stream` | New question continues the conversation |

---

### Story 5: Document Scoping (Narrow to Specific Document)

**When:** The AI returns `needs_document_selection: true`, OR the user manually clicks a document to scope to.

**Trigger A — AI asks for document selection:**
```
1. Response from /query has needs_document_selection: true
   └── available_documents[] contains the choices
2. UI shows a document picker modal/dropdown
3. User selects a document
4. POST /sessions/{id}/scope { drawing_title, drawing_name, document_type }
5. POST /query/stream { original query, session_id }   → Re-query with scope active
```

**Trigger B — User manually scopes via document sidebar:**
```
1. User clicks a document in the sidebar (from Story 2's document list)
2. POST /sessions/{id}/scope { drawing_title, drawing_name, document_type }
3. All subsequent queries automatically filter to that document
```

| Step | API | Purpose | Notes |
|------|-----|---------|-------|
| Set scope | `POST /sessions/{id}/scope` | Lock queries to one document | Pass `drawing_title`, `drawing_name`, `document_type` |
| Check scope | `GET /sessions/{id}/scope` | Show current scope state in UI | Use on session resume |
| Clear scope | `DELETE /sessions/{id}/scope` | Return to full project search | Show "Clear scope" button when scope is active |

**Frontend logic:**
- When scope is active → show a persistent badge: "Scoped to: M-401 Mechanical Roof Plan" with an (X) to clear
- Clearing scope = `DELETE /sessions/{id}/scope` → remove badge

---

### Story 6: Web Search & Hybrid Mode

**When:** User wants industry standards, code requirements, or information not in project documents.

**Option A — Pure web search (separate action, e.g., "Search Web" button):**
```
POST /web-search { query, project_id }
```

**Option B — Hybrid mode (RAG + Web combined, e.g., toggle in UI):**
```
POST /query/stream { query, project_id, session_id, search_mode: "hybrid" }
```

| Mode | API | Response Fields |
|------|-----|-----------------|
| Web only | `POST /web-search` | `result.answer`, `result.sources[]` (URLs) |
| Hybrid | `POST /query/stream` with `search_mode: "hybrid"` | `rag_answer` (project data), `web_answer` (web), `answer` (merged), `web_sources[]` |

**Frontend logic for hybrid:**
- Render `rag_answer` under a "From Project Documents" section
- Render `web_answer` under a "From Web" section
- Or render the merged `answer` as a single response with clear headers

---

### Story 7: Session History Panel (Sidebar)

**When:** User opens the chat history sidebar to view/resume/delete past sessions.

```
SEQUENCE:
  1. GET /sessions                              → List all sessions (sidebar items)

ON SESSION CLICK:
  2. GET /sessions/{id}/conversation            → Load chat history
  3. GET /sessions/{id}/scope                   → Check scope state

PARALLEL (optional detail view):
  ├── GET /sessions/{id}/stats                  → Show token usage, cost, engine stats
  └── GET /sessions/{id}/scope                  → Show scope state
```

| Step | API | Purpose | Notes |
|------|-----|---------|-------|
| 1 | `GET /sessions` | Populate sidebar list | Show: `session_id`, `message_count`, `last_accessed`, `project_id` |
| 2 | `GET /sessions/{id}/conversation` | Load full chat into the main panel | On click of a session item |
| 3 | `GET /sessions/{id}/stats` | Detail panel: token count, cost, engine usage | Optional — for power users |
| 4 | `DELETE /sessions/{id}` | Delete a session (with confirmation dialog) | Irreversible |

---

### Story 8: Session Context Update

**When:** User changes project mid-session, sets a default filter, or adds custom instructions.

```
POST /sessions/{id}/update { project_id?, filter_source_type?, custom_instructions? }
```

| Field | Use Case |
|-------|----------|
| `project_id` | User switches to a different project in the same session |
| `filter_source_type` | User toggles "Show only drawings" or "Show only specifications" |
| `custom_instructions` | User sets a persona or instruction like "Focus on electrical systems" |

---

### Story 9: Admin Dashboard

**When:** Admin user views system-wide session management.

```
PARALLEL:
  ├── GET /admin/sessions           → All sessions across all users
  ├── GET /config                   → Runtime configuration
  └── GET /health                   → System health
```

| Step | API | Purpose |
|------|-----|---------|
| 1 | `GET /admin/sessions` | Table of all sessions with metadata |
| 2 | `GET /config` | Show runtime config (models, fallback settings) |
| 3 | `POST /admin/cache/refresh` | "Refresh Cache" button for title cache invalidation |

---

## Complete API Call Sequence — Full User Journey

This is the end-to-end flow from app boot to asking questions:

```
APP BOOT
  ├── GET /health                                    [PARALLEL - Bootstrap]
  └── GET /                                          [PARALLEL - Bootstrap]

USER SELECTS PROJECT (project_id = 2361)
  ├── GET /projects/2361/documents                   [PARALLEL - Discovery]
  └── POST /sessions/create { project_id: 2361 }    [PARALLEL - Session Init]
       └── returns session_id = "session_abc123"

USER ASKS FIRST QUESTION
  └── POST /query/stream {                           [CORE - Streaming Q&A]
        query: "What HVAC units are on the roof?",
        project_id: 2361,
        session_id: "session_abc123"
      }
      └── SSE: token → token → token → done → [DONE]

AI SAYS "needs_document_selection: true"
  └── UI shows document picker from available_documents[]
      └── User picks "M-401 Mechanical Roof Plan"
          └── POST /sessions/session_abc123/scope {  [SCOPE - Set]
                drawing_title: "M-401 Mechanical Roof Plan",
                drawing_name: "M-401",
                document_type: "drawing"
              }
          └── POST /query/stream {                   [CORE - Re-query with scope]
                query: "What HVAC units are on the roof?",
                project_id: 2361,
                session_id: "session_abc123"
              }

USER ASKS FOLLOW-UP
  └── POST /query/stream {                           [CORE - Same session]
        query: "What are their BTU ratings?",
        project_id: 2361,
        session_id: "session_abc123"
      }

USER CLEARS SCOPE
  └── DELETE /sessions/session_abc123/scope           [SCOPE - Clear]

USER WANTS WEB INFO
  └── POST /query/stream {                           [CORE - Hybrid mode]
        query: "What are ASHRAE requirements for roof units?",
        project_id: 2361,
        session_id: "session_abc123",
        search_mode: "hybrid"
      }

USER OPENS HISTORY SIDEBAR
  └── GET /sessions                                  [SESSION - List]
      └── Click on old session
          ├── GET /sessions/{id}/conversation        [SESSION - Load history]
          └── GET /sessions/{id}/scope               [SCOPE - Check state]

USER DELETES OLD SESSION
  └── DELETE /sessions/{old_id}                      [SESSION - Cleanup]
```

---

## APIs NOT Needed in Production UI

| Endpoint | Why Skip |
|----------|----------|
| `GET /config` | Admin-only. Don't expose runtime config to regular users. |
| `POST /sessions/{id}/pin-document` | **Legacy** — replaced by Document Scope endpoints. |
| `DELETE /sessions/{id}/pin-document` | **Legacy** — replaced by Document Scope endpoints. |
| `GET /test-retrieve` | Debug/dev only. |
| `GET /debug-pipeline` | Debug/dev only. |
| `GET /metrics` | Prometheus scraping — not for UI consumption. |

---

## Quick Decision Table for Frontend Dev

| "I need to..." | Use This API |
|-----------------|-------------|
| Check if backend is alive | `GET /health` |
| List documents for a project | `GET /projects/{id}/documents` |
| Start a new chat | `POST /sessions/create` |
| Ask a question (chat) | `POST /query/stream` |
| Ask a question (widget/preview) | `POST /quick-query` |
| Get full response with metadata | `POST /query` |
| Search the web only | `POST /web-search` |
| Search project + web combined | `POST /query/stream` with `search_mode: "hybrid"` |
| Scope to a specific document | `POST /sessions/{id}/scope` |
| Check current scope | `GET /sessions/{id}/scope` |
| Clear document scope | `DELETE /sessions/{id}/scope` |
| Load chat history | `GET /sessions/{id}/conversation` |
| List all past sessions | `GET /sessions` |
| Get session details/cost | `GET /sessions/{id}/stats` |
| Update session context | `POST /sessions/{id}/update` |
| Delete a session | `DELETE /sessions/{id}` |
| Filter by drawings only | Add `filter_source_type: "drawing"` to query body |
| Filter by specifications only | Add `filter_source_type: "specification"` to query body |
| Filter by specific drawing | Add `filter_drawing_name: "M-401"` to query body |
| Force a specific engine | Add `engine: "traditional"` or `engine: "agentic"` to query body |

---

## Response Fields — What to Render Where

### Chat Bubble
- `answer` — main content (render as markdown)
- `is_clarification` — if `true`, style differently (question-style bubble)
- `confidence` — badge color (high=green, medium=amber, low=red)

### Source Panel (below answer)
- `source_documents[].display_title` — clickable source card title
- `source_documents[].download_url` — "View PDF" link
- `source_documents[].drawing_name` — subtitle (e.g., "M-401")
- `source_documents[].page` — "Page 3" label
- `web_sources[].url` — external link icon

### Suggestion Chips (below answer)
- `follow_up_questions[]` — clickable chips that auto-fill the query input

### Document Picker Modal (conditional)
- Triggered when `needs_document_selection: true`
- Show `available_documents[]` as selectable list
- On select → call `POST /sessions/{id}/scope` → re-query

### Chat Header / Status Bar
- `scoped_to` — "Scoped to: M-401 Mechanical Roof Plan" with clear (X) button
- `engine_used` — small label "agentic" / "traditional"
- `processing_time_ms` — "Answered in 8.1s"

### Session Sidebar
- `sessions[].message_count` — "8 messages"
- `sessions[].last_accessed` — relative time "2 hours ago"
- `sessions[].project_id` — project badge
