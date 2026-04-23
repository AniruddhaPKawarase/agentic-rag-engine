# Angular Frontend Integration Guide
## Unified RAG Agent + Document QA Bridge

**Audience:** Angular frontend developer implementing the dual-agent chat UI
**Target:** Angular dev server on `http://localhost:4200` → Sandbox API on `http://54.197.189.113:8001`
**Date:** 2026-04-17

---

## Table of Contents

1. [Setup & CORS](#1-setup--cors)
2. [Did the Old APIs Change?](#2-did-the-old-apis-change)
3. [API Overview](#3-api-overview)
4. [User Stories & API Sequence](#4-user-stories--api-sequence)
5. [Complete API Reference](#5-complete-api-reference)
6. [E2E Testing Flow](#6-e2e-testing-flow)
7. [Error Handling & Edge Cases](#7-error-handling--edge-cases)
8. [TypeScript Types](#8-typescript-types)

---

## 1. Setup & CORS

### Base URL

| Environment | URL |
|-------------|-----|
| **Sandbox (use this for dev)** | `http://54.197.189.113:8001` |
| Production | `https://ai5.ifieldsmart.com/rag` |

### CORS — Already Configured for Angular

The sandbox API is **pre-configured** to allow requests from `http://localhost:4200`. No special setup needed on your end.

**Server-side allowed origins (for reference):**
```
https://ifieldsmart.com
https://ai5.ifieldsmart.com
https://sandbox.ifieldsmart.ai
http://localhost:3000
http://localhost:4200   <-- Angular default
http://localhost:4300
http://localhost:8080
```

**Verification** — run this from your terminal:
```bash
curl -I -X OPTIONS "http://54.197.189.113:8001/query" \
  -H "Origin: http://localhost:4200" \
  -H "Access-Control-Request-Method: POST"
```
Should return `access-control-allow-origin: http://localhost:4200`.

### Authentication

**No auth currently required.** No `X-API-Key` header, no bearer token. This is intentional — security will be added in a later phase. For now, just send requests.

### HttpClient Setup (Angular)

```typescript
// environment.ts
export const environment = {
  apiBaseUrl: 'http://54.197.189.113:8001',
};

// app.config.ts
import { provideHttpClient } from '@angular/common/http';

export const appConfig: ApplicationConfig = {
  providers: [provideHttpClient()],
};

// rag.service.ts
@Injectable({ providedIn: 'root' })
export class RagService {
  private base = environment.apiBaseUrl;
  constructor(private http: HttpClient) {}
  // ... methods below
}
```

---

## 2. Did the Old APIs Change?

### Summary: **No breaking schema changes**, but **one important behavioral change**.

### 2.1 Schema Changes (Backward-Compatible Additions Only)

| Field | Type | Change | Impact |
|-------|------|--------|--------|
| `QueryRequest.docqa_document` | object/null | **NEW optional field** | None for old clients |
| `QueryRequest.search_mode` | string | Added `"docqa"` to valid values (was `"rag"\|"web"\|"hybrid"`) | None — old values still work |
| `Response.active_agent` | string | **NEW response field** | Old clients ignore; new UI reads it |
| `Response.suggest_switch` | string/null | **NEW response field** | Old clients ignore |
| `Response.selected_document` | string/null | **NEW response field** | Old clients ignore |

**All existing request fields and response fields are unchanged.** An Angular client using the old contract will continue to work — it just won't see the new dual-agent features.

### 2.2 Behavioral Change — Answer Format (IMPORTANT)

**Previously**, answer text from `/query` contained inline citations and structured blocks:

```
Direct answer: The mechanical drawings specify...

Supporting Details:
- "PROVIDE XVENT MODEL..." [Source: Page 41 / MECHANICAL SECOND FLOOR]

---

Citations:
- [Source: Page40 / ...]
- [Source: Page41 / ...]
```

**Now**, answer text is **natural prose with NO inline citations** or structure markers:

```
The mechanical drawings specify several XVENT models for different duct terminations:

- XVENT Model 6SEB-* for single outside air intake terminations
- XVENT Model DHEB-44-* for double exhaust terminations
- ...
```

### 2.3 What This Means for the Angular Developer

| Before | Now |
|--------|-----|
| Rendering answer text would show inline `[Source:]` tags | Render answer text as-is — it's clean prose |
| Citations were parsed from answer text | Render from `response.source_documents[]` array (as separate UI cards) |
| `---` separators appeared in answer | No separators — just paragraphs/bullets |
| "Direct answer" header appeared | No header |

**Action required:** If your UI was parsing citations from the answer text, **switch to rendering them from `source_documents[]`**. If you were already doing that, no changes needed.

### 2.4 What Hasn't Changed

- All endpoint paths (same URLs)
- All HTTP methods (GET/POST/DELETE)
- All required request fields
- The `follow_up_questions[]` array still works the same
- Session management (`session_id`, conversation history)
- Streaming format (`/query/stream` SSE)

---

## 3. API Overview

### 17 Endpoints — Grouped by Purpose

| # | Method | Endpoint | Purpose | Used In Story |
|---|--------|----------|---------|---------------|
| 1 | GET | `/` | Service info | Boot |
| 2 | GET | `/health` | Health check | Boot |
| 3 | GET | `/config` | Server config | (optional) |
| 4 | **POST** | **`/query`** | **Main query (RAG + DocQA)** | Stories 3-9 |
| 5 | POST | `/query/stream` | SSE streaming query | (optional) |
| 6 | POST | `/quick-query` | Lightweight query | (optional) |
| 7 | POST | `/web-search` | Web-only search | (optional) |
| 8 | POST | `/sessions/create` | Start session | Story 2 |
| 9 | GET | `/sessions` | List sessions | Story 10 |
| 10 | GET | `/sessions/{id}/stats` | Session stats | (optional) |
| 11 | GET | `/sessions/{id}/conversation` | Load history | Story 10 |
| 12 | POST | `/sessions/{id}/update` | Update context | (optional) |
| 13 | **DELETE** | **`/sessions/{id}`** | Delete session | Story 11 |
| 14 | POST | `/sessions/{id}/pin-document` | Pin doc (legacy) | (optional) |
| 15 | DELETE | `/sessions/{id}/pin-document` | Unpin doc (legacy) | (optional) |
| 16 | GET | `/test-retrieve` | Debug only | Never |
| 17 | GET | `/debug-pipeline` | Debug only | Never |

**Bold = the endpoints you'll use most.** `/query` alone handles almost everything because it multiplexes RAG and DocQA based on `search_mode`.

---

## 4. User Stories & API Sequence

### Story Map

```
[Boot] -> [Start Session] -> [Ask Question] -> [Get Answer + Sources]
                                                      |
                                   +------------------+------------------+
                                   |                                     |
                         [Click "Chat with Doc"]              [Ask follow-up / new Q]
                                   |                                     |
                         [DocQA Processing]                    [Stays in RAG mode]
                                   |
                         [Ask Doc-Specific Q]
                                   |
                    +--------------+--------------+
                    |              |              |
            [Doc Question]  [Project-Wide]  ["Back to Search"]
                    |              |              |
              [DocQA Answer] [Suggest Switch] [Exit DocQA Mode]
```

---

### STORY 1: Application Boot

**User action:** Opens the app.
**Goal:** Verify backend is reachable before showing chat UI.

**API call:**
```http
GET http://54.197.189.113:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "engines": { "agentic": { "initialized": true }, "traditional": { "faiss_loaded": false } },
  "fallback_enabled": true
}
```

**UI rendering:**
- Green dot if `status === "healthy"` — enable chat input
- Red dot otherwise — show "Backend unreachable" banner, disable send button

**Angular snippet:**
```typescript
checkHealth(): Observable<HealthResponse> {
  return this.http.get<HealthResponse>(`${this.base}/health`);
}
```

---

### STORY 2: Start a New Session

**User action:** Opens chat for project 2361.
**Goal:** Create a new session so follow-ups have context.

**API call:**
```http
POST http://54.197.189.113:8001/sessions/create
Content-Type: application/json

{ "project_id": 2361 }
```

**Response:**
```json
{ "success": true, "session_id": "session_c3d31259c0ed" }
```

**UI rendering:**
- Store `session_id` in a `BehaviorSubject` / signal
- Show session label in header: `"Session: session_c3d31259c0ed"`
- Initialize empty conversation view

**Angular snippet:**
```typescript
private sessionId = signal<string | null>(null);

startSession(projectId: number): Observable<SessionCreateResponse> {
  return this.http.post<SessionCreateResponse>(
    `${this.base}/sessions/create`,
    { project_id: projectId }
  ).pipe(tap(r => this.sessionId.set(r.session_id)));
}
```

---

### STORY 3: Ask a Simple Question (RAG Agent Answers)

**User action:** Types "What XVENT models are specified?" and hits Send.
**Goal:** Get an answer grounded in project documents.

**API call:**
```http
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
  "query": "What XVENT models are specified?",
  "project_id": 2361,
  "session_id": "session_c3d31259c0ed"
}
```

**Response (RAG answers successfully):**
```json
{
  "success": true,
  "answer": "The mechanical drawings specify several XVENT models for different duct terminations:\n- XVENT Model 6SEB-*...",
  "confidence": "high",
  "confidence_score": 0.92,
  "engine_used": "agentic",
  "active_agent": "rag",
  "fallback_used": false,
  "source_documents": [
    {
      "s3_path": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
      "file_name": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
      "display_title": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
      "drawing_name": "M-401",
      "drawing_title": "Mechanical Roof Plan",
      "page": 1,
      "download_url": "https://ifieldsmart.s3.amazonaws.com/..."
    }
  ],
  "follow_up_questions": [
    "Where is the condensate drain located?",
    "What are the ductwork specs?",
    "Which pages show ventilation details?"
  ],
  "processing_time_ms": 8104,
  "needs_document_selection": false,
  "suggest_switch": null
}
```

**UI rendering (CRITICAL):**

1. **Render `answer` as-is** — it's clean natural prose. Do NOT parse for `[Source:]` tags.
2. **Render `source_documents[]` as separate cards** below the answer. Each card should have:
   - Document icon
   - `display_title` (falls back to `file_name` or `s3_path`)
   - Page number (if present)
   - **"Chat with Document" button** ← triggers Story 5
3. **Render `follow_up_questions[]` as clickable chips** — clicking fills the input box.
4. Show `confidence` badge (high=green, medium=amber, low=red).

**Angular snippet:**
```typescript
interface QueryResponse {
  success: boolean;
  answer: string;
  confidence: 'high' | 'medium' | 'low';
  confidence_score: number;
  engine_used: 'agentic' | 'traditional' | 'docqa';
  active_agent: 'rag' | 'docqa';
  source_documents: SourceDoc[];
  follow_up_questions: string[];
  needs_document_selection: boolean;
  available_documents: AvailableDoc[];
  suggest_switch: 'rag' | null;
  processing_time_ms: number;
}

query(body: QueryRequest): Observable<QueryResponse> {
  return this.http.post<QueryResponse>(`${this.base}/query`, body);
}
```

---

### STORY 4: Ask a Question With No Good Match (Document Discovery)

**User action:** Asks something vague that the agent can't answer confidently.
**Goal:** Offer a list of documents the user can pick from.

**Response difference from Story 3:**
```json
{
  "success": true,
  "answer": "I couldn't find specific information... Your project has the following document groups — try selecting one for a focused search.",
  "confidence": "low",
  "needs_document_selection": true,
  "available_documents": [
    {
      "type": "drawing",
      "drawing_title": "M-401 Mechanical Roof Plan",
      "drawing_name": "M-401",
      "trade": "Mechanical",
      "pdf_name": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
      "fragment_count": 327
    },
    {
      "type": "specification",
      "drawing_title": "23 31 00 - HVAC Ducts",
      "pdf_name": "spec_23_31_00.pdf",
      "section_title": "HVAC Ducts",
      "fragment_count": 412
    }
  ],
  "source_documents": [],
  "follow_up_questions": ["Try asking...", "..."]
}
```

**UI rendering:**
- If `needs_document_selection === true`, show a **document picker** below the answer
- Each picker item has type badge ("drawing" / "specification"), title, and fragment count
- Clicking an item triggers Story 5 (using `pdf_name` as `s3_path`)

**Note on duplicates:** The `available_documents` list may contain near-duplicates (multiple pages of same drawing). The UI should deduplicate on the fly:
```typescript
deduplicateDocs(docs: AvailableDoc[]): AvailableDoc[] {
  const seen = new Set<string>();
  return docs.filter(d => {
    const key = (d.drawing_title || d.pdf_name || '').toLowerCase().trim();
    if (!key || key === 'specification' || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
```

---

### STORY 5: User Clicks "Chat with Document" — Activate DocQA Mode

**User action:** Clicks the "Chat with Document" button on a source card OR on a picker item.
**Goal:** Switch to deep-dive Q&A mode for that specific document.

**API call:**
```http
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
  "query": "Summarize what this document contains",
  "project_id": 2361,
  "session_id": "session_c3d31259c0ed",
  "search_mode": "docqa",
  "docqa_document": {
    "s3_path": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
    "file_name": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf"
  }
}
```

**What happens server-side:**
1. RAG orchestrator downloads PDF from S3 bucket
2. Uploads to DocQA Agent at port 8006
3. DocQA extracts, chunks, embeds, answers
4. RAG session is linked to the DocQA session
5. Returns answer through RAG response envelope

**Response (first-time takes ~17s due to PDF processing):**
```json
{
  "success": true,
  "answer": "The mechanical drawing titled 'ENLARGED MECHANICAL UNIT PLAN - 3RD FLR WEST' provides detailed layouts...",
  "confidence": "high",
  "confidence_score": 0.585,
  "engine_used": "docqa",
  "active_agent": "docqa",
  "selected_document": "0104202614084657M401MECHANICALROOFPLAN1-1.pdf",
  "source_documents": [{
    "s3_path": "...",
    "file_name": "...",
    "display_title": "...",
    "text_preview": "Full document context",
    "relevance_score": 1.0
  }],
  "follow_up_questions": ["...", "...", "..."],
  "processing_time_ms": 16295,
  "suggest_switch": null
}
```

**UI rendering:**
- Show a loading indicator — "Processing document..." (can take 15-20 seconds)
- When response arrives:
  - Update mode badge: **"RAG Mode" → "DocQA Mode"** (use a distinct color, e.g., purple)
  - Show a small header bar: `Chatting with: <selected_document>` + **"Back to Project Search" button**
  - Change input placeholder: `"Ask about this document..."`
  - Render answer same as Story 3 (prose + sources + follow-ups)

**Angular snippet:**
```typescript
chatWithDocument(doc: SourceDoc): void {
  this.mode.set('docqa');  // update UI mode signal
  this.selectedDoc.set(doc);

  this.query({
    query: 'Summarize what this document contains',
    project_id: this.projectId(),
    session_id: this.sessionId()!,
    search_mode: 'docqa',
    docqa_document: {
      s3_path: doc.s3_path,
      file_name: doc.file_name,
    },
  }).subscribe(response => {
    this.messages.update(m => [...m, { role: 'assistant', ...response }]);
    // active_agent from server is authoritative
    this.mode.set(response.active_agent);
  });
}
```

---

### STORY 6: Follow-up Question in DocQA Mode (Fast Path)

**User action:** After activating DocQA mode, asks another question about the same document.
**Goal:** Get answer without re-uploading the PDF.

**API call (note: NO `docqa_document` field):**
```http
POST http://54.197.189.113:8001/query
Content-Type: application/json

{
  "query": "What heat pump equipment is shown?",
  "project_id": 2361,
  "session_id": "session_c3d31259c0ed",
  "search_mode": "docqa"
}
```

**Response (4x faster — ~4s):**
```json
{
  "success": true,
  "answer": "The mechanical drawing specifies vertical, ducted-type water source heat pump units...",
  "confidence": "high",
  "confidence_score": 0.902,
  "engine_used": "docqa",
  "active_agent": "docqa",
  "selected_document": "...",
  "processing_time_ms": 3586,
  "suggest_switch": null
}
```

**UI rendering:** Same as Story 5 but stays in DocQA mode.

**Why fast?** The orchestrator reuses the cached DocQA session (PDF already processed). No S3 download, no re-indexing.

---

### STORY 7: Project-Wide Query While in DocQA Mode (Intent Switch)

**User action:** Types "Show all HVAC drawings across the project" while in DocQA mode.
**Goal:** Offer user the choice to switch back to project-wide search.

**API call (same as Story 6):**
```http
POST http://54.197.189.113:8001/query

{
  "query": "Show all HVAC drawings across the project",
  "project_id": 2361,
  "session_id": "session_c3d31259c0ed",
  "search_mode": "docqa"
}
```

**Response (intent classifier detects project-wide signal):**
```json
{
  "success": true,
  "suggest_switch": "rag",
  "active_agent": "docqa",
  "answer": "This question seems to be about the broader project, not just the selected document. Would you like to switch back to project search?",
  "processing_time_ms": 581
}
```

**UI rendering:**
- Show the answer
- Show an inline prompt with two buttons:
  - **"Switch to Project Search"** → switches mode to RAG, re-sends query with `search_mode` omitted (default RAG)
  - **"Stay in Document"** → keeps docqa mode, user can rephrase

**Angular snippet:**
```typescript
if (response.suggest_switch === 'rag') {
  this.showSwitchPrompt.set({
    message: response.answer,
    originalQuery: request.query,
  });
}

acceptSwitch(originalQuery: string): void {
  this.mode.set('rag');
  this.selectedDoc.set(null);
  // Re-send the original query in RAG mode
  this.query({
    query: originalQuery,
    project_id: this.projectId(),
    session_id: this.sessionId()!,
  }).subscribe(/* ... */);
}
```

---

### STORY 8: User Types "Back to Project Search" (Exit DocQA)

**User action:** Clicks "Back to Project Search" button OR types "back to search", "exit document", etc.
**Goal:** Return to RAG mode.

**Option A — Button Click (preferred, deterministic):**

Just reset local state — no API call needed:
```typescript
switchToRag(): void {
  this.mode.set('rag');
  this.selectedDoc.set(null);
  // show system message in chat
  this.messages.update(m => [...m, {
    role: 'system',
    content: 'Switched back to project search.'
  }]);
}
```

**Option B — User types exit keywords:**

Server detects the exit keywords and responds:
```json
{
  "success": true,
  "active_agent": "rag",
  "search_mode": "rag",
  "answer": "Switched back to project search mode. Ask any question about your project."
}
```

UI reads `active_agent: "rag"` from response and updates mode badge.

**Exit keywords detected server-side:**
- "back to search", "back to project"
- "exit document", "stop chatting with"
- "return to rag", "go back"
- "search project", "project search"

---

### STORY 9: Browse & Resume Past Sessions

**User action:** Clicks sidebar "Sessions" button.
**Goal:** Pick up where left off.

**API sequence:**

**9.1 List sessions:**
```http
GET http://54.197.189.113:8001/sessions
```

```json
{
  "success": true,
  "count": 3,
  "sessions": [
    {
      "session_id": "session_c3d31259c0ed",
      "created_at": 1744713220.123,
      "last_accessed": 1744714500.456,
      "message_count": 8,
      "total_tokens": 2450,
      "project_id": 2361
    }
  ]
}
```

**9.2 Load conversation when user clicks a session:**
```http
GET http://54.197.189.113:8001/sessions/session_c3d31259c0ed/conversation
```

```json
{
  "success": true,
  "session_id": "session_c3d31259c0ed",
  "message_count": 4,
  "conversation": [
    { "role": "user", "content": "What XVENT models?", "timestamp": 1744713220 },
    { "role": "assistant", "content": "...", "timestamp": 1744713228 }
  ]
}
```

**UI rendering:**
- Sidebar: list sessions with `session_id`, message count, relative time
- Active session highlighted
- Click → replace chat view with loaded conversation

---

### STORY 10: Delete a Session

**User action:** Clicks delete (X) on a session.

**API call:**
```http
DELETE http://54.197.189.113:8001/sessions/session_c3d31259c0ed
```

```json
{ "success": true, "session_id": "session_c3d31259c0ed", "deleted": true }
```

**UI rendering:**
- Optimistically remove from sidebar
- If it was the active session, clear chat view and start fresh

---

### STORY 11: Web Search Mode (Optional)

**User action:** Toggles search mode to "Web" and asks "ASHRAE HVAC standards".
**Goal:** Answer from public web (not project data).

**API call:**
```http
POST http://54.197.189.113:8001/query

{
  "query": "ASHRAE 90.1 requirements for HVAC",
  "project_id": 2361,
  "search_mode": "web"
}
```

Or use the dedicated endpoint:
```http
POST http://54.197.189.113:8001/web-search

{ "query": "ASHRAE 90.1 requirements for HVAC", "project_id": 2361 }
```

Response has `web_sources: [...]` instead of `source_documents`.

---

## 5. Complete API Reference

### 5.1 `POST /query` — Unified Query Endpoint

**This is the most important endpoint — used for ALL conversational queries (RAG + DocQA + Web + Hybrid).**

**Request:**
```typescript
interface QueryRequest {
  // Required
  query: string;                  // 1-2000 chars
  project_id: number;             // 1-999999

  // Session continuity
  session_id?: string;            // from /sessions/create
  conversation_history?: ConversationMessage[];

  // Mode selection
  search_mode?: 'rag' | 'web' | 'hybrid' | 'docqa';  // default: "rag"
  engine?: 'agentic' | 'traditional';                  // force specific engine

  // Source filtering
  filter_source_type?: 'drawing' | 'specification';
  filter_drawing_name?: string;

  // DocQA mode (only when search_mode='docqa')
  docqa_document?: {
    s3_path: string;              // required
    file_name?: string;
    download_url?: string;
  };

  // Advanced
  set_id?: number;
  generate_document?: boolean;    // default: true
}
```

**Response (same shape for all modes):**
```typescript
interface QueryResponse {
  // Core
  success: boolean;
  answer: string;                 // Natural prose, NO inline citations
  query: string;

  // Confidence
  confidence: 'high' | 'medium' | 'low';
  confidence_score: number;       // 0.0-1.0

  // Engine metadata (NEW fields marked)
  engine_used: 'agentic' | 'traditional' | 'docqa';
  fallback_used: boolean;
  agentic_confidence: string | null;
  active_agent: 'rag' | 'docqa';        // NEW — which agent answered
  suggest_switch: 'rag' | null;         // NEW — UI should offer switch
  selected_document: string | null;     // NEW — current DocQA document

  // Sources (dedup on frontend just in case)
  source_documents: SourceDoc[];
  s3_paths: string[];
  s3_path_count: number;
  web_sources: any[];

  // Follow-ups
  follow_up_questions: string[];
  improved_queries: string[];
  query_tips: string[];

  // Document discovery (fallback)
  needs_document_selection: boolean;
  available_documents: AvailableDoc[];
  scoped_to: string | null;

  // Timing
  processing_time_ms: number;

  // Session
  session_id: string | null;
  session_stats: any;

  // Debug
  model_used: string;
  token_usage: any;
  debug_info: any;
  error: string | null;
}

interface SourceDoc {
  s3_path: string;
  file_name: string;
  display_title: string;          // Prefer this for UI label
  download_url: string | null;    // HTTPS S3 URL (may be null)
  page: number | null;
  drawing_name: string;
  drawing_title: string;
  text_preview?: string;
  relevance_score?: number;
}

interface AvailableDoc {
  type: 'drawing' | 'specification';
  drawing_title: string;
  drawing_name?: string;
  section_title?: string;
  pdf_name: string;
  trade?: string;
  fragment_count: number;
}
```

### 5.2 `POST /sessions/create`

```typescript
// Request
{ project_id: number }

// Response
{ success: true, session_id: string }
```

### 5.3 `GET /sessions`

```typescript
// Response
{
  success: boolean,
  count: number,
  sessions: Array<{
    session_id: string,
    created_at: number,         // unix timestamp
    last_accessed: number,
    message_count: number,
    total_tokens: number,
    project_id: number,
  }>
}
```

### 5.4 `GET /sessions/{id}/conversation`

```typescript
// Response
{
  success: boolean,
  session_id: string,
  message_count: number,
  conversation: Array<{
    role: 'user' | 'assistant' | 'system',
    content: string,
    timestamp: number,
  }>
}
```

### 5.5 `DELETE /sessions/{id}`

```typescript
// Response
{ success: true, session_id: string, deleted: true }
```

### 5.6 `GET /health`

```typescript
// Response
{
  status: 'healthy' | 'unhealthy',
  engines: {
    agentic: { initialized: boolean },
    traditional: { faiss_loaded: boolean },
  },
  fallback_enabled: boolean,
}
```

---

## 6. E2E Testing Flow

### Run this sequence to verify your Angular integration works end-to-end.

| # | User Action | API Call | Expected UI Update |
|---|-------------|----------|-------------------|
| 1 | App loads | `GET /health` | Green dot, "Healthy" label |
| 2 | Click "New Chat" | `POST /sessions/create` | Session ID in header, empty chat |
| 3 | Type "What XVENT models?" → Send | `POST /query` | Answer + source cards + follow-ups |
| 4 | Click "Chat with Document" on M-401 card | `POST /query` (docqa mode) | Mode badge turns purple, "DocQA Mode" |
| 5 | Wait 15-20s | *(server processing)* | Loading spinner, then answer |
| 6 | Type "What heat pumps are shown?" | `POST /query` (docqa, no docqa_document) | Fast answer (~4s), stays in DocQA mode |
| 7 | Type "Show all HVAC across project" | `POST /query` (docqa) | Answer has `suggest_switch: "rag"` → show prompt |
| 8 | Click "Switch to Project Search" | Local state change, resend query | Mode badge back to blue, RAG answer |
| 9 | Click session in sidebar | `GET /sessions/{id}/conversation` | Chat history loads |
| 10 | Click X on session | `DELETE /sessions/{id}` | Session removed from sidebar |

### Critical Checkpoints

**CHECK 1 — Answer format is natural prose:**
- Answer text should NOT contain `[Source: ...]`, `---`, or `Direct answer:` headers
- Sources should appear ONLY in the card UI, not in answer text

**CHECK 2 — Mode switching works:**
- When response has `active_agent: "docqa"`, UI shows purple badge + "Back to Search" button
- When response has `suggest_switch: "rag"`, UI shows the switch prompt inline

**CHECK 3 — DocQA follow-ups are fast:**
- First DocQA query: ~17s (expected — S3 download + PDF processing)
- Follow-ups in same session: ~4s (NO `docqa_document` needed in request)

**CHECK 4 — Sources deduplicated on frontend:**
- Use the deduplication snippet from Story 4 for `available_documents`
- `source_documents` already deduplicated server-side

---

## 7. Error Handling & Edge Cases

### Common Error Scenarios

| Scenario | Response | UI Should Show |
|----------|----------|----------------|
| Backend down | Network error | "Backend unreachable" + retry button |
| Invalid `project_id` | HTTP 422 | "Invalid project ID" |
| Query too long (>2000 chars) | HTTP 422 | "Query too long" + char counter |
| Session expired | HTTP 404 on session ops | "Session expired — start new chat" |
| DocQA with no document selected | `needs_document_selection: true` | "Please select a document first" + link to sources |
| S3 download fails | `success: false, error: "S3 download failed"` | "Couldn't load document. Try another." |
| DocQA agent down | `success: false, error: "Document QA agent not running"` | "Document service unavailable" |
| DocQA timeout (>120s) | `success: false, error: "timed out"` | "Document too large, try a smaller one" |

### Angular Error Interceptor Pattern

```typescript
@Injectable()
export class ApiErrorInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    return next.handle(req).pipe(
      catchError((error: HttpErrorResponse) => {
        if (error.status === 0) {
          this.toast.error('Backend unreachable');
        } else if (error.status === 422) {
          this.toast.error('Invalid request: ' + error.error?.detail);
        } else if (error.status === 404 && req.url.includes('/sessions/')) {
          this.toast.warn('Session expired');
          this.sessionService.clearSession();
        }
        return throwError(() => error);
      })
    );
  }
}
```

### Handling 200 OK with `success: false`

Many errors return **HTTP 200** with `success: false` in the body (not HTTP 4xx/5xx). Always check the `success` field:

```typescript
query(body: QueryRequest): Observable<QueryResponse> {
  return this.http.post<QueryResponse>(`${this.base}/query`, body).pipe(
    tap(response => {
      if (!response.success) {
        this.toast.error(response.error || 'Query failed');
      }
    })
  );
}
```

---

## 8. TypeScript Types

Drop this into `src/app/models/rag-api.types.ts`:

```typescript
// ============================================================
// Enums
// ============================================================

export type SearchMode = 'rag' | 'web' | 'hybrid' | 'docqa';
export type ActiveAgent = 'rag' | 'docqa';
export type Confidence = 'high' | 'medium' | 'low';
export type Engine = 'agentic' | 'traditional' | 'docqa';
export type DocType = 'drawing' | 'specification';

// ============================================================
// Request Types
// ============================================================

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface DocQADocument {
  s3_path: string;
  file_name?: string;
  download_url?: string;
}

export interface QueryRequest {
  query: string;
  project_id: number;
  session_id?: string;
  search_mode?: SearchMode;
  engine?: 'agentic' | 'traditional';
  filter_source_type?: DocType;
  filter_drawing_name?: string;
  docqa_document?: DocQADocument;
  set_id?: number;
  conversation_history?: ConversationMessage[];
  generate_document?: boolean;
}

// ============================================================
// Response Types
// ============================================================

export interface SourceDoc {
  s3_path: string;
  file_name: string;
  display_title: string;
  download_url: string | null;
  page: number | null;
  drawing_name: string;
  drawing_title: string;
  text_preview?: string;
  relevance_score?: number;
}

export interface AvailableDoc {
  type: DocType;
  drawing_title: string;
  drawing_name?: string;
  section_title?: string;
  pdf_name: string;
  trade?: string;
  fragment_count: number;
}

export interface QueryResponse {
  // Core
  success: boolean;
  query: string;
  answer: string;
  error: string | null;

  // Confidence
  confidence: Confidence;
  confidence_score: number;

  // Agent state (NEW)
  engine_used: Engine;
  fallback_used: boolean;
  active_agent: ActiveAgent;
  suggest_switch: 'rag' | null;
  selected_document: string | null;
  agentic_confidence: string | null;

  // Sources
  source_documents: SourceDoc[];
  s3_paths: string[];
  s3_path_count: number;
  web_sources: any[];

  // Follow-ups
  follow_up_questions: string[];
  improved_queries: string[];
  query_tips: string[];

  // Document discovery
  needs_document_selection: boolean;
  available_documents: AvailableDoc[];
  scoped_to: string | null;

  // Session
  session_id: string | null;
  session_stats: any;

  // Metadata
  search_mode: SearchMode;
  processing_time_ms: number;
  model_used: string;
  token_usage: any;
  retrieval_count: number;
  s3_path_count: number;
  debug_info: any;
}

export interface Session {
  session_id: string;
  created_at: number;
  last_accessed: number;
  message_count: number;
  total_tokens: number;
  project_id: number;
}

export interface ConversationTurn {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  engines: {
    agentic: { initialized: boolean };
    traditional: { faiss_loaded: boolean };
  };
  fallback_enabled: boolean;
}
```

---

## 9. Quick Reference Card

### The One Endpoint You'll Call Most: `POST /query`

```typescript
// RAG mode (default)
{ query, project_id, session_id }

// Force web-only
{ query, project_id, session_id, search_mode: 'web' }

// Hybrid (RAG + Web)
{ query, project_id, session_id, search_mode: 'hybrid' }

// DocQA first query (upload + ask)
{
  query, project_id, session_id,
  search_mode: 'docqa',
  docqa_document: { s3_path, file_name }
}

// DocQA follow-up (cached session)
{ query, project_id, session_id, search_mode: 'docqa' }
```

### Response Fields to Check First

1. `success` — false means error, check `error` field
2. `active_agent` — `"rag"` or `"docqa"`, drives UI mode badge
3. `suggest_switch` — if `"rag"`, show switch prompt
4. `needs_document_selection` — if true, render `available_documents` picker
5. `answer` — natural prose, render as-is
6. `source_documents` — render as cards with "Chat with Document" buttons
7. `follow_up_questions` — render as clickable chips

### "Will My Old Code Still Work?"

**YES** — unless you were parsing `[Source:]` tags from the answer text. Old code that reads `answer`, `source_documents`, `follow_up_questions`, and `confidence` continues to work unchanged. New fields (`active_agent`, `suggest_switch`, `selected_document`) are additive — safe to ignore if you don't want the DocQA feature.

---

## 10. Contact / Issues

- Backend team (Aniruddha): See git log for changes
- Sandbox logs: `/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/agent.log`
- Full E2E test report: `test_results/e2e_summary_20260417_155619.md`
- Demo UI reference implementation: `docs/demo-ui.html` (vanilla JS)

**When opening a bug:** Include the request body, response body, session_id, and timestamp. The backend logs all requests with timing info.
