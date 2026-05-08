# Angular UI Integration — Sandbox APIs in User-Story Order

**Sandbox base URL:** `https://ai6.ifieldsmart.com/rag` ← **use this** (HTTPS, behind nginx)
**Direct VM port (debug only):** `http://54.197.189.113:8001` — do NOT hit from production UI; firewalled in some networks and HTTP-only.
**Auth:** none (open for now)


> **Why the `/rag/` prefix?** The sandbox VM runs nginx as an SSL-terminating reverse proxy (`ai6.ifieldsmart.com` → port 8001). The proxy routes by URL prefix:
>
> | Prefix | Port | Agent |
> |---|---:|---|
> | `/rag/*`         | 8001 | unified-rag-agent (this doc) |
> | `/docqa/*`       | 8006 | document-qa-agent (used internally by the bridge — UI rarely calls direct) |
> | `/construction/*`| 8003 | construction-intelligence-agent |
> | `/sql/*`         | 8002 | sql-intelligence-agent |
> | `/ingestion/*`   | 8004 | ingestion pipeline |
> | `/auth/*`        | 8010 | auth-service (placeholder) |
>
> nginx **strips the prefix** before forwarding, so a request to `https://ai6.ifieldsmart.com/rag/query` lands at the gateway as `POST /query`. Existing client code that hits the raw VM port (`http://...:8001/query`) still works for debugging but should NOT be used from the Angular UI in production — HTTPS only.
**Content-Type:** `application/json` for every POST except `/query/stream` (SSE)

This doc walks the **user story step-by-step**. Each step shows the exact request body the Angular client should send and the response shape to expect. Code samples are HttpClient TypeScript so you can paste them directly.

---

## TypeScript types (paste into `src/app/models/rag.types.ts`)

```typescript
export interface QueryRequest {
  query: string;
  project_id: number;
  session_id?: string | null;
  search_mode?: 'rag' | 'docqa' | 'web' | 'hybrid' | null;
  mode_hint?: 'rag' | 'docqa' | null;
  docqa_document?: DocQADocumentRef | null;
  conversation_history?: any[] | null;
  set_id?: number | null;
  filter_drawing_name?: string | null;
}

export interface DocQADocumentRef {
  s3_path: string;
  file_name: string;
  download_url: string;
  pdf_name?: string;
}

export interface SourceDocument {
  s3_path: string;
  file_name: string;
  display_title?: string;
  download_url: string;
  pdf_name?: string;
  page?: number;
  drawing_name?: string;
  drawing_title?: string;
}

export interface UnifiedResponse {
  success: boolean;
  answer: string;
  source_documents: SourceDocument[] | null;
  confidence: 'high' | 'medium' | 'low';
  session_id: string;
  active_agent: 'rag' | 'docqa';
  engine_used: 'agentic' | 'traditional' | 'docqa' | 'docqa_fallback' | 'classifier';
  needs_clarification: boolean;
  clarification_prompt?: string | null;
  selected_document?: DocQADocumentRef | null;
  docqa_session_id?: string | null;
  fallback_used?: boolean;
  follow_up_questions?: string[];
  groundedness_score?: number | null;
  // legacy
  sources?: any[];
  cost_usd?: number;
  elapsed_ms?: number;
  total_steps?: number;
  model?: string;
}
```

---

## Step 0 — App startup: health probe (optional but recommended)

**When:** before mounting the chat component.
**Why:** ensures the gateway is reachable so the UI can disable the chat box with a clear message instead of failing silently.

### `GET /health`

**Request:** none
**Response 200:**
```json
{
  "status": "healthy",
  "engines": {
    "agentic":     { "initialized": true },
    "traditional": { "faiss_loaded": false }
  },
  "fallback_enabled": true
}
```

```typescript
// rag.service.ts
private base = 'https://ai6.ifieldsmart.com/rag';
checkHealth() { return this.http.get<any>(`${this.base}/health`); }
```

---

## Step 1 — User asks first question (plain RAG)

**User story:** *"User asks a question and receives an answer with reference documents."*

### `POST /query`

**Request body (minimum):**
```json
{
  "query": "List HVAC drawings in the project",
  "project_id": 7222
}
```

> `session_id` is optional on first call — gateway creates one and returns it. **Save the returned `session_id` and use it on every subsequent request to keep history.**

**Response (success, RAG path):**
```json
{
  "success": true,
  "answer": "The HVAC drawings include M-101 mechanical first floor plan, M-201 second floor plan, M-301 mechanical schedules ...",
  "source_documents": [
    {
      "s3_path": "ifieldsmart/jrparkwayhotel.../Drawings/pdf251...",
      "file_name": "JRpkwyHotelMechPlans32-1",
      "display_title": "MECHANICAL FIRST FLOOR PLAN",
      "download_url": "https://ifieldsmart.s3.amazonaws.com/...?X-Amz-Signature=...",
      "pdf_name":   "JRpkwyHotelMechPlans32-1",
      "page": 3
    },
    { "...": "more sources" }
  ],
  "confidence":   "high",
  "session_id":   "rag_abc123",
  "active_agent": "rag",
  "engine_used":  "agentic",
  "follow_up_questions": ["What about plumbing?", "Show me the riser diagram"],
  "needs_clarification": false
}
```

```typescript
// chat.component.ts
sendQuery(text: string) {
  const body: QueryRequest = {
    query: text,
    project_id: this.projectId,
    session_id: this.sessionId   // null on the very first call
  };
  this.rag.query(body).subscribe(resp => {
    this.sessionId = resp.session_id;            // persist for whole session
    this.activeAgent = resp.active_agent;
    if (resp.needs_clarification) { this.showClarify(resp); return; }
    this.renderAnswer(resp);
    this.renderSources(resp.source_documents);   // dedupe + cards
  });
}
```

**UI rendering rules (cards):**
- Dedupe `source_documents` by `s3_path`. Same drawing returned twice → one card with pages aggregated (`pages: [3, 14, 22]`).
- Primary label = `file_name` (S3 basename). Subtitle = `display_title` only when distinct from primary.
- "Download" link → `<a href="{{download_url}}" target="_blank">`.
- "Chat with Document" button → triggers Step 2 below with the full source object.

---

## Step 2 — User clicks "Chat with Document" on a source

**User story:** *"User selects one or more reference documents from the response and clicks 'Chat with Document'. System processes the document. Show: 'Document(s) processed successfully. You can now ask questions.'"*

### `POST /query` *(same endpoint, different body)*

**Request body:**
```json
{
  "query": "Give me a brief overview of this document.",
  "project_id": 7222,
  "session_id": "rag_abc123",
  "search_mode": "docqa",
  "docqa_document": {
    "s3_path": "ifieldsmart/jrparkwayhotel.../Drawings/pdf251...",
    "file_name": "JRpkwyHotelMechPlans32-1",
    "download_url": "https://ifieldsmart.s3.amazonaws.com/...?X-Amz-Signature=...",
    "pdf_name": "JRpkwyHotelMechPlans32-1"
  }
}
```

**Response (success, DocQA path):**
```json
{
  "success": true,
  "answer": "This drawing covers the level-32 mechanical HVAC plan. Key items include: ...",
  "source_documents": [
    { "file_name": "JRpkwyHotelMechPlans32-1", "page": 3, "snippet": "..." }
  ],
  "active_agent":      "docqa",
  "selected_document": { "file_name": "JRpkwyHotelMechPlans32-1", "s3_path": "..." },
  "docqa_session_id":  "dq_xyz789",
  "engine_used":       "docqa",
  "confidence":        "high",
  "session_id":        "rag_abc123",
  "groundedness_score": 0.92
}
```

**Response (graceful fallback — DocQA unreachable or doc unindexable):**
```json
{
  "success": false,
  "answer": "Could not load document for deep-dive. Try selecting a different source or ask a general question to continue.",
  "active_agent":   "rag",
  "engine_used":    "docqa_fallback",
  "fallback_used":  true,
  "session_id":     "rag_abc123"
}
```

```typescript
// chat.component.ts
chatWithDocument(src: SourceDocument) {
  this.activeAgent = 'docqa';
  this.selectedDoc = src;
  this.appendSystemMsg(`Processing ${src.file_name}…`);

  const body: QueryRequest = {
    query: 'Give me a brief overview of this document.',
    project_id: this.projectId,
    session_id: this.sessionId,
    search_mode: 'docqa',
    docqa_document: {
      s3_path: src.s3_path,
      file_name: src.file_name,
      download_url: src.download_url,
      pdf_name: src.pdf_name ?? src.file_name
    }
  };
  this.rag.query(body).subscribe(resp => {
    if (resp.engine_used === 'docqa_fallback') {
      this.appendSystemMsg(`Could not load: ${resp.answer}`);
      this.activeAgent = 'rag';
      this.selectedDoc = null;
      return;
    }
    this.appendSystemMsg('Document processed. You can now ask questions.');
    this.docqaSessionId = resp.docqa_session_id!;
    this.renderAnswer(resp);
  });
}
```

---

## Step 3 — User asks follow-up questions on the document

**User story:** *"User starts asking questions. Document QA Agent answers strictly based on selected document(s) and provides page references, extracted snippets, context-grounded answers."*

### `POST /query` *(same shape as Step 2 — keep `search_mode` and `docqa_document`)*

**Request body:**
```json
{
  "query": "Where is fire damper mentioned in this drawing?",
  "project_id": 7222,
  "session_id": "rag_abc123",
  "search_mode": "docqa",
  "docqa_document": {
    "s3_path": "...",
    "file_name": "JRpkwyHotelMechPlans32-1",
    "download_url": "...",
    "pdf_name": "JRpkwyHotelMechPlans32-1"
  }
}
```

> The bridge **reuses `docqa_session_id`** (no re-upload) because the same `s3_path` is already in the session's `selected_documents`. Verified: `docqa_session_id` in the response is the **same** value as Step 2.

**Response:** identical shape to Step 2 success response, with the new answer scoped to the selected document.

```typescript
sendDocQAFollowup(text: string) {
  this.rag.query({
    query: text,
    project_id: this.projectId,
    session_id: this.sessionId,
    search_mode: 'docqa',
    docqa_document: this.selectedDoc!
  }).subscribe(resp => this.renderAnswer(resp));
}
```

---

## Step 4 — User asks a project-wide question → auto-switch back to RAG

**User story:** *"User asks: Show all missing scope in HVAC across project. System switches back to RAG Agent."*

### `POST /query` *(omit `search_mode` — let the classifier decide)*

**Request body:**
```json
{
  "query": "Show me all missing scope across the project",
  "project_id": 7222,
  "session_id": "rag_abc123"
}
```

> Phase 3 classifier sees "across the project" → +0.9 project-wide override → routes to RAG even though a doc is selected.

**Response:**
```json
{
  "success": true,
  "answer": "The following scope items are missing or incomplete across all project drawings: ...",
  "source_documents": [ /* multiple */ ],
  "active_agent": "rag",
  "engine_used":  "agentic",
  "session_id":   "rag_abc123",
  "selected_document": null
}
```

**UI behavior:** flip badge from DocQA → RAG. The selected_document is automatically cleared from the response (or you can clear `this.selectedDoc` when `active_agent === 'rag'`).

```typescript
// In your generic sendQuery() handler
if (resp.active_agent === 'rag' && this.activeAgent === 'docqa') {
  this.activeAgent = 'rag';
  this.selectedDoc = null;
  this.appendSystemMsg('Switched back to project-wide search.');
}
```

---

## Step 5 — Ambiguous query → Clarify prompt

**User story:** *"If query partially relates to both, system may ask clarification or combine context."*

### `POST /query` *(ambiguous pronoun + selected doc still in session)*

**Request body:**
```json
{
  "query": "is it missing?",
  "project_id": 7222,
  "session_id": "rag_abc123"
}
```

**Response (classifier confidence 0.3-0.7):**
```json
{
  "success": true,
  "answer": "",
  "needs_clarification":  true,
  "clarification_prompt": "Should I answer from the selected document (JRpkwyHotelMechPlans32-1) or search the whole project?",
  "active_agent":         "rag",
  "selected_document":    { "file_name": "JRpkwyHotelMechPlans32-1", "s3_path": "..." },
  "engine_used":          "classifier",
  "session_id":           "rag_abc123"
}
```

**UI rendering:** show a yellow/info banner with the `clarification_prompt` and two buttons. On either button click, **resubmit the same query** with `mode_hint`.

```typescript
// On "Answer from selected document"
resubmitWithMode(originalQuery: string, mode: 'rag' | 'docqa') {
  const body: QueryRequest = {
    query: originalQuery,
    project_id: this.projectId,
    session_id: this.sessionId,
    mode_hint: mode
  };
  if (mode === 'docqa' && this.selectedDoc) {
    body.search_mode = 'docqa';
    body.docqa_document = this.selectedDoc;
  }
  this.rag.query(body).subscribe(resp => this.renderAnswer(resp));
}
```

---

## Step 6 — Multi-document selection (optional power-user flow)

**User story:** *"User can select multiple documents and ask combined questions."*

Same `POST /query` endpoint. Call it **once per document** with `search_mode: 'docqa'` — the bridge calls DocQA's `/api/converse` N times with the **same `docqa_session_id`** so DocQA stores all docs in one session. Then ask combined questions normally.

```typescript
async loadMultipleDocs(srcs: SourceDocument[]) {
  for (const s of srcs) {
    await firstValueFrom(this.rag.query({
      query: 'load',          // any placeholder; answer is ignored
      project_id: this.projectId,
      session_id: this.sessionId,
      search_mode: 'docqa',
      docqa_document: { s3_path: s.s3_path, file_name: s.file_name,
                        download_url: s.download_url,
                        pdf_name: s.pdf_name ?? s.file_name }
    }));
  }
  // now user asks anything in DocQA mode and DocQA answers across all loaded docs
}
```

---

## Step 7 — Restore session on page reload (optional)

When the Angular app remounts (browser refresh), use the saved `session_id` to restore conversation history.

### `GET /sessions/{session_id}/conversation`

**Response 200:**
```json
{
  "session_id": "rag_abc123",
  "messages": [
    { "role": "user", "content": "list HVAC drawings", "timestamp": "..." },
    { "role": "assistant", "content": "...", "source_documents": [ "..." ] },
    { "role": "user", "content": "...", "timestamp": "..." }
  ]
}
```

```typescript
restoreSession(sessionId: string) {
  return this.http.get<any>(`${this.base}/sessions/${sessionId}/conversation`);
}
```

### `GET /sessions/{session_id}/stats` *(optional, for an "engine mix" sidebar)*

Returns `{ total_messages, engine_used: { agentic: 5, docqa: 3 }, active_agent, docqa_session_id }`.

### `DELETE /sessions/{session_id}` *(end-of-conversation cleanup)*

```typescript
endSession() { return this.http.delete(`${this.base}/sessions/${this.sessionId}`); }
```

---

## Streaming variant (optional — for token-by-token UX)

### `POST /query/stream` *(SSE)*

Same body as `/query`. Response is `text/event-stream` with `data: {…}\n\n` chunks ending in `data: [DONE]\n\n`. Use `EventSource` polyfill (Angular `HttpClient` doesn't support SSE natively):

```typescript
streamQuery(body: QueryRequest) {
  return new Observable<any>(observer => {
    fetch(`${this.base}/query/stream`,  // nginx has proxy_buffering off for /rag/query/stream {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(async r => {
      const reader = r.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) { observer.complete(); break; }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop()!;
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') return observer.complete();
          observer.next(JSON.parse(payload));
        }
      }
    }).catch(err => observer.error(err));
  });
}
```

---

## Summary — sequential API calls per user-story flow

| # | User action | Endpoint | Body shape | UI badge |
|---|---|---|---|---|
| 0 | App boots | `GET /health` (full: `GET https://ai6.ifieldsmart.com/rag/health`) | — | (verify reachable) |
| 1 | Asks first question | `POST /query` | `{query, project_id}` | RAG |
| 2 | Clicks "Chat with Document" | `POST /query` | `{query, project_id, session_id, search_mode:'docqa', docqa_document}` | DocQA |
| 3 | DocQA followup | `POST /query` | same as #2 (reuses bridge session) | DocQA |
| 4 | "Show all across project" | `POST /query` | `{query, project_id, session_id}` (no search_mode) | flips → RAG |
| 5 | Ambiguous pronoun | `POST /query` | `{query, project_id, session_id}` | shows clarify banner |
| 5b | User picks "from doc" | `POST /query` | `{query, …, mode_hint:'docqa', search_mode:'docqa', docqa_document}` | DocQA |
| 5c | User picks "whole project" | `POST /query` | `{query, …, mode_hint:'rag'}` | RAG |
| 6 | Multi-doc select | `POST /query` × N | one per doc with `search_mode:'docqa'` | DocQA |
| 7 | Page reload | `GET /sessions/{id}/conversation` | — | restore history |
| End | Session cleanup | `DELETE /sessions/{id}` | — | — |

---

## Behavioral guidance for Angular team

1. **One `session_id` per chat session.** Save it after first response, send on every subsequent request. Survives RAG↔DocQA mode switches.
2. **`docqa_session_id` is internal** — the gateway tracks it. UI doesn't need to send it; just keep sending the same `docqa_document` while in DocQA mode.
3. **Always send the full `docqa_document` object** in DocQA-mode requests (not just an ID). The bridge needs `download_url` for the S3 fallback path.
4. **Presigned URLs expire in 1 hour.** Don't cache `download_url` across long sessions — always use the latest from the most recent `/query` response. The gateway re-signs every time.
5. **`engine_used: docqa_fallback` ≠ failure** — render the `answer` field as a system message and flip the badge back to RAG. User can still continue the conversation.
6. **Dedupe sources by `s3_path`**, aggregate `page` numbers. Browser `<a href>` clicks on `download_url` work — they trigger GET (not HEAD).
7. **Schema is forward-compatible** — ignore unknown fields silently. Future phases may add fields without breaking existing clients.

---

## Quick test from the browser console

Paste at `https://ai6.ifieldsmart.com` (HTTPS works from any browser tab — no CORS preflight needed since the gateway is open) or while developing locally:

```javascript
// Step 1: plain RAG
let r = await fetch('https://ai6.ifieldsmart.com/rag/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ query: 'list HVAC drawings', project_id: 7222 })
}).then(r => r.json());
console.log('RAG:', r.engine_used, r.source_documents.length, 'sources');

// Step 2: chat with first doc
const src = r.source_documents[0];
let r2 = await fetch('https://ai6.ifieldsmart.com/rag/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: 'overview please',
    project_id: 7222,
    session_id: r.session_id,
    search_mode: 'docqa',
    docqa_document: { s3_path: src.s3_path, file_name: src.file_name,
                      download_url: src.download_url, pdf_name: src.file_name }
  })
}).then(r => r.json());
console.log('DocQA:', r2.engine_used, r2.docqa_session_id, '\n', r2.answer.slice(0,150));
```
