# Development Plan: RAG-to-Document QA Agent Bridge

**Date:** 2026-04-16
**Author:** JARVIS + Aniruddha
**Status:** AWAITING CONFIRMATION

---

## 1. User Story

**As a** Project Engineer
**I want** to ask questions, receive answers with reference documents, and then directly interact with those documents within the same session
**So that** I can validate, explore, and extract deeper insights without losing context or switching tools

---

## 2. Clarifying Q&A (Confirmed)

| # | Question | Answer |
|---|----------|--------|
| Q1 | Document delivery to DocQA | **Option A**: Download PDF from S3 via `s3_path` -> upload to DocQA `/api/converse` |
| Q2 | Answer format | Natural prose (bullet points, paragraphs). NO inline citations/sources/headers. Sources in UI cards only. Follow-ups as UI chips only. |
| Q3 | Session architecture | **Option A**: Single RAG session owns the flow. Orchestrator calls DocQA internally. User sees one continuous chat. |
| Q4 | Agent switching | **Option C (Hybrid)**: Manual "Chat with Document" button + auto-detect project-wide questions in DocQA mode with user prompt. |
| Q5 | Document selection display | Use **raw S3 key** (e.g., `0104202614084657M401MECHANICALROOFPLAN1-1.pdf`) to pick the document. Also fix `display_title` being null (root-cause investigation). |
| Q6 | download_url / SSL | Not actionable now — user lacks access to iField S3 bucket. Skip download_url for now. Use `s3_path` directly. |
| Q7 | Testing scope | **Functional E2E testing** only. Does the flow work end-to-end? |

---

## 3. Architecture

### 3.1 System Overview

```
User -> Demo UI (localhost:4200)
          |
          |--- POST /query (or /query/stream)
          |         |
          v         v
     RAG Agent (port 8001)
          |
          |--- MongoDB: drawingVision, drawing, specification
          |--- Returns: answer + source_documents[]
          |
          |--- IF user selects document -> "Chat with Document"
          |         |
          |         v
          |--- POST /query with mode="docqa"
          |         |
          |         v
          |    RAG Orchestrator internally:
          |      1. Downloads PDF from S3 using s3_path
          |      2. Uploads to DocQA Agent (port 8006) /api/converse
          |      3. Returns DocQA response through RAG session
          |
          |--- IF user asks project-wide question while in DocQA mode
          |         |
          |         v
          |    Intent Classifier detects "project scope" intent
          |    -> Suggests switching back to RAG mode
          |
     Document QA Agent (port 8006)
          |--- FAISS per-session vector store
          |--- gpt-4o generation
          |--- Session with processed document context
```

### 3.2 Session State Model

```
RAG Session {
  session_id: "session_abc123"
  project_id: 2361
  active_agent: "rag" | "docqa"        // NEW: tracks which agent is active
  docqa_session_id: null | "def456"     // NEW: DocQA session ID (created on first document selection)
  selected_document: null | {           // NEW: currently selected document
    s3_path: "0104202614084657M401..."
    file_name: "M401MECHANICALROOFPLAN1-1.pdf"
    display_title: "M-401 Mechanical Roof Plan"
  }
  conversation_history: [...]
}
```

### 3.3 Query Flow Decision Tree

```
User sends query
    |
    v
Is active_agent == "docqa"?
    |
    +--> YES: Intent classifier checks query
    |         |
    |         +--> "document-scoped" (e.g., "what notes on this page?")
    |         |      -> Forward to DocQA Agent
    |         |      -> Return DocQA response
    |         |
    |         +--> "project-wide" (e.g., "show all HVAC across project")
    |                -> Return suggestion: "This seems project-wide. Switch to RAG?"
    |                -> If user confirms -> switch active_agent to "rag", re-query
    |
    +--> NO: Normal RAG query
              -> Agentic engine -> MongoDB
              -> Return answer + source_documents
```

### 3.4 Document Selection -> DocQA Activation Flow

```
1. RAG returns source_documents[] in response
2. UI shows source cards with "Chat with Document" button
3. User clicks button on a specific document
4. UI sends: POST /query {
     query: "Process this document for Q&A",
     session_id: "session_abc123",
     search_mode: "docqa",              // NEW mode
     docqa_document: {                  // NEW field
       s3_path: "0104202614084657M401...",
       file_name: "M401MECHANICALROOFPLAN1-1.pdf"
     }
   }
5. RAG Orchestrator:
   a. Downloads PDF from S3 using boto3 + s3_path
   b. Uploads to DocQA Agent: POST http://localhost:8006/api/converse
      - multipart: files=[downloaded_pdf], query="Process document"
   c. Stores docqa_session_id in RAG session
   d. Sets active_agent = "docqa"
   e. Returns: { answer: "Document processed. You can now ask questions.",
                 active_agent: "docqa", selected_document: {...} }
6. UI shows: "Document processed. You can now ask questions."
   + "Back to Project Search" button
```

---

## 4. Changes Required

### Phase 0: Fix RAG Answer Format (Prompt Only)

**File:** `agentic/core/agent.py` (lines 186-190)

**Change:** Rewrite ANSWER FORMAT section in system prompt:
```
Before:  "Direct answer first, Supporting details with exact quotes,
          [Source: drawing_name] citations, 3 follow-up suggestions"

After:   "Answer naturally as a knowledgeable construction professional.
          Use bullet points or paragraphs as appropriate.
          Do NOT include source citations, references, or headers like 'Direct answer'.
          Do NOT add '---' separators, 'Citations:', 'Supporting Details:', or 'Note:' sections.
          Just answer the question clearly and completely."
```

**Files modified:** 1 (`agentic/core/agent.py`)
**Risk:** Low — prompt change only, no code logic change.

---

### Phase 1: RAG -> DocQA Bridge in Orchestrator

**New/Modified Files:**

| File | Change |
|------|--------|
| `gateway/models.py` | Add `search_mode: "docqa"` enum, add `docqa_document` field to `QueryRequest` |
| `gateway/orchestrator.py` | Add `_run_docqa()` method, add DocQA session tracking, add S3 download logic |
| `gateway/docqa_client.py` | **NEW** — HTTP client to call DocQA Agent API (upload + query) |
| `shared/s3_client.py` | **NEW** — Boto3 S3 download helper (download to temp file) |

**Key implementation:**

```python
# gateway/docqa_client.py (NEW)
class DocQAClient:
    """HTTP client for Document QA Agent at port 8006."""
    BASE_URL = "http://localhost:8006"

    async def upload_and_query(self, file_path, file_name, query, session_id=None):
        """Upload a document and ask a question."""
        # POST /api/converse with multipart form
        ...

    async def query(self, session_id, query):
        """Ask a follow-up question on an existing DocQA session."""
        # POST /api/chat with JSON body
        ...
```

```python
# In orchestrator.py — new method
async def _run_docqa(self, query, session_id, docqa_document, ...):
    """Route query to Document QA Agent."""
    # 1. If no docqa_session_id yet: download PDF from S3, upload to DocQA
    # 2. If docqa_session_id exists: forward query to DocQA chat endpoint
    # 3. Return DocQA response through RAG session
    ...
```

---

### Phase 2: Intent Detection for Agent Switching

**New File:** `gateway/intent_classifier.py`

**Logic:** Simple keyword + heuristic classifier (no LLM call needed):

```python
PROJECT_WIDE_SIGNALS = [
    "across project", "all drawings", "project-wide", "entire project",
    "all trades", "missing scope", "compare", "how many", "list all",
    "across all", "project summary", "total", "overall",
]

DOCUMENT_SCOPED_SIGNALS = [
    "this document", "this drawing", "this spec", "this page",
    "in this", "on this", "the selected", "current document",
    "what does it say", "page number", "section",
]

def classify_intent(query: str, active_agent: str) -> str:
    """Returns: 'rag', 'docqa', or 'suggest_switch'"""
    query_lower = query.lower()
    if active_agent == "docqa":
        if any(sig in query_lower for sig in PROJECT_WIDE_SIGNALS):
            return "suggest_switch"  # Suggest switching back to RAG
        return "docqa"  # Stay in DocQA mode
    return "rag"  # Default: RAG mode
```

**Files modified:** 1 new file + integration in orchestrator.

---

### Phase 3: Update Demo UI

**File:** `docs/demo-ui.html`

**Changes:**
- Point to **Sandbox APIs** (`http://54.197.189.113:8001`)
- Add "Chat with Document" button on each source card
- Add "Back to Project Search" button in DocQA mode
- Add visual mode indicator (RAG mode vs DocQA mode)
- Show document processing status message
- Handle `suggest_switch` response (prompt user)
- Use `s3_path` (raw S3 key) for document selection display

---

### Phase 4: Functional E2E Testing

**Test Scenarios:**

| # | Scenario | Steps | Expected |
|---|----------|-------|----------|
| T1 | Simple RAG query | Ask "What HVAC units?" | Natural prose answer, NO citations in text, sources in cards |
| T2 | RAG returns sources | Ask a question | `source_documents[]` populated with `s3_path` |
| T3 | Select document | Click "Chat with Document" on a source | "Document processed" message, mode switches to DocQA |
| T4 | DocQA query | Ask "What notes on this page?" | Answer grounded in selected document |
| T5 | Follow-up in DocQA | Ask another question | Stays in DocQA mode, answers from same document |
| T6 | Project-wide in DocQA | Ask "Show all HVAC across project" | Suggests switching back to RAG |
| T7 | Switch back to RAG | Click "Back to Project Search" | Mode switches, next query goes to RAG |
| T8 | Session continuity | Switch between modes | Conversation history preserved |

---

### Phase 5: Deploy to Sandbox VM

- SCP modified files to `54.197.189.113:/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/`
- Restart RAG agent on port 8001
- Verify DocQA agent running on port 8006
- Run E2E tests against sandbox

---

## 5. Files Changed Summary

| Phase | File | Action |
|-------|------|--------|
| 0 | `agentic/core/agent.py` | MODIFY — rewrite system prompt ANSWER FORMAT |
| 1 | `gateway/models.py` | MODIFY — add `docqa` search_mode, `docqa_document` field |
| 1 | `gateway/orchestrator.py` | MODIFY — add `_run_docqa()`, session state tracking |
| 1 | `gateway/docqa_client.py` | **NEW** — HTTP client for DocQA Agent |
| 1 | `shared/s3_client.py` | **NEW** — S3 download helper |
| 2 | `gateway/intent_classifier.py` | **NEW** — keyword-based intent classifier |
| 3 | `docs/demo-ui.html` | MODIFY — dual-agent UI with Sandbox APIs |
| 4 | `docs/E2E_TESTING_RAG_DOCQA.md` | **NEW** — test plan |

**Total: 5 modified files + 3 new files**

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| S3 download fails (no bucket access) | DocQA can't process documents | Fallback: use download_url if available, or show error |
| DocQA agent not running on sandbox | Bridge fails | Health check before attempting DocQA call |
| Intent classifier misroutes | Wrong agent answers | Hybrid approach — user always has manual switch button |
| Answer format regression | Bad UX | Test T1 specifically, lock down prompt |
| Large PDFs slow DocQA processing | Timeout | Set 120s timeout, show progress indicator |

---

## 7. What We Are NOT Doing (Explicitly Out of Scope)

- No security/auth middleware (per user request)
- No load testing
- No production deployment (sandbox only)
- No download_url SSL (no S3 bucket access yet)
- No multi-document selection in single DocQA session (one document at a time for v1)
- No display_title fix (using raw S3 key per user preference)

---

## 8. Execution Order

```
Phase 0 (30 min)  -> Fix answer format prompt
Phase 1 (3-4 hrs) -> Build RAG->DocQA bridge
Phase 2 (1-2 hrs) -> Intent detection
Phase 3 (2-3 hrs) -> Update demo UI
Phase 4 (1-2 hrs) -> E2E testing
Phase 5 (30 min)  -> Deploy to sandbox
```

**WAITING FOR CONFIRMATION: Proceed with this plan? (yes / modify / different approach)**
