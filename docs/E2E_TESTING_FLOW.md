# Unified RAG Agent — End-to-End UI Testing Flow

**Purpose:** Step-by-step testing script for QA and frontend developers to validate every user story against the live API.

**Base URL:** `https://ai5.ifieldsmart.com/rag`
**Test Project ID:** `2361` (use your project with uploaded drawings + specifications)

---

## Pre-requisites

1. Open `demo-ui.html` in Chrome/Edge (or your Angular app)
2. Have a valid API key ready
3. Project `2361` (or your test project) must have documents uploaded in MongoDB
4. Keep browser DevTools open (Network tab) to verify API calls alongside the UI

---

## Test Flow Overview

```
TEST 1: App Boot & Health ─────────────────────── (2 min)
TEST 2: Project Load & Document Discovery ─────── (3 min)
TEST 3: Simple Direct Question ────────────────── (3 min)
TEST 4: Question Requiring Document Selection ──── (5 min)
TEST 5: Follow-up / Multi-turn Conversation ───── (3 min)
TEST 6: Document Scoping (Manual) ─────────────── (4 min)
TEST 7: Clear Scope & Full Project Search ─────── (2 min)
TEST 8: Source Type Filtering ─────────────────── (3 min)
TEST 9: Web Search & Hybrid Mode ──────────────── (4 min)
TEST 10: Session History & Resume ──────────────── (4 min)
TEST 11: Session Management (Create/Delete) ────── (3 min)
TEST 12: Engine Forcing & Fallback ─────────────── (4 min)
TEST 13: Edge Cases & Error Handling ───────────── (5 min)
TEST 14: Streaming vs Non-Streaming ────────────── (3 min)
                                          TOTAL: ~48 min
```

---

## TEST 1: App Boot & Health Check

**Story:** App initialization — verify backend is alive before showing UI.

**APIs tested:** `GET /health`, `GET /`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1.1 | Open the demo UI in browser | Page loads, health dot is RED initially |
| 1.2 | Wait 2-3 seconds | Health dot turns GREEN |
| 1.3 | Check API Log panel (right side) | Two entries: `GET /health` (200) and `GET /` (200) — both tagged `S1:Boot` |
| 1.4 | Verify health label | Shows: `Healthy | Agentic: ON | Traditional: ON` (or `standby` for traditional) |

### What if it fails?

| Symptom | Likely Cause |
|---------|-------------|
| Health dot stays RED | Backend is down or CORS is blocking |
| `Traditional: standby` | Normal — FAISS lazy-loads on first fallback query |
| Network error in console | Check if `https://ai5.ifieldsmart.com/rag/health` is reachable |

---

## TEST 2: Project Load & Document Discovery

**Story:** User selects a project and sees available documents.

**APIs tested:** `GET /projects/{id}/documents`, `POST /sessions/create`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 2.1 | Enter API key in the top bar | Key field is filled |
| 2.2 | Enter project ID: `2361` | Project ID field shows `2361` |
| 2.3 | Click **"Load Project"** | Two API calls fire in PARALLEL (check API Log) |
| 2.4 | Verify left sidebar "Project Documents" section | Documents appear, grouped with blue `drawing` and amber `specification` badges |
| 2.5 | Verify left sidebar "Sessions" section | New session appears (e.g., `session_7b007fb191b2`) |
| 2.6 | Verify chat area | System messages: "Loading project 2361...", "Found X documents...", "Session created: session_..." |
| 2.7 | Check API Log | `S2:Docs GET /projects/2361/documents` (200) and `S2:Session POST /sessions/create` (200) |

### Verify document list contains

- At least some **drawings** (e.g., `M-100`, `M-401`, `E-003`)
- At least some **specifications** (e.g., `Section 23 00 00 - HVAC General`)
- Each item shows page count

---

## TEST 3: Simple Direct Question (Straightforward Answer)

**Story:** User asks a question that the agent can answer directly from the collection — no document selection needed.

**APIs tested:** `POST /query/stream` (or `POST /query`)

**Why this works without document selection:** The question is specific enough that the agentic engine finds the answer directly from the MongoDB collection without ambiguity.

### Questions to Test

| # | Question | Why It's Direct | Expected Behavior |
|---|----------|----------------|-------------------|
| 3a | `"What XVENT models are specified in the mechanical drawings?"` | Specific term "XVENT" narrows to exact documents | Direct answer with sources from M-401, M-402 |
| 3b | `"List all electrical panel schedules in the project"` | "Panel schedules" is a specific drawing type | Direct answer listing panels (AP4, AP5, etc.) |
| 3c | `"What is the total cooling capacity in tons for this project?"` | Specific engineering metric | Direct answer with tonnage and source drawings |
| 3d | `"What fire alarm devices are shown on the first floor?"` | Specific system + specific floor | Direct answer with device list |

### Steps (use question 3a)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 3.1 | Type: `What XVENT models are specified in the mechanical drawings?` | Text appears in input |
| 3.2 | Click **Send** (or press Enter) | User message bubble appears on right side |
| 3.3 | Watch for streaming response | Tokens appear one by one in assistant bubble (if SSE works) |
| 3.4 | Verify answer content | Mentions XVENT models (OHEB-44, OHEB-46, etc.) |
| 3.5 | Verify **confidence badge** | Shows `HIGH confidence (90%+)` in green |
| 3.6 | Verify **source documents** | Source cards appear: `M-401 Mechanical Roof Plan`, etc. |
| 3.7 | Verify **follow-up questions** | Clickable chips appear below answer (e.g., "What are the CFM ratings?") |
| 3.8 | Verify `needs_document_selection` is **false** | No document picker modal appears |
| 3.9 | Verify engine info | Shows `Engine: agentic` at bottom of answer |
| 3.10 | Check API Log | `S3:Query SSE /query/stream` or `S3:Query POST /query` (200) |

### Key Validation Points

- **`needs_document_selection: false`** — this is what makes it a "direct" question
- **`confidence: "high"`** — the agent was sure
- **`source_documents` is not empty** — sources were found
- **`fallback_used: false`** — agentic engine handled it directly

---

## TEST 4: Question Requiring Document Selection

**Story:** User asks an ambiguous question where the agent cannot determine which specific drawing to search — it asks the user to pick a document.

**APIs tested:** `POST /query/stream`, `POST /sessions/{id}/scope`, re-query

**Why this triggers document selection:** The question is broad enough that multiple drawings could contain the answer, and the agent needs the user to narrow down.

### Questions to Test

| # | Question | Why It Needs Selection | Expected Behavior |
|---|----------|----------------------|-------------------|
| 4a | `"What are the notes on this drawing?"` | "This drawing" is ambiguous — which one? | `needs_document_selection: true` with list |
| 4b | `"Summarize the general notes"` | Multiple drawings have general notes | Document picker appears |
| 4c | `"What equipment is shown?"` | Every discipline has equipment | Asks user to pick mechanical/electrical/plumbing |
| 4d | `"What are the specifications for insulation?"` | Multiple spec sections may cover insulation | Document picker with spec choices |

### Steps (use question 4a)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 4.1 | Type: `What are the notes on this drawing?` | |
| 4.2 | Click **Send** | User message appears |
| 4.3 | Wait for response | Response includes a **yellow-bordered document picker** |
| 4.4 | Verify `needs_document_selection: true` | Document picker box is visible inside the assistant message |
| 4.5 | Verify `available_documents` | List shows clickable document names with type badges |
| 4.6 | **Click on a document** (e.g., `M-401 Mechanical Roof Plan`) | Two API calls fire sequentially |
| 4.7 | Check API Log: Scope call | `S5:Scope POST /sessions/{id}/scope` (200) with `drawing_title: "M-401..."` |
| 4.8 | Check API Log: Re-query | `S3:Query` fires again with the same question, now scoped |
| 4.9 | Verify new answer | Specific answer about M-401's notes |
| 4.10 | Verify **scope badge** appears | Chat header shows: `Scoped to: M-401 Mechanical Roof Plan` with (X) |
| 4.11 | Verify confidence is now **higher** | Should be `HIGH` since the document is now specific |

### The Complete API Sequence for This Test

```
1. POST /query/stream { query: "What are the notes on this drawing?", session_id, project_id }
   → Response: needs_document_selection: true, available_documents: [...]

2. [User clicks "M-401 Mechanical Roof Plan"]

3. POST /sessions/{id}/scope { drawing_title: "M-401 Mechanical Roof Plan", drawing_name: "M-401", document_type: "drawing" }
   → Response: success: true, scope set

4. POST /query/stream { query: "What are the notes on this drawing?", session_id, project_id }
   → Response: specific answer about M-401 notes, scoped_to: "M-401 Mechanical Roof Plan"
```

---

## TEST 5: Follow-up / Multi-turn Conversation

**Story:** User asks follow-up questions that reference previous context — the session maintains conversation history.

**APIs tested:** `POST /query/stream` (same session_id)

### Conversation Flow

| Turn | Question | Why It Tests Multi-turn | Expected |
|------|----------|------------------------|----------|
| 1 | `"What HVAC units are on the first floor?"` | Baseline question — establishes context | Lists AHU units on first floor |
| 2 | `"What about the second floor?"` | Pronoun reference — "what about" refers to HVAC from turn 1 | Lists HVAC units on second floor (not asking "what about what?") |
| 3 | `"Which one has the highest CFM rating?"` | "Which one" refers to HVAC units mentioned in turns 1+2 | Identifies the specific unit with highest CFM |
| 4 | `"Show me the drawing where it appears"` | "It" refers to the unit from turn 3 | Returns the source drawing for that specific unit |

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 5.1 | Type turn 1: `What HVAC units are on the first floor?` | Direct answer with unit list |
| 5.2 | Verify `session_id` in API Log body | Same session_id as before |
| 5.3 | Type turn 2: `What about the second floor?` | Answer about SECOND floor HVAC (not generic) |
| 5.4 | Verify the agent understood context | Answer specifically mentions "second floor" without you repeating "HVAC" |
| 5.5 | Type turn 3: `Which one has the highest CFM rating?` | Specific unit name + CFM value |
| 5.6 | Type turn 4: `Show me the drawing where it appears` | Source document card with drawing reference |

### Key Validation Points

- Every query uses the **same `session_id`** — that's how context is maintained
- No `conversation_history` field is needed in the request body (session handles it)
- Follow-up answers should reference previous context correctly

---

## TEST 6: Document Scoping (Manual Selection from Sidebar)

**Story:** User clicks a document in the sidebar to scope all future queries to that specific document.

**APIs tested:** `POST /sessions/{id}/scope`, `GET /sessions/{id}/scope`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 6.1 | In left sidebar, click on a document (e.g., `M-100 Mechanical Lower Level Plan`) | Scope API fires |
| 6.2 | Check API Log | `S5:Scope POST /sessions/{id}/scope` (200) |
| 6.3 | Verify scope badge | Header shows: `Scoped to: M-100 Mechanical Lower Level Plan` |
| 6.4 | Verify system message | "Scoped to: M-100... All queries now filter to this document." |
| 6.5 | Ask: `"What equipment is shown?"` | Answer ONLY references equipment from M-100, not other drawings |
| 6.6 | Ask: `"What are the dimensions?"` | Answer ONLY from M-100 |
| 6.7 | Verify source documents in response | All sources reference M-100 only |

### Questions to Test While Scoped

| Question | Expected (scoped to M-100) |
|----------|---------------------------|
| `"List all equipment on this drawing"` | Only M-100 equipment |
| `"What are the general notes?"` | Only M-100 general notes |
| `"What ductwork is shown?"` | Only M-100 ductwork |
| `"What scale is this drawing?"` | M-100's scale |

---

## TEST 7: Clear Scope & Full Project Search

**Story:** User clears the document scope to return to project-wide search.

**APIs tested:** `DELETE /sessions/{id}/scope`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 7.1 | Click the **(X)** on the scope badge | API call fires |
| 7.2 | Check API Log | `S5:ClearScope DELETE /sessions/{id}/scope` (200) |
| 7.3 | Verify scope badge disappears | No scope badge in header |
| 7.4 | Verify system message | "Document scope cleared. Queries now search all project documents." |
| 7.5 | Ask: `"What equipment is shown across all mechanical drawings?"` | Answer references MULTIPLE drawings (M-100, M-200, M-401, etc.) |
| 7.6 | Verify source documents | Sources from multiple different drawings |

---

## TEST 8: Source Type Filtering

**Story:** User filters queries to only drawings or only specifications.

**APIs tested:** `POST /query` with `filter_source_type`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 8.1 | Set **Source Filter** dropdown to `"Drawings Only"` | |
| 8.2 | Ask: `"What are the insulation requirements?"` | Answer ONLY from drawing sources |
| 8.3 | Verify sources | All `source_documents` have drawing-related paths |
| 8.4 | Change **Source Filter** to `"Specifications Only"` | |
| 8.5 | Ask same question: `"What are the insulation requirements?"` | Answer from specification sections |
| 8.6 | Verify sources | Sources reference specification documents |
| 8.7 | Compare answers from 8.2 vs 8.5 | Different answers — drawings show insulation on drawings, specs show written requirements |
| 8.8 | Reset filter to `"All"` | |

---

## TEST 9: Web Search & Hybrid Mode

**Story:** User searches for industry standards or combines project data with web results.

**APIs tested:** `POST /web-search`, `POST /query` with `search_mode: "hybrid"`

### A. Pure Web Search

| Step | Action | Expected Result |
|------|--------|-----------------|
| 9.1 | Set **Mode** dropdown to `"Web Only"` | |
| 9.2 | Ask: `"What are the latest ASHRAE 90.1 energy code requirements for HVAC?"` | Web-sourced answer |
| 9.3 | Verify response has **web sources** | URLs to ashrae.org, energy.gov, etc. |
| 9.4 | Verify NO project source documents | No drawing/spec sources |

### B. Hybrid Mode (RAG + Web)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 9.5 | Set **Mode** dropdown to `"Hybrid (RAG + Web)"` | |
| 9.6 | Ask: `"What insulation is specified in our project and what does ASHRAE recommend?"` | Combined answer |
| 9.7 | Verify response has BOTH sections | `rag_answer` (project data) AND `web_answer` (web data) |
| 9.8 | Verify source documents | Mix of project sources AND web URLs |
| 9.9 | Reset **Mode** to `"Auto"` | |

### Web-Appropriate Questions

| Question | Mode | Expected |
|----------|------|----------|
| `"ASHRAE 90.1 duct insulation requirements"` | Web | Industry standards, code references |
| `"Latest fire code requirements for commercial buildings"` | Web | NFPA codes, IBC references |
| `"Compare our project HVAC specs with current energy codes"` | Hybrid | Project specs + current code requirements |
| `"What are the industry standard sizes for the ductwork shown in our drawings?"` | Hybrid | Project duct sizes + industry catalog data |

---

## TEST 10: Session History & Resume

**Story:** User browses past sessions and resumes an old conversation.

**APIs tested:** `GET /sessions`, `GET /sessions/{id}/conversation`, `GET /sessions/{id}/scope`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 10.1 | Create a second session: Click "Load Project" again with same project ID | New session created, appears in sidebar |
| 10.2 | Ask 2-3 questions in the new session | Build up conversation history |
| 10.3 | Click on the **first session** in the sidebar | Two parallel API calls fire |
| 10.4 | Check API Log | `S4:History GET /sessions/{id}/conversation` and `S5:GetScope GET /sessions/{id}/scope` |
| 10.5 | Verify chat area | Previous conversation from session 1 is loaded (all messages) |
| 10.6 | Verify scope state | If session 1 had a scope, it's restored |
| 10.7 | Ask a follow-up question | Answer references session 1's context (not session 2's) |
| 10.8 | Switch back to session 2 | Session 2's conversation loads, session 1's follow-up is in session 1 only |

---

## TEST 11: Session Lifecycle (Create / Update / Delete)

**Story:** Full session management operations.

**APIs tested:** `POST /sessions/create`, `POST /sessions/{id}/update`, `GET /sessions/{id}/stats`, `DELETE /sessions/{id}`

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 11.1 | Click "**+ New Session**" | `POST /sessions/create` (200), new session in sidebar |
| 11.2 | Ask 2 questions in the new session | Builds history |
| 11.3 | Check session stats (not in demo UI — test via curl): | |
| | `GET /sessions/{id}/stats` | Returns `message_count`, `engine_usage`, `total_cost_usd` |
| 11.4 | Delete the session: | |
| | (via curl) `DELETE /sessions/{id}` | `deleted: true` |
| 11.5 | Try to query with deleted session_id | Error or new session auto-created |
| 11.6 | Refresh session list | Deleted session no longer appears |

### Curl Commands for Manual Testing

```bash
# Create session
curl -X POST https://ai5.ifieldsmart.com/rag/sessions/create \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"project_id": 2361}'

# Get session stats
curl https://ai5.ifieldsmart.com/rag/sessions/SESSION_ID/stats \
  -H "X-API-Key: YOUR_KEY"

# Update session context
curl -X POST https://ai5.ifieldsmart.com/rag/sessions/SESSION_ID/update \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"filter_source_type": "drawing", "custom_instructions": "Focus on HVAC systems"}'

# Delete session
curl -X DELETE https://ai5.ifieldsmart.com/rag/sessions/SESSION_ID \
  -H "X-API-Key: YOUR_KEY"
```

---

## TEST 12: Engine Forcing & Fallback

**Story:** Test forcing specific engines and verifying fallback behavior.

**APIs tested:** `POST /query` with `engine` parameter

### A. Force Agentic Engine

| Step | Action | Expected Result |
|------|--------|-----------------|
| 12.1 | Set **Engine** dropdown to `"Force Agentic"` | |
| 12.2 | Ask: `"What electrical panels are in the project?"` | Answer with `engine_used: "agentic"` |
| 12.3 | Verify response | `engine_used: "agentic"`, `fallback_used: false` |

### B. Force Traditional Engine

| Step | Action | Expected Result |
|------|--------|-----------------|
| 12.4 | Set **Engine** dropdown to `"Force Traditional"` | |
| 12.5 | Ask same question: `"What electrical panels are in the project?"` | Answer with `engine_used: "traditional"` |
| 12.6 | Compare answer quality | May differ — traditional uses FAISS, agentic uses MongoDB |
| 12.7 | Reset to `"Auto"` | |

### C. Test Fallback (Trigger Low Confidence)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 12.8 | Set engine to Auto | |
| 12.9 | Ask a very vague question: `"Tell me everything"` | May trigger fallback |
| 12.10 | Check `fallback_used` in response | If `true`, traditional engine answered |
| 12.11 | Check `agentic_confidence` | Shows what the agentic engine's confidence was before fallback |

---

## TEST 13: Edge Cases & Error Handling

**Story:** Verify the system handles bad input gracefully.

### A. Authentication Errors

| Step | Action | Expected Result |
|------|--------|-----------------|
| 13.1 | Clear the API key field, try to query | HTTP 403: `{"detail": "Forbidden"}` |
| 13.2 | Enter an invalid API key: `"wrong-key-123"` | HTTP 403: `{"detail": "Forbidden"}` |
| 13.3 | Restore the correct API key | Queries work again |

### B. Validation Errors

| Step | Action | Expected Result |
|------|--------|-----------------|
| 13.4 | Send empty query (if possible) | HTTP 422: validation error — "String should have at least 1 character" |
| 13.5 | Send query with invalid project_id: `0` | Error response (project not found or validation) |
| 13.6 | Send query with non-existent session_id: `"session_fake123"` | Either error or new session auto-created |

### C. Long Queries

| Step | Action | Expected Result |
|------|--------|-----------------|
| 13.7 | Send a very long question (500+ characters) | Should work fine (max is 2000 chars) |
| 13.8 | Send a 2001+ character query | HTTP 422: validation error |

### D. No Documents in Project

| Step | Action | Expected Result |
|------|--------|-----------------|
| 13.9 | Load a project with no uploaded documents (e.g., project ID `9999`) | Document list empty, queries return low confidence or "no data found" |

---

## TEST 14: Streaming vs Non-Streaming Comparison

**Story:** Verify both query modes work and compare behavior.

### Steps

| Step | Action | Expected Result |
|------|--------|-----------------|
| 14.1 | Ask via streaming (default): `"What HVAC units are specified?"` | Tokens appear progressively, then full response |
| 14.2 | Note response time in API Log | e.g., 8000ms |
| 14.3 | If streaming fails, verify fallback | UI should automatically fallback to `POST /query` (check API Log for `S3:Query POST /query`) |
| 14.4 | Verify both modes return same answer quality | Same sources, same confidence |

---

## Test Results Checklist

Copy this checklist for each test run:

```
Date: ___________  Tester: ___________  Project ID: ___________

TEST 1:  App Boot & Health           [ ] PASS  [ ] FAIL  Notes: ___________
TEST 2:  Project Load & Discovery    [ ] PASS  [ ] FAIL  Notes: ___________
TEST 3:  Simple Direct Question      [ ] PASS  [ ] FAIL  Notes: ___________
TEST 4:  Document Selection Flow     [ ] PASS  [ ] FAIL  Notes: ___________
TEST 5:  Multi-turn Conversation     [ ] PASS  [ ] FAIL  Notes: ___________
TEST 6:  Manual Document Scoping     [ ] PASS  [ ] FAIL  Notes: ___________
TEST 7:  Clear Scope                 [ ] PASS  [ ] FAIL  Notes: ___________
TEST 8:  Source Type Filtering       [ ] PASS  [ ] FAIL  Notes: ___________
TEST 9:  Web Search & Hybrid         [ ] PASS  [ ] FAIL  Notes: ___________
TEST 10: Session Resume              [ ] PASS  [ ] FAIL  Notes: ___________
TEST 11: Session Lifecycle           [ ] PASS  [ ] FAIL  Notes: ___________
TEST 12: Engine Forcing/Fallback     [ ] PASS  [ ] FAIL  Notes: ___________
TEST 13: Edge Cases & Errors         [ ] PASS  [ ] FAIL  Notes: ___________
TEST 14: Streaming vs Non-Streaming  [ ] PASS  [ ] FAIL  Notes: ___________

Overall:  ___/14 PASSED
```

---

## Question Bank by Scenario

### Direct Questions (No Document Selection Needed)
These are specific enough that the agent finds the answer immediately.

| # | Question | Expected Source Type |
|---|----------|-------------------|
| 1 | "What XVENT models are specified in the mechanical drawings?" | Mechanical drawings |
| 2 | "List all electrical panel schedules and their ampere ratings" | Electrical schedules |
| 3 | "What is the total cooling capacity in tons?" | Mechanical schedules |
| 4 | "What fire alarm devices are shown on the first floor?" | Fire alarm drawings |
| 5 | "What type of roofing insulation is specified?" | Specifications |
| 6 | "What is the voltage for the main electrical service?" | Electrical one-line diagram |
| 7 | "List all plumbing fixture types in the project" | Plumbing drawings |
| 8 | "What are the ductwork sizes for the supply air on the second floor?" | Mechanical floor plans |

### Ambiguous Questions (Will Trigger Document Selection)
These are too broad — the agent needs the user to pick a specific document.

| # | Question | Why It's Ambiguous |
|---|----------|--------------------|
| 1 | "What are the notes on this drawing?" | Which drawing? |
| 2 | "Summarize the general notes" | Multiple drawings have general notes |
| 3 | "What equipment is shown?" | Every discipline has equipment |
| 4 | "What are the dimensions?" | Could be any drawing |
| 5 | "Explain the legend" | Multiple drawings have legends |
| 6 | "What does the schedule say?" | Which schedule? |

### Multi-turn Conversation Chains
Test context retention across turns.

**Chain A — HVAC Deep Dive:**
1. "What HVAC units are on the first floor?"
2. "What about the second floor?"
3. "Which unit has the highest CFM?"
4. "What drawing shows that unit?"
5. "Are there any notes about its installation?"

**Chain B — Electrical Investigation:**
1. "What electrical panels are in the project?"
2. "What is the main service voltage?"
3. "Which panel feeds the HVAC equipment?"
4. "What is the breaker size for that circuit?"

**Chain C — Cross-Reference:**
1. "What mechanical equipment is on the roof?"
2. "What electrical connections do those units need?"
3. "Are there any specifications for the roof units?"

### Web & Hybrid Questions

| # | Question | Mode | Expected |
|---|----------|------|----------|
| 1 | "What are ASHRAE 90.1 requirements for duct insulation?" | Web | Industry code references |
| 2 | "What are the latest NFPA fire code requirements?" | Web | Fire code references |
| 3 | "Compare our insulation specs with current ASHRAE standards" | Hybrid | Project specs + ASHRAE web data |
| 4 | "Are our electrical panel ratings compliant with NEC 2023?" | Hybrid | Project data + NEC code web data |
| 5 | "What are industry best practices for the HVAC system we specified?" | Hybrid | Project HVAC + industry best practices |

---

## API-to-Test Mapping

Quick reference: which test covers which API.

| API | Tested In |
|-----|-----------|
| `GET /` | TEST 1 |
| `GET /health` | TEST 1 |
| `GET /config` | TEST 11 (curl) |
| `POST /query` | TEST 14 (fallback) |
| `POST /query/stream` | TEST 3, 4, 5, 6, 7, 8, 9, 12 |
| `POST /quick-query` | Not in demo UI — test via curl |
| `POST /web-search` | TEST 9A |
| `POST /sessions/create` | TEST 2, 11 |
| `GET /sessions` | TEST 10, 11 |
| `GET /sessions/{id}/stats` | TEST 11 (curl) |
| `GET /sessions/{id}/conversation` | TEST 10 |
| `POST /sessions/{id}/update` | TEST 11 (curl) |
| `DELETE /sessions/{id}` | TEST 11 |
| `GET /projects/{id}/documents` | TEST 2 |
| `POST /sessions/{id}/scope` | TEST 4, 6 |
| `DELETE /sessions/{id}/scope` | TEST 7 |
| `GET /sessions/{id}/scope` | TEST 10 |
| `GET /admin/sessions` | Admin test (curl) |
| `POST /admin/cache/refresh` | Admin test (curl) |
