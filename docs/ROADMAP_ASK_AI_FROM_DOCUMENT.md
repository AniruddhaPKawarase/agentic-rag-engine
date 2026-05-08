# Roadmap: Ask AI from Document + Intelligent Follow-Up

**Date:** 2026-04-15
**Status:** DRAFT — Pending clarifying answers and user confirmation
**User Stories:** Story I (Follow-Up Questions & Query Enhancement) + Story II (PDF Document Interaction)
**Approach:** Drawing Title Scoping via DB-Level Filters (projectId + drawingName + drawingTitle)

---

## Executive Summary

Transform the unified RAG agent from one-shot Q&A into an interactive analysis platform. When the agent cannot answer a question in normal conversation, it discovers and presents available document groups (unique drawingTitles). The user selects a document, and all subsequent queries are scoped to that document at the MongoDB query level — the agent physically cannot see data outside the selected document.

**Key architectural change:** Remove `drawingVision` collection (testing only). Agent works with 2 collections: `drawing` (2.8M OCR fragments) and `specification` (80K docs).

---

## System Design Reference (from System Design Graph — 4,291 designs)

Production patterns informing this roadmap:

| Pattern Source | Components Applied | Why |
|---------------|-------------------|-----|
| **Search Engine** (cx:5) | Query Processor, Ranking Engine, Cache Layer, Spell Checker | Query enhancement, result ranking, caching scoped queries |
| **Chatbot Orchestration** (cx:9) | Delivery Engine, Queue Manager, Circuit Breaker | Orchestrator fallback chain, error handling, message delivery |
| **Document Management** (cx:5) | Document Store, Workflow Engine, State Machine, Audit Trail, RBAC | Document scope state machine, audit logging, access control |
| **Autocomplete** (cx:3) | Trie/Prefix Index, Ranking Service, Top-K Pattern | Drawing title suggestion, query refinement, prefix matching |
| **Recommendation Engine** (cx:5) | Feature Store, Candidate Generator, Two-Stage Ranking | Follow-up question generation, document suggestion ranking |
| **Rate Limiter** (cx:5) | API Gateway, Circuit Breaker | Request throttling, cost cap enforcement |
| **Notification System** (cx:4) | Priority Router, Template Engine | Follow-up suggestion delivery, user guidance templates |
| **Distributed Cache** (cx:4) | Cache-Aside, Consistent Hashing | Drawing title list caching, embedding cache |
| **Monitoring** (cx:4) | Metric Collectors, Alert Evaluator, Dashboard | Prometheus metrics, scope usage tracking, alert on failures |

**Unified Stack:** Redis (cache), Kafka/queue (async), PostgreSQL/MongoDB (state), Prometheus+Grafana (monitoring), Circuit Breaker + Audit Trail + State Machine (patterns).

---

## Phase Breakdown

### Phase 0: Cleanup — Remove drawingVision Collection
**Effort:** 2-3 hrs | **Risk:** LOW | **Dependencies:** None

| # | Task | Files |
|---|------|-------|
| 0.1 | Delete all vision tool functions (4 functions) | `agentic/tools/mongodb_tools.py` |
| 0.2 | Remove `vision_*` from TOOL_DEFINITIONS and TOOL_FUNCTIONS | `agentic/tools/registry.py` |
| 0.3 | Remove `VISION_COLLECTION` config var | `agentic/config.py` |
| 0.4 | Rewrite SYSTEM_PROMPT — remove all vision_* references, update routing guide | `agentic/core/agent.py` |
| 0.5 | Update `_extract_sources()` — remove `sourceFile` extraction (drawingVision specific) | `agentic/core/agent.py` |
| 0.6 | Update `.env` and `.env.example` — remove VISION_COLLECTION | `.env`, `.env.example` |
| 0.7 | Run existing tests — ensure nothing breaks | `tests/` |

**Acceptance:** Agent uses only 7 tools (legacy_* + spec_*). All existing tests pass.

---

### Phase 1: Document Discovery — List Unique Drawing Titles
**Effort:** 3-4 hrs | **Risk:** LOW | **Dependencies:** Phase 0

| # | Task | Files |
|---|------|-------|
| 1.1 | New function `list_unique_drawing_titles(project_id, set_id)` — aggregation pipeline returning unique drawingTitle + drawingName + trade + pdfName | `agentic/tools/drawing_tools.py` |
| 1.2 | New function `list_unique_spec_titles(project_id)` — aggregation for unique sectionTitle + pdfName + specificationNumber | `agentic/tools/specification_tools.py` |
| 1.3 | Cache layer for title lists — same project queried repeatedly within a session, cache for 5 min (Cache-Aside pattern from distributed cache design) | `agentic/core/cache.py` |
| 1.4 | Unit tests for both new functions | `tests/agentic/test_drawing_tools.py`, `tests/agentic/test_spec_tools.py` |

**MongoDB Aggregation (drawing collection):**
```
[
  {"$match": {"projectId": X}},
  {"$group": {
    "_id": {"drawingTitle": "$drawingTitle", "drawingName": "$drawingName"},
    "trade": {"$first": {"$ifNull": ["$setTrade", "$trade"]}},
    "pdfName": {"$first": "$pdfName"},
    "fragment_count": {"$sum": 1}
  }},
  {"$sort": {"_id.drawingTitle": 1}}
]
```

**Acceptance:** Given a projectId, returns deduplicated list of all drawingTitle/drawingName pairs. Cached per project for 5 min.

---

### Phase 2: Document Scope Filters on Existing Tools
**Effort:** 4-5 hrs | **Risk:** MEDIUM | **Dependencies:** Phase 1

Add optional `drawing_title` and `drawing_name` parameters to every search tool. When provided, MongoDB query includes these as additional filters — DB-level enforcement.

| # | Tool | Filter Added | MongoDB Query Change |
|---|------|-------------|---------------------|
| 2.1 | `legacy_search_text` | `drawing_title`, `drawing_name` | `+ {"drawingTitle": regex(...)}` and/or `{"drawingName": regex(...)}` |
| 2.2 | `legacy_search_trade` | `drawing_title`, `drawing_name` | `+ {"drawingTitle": regex(...)}` in `$match` stage |
| 2.3 | `legacy_list_drawings` | `drawing_title`, `drawing_name` | `+ {"drawingTitle": regex(...)}` in `$match` stage |
| 2.4 | `legacy_get_text` | No change — already scoped by `drawingId` | Already scoped |
| 2.5 | `spec_search` | `section_title`, `pdf_name` | `+ {"sectionTitle": regex(...)}` |
| 2.6 | `spec_list` | `section_title` | `+ {"sectionTitle": regex(...)}` |
| 2.7 | `spec_get_section` | No change — already accepts `section_title`/`pdf_name` | Already scoped |
| 2.8 | Update TOOL_DEFINITIONS in registry.py — add new optional params to function schemas | `agentic/tools/registry.py` |
| 2.9 | Unit tests — verify filter narrows results, verify empty results when doc doesn't contain query term | `tests/agentic/` |

**Acceptance:** `legacy_search_text(project_id=7222, search_text="fire damper", drawing_title="HVAC Duct Accessories")` returns ONLY results from that drawing title. Other drawings excluded at DB level.

---

### Phase 3: Session Scope State Management
**Effort:** 3-4 hrs | **Risk:** LOW | **Dependencies:** Phase 2

| # | Task | Files |
|---|------|-------|
| 3.1 | Add `scoped_drawing_title`, `scoped_drawing_name`, `scoped_document_type` to `ConversationContext` | `traditional/memory_manager.py` |
| 3.2 | Add `set_document_scope(session_id, drawing_title, drawing_name, doc_type)` method | `traditional/memory_manager.py` or `shared/session/manager.py` |
| 3.3 | Add `clear_document_scope(session_id)` method | Same |
| 3.4 | Add `get_document_scope(session_id)` method | Same |
| 3.5 | Scope persists in session JSON (survives page reload) | Session persistence layer |
| 3.6 | Unit tests for scope lifecycle: set → get → clear | `tests/test_session.py` |

**State Machine (from document management design pattern):**
```
[NO_SCOPE] → user selects drawing → [SCOPED: drawingTitle="X"]
                                          ↓
                              user asks question → scoped search
                                          ↓
                              user says "go back" → [NO_SCOPE]
                              agent can't find in doc → suggest other docs → [NO_SCOPE] or [SCOPED: drawingTitle="Y"]
```

**Acceptance:** Session correctly tracks scope state. Scope survives across queries within same session. Clear scope returns to full-project mode.

---

### Phase 4: Orchestrator — No Results → Document Discovery Path
**Effort:** 5-6 hrs | **Risk:** HIGH | **Dependencies:** Phase 1, 2, 3

This is the core logic change. When agent returns low confidence, instead of blind FAISS fallback, present document discovery.

| # | Task | Files |
|---|------|-------|
| 4.1 | New response fields: `needs_document_selection: bool`, `available_documents: list[dict]` | `gateway/models.py` |
| 4.2 | New method `_discover_documents(project_id, set_id)` — calls `list_unique_drawing_titles` + `spec_list`, merges into unified list | `gateway/orchestrator.py` |
| 4.3 | Modify `Orchestrator.query()` — when `_should_fallback()=True` AND session NOT already scoped: run document discovery + still run traditional fallback | `gateway/orchestrator.py` |
| 4.4 | Modify `Orchestrator.query()` — when session IS scoped: inject `drawing_title`/`drawing_name` filter into agentic engine call | `gateway/orchestrator.py` |
| 4.5 | Modify `_build_response()` — include `available_documents` and `needs_document_selection` in response | `gateway/orchestrator.py` |
| 4.6 | When scoped search returns empty — response includes "this document doesn't contain..." + suggest other titles | `gateway/orchestrator.py` |
| 4.7 | Integration tests: normal query → discovery → scoped query → unscope | `tests/test_orchestrator.py` |

**Flow:**
```
query() called
  → Check session: has scoped_drawing_title?
    YES → inject filter into agentic engine → run scoped
    NO  → run normal agentic
  → _should_fallback()?
    YES + not scoped → _discover_documents() → return with available_documents
    YES + scoped → "document doesn't contain this" → suggest alternatives
    NO → return agentic result
```

**Acceptance:** When agent can't answer, response includes `needs_document_selection=true` + list of unique drawingTitle/drawingName pairs. When scoped, agent only sees data from selected document.

---

### Phase 5: Tool Call Interception — Auto-Inject Scope
**Effort:** 3-4 hrs | **Risk:** MEDIUM | **Dependencies:** Phase 2, 3, 4

| # | Task | Files |
|---|------|-------|
| 5.1 | Modify `_execute_tool()` in agent.py — accept `scope` dict, auto-inject `drawing_title`/`drawing_name` into tool args for `legacy_*` tools | `agentic/core/agent.py` |
| 5.2 | Modify `run_agent()` — accept `scope` param from orchestrator, pass to `_execute_tool` | `agentic/core/agent.py` |
| 5.3 | Modify orchestrator → pass session scope to `AgenticEngine.query()` → pass to `run_agent()` | `gateway/orchestrator.py` |
| 5.4 | System prompt modification when scoped — append "DOCUMENT SCOPE ACTIVE" block | `agentic/core/agent.py` |
| 5.5 | Unit tests: verify tool args are modified when scope active, verify no modification when no scope | `tests/agentic/test_agent.py` |

**Dual enforcement:** Prompt tells agent it's scoped (correct reasoning). Wrapper enforces at DB level (can't escape).

**Acceptance:** When session has `scoped_drawing_title="HVAC Duct Accessories"`, every `legacy_search_text` call automatically includes `drawing_title="HVAC Duct Accessories"` regardless of what the agent requested.

---

### Phase 6: Agent System Prompt Rewrite
**Effort:** 2-3 hrs | **Risk:** LOW | **Dependencies:** Phase 0, 5

| # | Task | Files |
|---|------|-------|
| 6.1 | Remove all `vision_*` references from SYSTEM_PROMPT | `agentic/core/agent.py` |
| 6.2 | Update query routing guide for 2 collections only | `agentic/core/agent.py` |
| 6.3 | Add DOCUMENT-SCOPED MODE instructions | `agentic/core/agent.py` |
| 6.4 | Add follow-up suggestion format instructions (3 follow-ups with source citations) | `agentic/core/agent.py` |
| 6.5 | Add "no results" behavior: "say so clearly and the system will suggest documents" | `agentic/core/agent.py` |
| 6.6 | Test prompt with real queries — verify agent behavior in scoped vs unscoped mode | Manual testing |

**Acceptance:** Agent correctly uses only legacy_* and spec_* tools. In scoped mode, agent acknowledges scope in answers. Follow-ups are relevant and cited.

---

### Phase 7: Follow-Up Questions — Agentic Engine
**Effort:** 3-4 hrs | **Risk:** MEDIUM | **Dependencies:** Phase 6

Currently agentic engine returns `follow_up_questions: []`. Fix this.

| # | Task | Files |
|---|------|-------|
| 7.1 | Add `---FOLLOW_UP---` separator instruction to agentic SYSTEM_PROMPT | `agentic/core/agent.py` |
| 7.2 | Parse follow-ups from agent's final answer using existing `parse_follow_up_questions()` | `agentic/core/agent.py` |
| 7.3 | Include parsed follow-ups in `AgentResult` dataclass | `agentic/core/agent.py` |
| 7.4 | Update `_build_response()` in orchestrator — use agentic follow-ups instead of hardcoded `[]` | `gateway/orchestrator.py` |
| 7.5 | When scoped: follow-ups should reference the scoped document | Prompt engineering |
| 7.6 | When no results: follow-ups should suggest alternative documents from available_documents list | `gateway/orchestrator.py` |
| 7.7 | Unit tests: verify follow-ups parsed from agentic answer, verify empty gracefully handled | `tests/` |

**Acceptance:** Agentic engine returns 2-5 context-aware follow-up questions. 80%+ of responses have non-empty follow-ups.

---

### Phase 8: Query Enhancement on No Results
**Effort:** 3-4 hrs | **Risk:** MEDIUM | **Dependencies:** Phase 4, 7

When both engines fail (no results from any scope), generate improved queries.

| # | Task | Files |
|---|------|-------|
| 8.1 | New function `enhance_query(original_query, project_context)` — LLM call to rephrase/expand query with domain knowledge | New: `gateway/query_enhancer.py` or in `orchestrator.py` |
| 8.2 | Parse original query for missing attributes: trade, drawing ref, spec section, project stage | `gateway/query_enhancer.py` |
| 8.3 | Generate 2-3 improved queries: add context, clarify vague terms, expand domain keywords | `gateway/query_enhancer.py` |
| 8.4 | Add `improved_queries: list[str]`, `query_tips: list[str]` to UnifiedResponse | `gateway/models.py` |
| 8.5 | Wire into orchestrator: on double-failure (agentic + traditional both fail) → call enhance_query | `gateway/orchestrator.py` |
| 8.6 | Unit tests | `tests/test_query_enhancer.py` |

**Pattern:** Autocomplete design pattern — Query Aggregator + Ranking Service. Domain-specific expansion using construction trade taxonomy.

**Acceptance:** When "Check missing items" returns no results, system suggests: "Check missing HVAC scope items in mechanical drawings", "Identify missing fire protection components in spec section 21 00 00".

---

### Phase 9: Intent Detection — Drawing Title Matching
**Effort:** 2-3 hrs | **Risk:** LOW | **Dependencies:** Phase 1, 3

| # | Task | Files |
|---|------|-------|
| 9.1 | Enhance `document_chat` intent patterns to match drawing titles from available_documents | `traditional/rag/api/intent.py` |
| 9.2 | New function `match_drawing_title(user_input, available_titles)` — fuzzy match user's selection against known titles | `traditional/rag/api/intent.py` |
| 9.3 | Wire into orchestrator: when intent=document_chat, extract title → set scope | `gateway/orchestrator.py` |
| 9.4 | Handle `unpin_document` intent → clear scope | `gateway/orchestrator.py` |
| 9.5 | Unit tests for fuzzy matching | `tests/` |

**Acceptance:** User types "Check in Mechanical Lower Level Plan" → system matches to exact drawingTitle → enters scoped mode.

---

### Phase 10: Streamlit UI — Document Discovery & Scope Indicator
**Effort:** 4-5 hrs | **Risk:** LOW | **Dependencies:** Phase 4, 9

| # | Task | Files |
|---|------|-------|
| 10.1 | Render `available_documents` as clickable cards/buttons when `needs_document_selection=true` | `streamlit_app.py` |
| 10.2 | Group documents by trade (Mechanical, Electrical, Plumbing, Specifications) | `streamlit_app.py` |
| 10.3 | On click → set session scope → re-run query scoped | `streamlit_app.py` |
| 10.4 | Scope indicator badge: "Scoped to: HVAC Duct Accessories [x]" | `streamlit_app.py` |
| 10.5 | "Return to Full Project" button when in scoped mode | `streamlit_app.py` |
| 10.6 | Render `improved_queries` as clickable suggestions on no-results | `streamlit_app.py` |
| 10.7 | Render `query_tips` as guidance text | `streamlit_app.py` |
| 10.8 | Remove Document Pinning page dependency (replaced by scope feature) OR keep for advanced users | `streamlit_app.py` |

**Acceptance:** When agent can't answer, UI shows grouped document list. User clicks → enters scoped mode with visual indicator. Follow-ups and improved queries are clickable.

---

### Phase 11: Integration Testing
**Effort:** 3-4 hrs | **Risk:** LOW | **Dependencies:** Phase 0-10

| # | Task | Method |
|---|------|--------|
| 11.1 | End-to-end: normal query → answer (unscoped) | API test |
| 11.2 | End-to-end: no-results → document discovery → scoped query → answer | API test |
| 11.3 | End-to-end: scoped query → empty in doc → suggest alternatives | API test |
| 11.4 | End-to-end: unscope → return to full project | API test |
| 11.5 | Session persistence: scope survives across requests | API test |
| 11.6 | Follow-up questions: clickable, re-runnable | UI test |
| 11.7 | Query enhancement: improved queries on double failure | API test |
| 11.8 | Backward compatibility: existing /query requests unchanged | API test |
| 11.9 | Performance: scoped query latency < unscoped | Benchmark |
| 11.10 | Cost tracking: scoped queries reduce token usage | Metric check |

---

### Phase 12: Production Hardening
**Effort:** 4-5 hrs | **Risk:** MEDIUM | **Dependencies:** Phase 11

| # | Task | Pattern Reference |
|---|------|------------------|
| 12.1 | Circuit breaker on MongoDB aggregation (drawing title list) | Circuit Breaker (chatbot + API gateway) |
| 12.2 | Drawing title list cache with TTL (Redis or in-memory) | Cache-Aside (distributed cache) |
| 12.3 | Rate limiting on document discovery endpoint | Rate Limiter design |
| 12.4 | Prometheus metrics: scope_usage_total, discovery_latency_ms, scoped_query_latency_ms | Monitoring design |
| 12.5 | Structured audit log for scope events: set/clear/query | Audit Trail (document management) |
| 12.6 | Cost cap enforcement in scoped mode (same $0.50/req limit) | Existing cost control |
| 12.7 | Error handling: MongoDB timeout on aggregation → graceful degradation | Circuit Breaker |
| 12.8 | Security: validate drawing_title input (no regex injection) | Input validation |

---

## Timeline Summary

| Phase | Name | Effort | Cumulative |
|-------|------|--------|-----------|
| P0 | Remove drawingVision | 2-3 hrs | 2-3 hrs |
| P1 | Document Discovery query | 3-4 hrs | 5-7 hrs |
| P2 | Scope filters on tools | 4-5 hrs | 9-12 hrs |
| P3 | Session scope management | 3-4 hrs | 12-16 hrs |
| P4 | Orchestrator document discovery | 5-6 hrs | 17-22 hrs |
| P5 | Tool call interception | 3-4 hrs | 20-26 hrs |
| P6 | Agent prompt rewrite | 2-3 hrs | 22-29 hrs |
| P7 | Agentic follow-ups | 3-4 hrs | 25-33 hrs |
| P8 | Query enhancement | 3-4 hrs | 28-37 hrs |
| P9 | Intent detection | 2-3 hrs | 30-40 hrs |
| P10 | Streamlit UI | 4-5 hrs | 34-45 hrs |
| P11 | Integration testing | 3-4 hrs | 37-49 hrs |
| P12 | Production hardening | 4-5 hrs | 41-54 hrs |
| **TOTAL** | | | **~41-54 hrs** |

---

## User Story Mapping

| Acceptance Criteria | Phase | Status |
|--------------------|-------|--------|
| **Story I: Follow-up valid response** — 2+ relevant follow-ups | P7 | Planned |
| **Story I: Follow-up no results** — 2+ improved queries + 2+ tips | P8 | Planned |
| **Story I: Query enhancement quality** — adds context OR clarifies OR expands | P8 | Planned |
| **Story I: Clickable follow-ups re-run search** | P10 | Planned (existing for traditional, new for agentic) |
| **Story II: Reference source has action** | P10 | Planned (via document discovery cards) |
| **Story II: Context activation without re-upload** | P3, P4, P5 | Planned (session scope + auto-inject) |
| **Story II: Document-based responses with page refs** | P6, P7 | Planned (prompt + source extraction) |
| **Story II: Multi-document** | P3 | Optional (scope supports single title; multi later) |
| **Story II: Traceability** — page numbers, snippets | P2, P6 | Planned (legacy already has page; prompt cites) |

---

## What's NOT Changing

- Traditional RAG engine core (FAISS retrieval, generation pipeline) — untouched
- S3 utils, session persistence format — untouched
- Existing API endpoints — backward compatible
- FAISS indexes — same files, same structure
- MongoDB `drawing` + `specification` collections — no schema changes (read-only filters added)
- Existing test suites — extended, not rewritten
- Cost controls, rate limiting, auth — same enforcement

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| MongoDB aggregation slow on large projects | MEDIUM | Cache drawing title list per project (5 min TTL). Index on projectId + drawingTitle if needed. |
| Agent ignores scope in reasoning | LOW | Dual enforcement: prompt + tool wrapper. Agent physically can't see unscoped data. |
| Drawing title fuzzy matching false positives | LOW | Use exact match first, regex fallback, confirm with user on ambiguity. |
| Backward compatibility break | HIGH | All new fields are optional with defaults. Existing /query requests work unchanged. |
| LLM cost increase from query enhancement | LOW | Only fires on double failure (rare). Cap at 1 enhancement call per request. |
