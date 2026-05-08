# Design — `unified_rag_agent_with_docqa_extension`

**Date:** 2026-04-23
**Author:** Claude (with Aniruddha P. Kawarase)
**Baseline code:** `_versions/v1.2-hybrid-ship/` + yesterday's RRF/reranker/self-RAG additions already on `main`
**Target version:** `_versions/v2.0-docqa-extension/`
**Target env:** local + sandbox (54.197.189.113) only. NO VM deploy.
**Answers to clarifying questions:** `docs/superpowers/specs/2026-04-23-unified-rag-docqa-extension-CLARIFYING-QUESTIONS.md`

---

## 1. Problem

Current unified RAG agent has a DocQA bridge scaffold (~55% wired) but four defects block the user story:

1. **Hallucinated answer format** — contradictory system prompt leaks `[Source: …]`, `HIGH (90%)`, `Direct Answer` into the prose.
2. **Schema/wire drift** — `QueryRequest` rejects the `docqa_document` field the orchestrator needs; `UnifiedResponse` Pydantic model is out of sync with what the orchestrator actually emits (`source_documents`, `active_agent`, `selected_document`).
3. **S3-to-DocQA handoff incomplete** — client requires local file paths; no S3 fetch step visible.
4. **One-way intent classifier** — "this document" doesn't auto-route from RAG → DocQA; user is stuck until they manually click.

User story in one line: *"When RAG answers vaguely or with low confidence but returns source documents, user picks one, and the conversation continues inside DocQA on that specific file — seamlessly, same session, auto-switching back for project-wide questions."*

## 2. Non-goals

- Not rewriting DocQA agent. Zero code changes on port 8006.
- Not touching yesterday's RRF/reranker/self-RAG work on main. Those stay as they are.
- Not deploying to VM. Not changing nginx, systemd, or DNS.
- Not adding auth/middleware (user deferred). No rate limits, no JWT, no API keys.
- Not building MongoDB drawingTitle-scoped retrieval (ROADMAP_ASK_AI_FROM_DOCUMENT.md) — that's a different track.
- Not building CloudFront/SSL on presigned URLs yet — parked pending S3 credentials.

## 3. Hard constraints (from user's Q12)

1. **RAG baseline frozen:** v1.2-hybrid-ship + RRF + reranker + self-RAG stay as-is. DocQA is a *layer on top*.
2. **DocQA runs AFTER RAG**, never before. RAG always gets first attempt.
3. **Construction drawings:** DocQA must accept complex drawing PDFs. If DocQA's current vision/OCR is inadequate for drawings, surface as a *known limitation* — don't block the bridge.
4. **Wire schema frozen:** no renames, no removals. Only *additive optional default-null* fields. Sandbox UI is already integrated against existing wire shape.
5. **New APIs align with existing:** same envelope (`success`, `answer`, `source_documents`, `session_id`, `confidence`, `engine_used`).

## 4. Architecture

### 4.1 Current flow

```
User → Gateway /query → Orchestrator
          ├─► AgenticRAG (MongoDB tools, ReAct, GPT-4.1)
          │     └─► Fix #1 RRF hint → Fix #2 reranker → Fix #4 self-RAG → Fix #8 re-sign URLs
          └─► Fallback → Traditional RAG (FAISS)
```

### 4.2 New flow (DocQA layer)

```
User → Gateway /query → Orchestrator
          │
          ├─► RAG path (unchanged baseline)
          │     └─► Response includes source_documents[] with download_url presigned
          │
          │   [User picks one source → UI calls /query with docqa_document = {...}]
          │
          └─► DocQA path (NEW adapter layer)
                ├─► docqa_bridge.py  (NEW, thin adapter)
                │     ├─ resolve s3_path → download temp PDF → multipart → DocQA /api/upload
                │     ├─ persist docqa_session_id in MemoryManager
                │     └─ forward follow-up Q&A to DocQA /api/chat
                ├─► intent_classifier (bidirectional)  ← Q3 hybrid
                │     └─ score + clarify-if-confused
                └─► response normalizer (NEW)
                      └─ map DocQA fields → UnifiedResponse envelope unchanged
```

**Key design choice:** isolate the DocQA adapter in a single new module `gateway/docqa_bridge.py`. Existing modules (`orchestrator.py`, `router.py`, `models.py`) get *minimal*, *additive*, *guarded* changes only.

### 4.3 Module map

| Module | Change type | Lines changed | Purpose |
|---|---|---|---|
| `gateway/docqa_bridge.py` | **NEW** | ~200 | Single source of truth for S3-fetch + DocQA upload + Q&A + normalization |
| `gateway/docqa_client.py` | Minor edit | ~20 | Keep as HTTP transport; add `/api/chat` replay for existing session |
| `gateway/intent_classifier.py` | Rewrite | ~150 | Bidirectional classifier with confidence threshold + clarify option |
| `gateway/orchestrator.py` | Surgical additions | ~60 | Hook bridge into `_run_query`; don't touch RAG path |
| `gateway/models.py` | Additive | ~30 | Add optional `docqa_document`, `selected_document`, `active_agent`, `source_documents`, `needs_clarification`, `clarification_prompt` fields |
| `gateway/router.py` | None in v2.0 | 0 | No new endpoints — re-use `/query` with `search_mode` parameter. Avoids breaking existing clients. |
| `agentic/core/agent.py` | One-line fix | 1 | Delete contradictory line 155 (`Cite sources: [Source: …]`) |
| `docs/demo-ui.html` | Incremental | ~80 | Sources list shows S3 basename + drawing title, "Chat with Document" button, agent badge, clarification prompt |
| `shared/session/memory_manager.py` | Additive | ~40 | Add `active_agent`, `docqa_session_id`, `selected_documents[]` fields on session |
| `tests/test_docqa_bridge.py` | NEW | ~400 | Unit tests for bridge, classifier, schema |
| `tests/e2e/test_docqa_flow.py` | NEW | ~200 | E2E sandbox test |

## 5. Detailed design

### 5.1 Answer-format fix (Phase 1)

**Change:** Delete `agentic/core/agent.py:155` (the contradictory "cite [Source:]" instruction). Leave line 205 (`Do NOT include [Source: …]`) as the single source of truth.

**Verification approach:** Run 16 historical queries from `test_results/` baseline both before and after deletion. Expect zero `[Source:`/`Direct Answer`/`HIGH (90%)` occurrences in `answer` field post-fix. No source-document loss — those come from the `source_documents[]` list, which is separately populated by retrieval, not by the LLM.

**No post-processor.** Per Q2 = A only. We trust the fix; cheaper, deterministic, one line.

### 5.2 Schema additions (Phase 1)

`QueryRequest` (adds):
```python
docqa_document: Optional[dict] = None       # {s3_path, file_name, download_url, pdf_name}
mode_hint: Optional[str] = None              # "rag" | "docqa" | None → auto
```

`UnifiedResponse` (adds, all default-null):
```python
source_documents: Optional[list[dict]] = None   # aligns model with wire truth today
active_agent: Optional[str] = "rag"             # "rag" | "docqa"
selected_document: Optional[dict] = None
needs_clarification: Optional[bool] = False
clarification_prompt: Optional[str] = None
docqa_session_id: Optional[str] = None
```

**Backward compatibility:** every new field is `Optional` with a default. No existing field renamed or removed. `sources` stays. Sandbox UI keeps working.

### 5.3 DocQA bridge (Phase 2)

`gateway/docqa_bridge.py` — single file, single responsibility:

```python
class DocQABridge:
    def ensure_document_loaded(session_id, doc_ref) -> str:
        """
        Given {s3_path|download_url, file_name}, returns a docqa_session_id.
        - If already loaded in this session, reuse.
        - Else: download from S3 → temp file → POST to /api/upload.
        - Persist docqa_session_id + selected_documents[] in MemoryManager.
        - Temp file auto-deletes in finally block.
        """
    def ask(docqa_session_id, query) -> DocQAResponse:
        """POST /api/chat with session_id. Normalize to UnifiedResponse shape."""
    def list_loaded(session_id) -> list[dict]:
        """Return selected_documents from MemoryManager for this session."""
```

**S3 fetch:** `boto3.client.download_fileobj(Bucket, Key, fp)` into `tempfile.NamedTemporaryFile`. Context manager deletes on exit. If S3 unreachable, fall back to downloading the presigned `download_url` via `httpx.stream`.

**Multi-document:** If user selects 2+ docs (Q4 = A), bridge calls `/api/upload` N times with same `docqa_session_id`. DocQA treats them as one in-session corpus. Deferred enhancement: native multi-doc in DocQA agent (tracked separately per user note).

**Drawing-PDF handling:** Bridge passes raw PDF bytes as-is. DocQA's existing vision pipeline processes. If drawings have weak OCR accuracy, surfaced in final test report as a *known limitation* — not blocking this round.

**Error envelope on failure:**
```json
{
  "success": false,
  "answer": "Could not load document for deep-dive. Falling back to general search.",
  "active_agent": "rag",
  "engine_used": "docqa_fallback",
  ...
}
```
Graceful degrade to RAG — user never sees a 500.

### 5.4 Bidirectional intent classifier (Phase 3)

`gateway/intent_classifier.py` (rewritten):

```python
def classify(query: str, session: Session) -> IntentDecision:
    """
    Returns IntentDecision(target: 'rag'|'docqa'|'clarify', confidence: float, reason: str)
    """
```

**Signal table:**

| Signal | Direction | Weight |
|---|---|---|
| `session.selected_documents` is non-empty | → DocQA | baseline +0.3 |
| Query contains "this/that document", "in this spec", "in the selected drawing", "on page N", "what does it say about" | → DocQA | +0.4 |
| Query contains "across project", "all drawings", "every spec", "missing scope", "summarize project" | → RAG | +0.5 (overrides DocQA bias) |
| Query refers to entity by generic name ("HVAC", "fire damper") with no "this" | neutral | 0 |
| Query is a pronoun-heavy follow-up ("what about it", "and the one on page 5") AND last turn was DocQA | → DocQA | +0.4 |
| Explicit `mode_hint` in request | direct route | 1.0 |

**Threshold:** target with score ≥ 0.7 routes directly. 0.3 ≤ score < 0.7 returns `clarify` with a one-line prompt. < 0.3 defaults to RAG (safer).

**Clarify UX:** Gateway returns `needs_clarification=true`, `clarification_prompt="Should I answer from the selected spec ({file_name}) or search the whole project?"`. UI renders two buttons.

### 5.5 Session storage (Phase 2)

Extend existing `MemoryManager` (Q8 = A). New fields on session:

```python
active_agent: str = "rag"
docqa_session_id: Optional[str] = None
selected_documents: list[dict] = []   # each: {s3_path, file_name, loaded_at, docqa_session_id}
last_intent_decision: Optional[dict] = None
```

Persisted in memory only. Survives process lifetime, not restart — acceptable for sandbox per Q8.

### 5.6 UI changes (Phase 4)

`docs/demo-ui.html`:

1. **Source rendering** — primary label = `file_name` (S3 basename) from `source_documents[]`; secondary = `display_title`. Deduplicate by `s3_path`: if the same object appears twice, merge with pages aggregated.
2. **"Chat with Document" button** — on each source card. On click: POST `/query` with `docqa_document` = selected card's `{s3_path, file_name, download_url, pdf_name}`. Show loader: *"Processing document…"*. On success: system message *"Document processed. You can now ask questions."* and an *"Active agent"* badge flips to DocQA.
3. **Clarification prompt** — when response has `needs_clarification=true`, render inline two-choice buttons ("Ask from selected spec" / "Search whole project"). Selecting one re-submits the query with `mode_hint`.
4. **Back to RAG** — "Back to project search" button in DocQA mode → clears selected_documents → next query auto-routes to RAG.
5. **Backend URL** — stays on sandbox (`http://54.197.189.113:8001`) per Q10.

### 5.7 Download URLs (Phase 1/6)

No code changes — presigned SigV4 URLs already work ([orchestrator.py:175-258](PROD_SETUP/unified-rag-agent/gateway/orchestrator.py#L175-L258)). TTL 1 hour is fine. CloudFront + SSL subdomain parked until credentials arrive (per Q6 note).

Add a test: fetch each `download_url` from each source_doc in response, assert 200 OK. Catches signing regressions.

## 6. Wire contract examples

### 6.1 Normal RAG query (unchanged)

Request:
```json
{"query": "Is fire damper scope missing?", "project_id": 7222, "session_id": "abc"}
```

Response (existing shape + new additive fields):
```json
{
  "success": true,
  "answer": "No fire damper callouts were found in HVAC spec Section 23 00 00 or the mechanical drawings M-101 through M-501…",
  "source_documents": [
    {"s3_path": "agentic-ai-production/…/M-301.pdf", "file_name": "M-301.pdf",
     "display_title": "MECHANICAL FIRST FLOOR PLAN",
     "download_url": "https://…?X-Amz-Signature=…", "pdf_name": "M-301.pdf"}
  ],
  "confidence": "high",
  "active_agent": "rag",
  "selected_document": null,
  "engine_used": "agentic",
  "session_id": "abc"
}
```

### 6.2 User picks doc + asks follow-up

Request:
```json
{
  "query": "Where is fire damper mentioned in this spec?",
  "project_id": 7222,
  "session_id": "abc",
  "docqa_document": {"s3_path": "…/HVAC_SPEC.pdf", "file_name": "HVAC_SPEC.pdf", "download_url": "https://…"},
  "mode_hint": "docqa"
}
```

Response:
```json
{
  "success": true,
  "answer": "Fire damper is covered on page 14 under section 2.3. The spec calls for UL 555 rated…",
  "source_documents": [{"file_name": "HVAC_SPEC.pdf", "page": 14, "snippet": "…"}],
  "confidence": "high",
  "active_agent": "docqa",
  "selected_document": {"file_name": "HVAC_SPEC.pdf", "s3_path": "…"},
  "docqa_session_id": "dq_xyz",
  "engine_used": "docqa",
  "session_id": "abc"
}
```

### 6.3 Ambiguous follow-up (clarify)

Request: `{"query": "is it missing?", "project_id": 7222, "session_id": "abc"}`

Response:
```json
{
  "success": true,
  "answer": "",
  "needs_clarification": true,
  "clarification_prompt": "Should I answer from the selected spec (HVAC_SPEC.pdf) or search the whole project?",
  "active_agent": "rag",
  "selected_document": {"file_name": "HVAC_SPEC.pdf"},
  "session_id": "abc"
}
```

## 7. Data flow diagram

```
 User types question
       │
       ▼
 /query handler  ──► parse QueryRequest (schema-aligned)
       │
       ▼
 IntentClassifier.classify()
       │
       ├── target=rag  ────────────────► existing RAG path (unchanged)
       │                                       │
       │                                       ▼
       │                                  UnifiedResponse (active_agent="rag")
       │
       ├── target=clarify  ─────────────► return clarification prompt
       │
       └── target=docqa  ──────────────► DocQABridge
                                               │
                                               ▼
                                         ensure_document_loaded()
                                           │   │
                                           │   ├─ in session? reuse docqa_session_id
                                           │   └─ no: S3 → temp → /api/upload
                                           ▼
                                         bridge.ask() → /api/chat
                                               │
                                               ▼
                                         normalize → UnifiedResponse(active_agent="docqa")
```

## 8. Testing strategy (Q9 = B Standard)

**Phase gate: each phase must produce green tests before next phase starts.**

| Phase | Tests | Count target | Tooling |
|---|---|---|---|
| 1 | Schema additions (Pydantic) + answer-format regex | 20+ | pytest |
| 1 | Prompt regression: 16 historical queries — no `[Source:`/`HIGH (…)`/`Direct Answer` in answer text | 16 | pytest + LLM fixture |
| 2 | `DocQABridge` unit — mock S3 + mock DocQA HTTP | 25+ | pytest + respx |
| 2 | `docqa_client` session replay — mock DocQA | 10+ | pytest |
| 2 | Integration — real sandbox S3, mock DocQA | 5 | pytest -m integration |
| 3 | Intent classifier — 40-case table-driven | 40 | pytest parametrize |
| 4 | UI manual walk-through via Playwright MVP | 5 flows | manual + screenshots |
| 5 | E2E sandbox — RAG → pick doc → DocQA → back to RAG | 10 flows | pytest -m e2e |
| 5 | Bug-check subagent pass (parallel — 3 agents) | — | code-reviewer + security-reviewer + e2e-runner |
| 6 | Regression vs v1.2-hybrid-ship: latency, URL breakage, source coverage | 16 queries | comparison script |

**Coverage target:** ≥ 85% on `gateway/docqa_bridge.py`, `gateway/intent_classifier.py`; ≥ 80% overall.

**Bug-resolution rule:** do not advance phases while any CRITICAL/HIGH issue is open. Use code-reviewer + security-reviewer in parallel after every phase commit.

## 9. Rollout plan (pause-per-phase, per Q11/B2)

| Phase | Scope | Deliverable | Gate to next phase |
|---|---|---|---|
| 0 | Snapshot `_versions/v2.0-docqa-extension/` from current main | Full tree copy + README | User eyeball: "yes, proceed" |
| 1 | Answer-format fix + schema sync + prompt-cache message ordering (§11) | 1-line prompt fix + models.py additions + message-order refactor + 16-query regression showing `cache_hit_rate > 0.6` on turns 2+ | User: `[Source:]` eradicated; p50 latency drops 40-60% on turns 2+ |
| 2 | `docqa_bridge.py` + session plumbing + S3 fetch | New module + unit+integration tests (green) | User: single-doc upload + chat works in mock E2E |
| 3 | Bidirectional intent classifier | Rewritten classifier + 40 parametrized tests | User: "this document" auto-routes, clarify prompt fires |
| 4 | UI — Chat with Document, badges, clarification, dedupe sources | demo-ui.html updates + Playwright screenshots | User: visual walk-through in browser against sandbox |
| 5 | Full E2E + bug-check parallel subagents | E2E green + zero CRITICAL/HIGH open | User: signoff on test report |
| 6 | Regression diff vs v1.2-hybrid-ship + sign-off doc | comparison.docx + TEST_RESULTS_SUMMARY update | User: ship or list known limitations |

After each phase I stop, summarize, wait for your approval before starting the next. No autonomous multi-phase runs.

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DocQA vision pipeline weak on construction drawings | Medium | Medium | Phase 2 test includes 3 drawings from 7222/7223; if accuracy <60%, surface as known limitation, not blocker |
| S3 download timeouts on large PDFs (>50MB) | Low | Medium | Bridge uses 60s timeout + streaming download; returns graceful fallback |
| Intent classifier mis-routes | Medium | Low | Clarify mode catches ambiguous; 40 tests baseline; logs decision for post-hoc review |
| Presigned URL expires mid-DocQA session | Low | Low | Bridge re-signs on demand via `_ensure_signed_source_urls` |
| Schema additions break old sandbox UI | Low | High | All new fields `Optional` default-null; contract test asserts old `sources` field still present |
| Session memory loss on gateway restart | Known | Low (sandbox) | Documented; Phase 8-future work = MongoDB-backed sessions |
| Two gateway processes racing on same session | Low | Low | Single-worker uvicorn locally; not a sandbox concern |

## 11. OpenAI prompt caching (folded into Phase 1)

OpenAI auto-caches prompt prefixes ≥1024 tokens since Oct 2024 — no `cache_control` param needed (that's Anthropic). Cached tokens get **50% price discount + ~80% latency reduction** on the cached portion. No new library, no code risk; only *message-ordering discipline*.

### Current state audit
- `agentic/core/agent.py` system prompt is ~2000 tokens — well above the 1024 threshold
- ReAct tool definitions (~3000 tokens for 11 MongoDB tools) — also cacheable
- Per-query user message at the END — correct pattern already
- But: conversation_history is *prepended* inline with the user query in some paths (orchestrator.py) — breaks cache on every turn

### Changes (~40 lines total, all in Phase 1)

| File | Change | Benefit |
|---|---|---|
| `agentic/core/agent.py` | Lock message order: `[system_prompt, tool_defs, conversation_history, user_query]`. System prompt + tool defs NEVER change mid-session → always cache-hit after turn 1. | 40-60% latency drop on turns 2+ |
| `gateway/orchestrator.py` | When building multi-query RRF hints (Fix #1), append hint to the user-message *suffix*, never insert into the cached prefix | Preserves cache hit across hint variations |
| `gateway/docqa_bridge.py` (new) | DocQA /api/chat calls already reuse `session_id` so DocQA-side context is cached. Make sure we send same system prompt every time (not per-doc). | Cache hit on DocQA follow-ups |
| Logging | Log `cached_tokens` and `cache_hit_rate` from OpenAI response usage object in every call | Observability: proves the savings |

### Expected measurable impact (added to §8 test plan)
- Turn 1 of new session: **no change** (cold cache)
- Turns 2-N of same session: p50 latency drops from ~19s → ~10-12s
- Cost per session: ~35-45% drop on input tokens
- Success metric: `cache_hit_rate > 0.6` on turns 2+ in the 16-query regression test

### Guardrail
Prompt caching is best-effort — OpenAI may evict. Our fallback is uncached behavior (today's state). No regression risk.

## 12. Out of scope / future work

- Native multi-doc support in DocQA agent (user noted — separate track)
- MongoDB-scoped retrieval by `drawingTitle` (ROADMAP_ASK_AI_FROM_DOCUMENT.md — separate)
- CloudFront + SSL subdomain for download URLs (pending credentials)
- Persistent session store (Mongo/Redis)
- Auth middleware (user explicitly deferred)
- VM deploy (user explicitly deferred)
- Streaming for DocQA path (Phase 7 if needed)
- Redis LRU cache layer (Phase 7 if needed)
- boto3 connection pooling tuning (Phase 7 if needed)

## 13. Open questions (none remaining for build)

All 12 brainstorming questions answered. Additional clarifications B1, B2 answered. No blockers.

## 14. Approval checklist

Before I produce the implementation plan, please confirm:

- [ ] Architecture in §4 matches your mental model
- [ ] Non-goals (§2) and hard constraints (§3) captured correctly
- [ ] Phase split (§9) is the granularity you want
- [ ] Testing depth (§8) is "Standard" per Q9
- [ ] No deploy to VM; sandbox-only; local dev OK (per Q10)
- [ ] Anything missing or misframed — flag before I write the plan

Once confirmed, I'll invoke `writing-plans` to produce the phase-by-phase task list with acceptance criteria for each phase. No code until Phase 0 approval.
