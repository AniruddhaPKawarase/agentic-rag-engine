# Phase 6 Sign-off — unified-rag-agent v2.0 DocQA extension

**Date:** 2026-04-23
**Baseline:** `_versions/v1.2-hybrid-ship` (19s avg latency, 0 broken URLs, 61/61 unit tests)
**Target version:** `_versions/v2.0-docqa-extension-final`

## Scope delivered

### Phase 1 — Answer format + schema + cache ordering
- `agentic/core/agent.py`: deleted contradictory `[Source:]` citation instruction (commit `aa91888`)
- `gateway/models.py`: additive-only optional fields `docqa_document`, `mode_hint` on `QueryRequest`;
  `source_documents`, `active_agent`, `selected_document`, `needs_clarification`,
  `clarification_prompt`, `docqa_session_id`, `groundedness_score`, `flagged_claims`
  on `UnifiedResponse` (commit `d3fd28c`)
- `agentic/core/agent.py`: `build_react_messages()` + `_log_cache_metrics()` for OpenAI
  auto-cache prefix ordering (commit `09173e9`)
- Phase 1 regression harness — 16 historical queries + cache proxy (commit `fcce3be`)

### Phase 2 — DocQA bridge + session plumbing
- `traditional/memory_manager.py`: `ConversationSession` gets `active_agent`,
  `docqa_session_id`, `selected_documents`, `last_intent_decision` (commit `1d6c052`)
- `gateway/docqa_client.py`: `upload_only()` helper (commit `731032b`)
- `gateway/docqa_bridge.py`: `DocQABridge` class — S3 fetch, DocQA upload, session
  persistence, normalize to UnifiedResponse (commit `a9acb56`)
- `gateway/orchestrator.py`: `_run_docqa` refactored to use bridge (commit `2e5601e`)
- `gateway/router.py`: forward `docqa_document` to orchestrator (commit `174b179`)
- Phase 2 smoke tests (commit `2f450e0`)

### Phase 3 — Bidirectional intent classifier
- `gateway/intent_classifier.py`: rewrote as `classify()` returning `IntentDecision`
  with weighted scoring (project-wide +0.9 override, doc-scoped +0.4, selected-doc +0.3,
  pronoun-only +0.2, threshold 0.7 route, 0.3-0.7 clarify, <0.3 default RAG).
  Legacy `classify_intent` preserved. (commit `59a49d4`)
- `gateway/orchestrator.py`: classifier wired at top of `query()` with clarify envelope
  + auto-promote selected_documents into docqa_document (commit `4f96dfd`)

### Phase 4 — UI
- `docs/demo-ui.html`: source cards dedupe by s3_path, filename-primary label,
  `display_title` subtitle, aggregated pages, Download link (presigned), "Chat with
  Document" passes full source object, clarify-prompt two-button UX,
  `chatWithDocumentFull()`, `resubmitWithHint()`, `state.lastQuery` (commit `4521897`)

### Phase 5 — Testing + security fixes
- Phase 5.1: 10 E2E flows at `tests/e2e/test_docqa_flow.py` (commit `1966e9a`)
- Phase 5.2: 8 review findings resolved — SSRF validator, S3 bucket allowlist, session
  guards, download size cap, filename sanitization, classifier bypass, session
  persistence, DocQA upload telemetry (commit `e0a5d81`)

### Phase 6 — This round
- Regression harness vs baseline: `tests/test_phase6_regression_vs_v12.py`
- Final snapshot: `_versions/v2.0-docqa-extension-final/`
- This doc

## Test summary

| Suite | Count | Status |
|---|---:|---|
| Phase 1 unit (answer-format + schema + cache ordering) | 17 | PASS |
| Phase 2 unit (session, docqa_client, bridge, orchestrator route) | 22 | PASS |
| Phase 3 unit (classifier 37 parametrized + wiring) | 44 | PASS |
| Phase 4 UI (manual browser check) | — | engineer-run |
| Phase 5 security + intent-routing | 17 | PASS |
| Phase 1 regression (live, 16 queries) | 16 | **PASS — verified 2026-04-23 on :8002 local run** |
| Phase 2 smoke (live, 3 flows) | 3 | 1 PASS + 2 skip/fail (no DocQA 8006 in controller env) |
| Phase 5 E2E (live, 10 flows) | 10 | **engineer-run at gate** (requires DocQA reachable) |
| Phase 6 regression vs baseline (live, 16 queries) | 1 agg | **engineer-run at gate** |
| Pre-existing failures (aggregation_tools + rag_pipeline async) | 10 + 1 | unchanged, unrelated |

## Metrics (vs v1.2-hybrid-ship baseline)

Placeholder — engineer runs `tests/test_phase6_regression_vs_v12.py` and
pastes the printed aggregate below.

| Metric | v1.2-hybrid-ship | v2.0 | Delta |
|---|---:|---:|---:|
| Avg latency (cold) | 19.0 s | [run regression to populate] | — |
| p50 latency (cold) | — | [populate] | — |
| Turn 2+ latency (warm, same session) | 19.0 s | [populate] | target ≥40% drop (prompt cache) |
| Queries with broken URL | 0/16 | [populate] | must be 0 |
| Queries with zero sources | 0/16 | [populate] | must be ≤1 |
| [Source:] / HIGH (N%) / Direct Answer artifacts | present | **0** | cleared (verified Phase 1 local run) |
| Cache hit rate turn 2+ | n/a | [populate from OpenAI usage logs] | target >60% |

## User-story coverage

- [x] RAG answers with source_documents[] — unchanged
- [x] Source cards show filename + download link (Phase 4)
- [x] User picks doc → DocQA takes over (Phase 2 bridge, Phase 4 button)
- [x] Follow-up questions scoped to selected doc (bridge session persistence)
- [x] "Across project" auto-switches back to RAG (Phase 3 classifier with +0.9 override)
- [x] Clarify prompt on ambiguous queries (Phase 3 + UI yellow box)
- [x] Multi-doc upload same session (bridge uses same docqa_session_id)
- [x] No schema breakage for old UI (all new fields Optional + default)

## Known limitations (parked)

| # | Item | Why parked |
|---|---|---|
| 1 | DocQA agent multi-doc native support | User noted: track separately, add later |
| 2 | MongoDB-scoped retrieval by drawingTitle | ROADMAP_ASK_AI_FROM_DOCUMENT.md — different track |
| 3 | CloudFront + SSL subdomain for download URLs | Pending S3 credential rotation ("discuss later") |
| 4 | Persistent session store (Mongo/Redis) | In-memory OK for sandbox (Q8=A) |
| 5 | Auth middleware (JWT / OAuth / keys) | User explicitly deferred ("open for all for now") |
| 6 | VM deploy | Phase 6 ships local-only per Q10=A |
| 7 | `docqa_document` as Pydantic sub-model (typed) | MED finding parked post-ship |
| 8 | `search_mode` / `mode_hint` as `Literal[...]` | MED finding parked post-ship |
| 9 | `_format_docqa_response` dead code | LOW cleanup |
| 10 | `_last_*` instance-state race under concurrency | LOW, pre-existing |
| 11 | Hardcoded `application/pdf` MIME in multipart | LOW, no non-PDFs in pipeline |
| 12 | `__upload__` placeholder query sent to DocQA `/api/converse` | Telemetry added; verify DocQA team |
| 13 | Test pollution: `test_prompt_cache_ordering.py` fails in full-suite only | Fix via conftest isolation in post-ship pass |

## Revert plan

If Phase 1-6 needs rollback:
```bash
cd PROD_SETUP/unified-rag-agent
rsync -a _versions/v2.0-docqa-extension/ ./ \
  --exclude README.md --exclude VERSIONS.md
```

## Recommendation

**READY FOR CUMULATIVE GITHUB PUSH** after engineer confirms:
1. Live 10-flow E2E suite green on a box with DocQA 8006 reachable
2. Phase 6 regression harness populates the metrics table with acceptable numbers
3. Browser walk-through of `docs/demo-ui.html` against sandbox confirms UI

Push target per user decision:
- Rsync current tree → `agentic-ai-platform/agents/rag-engine/`
- New branch: `feature/docqa-extension-phase-1` (cut from `feature/phase-1-security-architecture-prereq`)
- Cumulative single push after all phases
- Bump parent submodule pointer on `agentic-ai-platform` and push parent

## Commit history (Phase 0-6 on this local branch)

```
e0a5d81 fix(security+correctness): Phase 5 review fixes (4 CRITICAL + 4 HIGH)
1966e9a test(e2e): Phase 5 — 10-flow end-to-end suite
4521897 feat(ui): Phase 4 source card dedupe + clarify prompt + download link
4f96dfd feat(orchestrator): auto-route via intent classifier + clarify path
59a49d4 feat(classifier): bidirectional intent routing with clarify mode
174b179 fix(router): forward docqa_document to orchestrator.query()
2f450e0 test: Phase 2 smoke harness for RAG->DocQA handoff
2e5601e feat(orchestrator): route search_mode=docqa through DocQABridge
a9acb56 feat(bridge): add DocQABridge adapter for S3→DocQA handoff
731032b feat(docqa_client): add upload_only helper for bridge
1d6c052 feat(session): track active_agent + docqa_session_id + selected_documents
fcce3be test: Phase 1 regression harness — 16 baseline queries + cache proxy
09173e9 perf(agent): lock message order for OpenAI auto-cache
d3fd28c feat(models): add additive optional fields for DocQA bridge
aa91888 fix(agent): remove contradictory [Source:] citation instruction
```
