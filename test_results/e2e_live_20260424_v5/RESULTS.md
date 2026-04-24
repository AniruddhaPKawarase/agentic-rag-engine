# Live E2E Run v5 — 14/14 PASS (all green)

**Date:** 2026-04-24
**Gateway:** `:8009` (fresh after Mongo reconnect)
**DocQA agent:** `:8006`
**Duration:** 18 min 34s
**Total tests:** 14 — **14 PASS / 0 FAIL** ✅

## Per-test results

| # | Test | Status |
|---|---|---|
| 1 | `test_handoff_rag_to_docqa_same_session` | **PASS** |
| 2 | `test_handoff_reuses_docqa_session_on_followup` | **PASS** |
| 3 | `test_docqa_error_returns_graceful_fallback` | **PASS** |
| 4 | `test_flow1_plain_rag_returns_sources` | **PASS** |
| 5 | `test_flow2_rag_then_docqa_handoff` | **PASS** |
| 6 | `test_flow3_docqa_followup_reuses_session` | **PASS** |
| 7 | `test_flow4_auto_switch_back_to_rag_on_project_wide_query` | **PASS** |
| 8 | `test_flow5_clarify_prompt_on_ambiguous_pronoun` | **PASS** |
| 9 | `test_flow6_mode_hint_docqa_overrides_classifier` | **PASS** |
| 10 | `test_flow7_download_url_is_live_and_fetchable` | **PASS** |
| 11 | `test_flow8_answer_format_clean_across_multiple_queries` | **PASS** |
| 12 | `test_flow9_docqa_error_returns_graceful_fallback` | **PASS** |
| 13 | `test_flow10_schema_backward_compatible` | **PASS** |
| 14 | `test_phase6_regression` | **PASS** |

## Phase 6 aggregate (all quality metrics GREEN)

```
avg latency:  17.0s   (baseline v1.2: 19.0s, threshold: 22.0s)    [-11%]
p50 latency:  14.6s   (baseline v1.2: ~18s)                        [-19%]
zero-source:  2/16    (threshold: <=4)                              PASS
broken URLs:  0       (was 282 in v3 — URL fix solid)               PASS
artifact:     0       (no [Source:]/HIGH (N%)/Direct Answer)        PASS
```

## Cumulative improvements across runs

| Run | Pass/Total | broken URLs | zero-src | p50 | Bugs fixed |
|---|---:|---:|---:|---:|---|
| v1 (20260423) | 12/14 | 278 | 2 | 12.6s | filename suffix |
| v2 (20260423) | 12/14 | 282 | 2 | 11.0s | (no change) |
| v3 (20260423) | 8/14 skip | 0 | 2 | 18.0s | HEAD→GET |
| v4 (20260423) | 13/14 | 0 | 3 | 15.8s | key construction, size guard |
| **v5 (20260424)** | **14/14** | **0** | **2** | **14.6s** | **(no new bugs) — CLEAN PASS** |

## User story — all flows verified live end-to-end

✅ RAG returns sources
✅ User picks doc → DocQA takes over
✅ DocQA follow-up reuses docqa_session_id (no re-upload)
✅ "Across project" auto-switches back to RAG (Phase 3 classifier)
✅ Ambiguous pronoun fires clarify prompt
✅ Explicit mode_hint overrides classifier
✅ Presigned download_urls reachable
✅ No answer-format artifacts across 16 queries
✅ Graceful docqa_fallback on broken s3_path (no 5xx)
✅ Schema backward-compatible (old UI clients unaffected)

## Ship-readiness: READY

All prior bugs resolved, zero regressions, latency 11% below v1.2-hybrid-ship baseline.

PRs open and ready to merge:
- Submodule: https://github.com/AniruddhaPKawarase/agentic-rag-engine/pull/2 (commit `da31592`)
- Parent:    https://github.com/AniruddhaPKawarase/agentic-ai-platform/pull/1 (commit `6dedc3f`)
