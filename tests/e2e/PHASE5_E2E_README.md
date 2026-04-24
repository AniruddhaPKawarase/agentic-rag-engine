# Phase 5 End-to-End Suite

10 tests covering the full user story end-to-end.

## How to run

Terminal 1:
```bash
cd PROD_SETUP/unified-rag-agent
python -m gateway.app   # wait for "Uvicorn running on http://0.0.0.0:8001"
```

Also ensure DocQA agent is reachable at `http://localhost:8006` (or wherever
`DOCQA_BASE_URL` points).

Terminal 2:
```bash
cd PROD_SETUP/unified-rag-agent
GATEWAY_URL=http://localhost:8001 \
  python -m pytest tests/e2e/test_docqa_flow.py -v -m e2e
```

Expected: 10 PASS. Flows 3, 5, 6 skip gracefully if DocQA agent is not
running (the other flows still exercise RAG + classifier + fallback).

## What's covered

1. **test_flow1_plain_rag_returns_sources** — baseline RAG with sources
2. **test_flow2_rag_then_docqa_handoff** — select source → DocQA engages
3. **test_flow3_docqa_followup_reuses_session** — docqa_session_id reuse
4. **test_flow4_auto_switch_back_to_rag_on_project_wide_query** — Phase 3
   classifier routes "across project" back to RAG
5. **test_flow5_clarify_prompt_on_ambiguous_pronoun** — "is it missing?"
   → needs_clarification envelope
6. **test_flow6_mode_hint_docqa_overrides_classifier** — explicit
   mode_hint=docqa wins
7. **test_flow7_download_url_is_live_and_fetchable** — presigned URLs work
8. **test_flow8_answer_format_clean_across_multiple_queries** — no
   [Source:], Direct Answer, HIGH (NN%) artifacts
9. **test_flow9_docqa_error_returns_graceful_fallback** — broken s3_path
   → engine_used=docqa_fallback, fallback_used=True
10. **test_flow10_schema_backward_compatible** — old UI field names present
