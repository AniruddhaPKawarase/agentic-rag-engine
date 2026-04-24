# Phase 2 Smoke Test — RAG→DocQA Handoff

Validates the bridge end-to-end against a running gateway + reachable DocQA
agent. Skipped by default; runs only when `GATEWAY_URL` is set.

## How to run

### Terminal 1 — start local gateway
```
cd PROD_SETUP/unified-rag-agent
python -m gateway.app
# wait for: "Uvicorn running on http://0.0.0.0:8001"
```

Also ensure the Document QA agent is reachable at `http://localhost:8006`
(or wherever `DOCQA_BASE_URL` points).

### Terminal 2 — run smoke
```
cd PROD_SETUP/unified-rag-agent
GATEWAY_URL=http://localhost:8001 \
  python -m pytest tests/test_phase2_smoke.py -v -m integration
```

Expected: 3 PASS.

## What's being asserted

1. **RAG→DocQA handoff** — select a source doc, DocQA engages (or falls
   back gracefully with `engine_used=docqa_fallback`).
2. **Session reuse** — selecting the same doc twice keeps the same
   `docqa_session_id` (no re-upload to DocQA).
3. **Graceful error** — broken s3_path returns a fallback response,
   NOT a 5xx. `engine_used=docqa_fallback`, `success=false`,
   `fallback_used=true`.

## If test 1 falls back (engine_used=docqa_fallback)

DocQA agent at port 8006 is unreachable OR the bridge can't download the
S3 object. Check:
- DocQA agent running: `curl http://localhost:8006/health`
- AWS credentials set in environment and S3 bucket accessible
- `download_url` in the source_document is still valid (TTL ~1h on presigned URLs)

## If test 2 fails with different docqa_session_ids

Bridge reuse logic broken. Check `DocQABridge.ensure_document_loaded` in
`gateway/docqa_bridge.py` — it compares by `s3_path` on
`session.selected_documents`.

## If test 3 returns 5xx

Exception handling in `orchestrator._run_docqa` broken. The `except
Exception as exc: bridge.normalize(error=...)` block should catch anything.
