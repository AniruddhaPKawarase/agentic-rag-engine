# Phase 1 Regression Harness

Validates that Phase 1's answer-format fix and prompt-cache ordering
changes do not regress baseline behavior.

## How to run

### Terminal 1 — start local gateway

```
cd PROD_SETUP/unified-rag-agent
python -m gateway.app
# wait for "Uvicorn running on http://0.0.0.0:8001"
```

### Terminal 2 — run harness

```
cd PROD_SETUP/unified-rag-agent
GATEWAY_URL=http://localhost:8001 \
  python -m pytest tests/test_phase1_regression.py -v -m integration
```

Expected: 16 artifact checks PASS. 1 cache-hit proxy check PASS (may show
variance — re-run once if it fails).

## What's being asserted

1. **No artifacts in answer prose** — none of `[Source: …]`, `Direct Answer`,
   `HIGH (NN%)` appears in `response.answer`.
2. **Cache hit proxy** — second query in the same session is ≥10% faster
   than the first (OpenAI auto-cache kicks in on the stable prefix).

## If an artifact check fails

Phase 1.1 fix was reverted or incomplete. Re-check `agentic/core/agent.py` —
no line should say `Cite sources: [Source: …]`. If found, re-apply Phase 1.1.

## If cache check fails

Phase 1.3 refactor may be broken. Check `agentic/core/agent.py` — the
`build_react_messages` function should be used in `run_agent()`, not inline
message construction. Also verify orchestrator.py does NOT prepend a
system message for RRF hints — it should pass them via `scope['rrf_hint']`.
