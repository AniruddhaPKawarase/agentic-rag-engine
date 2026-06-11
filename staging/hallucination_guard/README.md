# Hallucination Guard — Staging Workspace

Local staging dir for the **Tactical Patch v2** rollout to 8001 PROD.
Per roadmap §5.21: Pillars 1 + 4 + 6 + 7 + 11, additive, env-flag-gated.

## Status

| Deliverable | Status | Files |
|---|---|---|
| (a) Pre-flight + rollback | DONE | `scripts/preflight_check.sh`, `scripts/snapshot_baseline.sh`, `scripts/rollback.sh` |
| (b) Adversarial eval suite (50 Qs + grader) | DONE | `tests/adversarial_gold_set.jsonl`, `tests/grader.py`, `tests/run_eval.py` |
| (c) Pillar 6 anti-anchor module | DONE | `agentic/hallucination_guard/__init__.py`, `flags.py`, `pillar_6_anti_anchor.py`, `test_pillar_6.py` |
| Pillar 1 (anaphora resolver) | NEXT | _to be built_ |
| Pillar 4 (per-claim verifier) | NEXT | _to be built_ |
| Pillar 7 (citation-forcing schema) | NEXT | _to be built_ |
| Pillar 11 (sycophancy + injection) | NEXT | _to be built_ |

## Workspace Layout

```
hallucination_guard/
├── README.md                              ← you are here
├── scripts/
│   ├── preflight_check.sh                 ← pre-flight safety + state check
│   ├── snapshot_baseline.sh               ← create rollback snapshot
│   └── rollback.sh                        ← one-command rollback
├── tests/
│   ├── adversarial_gold_set.jsonl         ← 50 red-team queries
│   ├── grader.py                          ← rule-based + LLM-judge auto-grader
│   └── run_eval.py                        ← HTTP runner against 8001 /query
├── agentic/
│   └── hallucination_guard/               ← drop-in Python package for 8001 app dir
│       ├── __init__.py                    ← public API: is_enabled(), get_active_pillars(), ...
│       ├── flags.py                       ← env-flag plumbing
│       ├── pillar_6_anti_anchor.py        ← 6-rule prompt prepend
│       └── test_pillar_6.py               ← unit tests (run in place after copy)
└── docs/
    ├── HALLUCINATION_GUARD_DEPLOYMENT.md  ← full runbook (Day 0 → Day 4)
    └── PILLAR_6_INTEGRATION.md            ← integration guide for synthesizer
```

## Quick Start (the 4-day plan condensed)

| Day | Step | Time | Action |
|---|---|---|---|
| 0 | Pre-flight + snapshot | 2 hr | SCP staging tree → run `preflight_check.sh` → `snapshot_baseline.sh` → dry-run `rollback.sh` |
| 1A | Baseline eval | 6 hr | Run `run_eval.py` + `grader.py` against 8001 with all HG flags OFF |
| 1B | Ship Pillar 6 | 2 hr | Copy `agentic/hallucination_guard/` to app dir → integrate per `PILLAR_6_INTEGRATION.md` → restart → re-eval |
| 2 | Ship Pillar 7 | 8 hr | _Next deliverable batch_ |
| 3 | Ship Pillars 1 + 4 + 11 | 12 hr | _Next deliverable batch_ |
| 4 | Final eval + decision file | 4 hr | Master flag ON; full 50-Q eval; write decision file |

Target: **-50% hallucination** vs baseline.

## Doctrine (HARD RULES this code obeys)

From `HYBRID_RAG_v32_ROADMAP.md` §10.1, §10.3:

- 8001 PROD is never auto-modified — every change requires explicit user approval (got it 2026-06-03)
- VCSAI safety check before every service bounce (script auto-checks)
- All flags default OFF (this is the kill-switch)
- Reversible in 30 seconds via master flag
- `drawings_v2` / `specifications_v2` collections never touched (verified — no references in this package)
- `ai_assistant.pem` never used from this workspace (sandbox key only)
- Refusals are clean, never hedged (Pillar 6 rule 2 + rule 4)
- Every factual claim must cite (Pillar 6 rule 6 — pre-Pillar-7 enforcement via prompt; Pillar 7 enforces at decode)

## Next Deliverable Batch (when ready)

To complete the patch, four more pillars need code:

1. **Pillar 7 — Citation-forcing structured output** (1 day)
   - Pydantic schema: `AnswerClaim` with required `citation_chunk_id`
   - Decoder reject path (retry 3, then refuse)
   - Integration: replace `client.messages.create(...)` with structured-output call
2. **Pillar 1 — L1 anaphora resolver** (0.5 day)
   - One Haiku call before retrieval; rewrites query with prior turn context
   - Integration: pre-retrieval hook
3. **Pillar 4 — Minimal Lynx-style verifier** (1 day)
   - One Haiku call after synthesis; faithfulness check per atomic claim
   - Adds `[unverified]` tag on flagged claims
4. **Pillar 11 — Sycophancy + injection detector** (1 day)
   - Pattern matcher for "are you sure?", "ignore previous", etc.
   - Routes sycophancy to verify-existing path
   - Refuses on injection detection

Same env-flag pattern: each pillar has `HG_PILLAR_N`; master flag gates all.

When the next batch is ready, the integration into 8001 is incremental — flip one sub-flag, eval, repeat. Order: Pillar 6 → Pillar 7 → Pillar 1 → Pillar 11 → Pillar 4 (riskiest last, per latency overhead).

## Deployment Authority

The deployment runbook (`docs/HALLUCINATION_GUARD_DEPLOYMENT.md`) is the **single source of truth** for the rollout sequence. Do not deviate without writing a decision file.
