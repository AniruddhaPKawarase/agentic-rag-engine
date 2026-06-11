# Patch Plan — Pillar 6 + Pillar 11 Integration into 8001 PROD

**Date:** 2026-06-03
**Scope:** Tactical Patch v2, batch 1 of 2 (Pillars 6 + 11 of 5)
**Risk:** LOW — additive, env-flag-gated, default OFF, ~30-second reversal
**Approval needed:** explicit per HARD RULE before applying any diff

---

## Architecture Discovered

| Component | Path on sandbox | Role | Touched? |
|---|---|---|---|
| `/query` handler | `gateway/router.py:333` | Receives QueryRequest, fans out to orchestrator + multi-source | **YES** (Pillar 11 entry) |
| `/query/stream` handler | `gateway/router.py:737` | SSE wrapper — runs same path via `_run_query_offloop`, streams word-by-word | **YES** (Pillar 11 entry) |
| Primary synthesizer | `agentic/generation/synthesizer.py:85` | Compresses ReAct output → final response (Haiku-class model); called per query | **YES** (Pillar 6 wrap) |
| Multi-source merger | `gateway/synthesizer.py:91` | Only fires when ≥2 sources (RAG + email + meeting); GPT-4.1 | **YES** (Pillar 6 wrap, secondary) |
| Orchestrator | `gateway/orchestrator.py` | The deep RAG path called from router | NOT TOUCHED |

**Key insight:** the system prompt is assembled deep in `agentic/generation/synthesizer.py` (not in router). For Pillar 11 sycophancy addendum to reach the synthesizer would require threading through orchestrator → synthesizer (touches 3+ files). This batch **defers sycophancy-addendum threading** to a later batch and relies on Pillar 6 rule #3 to provide the "hold the line" behavior — overlap is ~70%. Pillar 11 in this batch handles INJECTION cleanly (early-return) and logs sycophancy detections for observability.

This simplification keeps the patch scope to **3 files + .env**.

---

## Files Modified — Exact Diff

### File 1 — `agentic/generation/synthesizer.py` (Pillar 6 wrap)

**Location:** lines 85-87 (system prompt assembly area).

**Diff:**

```diff
 from agentic.generation.llm_client import generate
 from agentic.generation.text_normalizer import normalize_chunks, normalize_output

+# Hallucination Guard v2 — Pillar 6 anti-anchor (env-flag gated; no-op when off)
+try:
+    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
+except Exception:  # pragma: no cover — package absent or import-time error → no-op
+    def _hg_p6_wrap(s: str) -> str:
+        return s
+
 logger = logging.getLogger("agentic_rag.generation.synthesizer")
```

Then in `synthesize()`, lines 85-87 become:

```diff
     system_prompt = _SYSTEM_PROMPT
     if shape_block:
         system_prompt = f"{shape_block}\n\n{system_prompt}"
+    # Pillar 6 anti-anchor wrap — prepends 6-rule doctrine when HG_PILLAR_6=true.
+    # No-op when flag is off (verified by 17 unit tests).
+    system_prompt = _hg_p6_wrap(system_prompt)
```

**Lines changed:** ~10 added at top (try/except import); 2 added in body. **Zero lines removed.**

### File 2 — `gateway/synthesizer.py` (Pillar 6 wrap, secondary)

**Location:** line 91 (Anthropic-style call uses OpenAI here; same pattern).

**Diff:**

```diff
 from openai import AsyncOpenAI

+# Hallucination Guard v2 — Pillar 6 anti-anchor (env-flag gated; no-op when off)
+try:
+    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
+except Exception:
+    def _hg_p6_wrap(s: str) -> str:
+        return s
+
 logger = logging.getLogger(__name__)
```

Inside `synthesize_answer()`:

```diff
     try:
         completion = await asyncio.wait_for(
             _openai().chat.completions.create(
                 model=model,
                 messages=[
-                    {"role": "system", "content": _SYSTEM},
+                    {"role": "system", "content": _hg_p6_wrap(_SYSTEM)},
                     {"role": "user", "content": prompt},
                 ],
                 max_tokens=600,
                 temperature=0,
             ),
             timeout=timeout,
         )
```

**Lines changed:** ~8 added at top; 1 line modified in body. Zero removals.

### File 3 — `gateway/router.py` (Pillar 11 pre-filter at both /query entry points)

**Location:** add helper at top of file; pre-filter at line 333 (`/query`) and line 737 (`/query/stream`).

**Step 3a — top-of-file helper (after existing imports):**

```diff
 # ... existing imports ...

+# ─── Hallucination Guard v2 — Pillar 11 adversarial pre-filter (env-flag gated) ─
+try:
+    from agentic.hallucination_guard.pillar_11_adversarial import (
+        pre_filter_query as _hg_p11_filter,
+        applied_metadata as _hg_p11_meta,
+    )
+    from agentic.hallucination_guard import get_active_pillars as _hg_active_pillars
+except Exception:  # pragma: no cover — package absent → all-allow stub
+    class _HG_AllowDecision:
+        action = "allow"
+        reason_class = "package_unavailable"
+        matched_patterns: list = []
+        refusal_text = None
+        prompt_addendum = None
+        regulatory_reframe_detected = False
+        pillar_applied = False
+    def _hg_p11_filter(q, has_history=False, force_enable=None):
+        return _HG_AllowDecision()
+    def _hg_p11_meta(decision=None):
+        return {"pillar": 11, "applied": False}
+    def _hg_active_pillars():
+        return set()
+
+
+def _hg_build_refusal_response(decision, body, resolved_session_id=None) -> dict:
+    """Build a /query response dict for a Pillar 11 injection refusal.
+    Returns a dict matching the /query response shape; skips retrieval + synthesis.
+    """
+    return {
+        "answer": decision.refusal_text,
+        "final_answer": decision.refusal_text,
+        "session_id": resolved_session_id,
+        "engine_used": "hallucination_guard_pillar_11",
+        "sources": [],
+        "all_retrieved_sources": [],
+        "verification_meta": {
+            "refused": True,
+            "reason_class": decision.reason_class,
+            "active_pillars": sorted(_hg_active_pillars()),
+            "pillar_11": _hg_p11_meta(decision),
+        },
+    }
```

**Step 3b — `/query` handler entry (insert after line 334, before line 336 imports):**

```diff
 @router.post("/query")
 async def query(request: Request, body: QueryRequest) -> dict:
     """Route query to selected sources, return per-source answers + synthesized final_answer."""
+    # ─── Pillar 11 pre-filter (Hallucination Guard v2) ───
+    _hg_has_history = bool(body.conversation_history)
+    _hg_decision = _hg_p11_filter(body.query, has_history=_hg_has_history)
+    if _hg_decision.action == "refuse_injection":
+        logger.info(
+            "[hg-p11] /query refused — reason=%s patterns=%d",
+            _hg_decision.reason_class, len(_hg_decision.matched_patterns),
+        )
+        return _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)
+
     from gateway.email_hybrid_search import search_emails
     from gateway.external_sources import search_meetings, search_rfis
     from gateway.synthesizer import synthesize_answer
```

**Step 3c — `/query/stream` handler entry (insert after line 751 `"""`, before `async def event_generator()`):**

```diff
 @router.post("/query/stream")
 async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
     """SSE streaming query — word-by-word token output. ..."""
+    # ─── Pillar 11 pre-filter (Hallucination Guard v2) ───
+    _hg_has_history = bool(body.conversation_history)
+    _hg_decision = _hg_p11_filter(body.query, has_history=_hg_has_history)
+    if _hg_decision.action == "refuse_injection":
+        logger.info(
+            "[hg-p11] /query/stream refused — reason=%s patterns=%d",
+            _hg_decision.reason_class, len(_hg_decision.matched_patterns),
+        )
+        refusal_dict = _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)
+        async def _hg_refusal_stream():
+            yield _sse("status", {"phase": "refused"})
+            yield _sse("token", {"delta": _hg_decision.refusal_text})
+            yield _sse("done", refusal_dict)
+        return StreamingResponse(_hg_refusal_stream(), media_type="text/event-stream")
+
     async def event_generator() -> Any:
```

**Step 3d — (optional) verification_meta annotation on the non-refused path:**

For non-refused responses, you can optionally annotate the response with which pillars fired. This requires a small modification AFTER the orchestrator response is built. Skip for tactical v2; add in v3.

**Files modified:** 3. **Lines added:** ~80. **Lines removed:** 0. **Lines modified in place:** 1.

---

### File 4 — `.env` (flag additions; default OFF)

Append at end:

```bash
# ──────────────────────────────────────────────────────────────────────────
# Hallucination Guard v2 — Tactical Patch (added 2026-06-03)
# Master flag — set to true to enable any pillar. Reversible in 30 seconds.
# ──────────────────────────────────────────────────────────────────────────
HALLUCINATION_GUARD_ENABLED=false
HG_PILLAR_1=false
HG_PILLAR_4=false
HG_PILLAR_6=false
HG_PILLAR_7=false
HG_PILLAR_11=false
```

**Why default OFF:** code lands first; we run the same baseline eval to confirm zero behavior change with flags OFF; then we flip flags per pillar.

---

## Deployment Sequence

Before any change:
1. ✅ VCSAI safety check (preflight_check.sh in staging/scripts/) — must show no active in-process jobs
2. ✅ Baseline snapshot already exists (`hg_v2_baseline_20260603T102905Z`)

Then:
1. Copy `staging/hallucination_guard/agentic/hallucination_guard/` → `agentic/hallucination_guard/`
2. Apply the 3 file patches above
3. Append the `.env` block
4. `python3 -m pytest agentic/hallucination_guard/ -v` — all 57 tests must pass in place
5. `sudo systemctl restart rag-agent.service`
6. Verify service healthy (`systemctl is-active`, `/health 200`)
7. **Smoke test with all flags OFF** — send 5 queries; confirm responses identical to pre-deploy
8. Re-run **baseline eval suite** — overall pass-rate must match the 44% baseline (±2%) to confirm no behavior change with flags OFF
9. Flip `HALLUCINATION_GUARD_ENABLED=true HG_PILLAR_6=true HG_PILLAR_11=true`
10. Restart + re-run eval — measure lift

---

## Expected Lift After Step 10

Based on baseline + Pillar 11 unit-test coverage:

| Category | Baseline | Target with P6+P11 | Mechanism |
|---|---|---|---|
| injection | 20% (1/5) | **80-100%** | Pillar 11 early-refuses all 5 patterns; unit tests catch AGS-046/047/050 |
| anti_anchor | 60% (6/10) | **80%+** | Pillar 6 rule #1 + rule #4 directly address carry-forward |
| sycophancy | 40% (2/5) | **60-80%** | Pillar 6 rule #3 partially covers; Pillar 11 logs but doesn't enhance prompt in v2 batch 1 |
| citation_forcing | 40% (4/10) | **50-60%** | Pillar 6 rule #6 ("cite every claim") — modest lift from prompt-only |
| anaphora | 50% (5/10) | **50-55%** | No direct intervention from P6/P11; Pillar 1 ships next batch |
| verifier_per_claim | 40% (4/10) | **50%** | Pillar 6 rule #1 helps marginally; Pillar 4 ships next batch |
| **OVERALL** | **44%** | **~62-68%** | Net **+18-24%pp** from this batch |

---

## Rollback (any of three levels)

1. **Flag-off (fastest, 30 sec):** `sudo sed -i 's/^HALLUCINATION_GUARD_ENABLED=.*/HALLUCINATION_GUARD_ENABLED=false/' .env && sudo systemctl restart rag-agent.service`
2. **Per-pillar flag-off:** flip just `HG_PILLAR_6=false` or `HG_PILLAR_11=false`, restart
3. **Full code rollback:** `staging/scripts/rollback.sh hg_v2_baseline_20260603T102905Z`

---

## Files NOT Touched (HARD RULE compliance check)

| File | Reason not touched |
|---|---|
| `drawings_v2` Mongo collection | READ-ONLY HARD RULE |
| `specifications_v2` Mongo collection | READ-ONLY HARD RULE |
| `gateway/orchestrator.py` | Deferred — would require sycophancy-addendum threading; not needed for batch 1 |
| `gateway/intent_classifier.py` | No-op for batch 1; Pillar 1 ships next batch |
| Any persona-related code | Persona is 8001-OFF per HARD RULE |
| `gateway/feedback_router.py` | Out of scope |
| `agentic/tools/registry.py` | ENABLE_SPECIFICATIONS gate untouched |
| `_backups/` directory | Read-only reference |

---

## Approval Required

Approve THIS exact diff before I apply it on the sandbox. If any block above looks wrong, point to the diff section and we revise before touching PROD.

If approved, the on-VM apply takes ~10 minutes (3 file edits + env append + pytest in place + restart + flag-off smoke + baseline re-run). The lift measurement after flag-on takes another ~40 min (full 50-Q eval).

Total time from approval → measured lift: **~1 hour**.
