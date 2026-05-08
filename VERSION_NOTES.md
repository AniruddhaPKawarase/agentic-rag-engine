# v3.1 — Generation Chain + Conversation Memory

**Status:** Active development branch (NOT deployed). Live service still runs v3.0.
**Started:** 2026-04-30
**Source:** snapshot of v3.0-ckg-integration (post Tier 1+2+3 citation work) at the repo root after the 2026-04-30 deploy.

## What this version adds

A 6-agent **generation chain** running after the existing ReAct retrieval agent, plus persistent **conversation memory** with semantic recall.

| # | Agent | Module | Model |
|---|---|---|---|
| 1 | Memory Recall | `agentic/memory/recall_agent.py` | gpt-4o-mini |
| 2 | Query Rewriter | `agentic/memory/query_rewriter.py` | gpt-4o-mini |
| 3 | ReAct Retrieval | `agentic/core/agent.py` (UNCHANGED) | gpt-4.1 |
| 4 | Answer Synthesizer | `agentic/generation/synthesizer.py` | claude-haiku-4-5 (fallback gpt-4o-mini) |
| 5 | Style/Tone Rewriter | `agentic/generation/stylist.py` | claude-haiku-4-5 (fallback gpt-4o-mini) |
| 5b | Cache Re-Expression | inside Stylist (different prompt) | same as Stylist |
| 6 | Memory Writer (async) | `agentic/memory/writer.py` | gpt-4o-mini |

## Storage

- **Primary:** Mongo Atlas vector index on `iField.session_turn_embeddings`
- **Cache:** local FAISS at `./conversation_sessions/embeddings/<session_id>.faiss`
- **Read path:** FAISS → Atlas (on miss, also rebuilds FAISS async)
- **Write path:** Atlas (durable) + FAISS (hot read cache)

## Feature flags (default OFF — behaviour identical to v3.0)

```
MEMORY_RECALL_ENABLED=false
QUERY_REWRITER_ENABLED=false
ANSWER_SYNTHESIZER_ENABLED=false
STYLE_REWRITER_ENABLED=false
MEMORY_WRITER_VECTOR_ENABLED=false
CACHE_REEXPRESSION_ENABLED=false

RECALL_MODEL=gpt-4o-mini
REWRITER_MODEL=gpt-4o-mini
SYNTHESIZER_MODEL=claude-haiku-4-5
SYNTHESIZER_MODEL_FALLBACK=gpt-4o-mini
STYLIST_MODEL=claude-haiku-4-5
STYLIST_MODEL_FALLBACK=gpt-4o-mini
WRITER_MODEL=gpt-4o-mini

MEMORY_PRIMARY_BACKEND=mongo_atlas
MEMORY_CACHE_BACKEND=faiss
MEMORY_ATLAS_TIMEOUT_MS=150
ROLLING_SUMMARY_INTERVAL=10

QUERY_REWRITER_SKIP_HEURISTIC=true
STYLE_SKIP_THRESHOLD_CHARS=300

SYNTHESIZER_MODEL_EVAL=claude-sonnet-4-6
EVAL_MODE=false
```

## Latency target

Today's v3.0 baseline: ~17–19 s
v3.1 target with all flags ON: ~17.5–19.5 s

Achieved through:
- Memory Recall (Agent 1) runs in parallel with ReAct (Agent 3) via `asyncio.gather`
- Query Rewriter (Agent 2) skipped on queries with no anaphoric markers (~60% of traffic)
- Stylist (Agent 5) skipped when answer is already <300 chars (~30% of traffic)
- Synthesizer + Stylist stream output to client (hides ~400ms perceived latency)
- FAISS local cache hits in 80% of memory-recall calls (~5ms vs ~80ms Atlas)

## Build phases

| Phase | Deliverable | Status |
|---|---|---|
| P0 | Worktree + module scaffolding | ✓ done |
| P1 | Memory Writer + Mongo Atlas + FAISS dual-layer | in progress (parallel with P3) |
| P3 | Synthesizer + Stylist + Cache Re-Expression | in progress (parallel with P1) |
| P2 | Memory Recall + Query Rewriter | pending P1 |
| P4 | Orchestrator wiring + streaming + skip-rules | pending P1+P2+P3 |
| P5 | Eval (length / tone / multi-turn coherence) | pending P4 |

## Rollback

This is a sandboxed worktree under `_versions/`. Live service runs from the repo root (unchanged). To delete v3.1 entirely:
```
rm -rf _versions/v3.1-generation-chain/
```

To deploy v3.1 (only after eval signs off):
- Same pattern as v3.0 deploy: backup live files, copy patched modules, set new env vars in `/etc/systemd/system/rag-agent.service` env file, `systemctl restart rag-agent.service`.

## Out of scope

- Document-FAISS backfill for additional projects (separate ticket).
- API contract changes (none — only new optional response fields).
- LangChain/LangGraph (using plain asyncio + OpenAI/Anthropic SDKs).

---

## Phase P4 — Generation Chain Orchestrator (2026-05-03)

**New module:** `gateway/generation_chain.py`. Single public entrypoint
`run_generation_chain(...)` that wires Recall → Rewriter → ReAct → Synthesizer
→ Stylist → Memory Writer (or Cache Re-Expression for cached hits).

**Master kill switch:** `V31_CHAIN_ENABLED=false` (default). With the flag off,
`gateway/orchestrator.py` runs byte-identically to v3.0 — the chain
dispatch is a single early-return guarded by `chain_enabled()` placed
before the legacy "Try agentic" block.

### Serial vs parallel front-end (design decision)

The original spec floated "Memory Recall in parallel with ReAct retrieval."
We rejected that because **rewrite depends on recall**, and **retrieval
depends on the rewritten query** — so the front-end is inherently serial:

```
Recall (~150ms) → Rewriter (~200ms, often skipped) → ReAct (~17s)
                  → Synthesizer (~600ms, streamable)
                  → Stylist     (~400ms, streamable)
                  → Memory Writer (background, 0ms blocking)
```

Total added latency on top of ReAct: ~1.4s buffered, ~0.4s with token
streaming hiding most of synth+stylist. Acceptable.

### Streaming protocol (when `stream=True`)

The chain returns an async generator yielding three event kinds:

* `{"event": "metadata", "metadata": {...}}` — emitted exactly once
  before any token. Carries `memory_context_used`, `query_rewritten`,
  `synthesizer_planned`, `stylist_planned`, `contextualized_query`,
  `cache_hit`. Lets the UI render badges immediately.
* `{"event": "token", "delta": "<chunk>"}` — zero or more, in order.
  When stylist is enabled we emit *stylist* tokens (the synthesizer's
  output is buffered to feed the stylist). When stylist is disabled but
  synthesizer is enabled, synthesizer tokens are streamed instead.
* `{"event": "done", "metadata": {final response dict}}` — exactly one,
  last event. Carries the assembled response in the same shape as the
  buffered path returns.

### Failure handling

Every sub-agent call (recall, rewrite, run_agent, synthesize, stylize,
reexpress_cached, MemoryWriter) is wrapped in its own `try/except`. On
failure the chain falls through to the previous stage's output. A
total Recall + Rewriter + Synthesizer + Stylist failure still returns
the raw ReAct answer. The Memory Writer is always fire-and-forget — its
failure is logged but never propagates to the user response.

### New env vars

| Name | Default | Effect |
|------|---------|--------|
| `V31_CHAIN_ENABLED` | `false` | Master kill switch — flag must be ON for the chain to run at all. |

The chain still respects the existing per-agent flags (`MEMORY_RECALL_ENABLED`,
`QUERY_REWRITER_ENABLED`, `ANSWER_SYNTHESIZER_ENABLED`, `STYLE_REWRITER_ENABLED`,
`CACHE_REEXPRESSION_ENABLED`) so ops can re-disable any single stage without
redeploying.

### New response fields (additive)

`memory_context_used`, `query_rewritten`, `synthesizer_used`, `stylist_used`,
`cache_reexpressed`, `contextualized_query`. All boolean-or-string,
backward-compatible (frontends that ignore them keep working).

### Tests

`tests/v3_1/test_generation_chain.py` — 17 tests covering: master kill
switch, full pipeline happy path, cache-hit branches (flag off + on),
every skip rule, writer fire-and-forget, streaming protocol ordering,
and graceful failure of every sub-agent. Total v3.1 suite: **75 tests
passing** (58 from P1+P2+P3 + 17 from P4).

### Surgical edit to `gateway/orchestrator.py`

Two additions, both gated by `chain_enabled()`:

1. Top-of-file: `from gateway.generation_chain import run_generation_chain, chain_enabled`.
2. Inside `Orchestrator.query`, just before the existing "Try agentic"
   block: a new `_maybe_run_v31_chain(...)` call that early-returns the
   chain's response when the flag is on. When the flag is off the
   helper is never reached and the legacy code path is unchanged.

The helper itself replicates the agent's existing cache lookup so a
cached answer flows through Branch A (re-expression) rather than
re-running ReAct.

---

## P5 — Agent 0.5: Answer Shape Classifier

`agentic/generation/answer_shape.py` adds a tiny upstream agent that
buckets the user's query into one of five answer-length shapes so
downstream Synthesizer + Stylist can produce appropriately-sized
answers — terse factoids vs. full explanations — instead of a
one-size-fits-all paragraph.

### Bucket taxonomy

| shape | typical query | default cap |
|-------|---------------|-------------|
| `factoid`     | "What is the duct size?" | 80 chars |
| `count`       | "How many DOAS units?"   | 60 chars |
| `list`        | "List all electrical panels" | 300 chars |
| `comparison`  | "Compare A-101 with S-101"   | 400 chars |
| `explanation` | "Explain how the HVAC system works" | 600 chars |
| `default`     | (flag off / unclassified) | n/a — legacy behaviour preserved |

### Strategy (latency-first)

1. **Skip rule** — `ANSWER_SHAPE_CLASSIFIER_ENABLED=false` (default) →
   returns `shape='default'` with `skip_reason='flag_off'`. Downstream
   agents read `default` as "use existing prompts unchanged" → byte-
   identical behaviour to today.
2. **Image override** — `has_image=True` + short query + level/floor
   regex → forced `factoid` (high confidence, no LLM call).
3. **Regex fast-path** — five ordered pattern groups (count → factoid →
   list → comparison → explanation), first match wins, confidence 0.9.
   Patterns target dimensional keywords (size, height, voltage, CFM,
   …), counting verbs (`how many`, `number of`, …), enumeration verbs
   (`list all`, `name the …`), and comparison/explanation lead-ins.
4. **LLM fallback** — only when query is non-trivial (>20 chars) and
   no regex matched. Single 1-token call to `SHAPE_CLASSIFIER_MODEL`
   (default `gpt-4o-mini`), max_tokens=20, temperature=0.0. Any
   unexpected response or exception → `shape='default'` (transparent).

### Env vars

| name | default | purpose |
|------|---------|---------|
| `ANSWER_SHAPE_CLASSIFIER_ENABLED` | `false` | master kill switch |
| `SHAPE_CLASSIFIER_MODEL` | `gpt-4o-mini` | model for LLM fallback |
| `SHAPE_FACTOID_MAX_CHARS` | `80` | factoid cap |
| `SHAPE_COUNT_MAX_CHARS` | `60` | count cap |
| `SHAPE_LIST_MAX_CHARS` | `300` | list cap |
| `SHAPE_COMPARISON_MAX_CHARS` | `400` | comparison cap |
| `SHAPE_EXPLANATION_MAX_CHARS` | `600` | explanation cap |

### Downstream wiring

Both `synthesizer.synthesize(...)` and `stylist.stylize(...)` /
`stylist.reexpress_cached(...)` accept a new optional
`answer_shape: dict | None = None` kwarg (additive, no signature
break). When set to a non-default shape they:

* Inject an `ANSWER_SHAPE: <bucket>` + `LENGTH_BUDGET` + `FORMAT_RULE`
  block at the top of the system prompt.
* Reduce `max_tokens` to `max(80, target_word_count * 2)` — the LLM
  can't blow the budget even on a hallucinated long-form answer.
* Tighten the synth/stylist skip thresholds (so a 100-char draft no
  longer auto-passes through when the budget is 80).

`gateway.generation_chain.run_generation_chain(...)` calls the
classifier exactly once, after Memory Recall (so the rewriter
cannot mask the original intent), then threads the result through
to `synthesize`, `stylize`, and `reexpress_cached`. The chain wraps
`classify_shape(...)` in try/except → on failure it returns the
canonical `default` dict and never crashes the response. The shape
is also surfaced in the streaming `metadata` event and the final
response dict (`answer_shape`, `target_length_chars` keys).

### Byte-identical-when-off guarantee

The `_shape_is_active(...)` helper inside `generation_chain.py`
ensures the new `answer_shape=` kwarg is **only** passed to
underlying generators when the classifier is enabled AND returned
a non-default shape. With the flag off, downstream call signatures
are unchanged, the synth/stylist system prompts are byte-identical
to today, and pre-existing test mocks (whose signatures predate
this kwarg) continue to work without modification.

### Tests

`tests/v3_1/test_answer_shape.py` — 18 tests covering: master flag
off, image override, every regex bucket, LLM fallback (label parse
+ invalid response + exception swallow), env-var overrides for
target lengths, synthesizer + stylist consuming the shape rule in
the system prompt, the byte-identical-when-off guarantee, and full
chain integration verifying the shape dict reaches both downstream
calls plus the response telemetry. Total v3.1 suite: **105 tests
passing** (87 prior + 18 new). Zero regressions.

### Expected impact

Shorter, sharper factoid/count answers ("12 AHUs across the
project." vs. a 300-char paragraph). Unchanged length on
explanations and comparisons (the budgets are sized for those).
Negligible added latency on the regex fast-path (single
compiled-regex search ~5µs); the LLM fallback is only on
non-trivial unrecognised queries and uses the cheapest model with
max_tokens=20.

