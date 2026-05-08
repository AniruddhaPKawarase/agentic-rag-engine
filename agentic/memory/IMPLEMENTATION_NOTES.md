# v3.1 Memory Layer — Implementation Notes

Phase **P1** of the Unified RAG Agent v3.1 Generation Chain. This subsystem
adds semantic recall + rolling summarisation on top of the existing
`traditional/memory_manager.py` (which keeps doing what it already does —
in-RAM session cache + JSON-on-disk / S3 mirror).

## Module map

| File | Role |
|------|------|
| `agentic/memory/embeddings.py`   | Single `embed_text(str) -> list[float]` helper. Sync OpenAI client cached at module level. 2-attempt retry with exp backoff. 2000-char input cap. |
| `agentic/memory/vector_store.py` | `SessionVectorStore` — dual-layer FAISS-local + Mongo Atlas vector storage. FAISS-first reads, async rebuild on Atlas fallback. Never raises from `search()`. |
| `agentic/memory/writer.py`       | `MemoryWriter` (Agent 6) — fire-and-forget background writer. Embeds, persists to `MemoryManager`, writes both vector layers, regenerates rolling summary every N turns. All steps after the manager write are gated behind `MEMORY_WRITER_VECTOR_ENABLED`. |

## Mongo Atlas — `iField.session_turn_embeddings`

One document per conversation turn (user OR assistant — separate rows).

```jsonc
{
  "session_id":   "session_abc123",          // FK to ConversationSession.session_id
  "turn_index":   7,                          // 0-based ordinal in messages[]
  "role":         "user",                     // "user" | "assistant"
  "text_excerpt": "first 500 chars of turn",  // for log/debug; full text stays in JSON session
  "embedding":    [0.012, -0.044, ...],       // 1536-dim text-embedding-3-small
  "created_at":   ISODate("2026-05-03..."),
  "metadata":     { "project_id": 7166, "set_id": "abc" }
}
```

### Standard btree index (created automatically by `ensure_indexes()`)

```js
db.session_turn_embeddings.createIndex(
  { session_id: 1, turn_index: 1 }
)
```

### Atlas Vector Search index (create manually in the Atlas UI / API)

Index name **must match** the `MEMORY_ATLAS_VECTOR_INDEX` env var
(default: `session_turn_embeddings_vec_idx`).

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "session_id"
    }
  ]
}
```

The `session_id` filter field is what allows `$vectorSearch ... filter:
{session_id: ...}` to scope each query to a single conversation. Without
it, every search would scan every session's vectors.

## FAISS local layout

Files live next to the existing JSON sessions:

```
./conversation_sessions/
    session_abc123.json                    (existing — managed by MemoryManager)
    embeddings/
        session_abc123.faiss               (faiss_cpu IndexFlatIP, 1536-d)
        session_abc123.meta.jsonl          (one JSONL row per FAISS vector)
```

Vectors are L2-normalised before insertion so the inner-product score
returned by `IndexFlatIP` equals cosine similarity.

The `.meta.jsonl` sidecar maps FAISS row → turn metadata
(`turn_index`, `role`, `text_excerpt`, `metadata`). It MUST stay in
lockstep with the FAISS index — the rebuild path (`_rebuild_faiss_from_atlas`)
deletes and rewrites both files atomically under `_fs_lock`.

## Rolling summary

* Trigger: every `ROLLING_SUMMARY_INTERVAL` user turns (default **10**).
* Window: last `ROLLING_SUMMARY_TURN_WINDOW` messages (default **30**).
* Model: `ROLLING_SUMMARY_MODEL` (default **gpt-4o-mini**).
* Output: ≤200-word prose summary preserving construction-domain
  entities (drawing names, project IDs, trades, equipment names).
* Storage: written to `ConversationContext.rolling_summary` if the
  field exists on the dataclass; otherwise falls back to
  `custom_instructions` via `MemoryManager.update_context()` for
  backward compatibility.

A non-blocking lock (`threading.Lock.acquire(blocking=False)`) prevents
two concurrent writers from regenerating the same summary.

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | _(required)_ | Used by embeddings + summariser. |
| `MEMORY_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model id. |
| `MONGODB_URI` | _(empty)_ | If unset, Atlas writes/reads silently no-op (FAISS still works). |
| `MONGO_DB` | `iField` | Database holding `session_turn_embeddings`. |
| `MEMORY_ATLAS_TIMEOUT_MS` | `150` | Hard `maxTimeMS` budget for `$vectorSearch`. |
| `MEMORY_ATLAS_VECTOR_INDEX` | `session_turn_embeddings_vec_idx` | Name of the Atlas Vector Search index. |
| `MEMORY_WRITER_VECTOR_ENABLED` | `false` | Master switch: when false, only the existing `MemoryManager.add_to_session` runs (today's behaviour). |
| `ROLLING_SUMMARY_INTERVAL` | `10` | User-turn cadence for summary regeneration. |
| `ROLLING_SUMMARY_TURN_WINDOW` | `30` | Number of recent messages fed to the summariser. |
| `ROLLING_SUMMARY_MODEL` | `gpt-4o-mini` | Chat model for summary calls. |

## Latency budget per operation (target / on-success)

| Operation | Target | Notes |
|---|---|---|
| `embed_text` | 60–150 ms | Network-bound. Capped at 2000 chars. |
| `SessionVectorStore.add` | 5–15 ms (FAISS) + 20–60 ms (Atlas insert, async-friendly) | Atlas write is sync inside the worker thread but the worker itself runs after the user response. |
| `SessionVectorStore.search` (FAISS hit) | < 10 ms | Local index, no network. |
| `SessionVectorStore.search` (Atlas fallback) | ≤ `MEMORY_ATLAS_TIMEOUT_MS` (150 ms) | Hard cap via `maxTimeMS`. |
| `MemoryWriter.write_turn_async` (caller-visible) | < 1 ms | Pure `Executor.submit` — actual work happens off-thread. |
| Rolling summary regeneration | 1–3 s | gpt-4o-mini call; runs at most once per `ROLLING_SUMMARY_INTERVAL` turns. |

The user-visible response path pays **none** of these costs — the writer
is a side channel scheduled after `await` of the response.

## Failure modes

1. **OpenAI down** — `embed_text` raises after 2 retries → writer logs &
   skips vector storage for that turn. `MemoryManager` write still
   succeeds.
2. **Atlas down** — `vector_store.add` logs a warning, FAISS write still
   succeeds. `search` falls back to FAISS-only.
3. **FAISS file corruption** — `_load_faiss_index` returns `None`,
   `search` falls through to Atlas, async rebuild kicks in.
4. **Both stores down** — `search` returns `[]`, never raises.
5. **Summariser failure** — caught in `_regenerate_summary`, logged, no
   retry until the next interval tick.

## Open follow-ups (out of scope for P1)

* Wire the writer into the live FastAPI handlers (gateway / orchestrator).
* Add a Prometheus counter for `memory_writer_failures_total`.
* Decide on retention policy for `session_turn_embeddings` (TTL? per-tenant?).
* Add a Memory Reader (Agent 7) that surfaces `search()` hits into the
  generation prompt.

---

# v3.1 P2 — Memory Recall + Query Rewriter (front of generation chain)

These two agents run BEFORE the existing ReAct retrieval agent. They give
the retriever a self-contained query plus session context so follow-up
questions like "and what about the second one?" produce correct
retrieval results.

## Module map (P2 additions)

| File | Role |
|------|------|
| `agentic/memory/recall_agent.py` | **Agent 1 — Memory Recall.** `recall(...)` returns rolling_summary, recent_turns, semantic_turns, topic_tags, had_context. Pulls from `MemoryManager.get_session()` + `SessionVectorStore.search()`. Never raises. |
| `agentic/memory/query_rewriter.py` | **Agent 2 — Query Rewriter.** `rewrite(...)` resolves coreference using the recall payload via a low-temperature LLM call. Latency-first: skips the LLM when a heuristic shows the query is already self-contained. |

## `recall_agent.recall(...)` return schema

```python
{
    "rolling_summary": str | None,    # session-level summary (writer maintains)
    "recent_turns":   list[{"role": str, "content": str[:500], "turn_index": int}],
    "semantic_turns": list[{"turn_index": int, "role": str, "text_excerpt": str, "score": float, "metadata": dict}],
    "topic_tags":     list[str],      # drawing names + trade words, max 10 unique
    "had_context":    bool,           # False ⇒ all collections empty
}
```

* `recent_turns` slice: last `top_k_recent` (default 6) entries from
  `session.messages`, with absolute `turn_index` preserved so the
  rewriter and downstream consumers can reconcile against vector-store
  hits.
* `semantic_turns`: dedup'd against `recent_turns` by `turn_index` so
  the prompt budget is not wasted on duplicates.
* `rolling_summary` precedence: explicit `context.rolling_summary`
  field first, falling back to `context.custom_instructions` for older
  session payloads (backward-compat with pre-P1 sessions).
* Topic tags: `\b[A-Z]{1,4}-?\d+[a-zA-Z]?\b` for drawing identifiers
  (`S-101`, `M101`, `E604a`) plus a fixed trade list (Mechanical,
  Electrical, Plumbing, Structural, Architectural, Civil, Fire,
  Concrete, Steel, HVAC). Scans the last 30 turns. Cap of 10 tags.

## `query_rewriter.rewrite(...)` skip rules

In order — first match wins:

1. `QUERY_REWRITER_ENABLED != "true"` → `skip_reason="flag_off"`.
2. `memory_context.had_context is False` → `"no_session_context"`.
3. **Anaphora heuristic** (`QUERY_REWRITER_SKIP_HEURISTIC == "true"`,
   default on): if no markers from the list below appear in the
   lower-cased query AND `len(query) > 30`, skip with
   `"no_anaphora_markers"`.

Anaphora markers (leading/trailing spaces are intentional — they
prevent matching e.g. `"it"` inside `"italic"`):

```
"it ", "its ", "that ", "those ", "these ", "this ",
"second one", "first one", "other one",
"same", "also", "again",
"what about", "how about",
"and ", " them ", "they ", " he ", " she "
```

Quick coverage check:
* `"what about the second one?"` → both `"what about"` AND
  `"second one"` match. Proceeds to LLM. ✓
* `"and the next floor?"` → `"and "` matches. Proceeds. ✓
* `"What are the fire safety requirements for atrium smoke control?"`
  → no markers, length > 30. Skipped. ✓
* `"yes"` → no markers, but length ≤ 30 → proceeds (cheap to be
  defensive on micro-replies). ✓

## Sanity guards on rewriter output

After the LLM call we discard the rewritten query and pass through
the original when:

1. Output length > 3 × original query length
   → `skip_reason="rewriter_output_invalid"`.
2. Output (lower-cased, lstripped) starts with any of:
   `"here is"`, `"here's"`, `"rewritten:"`, `"rewritten query:"`,
   `"the rewritten"`, `"sure,"`, `"sure!"`, `"i've rewritten"`,
   `"i have rewritten"` → `skip_reason="rewriter_output_invalid"`.
3. Output equals the original (whitespace-collapsed, case-folded)
   → `skip_reason="already_self_contained"`, `was_rewritten=False`.
4. LLM raises → swallowed, `skip_reason="rewriter_llm_error"`.

The output is always also stripped of one outer pair of matching
single or double quotes (LLMs love to wrap rewrites in quotes).

## Env vars consumed by P2

| Var | Default | Used by |
|-----|---------|---------|
| `MEMORY_RECALL_ENABLED` | `true` | recall_agent — global kill switch |
| `MEMORY_ATLAS_TIMEOUT_MS` | `150` | recall_agent — vector search timeout |
| `QUERY_REWRITER_ENABLED` | `true` | query_rewriter — global kill switch |
| `QUERY_REWRITER_SKIP_HEURISTIC` | `true` | query_rewriter — toggle anaphora skip |
| `REWRITER_MODEL` | `gpt-4o-mini` | query_rewriter — primary LLM model |

All have safe defaults; running with no env vars set is fine.

## Latency budget (P2 alone)

| Step | When | Cost |
|------|------|------|
| `recall(...)` no-session path | every no-context call | < 1 ms |
| `recall(...)` with FAISS hit | most sessions | 60–160 ms (embed + FAISS) |
| `recall(...)` Atlas fallback | cold FAISS | + ≤ 150 ms |
| `rewrite(...)` skip path | self-contained queries | < 1 ms |
| `rewrite(...)` LLM path | follow-ups | 250–600 ms (gpt-4o-mini, 200 max_tokens, T=0.1) |

The heuristic-skip exists specifically to keep the typical first-question
case under the 1ms bar — only real follow-ups pay the LLM cost.

## Open follow-ups (out of scope for P2)

* Wire `recall(...)` + `rewrite(...)` into the gateway / orchestrator
  in front of the ReAct retrieval agent.
* Tune `_FORBIDDEN_OUTPUT_PREFIXES` based on observed live traffic.
* Consider a tiny LLM-based intent classifier that beats the regex
  heuristic on accuracy without losing too much latency.
* Surface `topic_tags` to retrieval as soft filters (e.g. boost
  drawings whose name matches a tag).
