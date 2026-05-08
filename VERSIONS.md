# Unified RAG Agent — Version Log & Revert Procedures

## Named versions

Full per-file snapshots live under `_versions/<name>/` so you can revert with a single `cp` or `rsync`.

| Version | State | What's in it | Revert command |
|---|---|---|---|
| **`v2.0-docqa-extension-final`** ⭐ | **Post-change (2026-04-23)** | Phases 0-5 shipped: bridge, classifier, UI, answer-format fix, prompt-cache ordering, 8 security/correctness fixes. See `docs/PHASE6_SIGNOFF.md`. | `rsync -a PROD_SETUP/unified-rag-agent/_versions/v2.0-docqa-extension-final/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md` |
| **`v2.0-docqa-extension`** | Pre-change baseline (2026-04-23) | Snapshot of main before DocQA-extension work. Revert target for Phases 1-6 of docqa-extension rollout. | `rsync -a PROD_SETUP/unified-rag-agent/_versions/v2.0-docqa-extension/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md` |
| **`v1.2-hybrid-ship`** ⭐ | **RECOMMENDED CURRENT** | v1.1 code run with SELF_RAG off + RERANKER_INCLUDE_SCORE off. Identical output schema to v1.0, matching 19s latency, 0 broken URLs, Fix #3's counting/typical wins preserved. 61/61 unit tests pass. | `rsync -a PROD_SETUP/unified-rag-agent/_versions/v1.2-hybrid-ship/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md` |
| **`v1.1-stable-fix3-fix4`** | Stable (local, tested) | v1.0 + Fix #3 (aggregation tools) + Fix #4 (self-RAG groundedness). 60/60 unit tests pass. UI contract unchanged. Live-tested DOAS count returns correct 2 tags. Latency 30s. | `rsync -a PROD_SETUP/unified-rag-agent/_versions/v1.1-stable-fix3-fix4/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md` |
| **`v1.0-stable-fixes-1-2-5-8`** | Stable (local, tested) | Fixes #8 + #5 + #1 + #2. 38/38 unit tests pass. UI contract unchanged. | `rsync -a PROD_SETUP/unified-rag-agent/_versions/v1.0-stable-fixes-1-2-5-8/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md` |
| **`v1.1-dev-fix3-aggregation`** | Superseded (design doc only) | Early design sketch, kept for history. | Use a stable version above. |

Each version folder ships its own `README.md` with what it contains and exact
copy-back commands. Start with `_versions/v1.0-stable-fixes-1-2-5-8/README.md`.

---

## 2026-04-22 — Fixes #1 + #2 (local-only, feature-flagged, NOT yet on VM)

**Goal (Fix #1):** improve recall on ambiguous/terminology-mismatched questions
by decomposing the user query into diverse sub-queries, fanning them out
across the agent's own MongoDB search tools, and fusing the ranked results
with Reciprocal Rank Fusion (RRF). The top candidates become a system-prompt
hint prepended to the agent's `conversation_history`.

**Goal (Fix #2):** reorder the final `source_documents[]` best-first using a
cheap LLM-as-judge (`gpt-4.1-mini` scores each candidate 0–10). UI sees the
same shape, just with the most relevant sources at the top.

### Files added (local only, new modules — no existing code replaced)

| File | Purpose |
|---|---|
| `gateway/retrieval_enrichment.py` | `decompose_query()`, `multi_query_retrieve()`, `reciprocal_rank_fusion()`, `build_context_hint()`, `format_hint_for_agent()` |
| `gateway/reranker.py` | `rerank_source_documents()` — LLM-as-judge scoring + reorder |
| `tests/test_retrieval_enrichment.py` | 19 unit tests (stubbed OpenAI, stubbed tools) |
| `tests/test_reranker.py` | 11 unit tests (stubbed OpenAI) |

### Files modified (additive only, no breaking changes)

| File | Change | Revert |
|---|---|---|
| `gateway/orchestrator.py` | Added 3 env flags (`MULTI_QUERY_RRF_ENABLED`, `RERANKER_ENABLED`, `RERANKER_KEEP_TOP_K`); added `_maybe_rerank()` helper; added pre-ReAct hint injection; added post-retrieval rerank in 3 `_build_response` branches and in the agentic success path | Unset the env flags → both fixes become no-ops |

### Feature flags (default OFF — zero behavioural change until flipped)

| Env var | Default | What it does |
|---|---|---|
| `MULTI_QUERY_RRF_ENABLED` | `false` | Turn on Fix #1 (multi-query decomposition + RRF hint) |
| `RERANKER_ENABLED` | `false` | Turn on Fix #2 (LLM-as-judge reorder) |
| `RERANKER_KEEP_TOP_K` | `0` (keep all) | If set, truncate reranked list to top-K |
| `MULTI_QUERY_MODEL` | `gpt-4.1-mini` | Model used for decomposition |
| `MULTI_QUERY_COUNT` | `4` | Number of sub-queries |
| `MULTI_QUERY_PER_TOOL_LIMIT` | `10` | Per-(sub_query, tool) result cap |
| `MULTI_QUERY_FUSED_TOP_K` | `15` | Size of fused candidate list passed to the agent hint |
| `RERANKER_MODEL` | `gpt-4.1-mini` | Model used for scoring |
| `RERANKER_CANDIDATE_CAP` | `30` | Max items scored per query (keep-tail for the rest) |

### Measurable delta (projects 7222 + 7223, 16 queries)

Baseline = Fix #8 + Fix #5 only (flags off). Comparison = all four fixes on.

| Metric | Flags off | Flags on |
|---|---:|---:|
| Avg latency | 15.0 s | **19.0 s** (+4 s for decomposition + rerank LLM calls) |
| Queries with broken URL | 0 / 16 | 0 / 16 |
| Queries with zero sources | 0 / 16 | 0 / 16 |
| Engine mix | 100% agentic | 100% agentic |
| Confidence = high | 16 / 16 | 16 / 16 |

**Interpretation:** both fixes execute correctly, never regress URL or
source-count correctness, and shift answer wording and source ordering as
designed. On the 8 manager-reported failures specifically, the remaining
gaps are in **counting** (valve count, DOAS count) and **domain ontology**
("typical levels"), which require Fix #3 (dedicated aggregation primitive) —
neither Fix #1 nor Fix #2 was ever going to solve those. Fix #1 + #2 are
most useful for queries with (a) terminology mismatch between user language
and document language, or (b) dozens of marginally-relevant sources that
the frontend is struggling to display.

### Artifacts

- Unit tests: `tests/test_retrieval_enrichment.py` (19/19), `tests/test_reranker.py` (11/11)
- Flags-off run: `test_results/local_after_fixes_local_20260422_152341/`
- Flags-on run: `test_results/local_after_all_fixes_20260422_154725/`
- Comparison: `test_results/comparison_fix1_fix2_*/{comparison.json,comparison.docx}`

### Revert (local)

Nothing destructive was changed. To disable:

```bash
# Leave the env vars off (default). Done.
unset MULTI_QUERY_RRF_ENABLED RERANKER_ENABLED

# To fully remove the code, delete the two new modules and un-wire the helper:
rm PROD_SETUP/unified-rag-agent/gateway/retrieval_enrichment.py
rm PROD_SETUP/unified-rag-agent/gateway/reranker.py
# Then restore orchestrator.py from backup (choose the right timestamp):
cp PROD_SETUP/unified-rag-agent/_backups/orchestrator_fixAB_v2_20260420_111459.py \
   PROD_SETUP/unified-rag-agent/gateway/orchestrator.py
# Re-apply Fix #8 patches only (see the 2026-04-22 Fix #8/#5 section below).
```

### Revert (VM, once deployed)

**Not yet deployed. When rolling out, the safe path is:**

1. Deploy the new modules via `scp` (additive).
2. Deploy patched `orchestrator.py` with backup to `_backups/orchestrator_preFix12_<ts>.py`.
3. Restart `rag-agent.service` with flags **off** — behaviour identical to today.
4. Flip flags one at a time, 1 hour apart, by editing `/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/.env` and restarting.
5. If issues: unset the flag and restart — no code revert needed.
6. To fully revert: restore `orchestrator.py` from backup, remove the two new modules, restart.

---



This file tracks every intentional change to production-relevant code plus
the timestamped backup that lets us revert. Every fix leaves a rollback path.

**Rule:** before touching `gateway/*.py`, `agentic/**/*.py`, or `shared/*.py`
in a way that could affect responses, drop the existing file into
`_backups/` with a `<name>_<feature>_<YYYYMMDD_HHMMSS>.py` filename. Repeat
on the VM when deploying (same `_backups/` folder under
`/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/`).

## Convention

| Filename pattern | Meaning |
|---|---|
| `<name>_preFix<N>_<ts>.py` | Snapshot taken **before** the change landed |
| `<name>_fix<N>_<ts>.py` | Snapshot taken **after** the change landed |
| `<name>_preSync_<ts>.py` | Snapshot before we overwrote local with VM's version |

---

## 2026-04-22 — Fixes #8 + #5 (local-only; NOT yet on VM)

**Goal:** Every source in `/query` responses returns a working signed
download URL, and spec-based questions get full-section context instead of
truncated 500/2000-char fragments.

### Files changed (local only)

| File | Change | Local backup |
|---|---|---|
| `gateway/orchestrator.py` | Added `_ensure_signed_source_urls()` helper + calls in `_build_response()` dict/model-dump branches; re-signing guard inside `_extract_source_documents()` | `_backups/orchestrator_fixAB_v2_20260420_111459.py` (previous version) — no new backup needed because the change is additive |
| `agentic/tools/specification_tools.py` | Added `get_full_specification_text()` (schema-A + schema-B aware) | `_backups/specification_tools_FixS3_20260421_212038.py` |
| `agentic/tools/registry.py` | Imported + registered new tool `spec_get_full_text` | covered by sync backup (see below) |
| `agentic/core/agent.py` | Synced from VM (was behind: missing `scope` kwarg, no `source_docs` field) | `_backups/local_presync_20260422_152122/agent.py` |
| `gateway/title_cache.py` | Synced from VM (file didn't exist locally) | (new file) |
| `gateway/query_enhancer.py` | Synced from VM (file didn't exist locally) | (new file) |
| `gateway/auth.py` | Synced from VM (file didn't exist locally) | (new file) |

### Verification artifacts

| Artifact | Purpose |
|---|---|
| `tests/test_url_resigning.py` | 8 unit tests for `_ensure_signed_source_urls` — incl. live S3 probe. Status: 8/8 pass |
| `tests/e2e/reproduce_manager_failures.py` | Dumps traces for the 8 manager questions × N projects |
| `tests/e2e/local_fix_harness.py` | Runs the full Orchestrator.query() in-process (no VM) |
| `tests/e2e/compare_before_after.py` | Side-by-side JSON + DOCX report |
| `test_results/manager_before_fixes_GETprobe_20260422_151116/` | Sandbox baseline (pre-fix) |
| `test_results/local_after_fixes_local_20260422_152341/` | Local run after fixes |
| `test_results/comparison_fix8_fix5_20260422_152911/` | Before/after comparison |

### Measurable delta (projects 7222 + 7223, 16 queries)

| Metric | Before (sandbox) | After (local) |
|---|---|---|
| Queries with any broken URL (GET probe) | **6 / 16** | **0 / 16** |
| Queries with zero sources | **1 / 16** | **0 / 16** |
| Engine mix | 10 agentic / 6 traditional | 16 agentic / 0 traditional |
| Avg latency | 11.4 s (fresh) | 15.0 s |

Latency regression explained by Fix #5 giving the agent an extra tool it often
picks, plus spec queries that previously hit traditional FAISS now stay in the
multi-step ReAct path. Acceptable given the accuracy/URL-validity wins.

### Revert (local)

Each file can be reverted individually — everything else keeps working:

```bash
cd PROD_SETUP/unified-rag-agent

# Revert agent.py to pre-sync state
cp _backups/local_presync_20260422_152122/agent.py agentic/core/agent.py

# Revert specification_tools.py to the pre-Fix-#5 state
cp _backups/specification_tools_FixS3_20260421_212038.py agentic/tools/specification_tools.py

# Revert registry.py — just remove the spec_get_full_text lines manually, no
# backup required (additive change only); or grep and delete:
grep -v 'spec_get_full_text\|get_full_specification_text as spec_get_full_text' agentic/tools/registry.py > /tmp/r.py && mv /tmp/r.py agentic/tools/registry.py

# Revert orchestrator.py — Fix #8 can be toggled by removing the
# _ensure_signed_source_urls definition and the two calls in _build_response.
# Or restore from any previous backup.
```

### Revert (VM, once deployed)

**Not yet deployed.** When rolling out:

```bash
# On VM
cd /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent
TS=$(date +%Y%m%d_%H%M%S)

# Pre-deploy snapshot (mandatory)
cp gateway/orchestrator.py _backups/orchestrator_preFix8_${TS}.py
cp agentic/tools/specification_tools.py _backups/specification_tools_preFix5_${TS}.py
cp agentic/tools/registry.py _backups/registry_preFix5_${TS}.py

# Deploy via scp from local (run from repo root on workstation)
scp -i <pem> gateway/orchestrator.py ubuntu@54.197.189.113:/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/gateway/orchestrator.py
scp -i <pem> agentic/tools/specification_tools.py ubuntu@54.197.189.113:/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/agentic/tools/specification_tools.py
scp -i <pem> agentic/tools/registry.py ubuntu@54.197.189.113:/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/agentic/tools/registry.py

# Restart
ssh -i <pem> ubuntu@54.197.189.113 "sudo systemctl restart rag-agent.service && sleep 3 && curl -s http://localhost:8001/health"

# Rollback (any time later)
ssh -i <pem> ubuntu@54.197.189.113 "cd /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent && cp _backups/orchestrator_preFix8_${TS}.py gateway/orchestrator.py && cp _backups/specification_tools_preFix5_${TS}.py agentic/tools/specification_tools.py && cp _backups/registry_preFix5_${TS}.py agentic/tools/registry.py && sudo systemctl restart rag-agent.service"
```

---

## UI Contract (do not break)

These fields are read by the Angular frontend and must remain present +
same-typed in every `/query` response:

- `answer` (string), `rag_answer` (string|null)
- `confidence` (string: high|medium|low), `confidence_score` (float)
- `source_documents[]` with keys: `s3_path, file_name, display_title, download_url, pdf_name, drawing_name, drawing_title, page`
- `s3_paths[]`, `s3_path_count` (int)
- `engine_used` (string: agentic|traditional|docqa|hybrid), `fallback_used` (bool)
- `agentic_confidence` (string|null)
- `follow_up_questions[]`, `processing_time_ms` (int)
- `session_id`, `session_stats`, `search_mode`
- `pin_status`, `needs_document_selection`, `available_documents[]`, `scoped_to`
- `web_sources[]`, `web_source_count`
- `active_agent`, `suggest_switch`, `selected_document`
- `success`, `error`

Fixes #8 and #5 are **strictly additive internal changes**: Fix #8 rewrites
the value of `download_url` (same field, better content); Fix #5 adds a new
agent tool (internal), no schema change. UI code does not need any edits.

---

## Historical changes

| Date | Change | Files touched | Revert instructions |
|---|---|---|---|
| 2026-04-20 | Fix A — restore traditional RAG fallback in orchestrator | `gateway/orchestrator.py` | `cp _backups/orchestrator_fixA_20260420_104533.py gateway/orchestrator.py` |
| 2026-04-20 | Fix B — evasive-answer heuristic in `_should_fallback()` | `gateway/orchestrator.py` | `cp _backups/orchestrator_fixAB_20260420_111022.py gateway/orchestrator.py` |
| 2026-04-20 | Fix B v2 — expanded evasive patterns | `gateway/orchestrator.py` | `cp _backups/orchestrator_fixAB_v2_20260420_111459.py gateway/orchestrator.py` |
| 2026-04-21 | Fix spec `s3BucketPath` projection so source s3_path populates | `agentic/tools/specification_tools.py` | `cp _backups/specification_tools_FixS3_20260421_212038.py agentic/tools/specification_tools.py` |
| 2026-04-20 | Provisioned `ai6.ifieldsmart.com` nginx site + Let's Encrypt cert | `/etc/nginx/sites-enabled/ai6.ifieldsmart.com` on VM | `sudo rm /etc/nginx/sites-enabled/ai6.ifieldsmart.com && sudo systemctl reload nginx` |

---

## Pending fixes (design sketches only, not yet implemented)

### Fix #1 — Multi-Query + RAG-Fusion (RRF)

- Add `gateway/retrieval_enrichment.py` with two pure functions:
  - `decompose_query(query: str, n: int = 4) -> list[str]` — calls gpt-4.1-mini to rewrite the user query into N semantically diverse sub-queries (synonyms, trade variants, sheet-type variants).
  - `rag_fusion(query_variants: list[str], tools: list[Callable], k: int = 40) -> list[dict]` — runs each variant through each tool, accumulates ranked lists, merges with Reciprocal Rank Fusion.
- Wire into `Orchestrator.query()` **before** the ReAct loop: produce a "pre-retrieved context hint" (top-20 fused results), inject as a system-prompt addition so the agent starts already knowing the best candidate sources. Agent's own tool calls still happen — RRF just bootstraps.
- **UI impact:** none. `answer` and `source_documents[]` remain the contract.
- **Feature flag:** `MULTI_QUERY_RRF_ENABLED` env var, default false; flip to true after parity testing.

### Fix #2 — Cross-encoder / LLM-as-judge reranker

- Add `gateway/reranker.py::rerank(query, candidates, top_k=5)` — either:
  - A) `bge-reranker-v2-m3` via Hugging Face Inference API, or
  - B) gpt-4.1-mini as judge (score each candidate 0–10 vs the query, keep top-k).
- Apply in `_build_response()` right before emitting `source_documents`: take the raw 20–40 agentic sources, compress to the 5 most relevant. The `_source_documents_full[]` internal list can stay for debugging.
- **UI impact:** none. `source_documents[]` gets shorter and more relevant.
- **Feature flag:** `RERANKER_ENABLED` env var, default false.

Both #1 and #2 are scoped for the next session. I'll write unit tests first,
then wire them into the orchestrator behind their flags, then re-run the
`local_fix_harness.py` harness and compare again.
