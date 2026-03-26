# CLAUDE.md — RAG Agent VCS (Production)

---

## Conversation Intelligence Upgrade v3 — 2026-03-26 (APPROVED, PENDING DEVELOPMENT)

**Design doc:** [DEVELOPMENT_PLAN_CONVERSATION_UPGRADE.md](DEVELOPMENT_PLAN_CONVERSATION_UPGRADE.md)

### Problem
Agent treats every query independently. Follow-ups ("tell me more"), answer references ("what size you mentioned"), and drawing-specific queries ("notes on drawing A0.01") all fail because:
1. No follow-up intent detection — "tell me more" triggers fresh retrieval with no prior context
2. No drawing name/title filtering in FAISS retrieval
3. Previous answers truncated to 150 chars — LLM can't reference details like "1/2 inch"
4. Conversation context budget too small (2000 tokens)

### Confirmed Design Decisions
| Decision | Choice |
|----------|--------|
| Follow-up handling | Augment query with previous Q+A + re-retrieve from FAISS |
| Drawing name filtering | Soft filter with 1.5x similarity boost (not hard filter) |
| Previous answer context | Full last answer + 300 chars for older messages |
| Project scope | All projects (universal) |
| Conversation token budget | 3000 tokens (was 2000) |
| Intent detection method | Pure regex (0ms latency) |

### Files to Modify
| File | Change |
|------|--------|
| `rag/api/intent.py` | Add follow_up, reference_previous, drawing_specific intent types |
| `rag/api/generation_unified.py` | Query augmentation + enhanced conversation context builder |
| `rag/retrieval/engine.py` | Drawing name/title soft filter with boost |
| `rag/api/prompts.py` | Add conversation awareness rules |
| `rag/api/helpers.py` | Add drawing name extraction helper |
| `memory_manager.py` | Add get_last_user_message(), get_last_assistant_message() |

### New Intent Types (regex, 0ms)
| Intent | Patterns | Action |
|--------|----------|--------|
| `follow_up` | "tell me more", "continue", "elaborate", "expand on that" | Augment query with prior Q+A |
| `reference_previous` | "what size you mentioned", "you said", "as you mentioned" | Include full previous answer |
| `drawing_specific` | Regex: `[A-Z]{1,3}[-.]?\d{1,4}[.-]?\d{0,3}` | Soft-boost matching drawing chunks |
| `drawing_title_specific` | "Partition Schedule", "Floor Plan" etc. | Soft-boost matching title chunks |

---

## Project Identity
- **Name:** Construction Documentation QA API
- **Version:** 2.0.0 (v1.4.0 CHANGELOG)
- **Type:** FastAPI RAG + Web Search + Hybrid Agent
- **Entry Point:** `generate.py` (backward-compat) or `uvicorn rag.api.state:app`
- **Port:** 8000 (configurable via `PORT` env var)
- **Platform:** Ubuntu Linux (production), Windows (development)

---

## Architecture Overview

```
generate.py (entry)
  └── rag/api/state.py (FastAPI app, CORS, OpenAI client)
        ├── rag/api/routes.py (13+ endpoints)
        │     ├── rag/api/intent.py (regex intent detection, <1ms)
        │     ├── rag/api/generation_unified.py (RAG/Web/Hybrid pipeline)
        │     ├── rag/api/generation_web.py (web-only generation)
        │     ├── rag/api/streaming.py (SSE streaming)
        │     ├── rag/api/prompts.py (4 prompt templates)
        │     ├── rag/api/helpers.py (confidence, trade detection, token budget)
        │     └── rag/api/models.py (Pydantic schemas)
        ├── rag/retrieval/ (FAISS search layer)
        │     ├── state.py (project registry: 7166,7201,7212,7222,7223,7277,7292)
        │     ├── loaders.py (index + metadata loading)
        │     ├── engine.py (retrieve_context — core search)
        │     ├── embeddings.py (LRU-cached embedding, maxsize=512)
        │     ├── metadata.py (title/URL builders)
        │     ├── diagnostics.py (stats, test queries)
        │     └── cli.py (CLI entry)
        ├── services/web_search.py (OpenAI web_search tool, 5-min TTL cache)
        ├── memory_manager.py (session persistence, 200 max, LRU eviction)
        ├── config/settings.py (WEB_SEARCH_MODEL loader)
        └── utils/logger.py (centralized logging)
```

---

## Environment Variables (.env)

| Variable | Default | Purpose | Impact |
|----------|---------|---------|--------|
| `OPENAI_API_KEY` | *required* | OpenAI API key | All LLM + embedding calls |
| `LLM_MODEL` | `gpt-4o` | Main generation model | Answer quality + latency |
| `WEB_SEARCH_MODEL` | `gpt-4.1` | Web search model | Web/hybrid answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Query embedding (1536-dim) | Retrieval accuracy |
| `HOST` | `0.0.0.0` | Server bind address | Network access |
| `PORT` | `8000` | Server port | Endpoint URL |
| `INDEX_ROOT` | `/home/ubuntu/vcsai/index` | FAISS index directory | Where indices live |
| `SESSION_STORAGE_PATH` | `./conversation_sessions` | Session persistence dir | Conversation memory |
| `MAX_SESSIONS` | `200` | Max in-memory sessions | Memory usage |
| `MAX_TOKENS_PER_SESSION` | `10000` | Session summarization trigger | Context quality |
| `LOG_LEVEL` | `INFO` | Logging verbosity | Debug output |
| `LOG_TO_FILE` | `false` | File logging toggle | Disk usage |
| `LOG_FILE_PATH` | `./logs/api.log` | Log file location | Log persistence |

---

## Key Thresholds & Constants

| Constant | Value | Location | What It Controls |
|----------|-------|----------|------------------|
| `CONFIDENCE_THRESHOLD` | `0.30` | `helpers.py` | Below this triggers clarification instead of answer |
| `MAX_CONTEXT_TOKENS` | `4000` | `helpers.py` | Max tokens in LLM context from chunks |
| `MAX_CHUNKS` | `10` | `helpers.py` | Max chunks per response |
| `over_fetch_multiplier` | `3` | `engine.py` | Fetch 3x top_k for filtering headroom |
| `LRU embedding cache` | `512` | `embeddings.py` | Cached query embeddings (avoid repeated API calls) |
| `Web cache TTL` | `300s` | `web_search.py` | Web results cached 5 minutes |
| `Web cache max` | `200` | `web_search.py` | Max cached web responses |
| `max_messages_before_summary` | `15` | `memory_manager.py` | Triggers session summarization |
| `Session cleanup` | `24h` | `memory_manager.py` | Auto-delete old sessions on startup |
| `_INPUT_COST_PER_1K` | `0.0025` | `helpers.py` | Cost tracking ($2.50/1M input) |
| `_OUTPUT_COST_PER_1K` | `0.01` | `helpers.py` | Cost tracking ($10/1M output) |

---

## API Endpoints

### Core Query
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query` | Full RAG/Web/Hybrid generation |
| `POST` | `/query/stream` | SSE streaming response |
| `POST` | `/quick-query` | Simplified UI endpoint |
| `POST` | `/web-search` | Web search only |

### Session Management
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions/create` | Create session |
| `GET` | `/sessions` | List sessions |
| `GET` | `/sessions/{id}/stats` | Session stats |
| `GET` | `/sessions/{id}/conversation` | Full history |
| `POST` | `/sessions/{id}/update` | Update context |
| `DELETE` | `/sessions/{id}` | Delete session |

### System
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Version + feature info |
| `GET` | `/health` | Health check |
| `GET` | `/config` | Current configuration |
| `GET` | `/test-retrieve` | Debug: test FAISS |
| `GET` | `/debug-pipeline` | Debug: full pipeline |
| `GET` | `/test-web-search` | Debug: test web search |

---

## Pipeline Flow (POST /query)

```
1. Intent Detection (regex, <1ms) → greeting/small_talk/thanks/farewell/meta → instant response
2. Session load/create (if session_id provided)
3. Retrieval:
   - RAG: embed query → FAISS search → filter by score/trade/source_type → budget tokens
   - Web: OpenAI responses.create with web_search tool (cached 5 min)
   - Hybrid: both in parallel
4. Confidence check: avg_sim*0.6 + max_sim*0.4 → if < 0.30, clarification mode
5. Build prompt (rag/web/hybrid template from prompts.py)
6. LLM generation (gpt-4o, temperature from request)
7. Parse follow-up questions (split on ---FOLLOW_UP---)
8. Session memory update + optional summarization
9. Return QueryResponse with sources, tokens, confidence
```

---

## Project Registry

**File:** `rag/retrieval/state.py`

| Project ID | Index File | Metadata File |
|------------|-----------|---------------|
| 7166 | `faiss_index_7166.bin` | `metadata_7166.jsonl` |
| 7201 | `faiss_index_7201.bin` | `metadata_7201.jsonl` |
| 7212 | `faiss_index_7212.bin` | `metadata_7212.jsonl` |
| 7222 | `faiss_index_7222.bin` | `metadata_7222.jsonl` |
| 7223 | `faiss_index_7223.bin` | `metadata_7223.jsonl` |
| 7277 | `faiss_index_7277.bin` | `metadata_7277.jsonl` |
| 7292 | `faiss_index_7292.bin` | `metadata_7292.jsonl` |

**To add a new project:** Add entry to `PROJECTS` dict in `rag/retrieval/state.py` with `ProjectConfig`. Ensure FAISS index + metadata files exist at `INDEX_ROOT`.

---

## Search Modes

| Mode | Prompt Builder | Context Source | Use Case |
|------|---------------|----------------|----------|
| `rag` | `build_rag_prompt()` | FAISS chunks only | Project-specific questions |
| `web` | `build_web_prompt()` | Web search only | General construction knowledge |
| `hybrid` | `build_hybrid_prompt()` | FAISS + Web (parallel) | Combined project + general |

---

## Modification Guide

### Change the LLM model
1. Update `LLM_MODEL` in `.env` (for generation)
2. Update `WEB_SEARCH_MODEL` in `.env` (for web search)
3. Restart server

### Change embedding model
1. Update `EMBEDDING_MODEL` in `.env`
2. **Warning:** Requires re-indexing ALL projects (dimension change)
3. Rebuild FAISS indices via ingestion pipeline

### Add a new project
1. Run ingestion pipeline for the new project ID
2. Place `faiss_index_{pid}.bin` and `metadata_{pid}.jsonl` in `INDEX_ROOT`
3. Add `ProjectConfig` entry to `rag/retrieval/state.py` → `PROJECTS` dict
4. Restart server

### Adjust retrieval quality
- `CONFIDENCE_THRESHOLD` in `helpers.py`: lower = fewer clarifications, higher = more
- `MAX_CONTEXT_TOKENS` in `helpers.py`: increase for more context, decrease for faster responses
- `MAX_CHUNKS` in `helpers.py`: more chunks = broader context
- `over_fetch_multiplier` in `engine.py`: higher = better filtering, more compute
- `min_score` in `QueryRequest` default: raise to filter low-quality chunks

### Adjust session memory
- `MAX_SESSIONS` in `.env`: memory cap for concurrent sessions
- `MAX_TOKENS_PER_SESSION` in `.env`: when summarization kicks in
- `max_messages_before_summary` in `memory_manager.py`: message count trigger

### Add a new intent
- Edit `rag/api/intent.py` → add regex pattern to `INTENT_PATTERNS` + response handler
- No LLM call needed; intent detection is pure regex

### Modify prompt templates
- Edit `rag/api/prompts.py` → functions: `build_rag_prompt()`, `build_web_prompt()`, `build_hybrid_prompt()`, `build_clarification_prompt()`
- All prompts include: 20+ years expertise persona, hallucination guard, follow-up requirement

### Add a new API endpoint
- Add route handler in `rag/api/routes.py`
- Add Pydantic models in `rag/api/models.py` if needed
- Route is auto-registered via the FastAPI app in `rag/api/state.py`

### Tune web search caching
- `_WEB_CACHE_TTL` in `services/web_search.py`: cache duration (default 300s)
- `_WEB_CACHE_MAX` in `services/web_search.py`: max cached entries (default 200)

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `generate.py` | ~85 | Backward-compat entry + uvicorn |
| `retrieve.py` | ~22 | Backward-compat retrieval entry |
| `memory_manager.py` | ~732 | Session persistence + summarization |
| `rag/api/state.py` | ~93 | FastAPI app, CORS, client init |
| `rag/api/routes.py` | ~530 | All 13+ route handlers |
| `rag/api/models.py` | ~197 | Pydantic request/response schemas |
| `rag/api/generation_unified.py` | ~450 | Core RAG/Web/Hybrid pipeline |
| `rag/api/generation_web.py` | ~200 | Web-search-only path |
| `rag/api/streaming.py` | ~200 | SSE async generator |
| `rag/api/intent.py` | ~134 | Regex intent detection |
| `rag/api/prompts.py` | ~203 | 4 prompt templates |
| `rag/api/helpers.py` | ~264 | Confidence, trade, budget, parsing |
| `rag/retrieval/state.py` | ~63 | Project registry + config |
| `rag/retrieval/engine.py` | ~150 | Core FAISS retrieval |
| `rag/retrieval/embeddings.py` | ~27 | LRU-cached embedding |
| `rag/retrieval/loaders.py` | ~90 | Index + metadata loading |
| `rag/retrieval/metadata.py` | ~123 | Title/URL builders |
| `rag/retrieval/diagnostics.py` | varies | Stats + test queries |
| `services/web_search.py` | ~115 | Web search + TTL cache |
| `config/settings.py` | ~11 | Settings class |
| `utils/logger.py` | ~11 | Logger setup |

---

## Metadata Schema (JSONL records)

Key fields per chunk record:
- `project_id`, `source_type` ("drawing"/"specification"/"sql")
- `text`, `token_count`, `created_at`
- `pdfName`, `drawingTitle`, `drawing_id`, `drawing_name`, `page`
- `s3_path`, `s3BucketPath` (S3 location)
- `section_title`, `trade_name`, `trade_id`, `set_name`, `set_trade`
- `csi_division`, `trades` (arrays)
- `material`, `quantity`, `unit`

**Display title resolution:** `drawingTitle` > `drawing_title` > `pdfName` > `section_title` > first 6 words > `Document {id}`

**Download URL format:** `https://{bucket}.s3.amazonaws.com/{key_prefix}/{pdfName}.pdf`

---

## Dependencies (requirements.txt)

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
openai>=1.30.0
faiss-cpu>=1.7.4
numpy>=1.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## Known Limitations
- CORS open to all origins (restrict `allow_origins` in `rag/api/state.py` for production)
- No authentication/authorization on endpoints
- No rate limiting
- Session storage capped at 200 in-memory (LRU eviction)
- Session IDs are MD5 hashes (not cryptographically secure)

---

## Startup Sequence

```python
@app.on_event("startup")
1. Clean old sessions (>24h)
2. Initialize FAISS indices for all registered projects
3. Test retrieval for projects 7212, 7223
4. Log readiness
```

**Command:** `python generate.py` or `uvicorn rag.api.state:app --host 0.0.0.0 --port 8000`

---

## Phase S3: Storage Migration to AWS S3 (2026-03-20)

### Status: COMPLETED & ACTIVATED (2026-03-23)
- **STORAGE_BACKEND=s3** set in `.env` — S3 writes are LIVE
- **Bucket:** `agentic-ai-production` | **Region:** `us-east-1` | **Prefix:** `rag-agent/`
- **AWS Key:** `AKIATXJLUBGKPCIQ5TMN` (configured in `.env`)
- Modified: `memory_manager.py` (S3 write-through save, S3 fallback load, S3 delete)
- Modified: `rag/retrieval/loaders.py` (S3 fallback for FAISS index download)
- Migration script: `scripts/migrate_sessions_to_s3.py`
- Test suite: `tests/test_s3_rag.py` (8 tests, all PASS)
- Live S3 connectivity verified

### Objective
Move conversation sessions and FAISS index backups from local VM storage to S3. Keep old code commented out for rollback.

### Current Local Storage (what moves to S3)

| Data Type | Current Location | File Pattern | S3 Target |
|-----------|-----------------|--------------|-----------|
| Conversation sessions | `./conversation_sessions/` | `session_{hash}.json` | `rag-agent/conversation_sessions/session_{hash}.json` |
| FAISS indexes (read-only at runtime) | `INDEX_ROOT/` | `faiss_index_{pid}.bin` | `rag-agent/faiss_indexes/faiss_index_{pid}.bin` (backup/sync) |
| Metadata JSONL (read-only at runtime) | `INDEX_ROOT/` | `metadata_{pid}.jsonl` | `rag-agent/faiss_indexes/metadata_{pid}.jsonl` (backup/sync) |
| Logs | Console only | N/A | `rag-agent/api_logs/{YYYY-MM-DD}/rag_agent.log` |

### S3 Folder Structure

```
rag-agent/
├── faiss_indexes/
│   ├── faiss_index_7166.bin      (~222 MB)
│   ├── metadata_7166.jsonl       (~33 MB)
│   ├── faiss_index_7201.bin      (~174 MB)
│   ├── metadata_7201.jsonl       (~26 MB)
│   └── ... (7 projects total, ~800 MB)
├── conversation_sessions/
│   ├── session_43d5ff4a46e3.json
│   └── ... (up to 200 active sessions)
└── api_logs/
    └── 2026-03-20/
        └── rag_agent.log
```

### Files to Modify

| File | Change | Details |
|------|--------|---------|
| `config/settings.py` | Add S3 settings | `STORAGE_BACKEND`, `S3_BUCKET_NAME`, `S3_REGION`, AWS creds, `S3_AGENT_PREFIX=rag-agent` |
| `memory_manager.py` | `_save_session()`: upload JSON to S3 after local save | Line ~606: add S3 upload after `json.dump()` |
| `memory_manager.py` | `_load_sessions()`: if local empty, list+download from S3 | Line ~618: add S3 fallback for session loading |
| `memory_manager.py` | Session delete: remove from S3 too | Line ~549: add S3 delete after `unlink()` |
| `rag/retrieval/loaders.py` | FAISS index loading: if local file missing, download from S3 to local cache | Lines 13-48: add S3 download fallback before `faiss.read_index()` |
| `rag/api/routes.py` | Startup: after loading indexes, upload to S3 as backup (async, non-blocking) | Line ~514: add background S3 backup task |

### New Files

| File | Purpose |
|------|---------|
| `scripts/migrate_sessions_to_s3.py` | Upload all `conversation_sessions/session_*.json` to S3 |
| `scripts/backup_indexes_to_s3.py` | Upload all FAISS `.bin` + `.jsonl` files from `INDEX_ROOT` to S3 |
| Uses shared `PROD_SETUP/s3_utils/` module | |

### FAISS Index Strategy
FAISS indexes are **large** (~800 MB total) and **read-only at runtime**. Strategy:
1. **Primary:** Always serve from local disk (fast reads, FAISS requires local file)
2. **S3 = backup/sync:** Upload to S3 after ingestion completes (done by Ingestion Agent)
3. **S3 → local recovery:** If VM loses index files, download from S3 to `INDEX_ROOT` on startup
4. **No real-time S3 reads** for FAISS — too slow for query-time retrieval

### Session Strategy
Sessions are **small** (~1-10 KB each) and **read/write frequently**. Strategy:
1. **Write-through:** Save to local + S3 on every session update
2. **Read:** Load from local first; S3 fallback if local file missing
3. **Delete:** Remove from both local and S3
4. **Startup:** Load from local dir; if empty, list and download from S3

### New .env Variables
```
STORAGE_BACKEND=s3
S3_BUCKET_NAME=vcs-ai-agents-data
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=<from user>
AWS_SECRET_ACCESS_KEY=<from user>
S3_AGENT_PREFIX=rag-agent
```

### New Dependency
```
boto3>=1.34.0   # add to requirements.txt
```

### Code Pattern for Sessions (memory_manager.py)
```python
# _save_session() — after existing local save:
# --- S3 UPLOAD (active when STORAGE_BACKEND=s3) ---
if settings.storage_backend == "s3":
    from s3_utils.operations import upload_bytes
    s3_key = f"{settings.s3_agent_prefix}/conversation_sessions/{session_file.name}"
    upload_bytes(json.dumps(session_data, ensure_ascii=False).encode('utf-8'), s3_key)
# --- END S3 UPLOAD ---

# _load_sessions() — after local load attempt:
# --- S3 FALLBACK (active when STORAGE_BACKEND=s3) ---
if settings.storage_backend == "s3" and not local_sessions_found:
    from s3_utils.operations import list_objects, download_bytes
    s3_prefix = f"{settings.s3_agent_prefix}/conversation_sessions/"
    for obj in list_objects(s3_prefix):
        data = json.loads(download_bytes(obj['Key']))
        # restore session from S3 data...
# --- END S3 FALLBACK ---
```

### Rollback
Set `STORAGE_BACKEND=local` in `.env` → restart → sessions use local-only I/O.

### Activation Instructions
When user says **"use the new S3 code"**:
1. Set `STORAGE_BACKEND=s3` in `.env`
2. Run `python scripts/migrate_sessions_to_s3.py`
3. Run `python scripts/backup_indexes_to_s3.py`
4. Restart: `sudo systemctl restart rag-agent`
5. Verify: create session → check S3 → restart → verify session persists
