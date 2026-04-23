# Unified RAG Agent — Design Spec

**Date:** 2026-04-05
**Status:** APPROVED — Ready for implementation planning
**Approach:** Facade Gateway (Approach A)
**Port:** 8001 (replaces old RAG agent)
**Gateway prefix:** `/rag/`

---

## 1. Executive Summary

Merge AgenticRAG (GPT-4.1 + MongoDB tools, port 8010) and Traditional RAG (FAISS + OpenAI embeddings, port 8001) into a single `unified-rag-agent` on port 8001. AgenticRAG is the primary engine; Traditional RAG is the automatic fallback when agentic confidence is low, answer is empty, or an exception occurs. FAISS indexes are lazy-loaded only on first fallback to save ~2GB RAM.

### Key Decisions

| Decision | Choice |
|----------|--------|
| Port | 8001 with `/rag/` gateway prefix |
| Folder | `PROD_SETUP/unified-rag-agent/` |
| Fallback trigger | Low confidence OR empty answer OR exception OR no sources |
| Endpoints | Existing endpoints preserved, routed through orchestrator |
| FAISS loading | Lazy — only on first fallback call |
| Sessions | Unified — one session system for both engines |
| Engine override | Optional `engine` field: "agentic", "traditional", or null (auto) |
| Backward compat | 100% — old request format works unchanged |

---

## 2. Folder Structure

```
PROD_SETUP/unified-rag-agent/
├── gateway/                          ← NEW (4 files)
│   ├── __init__.py
│   ├── app.py                        ← FastAPI on port 8001, lifespan, CORS
│   ├── router.py                     ← All 18 endpoints (backward compatible)
│   ├── orchestrator.py               ← Agentic-first → Traditional fallback
│   └── models.py                     ← Unified request/response schemas
│
├── agentic/                          ← MOVED from AgenticRAG/
│   ├── __init__.py
│   ├── core/
│   │   ├── agent.py                  ← ReAct agent (GPT-4.1, 8-step, 11 tools)
│   │   ├── cache.py                  ← Thread-safe two-level cache
│   │   ├── db.py                     ← MongoDB connection + indexes
│   │   ├── audit.py                  ← Structured audit logging
│   │   └── text_reconstruction.py    ← Spatial OCR fragment reconstruction
│   └── tools/
│       ├── registry.py               ← 11 tools unified registry
│       ├── mongodb_tools.py          ← drawingVision queries (4 tools)
│       ├── drawing_tools.py          ← Legacy drawing queries (4 tools)
│       ├── specification_tools.py    ← Specification queries (3 tools)
│       └── validation.py             ← Input validation
│
├── traditional/                      ← MOVED from RAG_agent_VCS/RAG/
│   ├── __init__.py
│   ├── rag/
│   │   ├── retrieval/
│   │   │   ├── engine.py             ← FAISS retrieve_context()
│   │   │   ├── loaders.py            ← Index loading + S3 fallback
│   │   │   ├── embeddings.py         ← LRU-cached OpenAI embeddings
│   │   │   ├── metadata.py           ← Title/URL builders
│   │   │   ├── state.py              ← Project registry (8 projects)
│   │   │   └── diagnostics.py        ← Stats + test queries
│   │   └── api/
│   │       ├── generation_unified.py ← RAG/Web/Hybrid pipeline
│   │       ├── generation_web.py     ← Web-only generation
│   │       ├── streaming.py          ← SSE async generator
│   │       ├── intent.py             ← Regex intent detection (9 types)
│   │       ├── prompts.py            ← 4 prompt templates
│   │       ├── greeting_agent.py     ← Greeting responses
│   │       └── helpers.py            ← Context formatting
│   ├── services/
│   │   └── web_search.py             ← OpenAI web_search tool
│   └── config/
│       └── settings.py               ← WEB_SEARCH_MODEL loader
│
├── shared/                           ← NEW (common utilities)
│   ├── __init__.py
│   ├── config.py                     ← Unified environment config
│   ├── s3_utils/                     ← Single copy (both engines)
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── config.py
│   │   ├── helpers.py
│   │   └── operations.py
│   └── session/                      ← Unified session manager
│       ├── __init__.py
│       ├── manager.py                ← Extended MemoryManager
│       └── models.py                 ← Session data models
│
├── tests/
│   ├── __init__.py
│   ├── test_orchestrator.py          ← Fallback logic (5+ tests)
│   ├── test_gateway.py               ← Endpoint tests (10+ tests)
│   ├── test_config.py                ← Config loading tests
│   ├── agentic/                      ← Existing 57 tests
│   │   ├── test_validation.py
│   │   ├── test_text_reconstruction.py
│   │   ├── test_cache.py
│   │   └── test_audit.py
│   └── traditional/
│       └── test_s3_rag.py            ← Existing S3 tests
│
├── .env                              ← Unified environment
├── .env.example
├── requirements.txt                  ← Combined dependencies
├── CLAUDE.md
└── ARCHITECTURE.md
```

---

## 3. Orchestrator

### Flow

```
Query → Intent Detection (regex, 0ms)
  │
  ├── Greeting/SmallTalk → Short-circuit response (no engine)
  │
  ├── engine="traditional" → Traditional RAG directly
  │
  ├── engine="agentic" → AgenticRAG only (no fallback)
  │
  └── engine=null (default) → Orchestrator:
        │
        ├── 1. Run AgenticRAG (MongoDB tools, ReAct loop)
        │
        ├── 2. Evaluate: confidence, answer length, sources, escalation
        │
        ├── 3a. SUCCESS (high/medium confidence + answer + sources)
        │       → Return agentic result
        │
        └── 3b. FAIL (low confidence OR empty OR error OR no sources)
                → Lazy-load FAISS (if first time)
                → Run Traditional RAG (FAISS + LLM)
                → Return traditional result with fallback_used=true
```

### Fallback Conditions

```python
def _should_fallback(result) -> bool:
    if result is None:                              return True  # Exception
    if not result.answer or len(result.answer) < 20: return True  # Empty
    if result.confidence == "low":                   return True  # Low confidence
    if not result.sources:                           return True  # No sources
    if result.needs_escalation:                      return True  # Max steps hit
    return False
```

### FAISS Lazy Loading

- First fallback call: loads FAISS index for requested project (~3-5s from disk, or S3)
- Subsequent calls: instant (index stays in memory)
- Thread-safe with double-checked locking
- Per-project loading (not all 8 projects at once)

### Streaming

- AgenticRAG streams tokens via SSE
- If fallback triggers after stream completes → emit `{"type": "fallback"}` event → traditional result as single `{"type": "done"}` event
- If `engine="traditional"` → traditional streaming pipeline (existing SSE)

---

## 4. API Endpoints

| # | Method | Endpoint | Engine | Purpose |
|---|--------|----------|--------|---------|
| 1 | POST | `/query` | Orchestrator | Agentic-first → fallback |
| 2 | POST | `/query/stream` | Orchestrator | SSE streaming with fallback |
| 3 | POST | `/quick-query` | Orchestrator | Simplified response |
| 4 | POST | `/web-search` | Traditional | Web search only |
| 5 | GET | `/health` | Both | Combined engine health |
| 6 | GET | `/config` | Gateway | Unified config |
| 7 | GET | `/` | Gateway | API info |
| 8 | POST | `/sessions/create` | Shared | Create session |
| 9 | GET | `/sessions` | Shared | List sessions |
| 10 | GET | `/sessions/{id}/stats` | Shared | Stats + engine usage |
| 11 | GET | `/sessions/{id}/conversation` | Shared | Message history |
| 12 | POST | `/sessions/{id}/update` | Shared | Update context |
| 13 | DELETE | `/sessions/{id}` | Shared | Delete session |
| 14 | POST | `/sessions/{id}/pin-document` | Traditional | Pin docs for FAISS scoped search |
| 15 | DELETE | `/sessions/{id}/pin-document` | Traditional | Unpin docs |
| 16 | GET | `/test-retrieve` | Traditional | Test FAISS directly |
| 17 | GET | `/debug-pipeline` | Both | Debug both engines |
| 18 | GET | `/metrics` | Gateway | Prometheus metrics |

### Request Schema (backward compatible)

```python
class QueryRequest(BaseModel):
    query: str                                    # Required
    project_id: int                               # Required
    session_id: Optional[str] = None
    search_mode: Optional[str] = None             # "rag" | "web" | "hybrid"
    generate_document: bool = True
    filter_source_type: Optional[str] = None
    filter_drawing_name: Optional[str] = None
    set_id: Optional[int] = None                  # AgenticRAG MongoDB filter
    conversation_history: Optional[list] = None   # Sessionless mode
    engine: Optional[str] = None                  # "agentic" | "traditional" | null
```

### Response Schema (backward compatible + new fields)

```python
class UnifiedResponse(BaseModel):
    # Backward-compatible
    success: bool = True
    answer: str
    sources: list[dict] = []
    confidence: str = "high"
    session_id: str = ""
    follow_up_questions: list[str] = []
    needs_clarification: bool = False
    
    # New
    engine_used: str = "agentic"
    fallback_used: bool = False
    agentic_confidence: Optional[str] = None
    
    # Metrics
    cost_usd: float = 0.0
    elapsed_ms: int = 0
    total_steps: int = 0
    model: str = ""
```

---

## 5. Shared Config

Single `.env` file with sections:
- **Server:** HOST, PORT, LOG_LEVEL
- **OpenAI:** OPENAI_API_KEY (shared)
- **AgenticRAG:** AGENTIC_MODEL, max steps, cost caps, rate limit
- **MongoDB:** MONGODB_URI, MONGO_DB
- **Traditional RAG:** TRADITIONAL_MODEL, EMBEDDING_MODEL, INDEX_ROOT, confidence threshold
- **S3:** Shared bucket, region, credentials, prefix
- **Orchestrator:** FALLBACK_ENABLED, FALLBACK_TIMEOUT_SECONDS, FAISS_LAZY_LOAD
- **Auth:** API_KEY

---

## 6. Unified Sessions

Extends traditional `MemoryManager` with engine tracking:

```python
class UnifiedSession:
    session_id: str
    messages: list[Message]
    context: ConversationContext        # project_id, filters, pinned docs
    summaries: list[ConversationSummary]
    total_tokens: int
    engine_usage: dict                  # {"agentic": 5, "traditional": 2, "fallback": 1}
    last_engine: str                    # Which engine answered last
    total_cost_usd: float              # Cumulative cost
```

Persistence: local JSON + S3 write-through (existing pattern).

---

## 7. Import Path Changes

Both engines need import path adjustments since they're moving into subdirectories.

### AgenticRAG imports

Old: `from core.agent import run_agent`
New: `from agentic.core.agent import run_agent`

**Strategy:** Add `sys.path` manipulation in `agentic/__init__.py` OR use relative imports within the agentic package. Minimal code changes — just the import roots.

### Traditional RAG imports

Old: `from rag.retrieval.engine import retrieve_context`
New: `from traditional.rag.retrieval.engine import retrieve_context`

**Strategy:** Same — `sys.path` in `traditional/__init__.py` OR relative imports.

### Shared imports

Both engines import S3 utils:
- Old agentic: `from s3_utils.client import get_s3_client`
- Old traditional: `from s3_utils.client import get_s3_client`
- New: `from shared.s3_utils.client import get_s3_client`

**Strategy:** `shared/__init__.py` adds shared path. Both engines get a thin shim that redirects `s3_utils` imports to `shared.s3_utils`.

---

## 8. 12-Point Production Review

| # | Dimension | Rating | Key Design |
|---|-----------|--------|-----------|
| 1 | Scaling | PASS | 2 workers, lazy FAISS, MongoDB pool 20/worker |
| 2 | Optimization | PASS | Agentic-first skips FAISS 85% of time, two-level cache, lazy loading |
| 3 | Performance | PASS | Per-request timing, engine attribution, Prometheus /metrics |
| 4 | Request Handling | PASS | Rate limiting, 120s timeout, $0.50 cost cap, engine override |
| 5 | Vulnerability | PASS | Project ID hard-override, input validation, timing-safe auth, prompt defense |
| 6 | SDLC | PASS | 57+ agentic tests + new gateway/orchestrator tests, 80%+ target |
| 7 | Compliance | PASS | Structured audit logging with engine_used field, S3 persistence |
| 8 | Disaster Recovery | PASS | MongoDB down → FAISS fallback. OpenAI down → model fallback. S3 fallback for indexes. |
| 9 | Support | PASS | /health (both engines), /debug-pipeline, engine attribution in responses |
| 10 | Maintenance | PASS | All config via .env, independent engine updates, lazy index management |
| 11 | Network | PASS | TLS at Nginx, internal-only port, restricted CORS, bearer auth |
| 12 | Resources | PASS | Lazy FAISS (~2GB saved), cost caps, connection pools, Prometheus monitoring |

---

## 9. Implementation Phases

| Phase | What | Depends On | Effort |
|-------|------|-----------|--------|
| **P1** | Create folder structure + move files | Nothing | 1 hr |
| **P2** | Shared config + S3 utils + import shims | P1 | 2 hrs |
| **P3** | Unified session manager | P2 | 2 hrs |
| **P4** | Gateway app + router (all 18 endpoints) | P2, P3 | 3 hrs |
| **P5** | Orchestrator (fallback logic) | P4 | 3 hrs |
| **P6** | Tests (orchestrator + gateway + config) | P5 | 3 hrs |
| **P7** | Import path fixes in both engines | P2 | 2 hrs |
| **P8** | Integration test (full pipeline, real data) | P7 | 2 hrs |
| **P9** | Deploy to sandbox VM (port 8001) | P8 | 1 hr |
| **P10** | Test on sandbox + fix bugs | P9 | 2 hrs |
| **P11** | Push to GitHub | P10 | 30 min |
| **Total** | | | **~21 hrs** |

---

## 10. What's NOT Changing

- AgenticRAG core logic (`agent.py`, all tools, `text_reconstruction.py`) — untouched
- Traditional RAG core logic (`engine.py`, `generation_unified.py`, `streaming.py`) — untouched
- FAISS indexes (same files, same S3 paths)
- MongoDB collections (same queries)
- S3 bucket structure (prefix changes from `rag-agent/` to `unified-rag-agent/`)
- Existing test suites (moved, not rewritten)

---

## 11. Success Criteria

| Metric | Target |
|--------|--------|
| All old `/query` requests work unchanged | 100% backward compatible |
| AgenticRAG answers most queries without fallback | >80% |
| Fallback latency (first call, FAISS cold load) | <5 seconds |
| Fallback latency (subsequent calls) | <3 seconds |
| Total test count | 70+ (57 agentic + 15 gateway/orchestrator) |
| Memory (no fallback triggered) | <300MB |
| Memory (FAISS loaded for 1 project) | <2.3GB |
| Sandbox deployment | Running on port 8001 |
| Gateway health check | Both engines report status |
