# CLAUDE.md — Unified RAG Agent

## Architecture

Merged AgenticRAG (MongoDB tools, GPT-4.1) + Traditional RAG (FAISS + OpenAI embeddings) into a single service.

```
Query → Gateway (port 8001)
  → AgenticRAG (primary): MongoDB tools, ReAct loop, 7 tools (4 legacy + 3 spec), 2 collections
  → If low confidence/empty/error:
    → Document discovery aggregation (replaces FAISS fallback)
    → Traditional RAG (fallback): FAISS vector search, 8 projects, LRU embeddings
  → Unified response with engine_used attribution
```

## Engines

| Engine | Data Source | Model | When Used |
|--------|-----------|-------|-----------|
| AgenticRAG | MongoDB (drawing 2.8M, specification), 7 tools (4 legacy + 3 spec) | GPT-4.1 | Primary — all queries |
| Document Discovery | MongoDB aggregation on drawing/specification collections | N/A | When agent cannot answer — surfaces document groups for user scoping |
| Traditional RAG | FAISS indexes (8 projects, OpenAI embeddings) | GPT-4o | Fallback on low confidence, or explicit engine="traditional" |

## Folder Structure

```
unified-rag-agent/
├── gateway/         ← FastAPI app, router, orchestrator, models, title_cache
├── agentic/         ← AgenticRAG engine (core/ + tools/), 7 tools (4 legacy + 3 spec)
├── traditional/     ← Traditional RAG engine (rag/ + services/)
├── shared/          ← Config, S3 utils, session manager, document scope
└── tests/           ← Combined test suite
```

## API Endpoints (port 8001, prefix /rag/ via gateway)

| Method | Path | Purpose |
|--------|------|---------|
| POST | /query | Agentic-first → fallback |
| POST | /query/stream | SSE streaming |
| POST | /quick-query | Simplified response |
| POST | /web-search | Web search (traditional only) |
| GET | /health | Combined engine health |
| GET | /config | Settings (no secrets) |
| GET | / | API info |
| POST | /sessions/create | Create session |
| GET | /sessions | List sessions |
| GET | /sessions/{id}/stats | Session stats + engine usage |
| GET | /sessions/{id}/conversation | Message history |
| POST | /sessions/{id}/update | Update context |
| DELETE | /sessions/{id} | Delete session |
| POST | /sessions/{id}/pin-document | Pin for FAISS scoped search |
| DELETE | /sessions/{id}/pin-document | Unpin |
| POST | /sessions/{id}/scope | Set document scope (drawing_title, drawing_name) |
| DELETE | /sessions/{id}/scope | Clear document scope |
| GET | /sessions/{id}/scope | Get current scope state |
| GET | /projects/{id}/documents | Document discovery — list drawing titles and spec sections |
| GET | /admin/sessions | Admin: list all sessions with scope state |
| POST | /admin/cache/refresh | Admin: invalidate title cache (by project or all) |
| GET | /test-retrieve | Test FAISS directly |
| GET | /debug-pipeline | Debug both engines + title cache stats |
| GET | /metrics | Prometheus |

## Key Environment Variables

```
PORT=8001
OPENAI_API_KEY=...
AGENTIC_MODEL=gpt-4.1
TRADITIONAL_MODEL=gpt-4o
MONGODB_URI=mongodb+srv://...
INDEX_ROOT=./index
FALLBACK_ENABLED=true
FAISS_LAZY_LOAD=true
STORAGE_BACKEND=s3
```

## Fallback Logic

AgenticRAG runs first. Falls back to Traditional RAG when:
- Confidence = "low"
- Answer is empty or < 20 chars
- No sources found
- Agent hit max steps (needs_escalation)
- Exception/timeout

FAISS indexes are lazy-loaded on first fallback (saves ~2GB RAM).

## Running

```bash
# Development
cd unified-rag-agent
python -m gateway.app

# Production (systemd)
uvicorn gateway.app:app --host 0.0.0.0 --port 8001 --workers 1
```

## Tests

```bash
python -m pytest tests/ -v
```
