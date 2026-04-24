# CLAUDE.md — Unified RAG Agent

## Architecture

Merged AgenticRAG (MongoDB tools, GPT-4.1) + Traditional RAG (FAISS + OpenAI embeddings) into a single service.

```
Query → Gateway (port 8001)
  → AgenticRAG (primary): MongoDB tools, ReAct loop, 11 tools, 3 collections
  → If low confidence/empty/error:
    → Traditional RAG (fallback): FAISS vector search, 8 projects, LRU embeddings
  → Unified response with engine_used attribution
```

## Engines

| Engine | Data Source | Model | When Used |
|--------|-----------|-------|-----------|
| AgenticRAG | MongoDB (drawingVision, drawing 2.8M, specification) | GPT-4.1 | Primary — all queries |
| Traditional RAG | FAISS indexes (8 projects, OpenAI embeddings) | GPT-4o | Fallback on low confidence, or explicit engine="traditional" |

## Folder Structure

```
unified-rag-agent/
├── gateway/         ← FastAPI app, router, orchestrator, models
├── agentic/         ← AgenticRAG engine (core/ + tools/)
├── traditional/     ← Traditional RAG engine (rag/ + services/)
├── shared/          ← Config, S3 utils, session manager
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
| GET | /test-retrieve | Test FAISS directly |
| GET | /debug-pipeline | Debug both engines |
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
