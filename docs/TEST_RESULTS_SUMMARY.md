# Unified RAG Agent - E2E Test Results Summary

| Field | Value |
|-------|-------|
| **Date** | 2026-04-15 |
| **Environment** | Sandbox VM (54.197.189.113:8001) |
| **Service** | Unified RAG Agent v1.0.0 |
| **Engines** | agentic (initialized), traditional (FAISS not loaded) |
| **Model** | gpt-4.1 (primary), gpt-4.1-mini (fallback) |
| **Storage** | S3 backend, MongoDB (iField) |

---

## Test Results Overview

| Metric | Value |
|--------|-------|
| Total Tests | 8 |
| Passed | 8 |
| Failed | 0 |
| Pass Rate | **100%** |

---

## Per-Test Results

| # | Endpoint | Method | Status | Latency (ms) | Token Usage | Est. Cost |
|---|----------|--------|--------|--------------|-------------|-----------|
| 1 | `/health` | GET | 200 | 581.69 | n/a | $0.00 |
| 2 | `/config` | GET | 200 | 599.93 | n/a | $0.00 |
| 3 | `/` | GET | 200 | 583.86 | n/a | $0.00 |
| 4 | `/query` | POST | 200 | 9,458.65 | 0 (not tracked) | $0.00 |
| 5 | `/quick-query` | POST | 200 | 11,656.09 | 0 (not tracked) | $0.00 |
| 6 | `/projects/7222/documents` | GET | 200 | 569.91 | n/a | $0.00 |
| 7 | `/debug-pipeline` | GET | 200 | 569.91 | n/a | $0.00 |
| 8 | `/admin/sessions` | GET | 200 | 578.54 | n/a | $0.00 |

---

## Response Time Statistics

| Metric | Value (ms) |
|--------|-----------|
| **Min** | 569.91 |
| **Max** | 11,656.09 |
| **Average** | 3,003.58 |
| **P95 (estimated)** | ~11,106.35 |
| **Median** | 583.86 |

### Breakdown by Category

| Category | Endpoints | Avg Latency (ms) |
|----------|-----------|-------------------|
| Health/Info (GET) | `/health`, `/config`, `/`, `/debug-pipeline`, `/admin/sessions` | 582.79 |
| Document Listing (GET) | `/projects/7222/documents` | 569.91 |
| RAG Query (POST) | `/query` | 9,458.65 |
| Quick Query (POST) | `/quick-query` | 11,656.09 |

---

## Token Usage Summary

| Endpoint | Prompt Tokens | Completion Tokens | Total Tokens |
|----------|--------------|-------------------|-------------|
| `/query` (HVAC) | 0 | 0 | 0 |
| `/quick-query` (Electrical) | not reported | not reported | not reported |

**Note:** Token tracking returned zeros for the `/query` endpoint (`token_usage.total_tokens: 0`). The `token_tracking` field was `null`. This suggests token metering is not yet wired in the agentic engine, or the OpenAI usage metadata is not being captured. The agentic debug info reports `agentic_cost_usd: 0.0` and `agentic_steps: 5`.

---

## Follow-Up Questions Generated

The `/query` endpoint (HVAC query) generated these follow-up suggestions:

1. What are the technical requirements for the Variable Refrigerant Flow HVAC systems?
2. Are there specific manufacturers or models listed for the Dedicated Outdoor Air Units?
3. What are the control and monitoring features specified for the Direct Digital Control System?

---

## Source Documents (Sample from /query Response)

| # | File Name | Page |
|---|-----------|------|
| 1 | `Table-of-Contents_Rev_14` | 1 |
| 2 | `Vibration-and-Seismic-Controls-for-HVAC_Rev_01` | 1 |
| 3 | `Commissioning-of-HVAC_Rev_04` | 1 |
| 4 | `Variable-Refrigerant-Flow-HVAC-Systems_Rev_01` | 1 |
| 5 | `Commissioning-of-HVAC_Rev_01` | 1 |
| 6 | `HVAC-Air-Distribution-System-Cleaning_Rev_02` | 1 |
| 7 | `Dedicated-Outdoor-Air-Units_Rev_01` | 1 |
| 8 | `HVAC-Air-Distribution-System-Cleaning_Rev_01` | 1 |
| 9 | `Demonstration-and-Training_Rev_02` | 1 |
| 10 | `Direct-Digital-Control-System-for-HVAC_Rev_07` | 1 |

**Download URLs:** All `download_url` fields returned `null`. S3 paths are empty strings. This indicates the download URL generation is not configured for this project's documents.

---

## Project Document Index (from /projects/7222/documents)

| Metric | Value |
|--------|-------|
| Total Documents | 2,609 |
| Response Size | ~544 KB |

### Trade Distribution (Sample)

| Trade | Example Drawing |
|-------|----------------|
| HVAC | JRpkwyHotelMechPlans1-1 |
| Architecture | A001 - BASEMENT SLAB EDGE PLAN |
| Electrical | E107 - 7TH FLOOR PLAN ELECTRICAL |
| Structural | S108 - 7TH FLOOR FRAMING PLAN |
| Mechanical | M107a - 7TH FLOOR PLAN HVAC |

---

## Pipeline Debug State

| Component | Status |
|-----------|--------|
| Orchestrator fallback | Enabled (30s timeout) |
| Agentic engine | Initialized |
| Traditional engine | FAISS not loaded |
| Title cache | 1 project cached (ID: 7222), TTL: 3600s |

---

## Observations and Issues

### Working Correctly
- All 8 endpoints returned HTTP 200
- Agentic engine is initialized and answering queries with high confidence (0.9)
- Document index returns full project catalog (2,609 documents)
- Follow-up questions are contextually relevant
- Fallback mechanism is enabled and configured
- Pipeline debug endpoint provides useful operational visibility

### Issues Found
1. **Token tracking not populated** - `/query` returns `total_tokens: 0` and `token_tracking: null`. Cost tracking (`agentic_cost_usd: 0.0`) also appears non-functional despite making OpenAI API calls.
2. **Download URLs missing** - All source documents have `download_url: null` and empty `s3_path`. Users cannot download referenced documents.
3. **Empty `query` field** - The `/query` response returns `"query": ""` instead of echoing back the input query.
4. **Traditional engine FAISS not loaded** - The traditional RAG engine shows `faiss_loaded: false`. Fallback to traditional will not work until FAISS index is built.
5. **Quick-query response schema differs** - `/quick-query` returns a simpler schema (`answer`, `sources`, `confidence`, `engine_used`) compared to `/query`. No token usage, no follow-up questions, no source documents.
6. **No active sessions** - `/admin/sessions` returns 0 sessions, suggesting session tracking may not be persisting across queries.

### Performance Notes
- GET endpoints respond consistently in ~570-600ms (network round-trip to AWS us-east-1)
- RAG queries take 9.5-11.7 seconds, which is within acceptable range for multi-step agentic retrieval (5 steps observed)
- The `/projects/7222/documents` endpoint returns 544KB of JSON - pagination should be considered for production use
