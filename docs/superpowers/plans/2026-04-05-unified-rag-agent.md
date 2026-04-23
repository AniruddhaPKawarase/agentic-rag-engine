# Unified RAG Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge AgenticRAG and Traditional RAG into a single unified-rag-agent on port 8001 with agentic-first + automatic fallback to traditional FAISS on low confidence.

**Architecture:** Facade Gateway pattern — `gateway/app.py` is the single FastAPI entry point. `gateway/orchestrator.py` runs AgenticRAG first, evaluates confidence, and falls back to Traditional RAG if needed. Both engine codebases are moved (not rewritten) into `agentic/` and `traditional/` subdirectories. Shared config, S3, and sessions in `shared/`.

**Tech Stack:** Python 3.10+, FastAPI, OpenAI (GPT-4.1 + text-embedding-3-small), MongoDB (pymongo), FAISS (faiss-cpu), boto3, pydantic, uvicorn, pytest

**Design Spec:** `docs/superpowers/specs/2026-04-05-unified-rag-design.md`

---

## File Map

### New Files (gateway + shared)

| File | Responsibility |
|------|---------------|
| `gateway/__init__.py` | Package init |
| `gateway/app.py` | FastAPI app on port 8001, lifespan, CORS, Prometheus |
| `gateway/router.py` | All 18 endpoints, backward compatible |
| `gateway/orchestrator.py` | Agentic-first → fallback logic, lazy FAISS loading |
| `gateway/models.py` | Unified request/response schemas |
| `shared/__init__.py` | Package init |
| `shared/config.py` | Unified env config for both engines |
| `shared/s3_utils/` | Copied from old RAG s3_utils/ (single source of truth) |
| `shared/session/__init__.py` | Session package init |
| `shared/session/manager.py` | Extended MemoryManager with engine tracking |
| `shared/session/models.py` | Session data models |

### Move Operations (no code changes)

| Source | Destination |
|--------|------------|
| `PROD_SETUP/AgenticRAG/core/` | `unified-rag-agent/agentic/core/` |
| `PROD_SETUP/AgenticRAG/tools/` | `unified-rag-agent/agentic/tools/` |
| `PROD_SETUP/AgenticRAG/config.py` | `unified-rag-agent/agentic/config.py` |
| `PROD_SETUP/RAG_agent_VCS/RAG/rag/` | `unified-rag-agent/traditional/rag/` |
| `PROD_SETUP/RAG_agent_VCS/RAG/services/` | `unified-rag-agent/traditional/services/` |
| `PROD_SETUP/RAG_agent_VCS/RAG/config/` | `unified-rag-agent/traditional/config/` |
| `PROD_SETUP/RAG_agent_VCS/RAG/memory_manager.py` | `unified-rag-agent/traditional/memory_manager.py` |
| `PROD_SETUP/RAG_agent_VCS/RAG/s3_utils/` | `unified-rag-agent/shared/s3_utils/` |

### Test Files

| File | Tests |
|------|-------|
| `tests/__init__.py` | Package init |
| `tests/test_orchestrator.py` | Fallback logic, engine selection, lazy loading |
| `tests/test_gateway.py` | Endpoint registration, request routing |
| `tests/test_config.py` | Config loading, defaults, validation |
| `tests/test_models.py` | Request/response schema validation |
| `tests/agentic/` | Existing 57 tests (moved from AgenticRAG/tests/) |
| `tests/traditional/` | Existing S3 tests (moved from RAG/tests/) |

---

## Task Overview (11 Tasks)

| # | Task | Depends On | Effort |
|---|------|-----------|--------|
| 1 | Create folder structure + move files | - | 1 hr |
| 2 | Shared config | 1 | 1 hr |
| 3 | Shared S3 utils + import shims | 1, 2 | 1 hr |
| 4 | Unified session manager | 2, 3 | 2 hrs |
| 5 | Gateway models (request/response) | 2 | 1 hr |
| 6 | Orchestrator (fallback logic) | 2, 5 | 3 hrs |
| 7 | Gateway app + router (18 endpoints) | 4, 5, 6 | 3 hrs |
| 8 | Import path fixes in both engines | 3 | 2 hrs |
| 9 | Integration test + CLAUDE.md | 7, 8 | 2 hrs |
| 10 | Deploy to sandbox VM | 9 | 2 hrs |
| 11 | Test on sandbox + push to GitHub | 10 | 2 hrs |

---

## Task 1: Create Folder Structure + Move Files

**Files:**
- Create: directory tree for `unified-rag-agent/`
- Move: files from `AgenticRAG/` → `agentic/`, `RAG_agent_VCS/RAG/` → `traditional/`

- [ ] **Step 1: Create the unified-rag-agent directory tree**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"

# Create all directories
mkdir -p unified-rag-agent/{gateway,agentic,traditional,shared/s3_utils,shared/session,tests/agentic,tests/traditional}

# Create __init__.py files
touch unified-rag-agent/__init__.py
touch unified-rag-agent/gateway/__init__.py
touch unified-rag-agent/agentic/__init__.py
touch unified-rag-agent/traditional/__init__.py
touch unified-rag-agent/shared/__init__.py
touch unified-rag-agent/shared/s3_utils/__init__.py
touch unified-rag-agent/shared/session/__init__.py
touch unified-rag-agent/tests/__init__.py
touch unified-rag-agent/tests/agentic/__init__.py
touch unified-rag-agent/tests/traditional/__init__.py
```

- [ ] **Step 2: Copy AgenticRAG engine into agentic/**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"

# Core engine
cp -r AgenticRAG/core unified-rag-agent/agentic/core
cp -r AgenticRAG/tools unified-rag-agent/agentic/tools
cp AgenticRAG/config.py unified-rag-agent/agentic/config.py

# Ensure __init__.py exists
touch unified-rag-agent/agentic/__init__.py
touch unified-rag-agent/agentic/core/__init__.py
touch unified-rag-agent/agentic/tools/__init__.py
```

- [ ] **Step 3: Copy Traditional RAG engine into traditional/**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"

# RAG module (retrieval + api)
cp -r RAG_agent_VCS/RAG/rag unified-rag-agent/traditional/rag

# Supporting files
cp RAG_agent_VCS/RAG/memory_manager.py unified-rag-agent/traditional/memory_manager.py
cp -r RAG_agent_VCS/RAG/services unified-rag-agent/traditional/services
cp -r RAG_agent_VCS/RAG/config unified-rag-agent/traditional/config

# Compatibility shims
cp RAG_agent_VCS/RAG/generate.py unified-rag-agent/traditional/generate.py
cp RAG_agent_VCS/RAG/retrieve.py unified-rag-agent/traditional/retrieve.py

# Ensure __init__.py
touch unified-rag-agent/traditional/__init__.py
```

- [ ] **Step 4: Copy S3 utils into shared/**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"

# Use the old RAG's s3_utils (more complete)
cp RAG_agent_VCS/RAG/s3_utils/client.py unified-rag-agent/shared/s3_utils/client.py
cp RAG_agent_VCS/RAG/s3_utils/config.py unified-rag-agent/shared/s3_utils/config.py
cp RAG_agent_VCS/RAG/s3_utils/helpers.py unified-rag-agent/shared/s3_utils/helpers.py
cp RAG_agent_VCS/RAG/s3_utils/operations.py unified-rag-agent/shared/s3_utils/operations.py
touch unified-rag-agent/shared/s3_utils/__init__.py
```

- [ ] **Step 5: Copy test files**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"

# Agentic tests
cp AgenticRAG/tests/conftest.py unified-rag-agent/tests/agentic/conftest.py
cp AgenticRAG/tests/test_validation.py unified-rag-agent/tests/agentic/test_validation.py
cp AgenticRAG/tests/test_text_reconstruction.py unified-rag-agent/tests/agentic/test_text_reconstruction.py
cp AgenticRAG/tests/test_cache.py unified-rag-agent/tests/agentic/test_cache.py
cp AgenticRAG/tests/test_audit.py unified-rag-agent/tests/agentic/test_audit.py

# Traditional tests
cp RAG_agent_VCS/RAG/tests/test_s3_rag.py unified-rag-agent/tests/traditional/test_s3_rag.py
```

- [ ] **Step 6: Copy .env and requirements**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"

# Take the agentic .env as base (we'll merge in Task 2)
cp AgenticRAG/.env unified-rag-agent/.env
cp AgenticRAG/.env.example unified-rag-agent/.env.example

# Merge requirements (deduplicate later in Task 2)
cat AgenticRAG/requirements.txt RAG_agent_VCS/RAG/requirements.txt | sort -u > unified-rag-agent/requirements.txt
```

- [ ] **Step 7: Verify structure**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent"
find . -type f -name "*.py" | wc -l
# Expected: ~50+ Python files
ls gateway/ agentic/core/ agentic/tools/ traditional/rag/ shared/s3_utils/
```

- [ ] **Step 8: Commit**

```bash
git add unified-rag-agent/
git commit -m "feat(unified-rag): create folder structure + copy both engines"
```

---

## Task 2: Shared Config

**Files:**
- Create: `shared/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write test**

```python
"""tests/test_config.py — Unified config tests."""
import os
import pytest


def test_config_loads_defaults():
    """Config loads with sensible defaults when env vars are minimal."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
    
    # Clear cached config if any
    from shared.config import get_config
    get_config.cache_clear()
    
    config = get_config()
    assert config.port == 8001
    assert config.agentic_model == "gpt-4.1"
    assert config.traditional_model == "gpt-4o"
    assert config.fallback_enabled is True
    assert config.faiss_lazy_load is True
    assert config.agentic_max_steps == 8
    assert config.confidence_threshold == 0.30


def test_config_reads_env_overrides():
    """Config respects environment variable overrides."""
    os.environ["PORT"] = "9999"
    os.environ["AGENTIC_MODEL"] = "gpt-5"
    os.environ["FALLBACK_ENABLED"] = "false"
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
    
    from shared.config import get_config
    get_config.cache_clear()
    
    config = get_config()
    assert config.port == 9999
    assert config.agentic_model == "gpt-5"
    assert config.fallback_enabled is False
    
    # Cleanup
    os.environ.pop("PORT", None)
    os.environ.pop("AGENTIC_MODEL", None)
    os.environ.pop("FALLBACK_ENABLED", None)


def test_config_is_immutable():
    """Config dataclass is frozen."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
    
    from shared.config import get_config
    get_config.cache_clear()
    
    config = get_config()
    with pytest.raises(AttributeError):
        config.port = 1234
```

- [ ] **Step 2: Run tests (expect fail)**

Run: `cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent" && python -m pytest tests/test_config.py -v`
Expected: FAIL (cannot import shared.config)

- [ ] **Step 3: Implement shared/config.py**

```python
"""
shared/config.py — Unified configuration for both RAG engines.

Single .env file drives everything. Both engines read from this config.
"""

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class UnifiedConfig:
    """Immutable unified configuration."""
    
    # ── Server ──────────────────────────────────
    host: str
    port: int
    log_level: str
    
    # ── OpenAI (shared) ─────────────────────────
    openai_api_key: str
    
    # ── AgenticRAG ──────────────────────────────
    agentic_model: str
    agentic_model_fallback: str
    agentic_max_steps: int
    agentic_max_context_tokens: int
    agentic_max_request_cost: float
    agentic_daily_budget: float
    agentic_rate_limit: int
    
    # ── Traditional RAG ─────────────────────────
    traditional_model: str
    traditional_embedding_model: str
    web_search_model: str
    index_root: str
    max_sessions: int
    max_tokens_per_session: int
    confidence_threshold: float
    
    # ── MongoDB (agentic) ───────────────────────
    mongodb_uri: str
    mongo_db: str
    
    # ── S3 (shared) ─────────────────────────────
    storage_backend: str
    s3_bucket_name: str
    s3_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_agent_prefix: str
    
    # ── Orchestrator ────────────────────────────
    fallback_enabled: bool
    fallback_timeout_seconds: int
    faiss_lazy_load: bool
    
    # ── Auth ────────────────────────────────────
    api_key: str


@lru_cache()
def get_config() -> UnifiedConfig:
    """Build config from environment variables."""
    return UnifiedConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        
        agentic_model=os.getenv("AGENTIC_MODEL", "gpt-4.1"),
        agentic_model_fallback=os.getenv("AGENTIC_MODEL_FALLBACK", "gpt-4.1-mini"),
        agentic_max_steps=int(os.getenv("AGENTIC_MAX_STEPS", "8")),
        agentic_max_context_tokens=int(os.getenv("AGENTIC_MAX_CONTEXT_TOKENS", "100000")),
        agentic_max_request_cost=float(os.getenv("AGENTIC_MAX_REQUEST_COST", "0.50")),
        agentic_daily_budget=float(os.getenv("AGENTIC_DAILY_BUDGET", "50.0")),
        agentic_rate_limit=int(os.getenv("AGENTIC_RATE_LIMIT", "20")),
        
        traditional_model=os.getenv("TRADITIONAL_MODEL", "gpt-4o"),
        traditional_embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        web_search_model=os.getenv("WEB_SEARCH_MODEL", "gpt-4.1"),
        index_root=os.getenv("INDEX_ROOT", "./index"),
        max_sessions=int(os.getenv("MAX_SESSIONS", "200")),
        max_tokens_per_session=int(os.getenv("MAX_TOKENS_PER_SESSION", "10000")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.30")),
        
        mongodb_uri=os.environ.get("MONGODB_URI", ""),
        mongo_db=os.getenv("MONGO_DB", "iField"),
        
        storage_backend=os.getenv("STORAGE_BACKEND", "s3"),
        s3_bucket_name=os.getenv("S3_BUCKET_NAME", "agentic-ai-production"),
        s3_region=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        s3_agent_prefix=os.getenv("S3_AGENT_PREFIX", "unified-rag-agent"),
        
        fallback_enabled=os.getenv("FALLBACK_ENABLED", "true").lower() == "true",
        fallback_timeout_seconds=int(os.getenv("FALLBACK_TIMEOUT_SECONDS", "30")),
        faiss_lazy_load=os.getenv("FAISS_LAZY_LOAD", "true").lower() == "true",
        
        api_key=os.getenv("API_KEY", ""),
    )
```

- [ ] **Step 4: Run tests (expect pass)**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3/3 PASS

- [ ] **Step 5: Write unified .env file**

Create `unified-rag-agent/.env` merging both engines' env vars. Copy the actual secrets from the old RAG .env and the AgenticRAG .env.

- [ ] **Step 6: Write requirements.txt**

```
# Unified RAG Agent dependencies
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.30.0,<1.0.0
openai>=1.30.0,<2.0.0
pymongo>=4.6.0,<5.0.0
faiss-cpu>=1.7.4
numpy>=1.24.0
pydantic>=2.0.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0
cachetools>=5.3.0,<6.0.0
boto3>=1.34.0
prometheus-fastapi-instrumentator>=7.0.0,<8.0.0
```

- [ ] **Step 7: Commit**

```bash
git add shared/config.py tests/test_config.py .env requirements.txt
git commit -m "feat(unified-rag): shared config — unified env for both engines"
```

---

## Task 3: Shared S3 Utils + Import Shims

**Files:**
- Create: `agentic/compat.py` — import shim for agentic engine
- Create: `traditional/compat.py` — import shim for traditional engine

- [ ] **Step 1: Create agentic import shim**

```python
"""
agentic/compat.py — Import compatibility for AgenticRAG.

AgenticRAG was originally at the repo root. Now it's in agentic/.
This shim ensures internal imports still work.
"""

import sys
from pathlib import Path

# Add unified-rag-agent root to path so 'shared' is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Add agentic/ to path so 'core' and 'tools' are importable from within agentic
_agentic = Path(__file__).resolve().parent
if str(_agentic) not in sys.path:
    sys.path.insert(0, str(_agentic))
```

- [ ] **Step 2: Create traditional import shim**

```python
"""
traditional/compat.py — Import compatibility for Traditional RAG.

Traditional RAG was originally at RAG_agent_VCS/RAG/. Now it's in traditional/.
This shim ensures internal imports still work.
"""

import sys
from pathlib import Path

# Add unified-rag-agent root to path so 'shared' is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Add traditional/ to path so 'rag', 'services', 'config' are importable
_traditional = Path(__file__).resolve().parent
if str(_traditional) not in sys.path:
    sys.path.insert(0, str(_traditional))

# Redirect s3_utils imports to shared.s3_utils
import importlib
import shared.s3_utils as _s3
sys.modules["s3_utils"] = _s3
sys.modules["s3_utils.client"] = importlib.import_module("shared.s3_utils.client")
sys.modules["s3_utils.config"] = importlib.import_module("shared.s3_utils.config")
sys.modules["s3_utils.helpers"] = importlib.import_module("shared.s3_utils.helpers")
sys.modules["s3_utils.operations"] = importlib.import_module("shared.s3_utils.operations")
```

- [ ] **Step 3: Update agentic/__init__.py to auto-load compat**

```python
"""AgenticRAG engine — MongoDB-based agentic document Q&A."""
from agentic import compat  # noqa: F401 — sets up import paths
```

- [ ] **Step 4: Update traditional/__init__.py to auto-load compat**

```python
"""Traditional RAG engine — FAISS-based vector search document Q&A."""
from traditional import compat  # noqa: F401 — sets up import paths
```

- [ ] **Step 5: Verify agentic imports work**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent"
python -c "import agentic; from agentic.core.cache import get_tool_result; print('Agentic imports OK')"
```

- [ ] **Step 6: Verify shared S3 imports work**

```bash
python -c "from shared.s3_utils.config import get_s3_config; print('Shared S3 OK')"
```

- [ ] **Step 7: Commit**

```bash
git add agentic/compat.py agentic/__init__.py traditional/compat.py traditional/__init__.py
git commit -m "feat(unified-rag): import shims — redirect s3_utils to shared, fix paths"
```

---

## Task 4: Unified Session Manager

**Files:**
- Create: `shared/session/models.py`
- Create: `shared/session/manager.py`

> This extends the existing MemoryManager from traditional/memory_manager.py with engine tracking fields. Due to the complexity of MemoryManager (732 lines), we import and extend it rather than rewrite.

- [ ] **Step 1: Create session models**

```python
"""shared/session/models.py — Extended session models with engine tracking."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EngineUsage:
    """Tracks which engine answered queries in a session."""
    agentic: int = 0
    traditional: int = 0
    fallback: int = 0
    
    def record(self, engine: str) -> None:
        if engine == "agentic":
            self.agentic += 1
        elif engine == "traditional":
            self.traditional += 1
        elif engine == "traditional_fallback":
            self.fallback += 1
    
    def to_dict(self) -> Dict[str, int]:
        return {"agentic": self.agentic, "traditional": self.traditional, "fallback": self.fallback}


@dataclass
class UnifiedSessionMeta:
    """Extra metadata for unified sessions."""
    engine_usage: EngineUsage = field(default_factory=EngineUsage)
    last_engine: str = ""
    total_cost_usd: float = 0.0
```

- [ ] **Step 2: Create session manager wrapper**

```python
"""
shared/session/manager.py — Unified session manager.

Wraps the traditional MemoryManager and adds engine tracking.
The MemoryManager handles all the heavy lifting (session CRUD, S3 persistence,
summarization, token tracking). We just extend it.
"""

import logging
from typing import Optional, Dict, Any

from shared.session.models import EngineUsage, UnifiedSessionMeta

logger = logging.getLogger("unified_rag.session")

# Session metadata stored alongside traditional sessions
_session_meta: Dict[str, UnifiedSessionMeta] = {}


def get_meta(session_id: str) -> UnifiedSessionMeta:
    """Get or create engine metadata for a session."""
    if session_id not in _session_meta:
        _session_meta[session_id] = UnifiedSessionMeta()
    return _session_meta[session_id]


def record_engine_use(session_id: str, engine: str, cost_usd: float = 0.0) -> None:
    """Record which engine answered a query in this session."""
    meta = get_meta(session_id)
    meta.engine_usage.record(engine)
    meta.last_engine = engine
    meta.total_cost_usd += cost_usd
    logger.info(
        "Session %s: engine=%s, total_cost=$%.4f, usage=%s",
        session_id, engine, meta.total_cost_usd, meta.engine_usage.to_dict(),
    )


def get_session_stats_extended(session_id: str) -> Dict[str, Any]:
    """Get extended stats including engine usage."""
    meta = get_meta(session_id)
    return {
        "engine_usage": meta.engine_usage.to_dict(),
        "last_engine": meta.last_engine,
        "total_cost_usd": round(meta.total_cost_usd, 4),
    }


def clear_meta(session_id: str) -> None:
    """Clear engine metadata for a session."""
    _session_meta.pop(session_id, None)
```

- [ ] **Step 3: Write tests**

```python
"""tests/test_session.py — Unified session manager tests."""

import pytest


def test_engine_usage_tracking():
    from shared.session.manager import record_engine_use, get_meta, clear_meta
    
    sid = "test_session_1"
    clear_meta(sid)
    
    record_engine_use(sid, "agentic", cost_usd=0.005)
    record_engine_use(sid, "agentic", cost_usd=0.003)
    record_engine_use(sid, "traditional_fallback", cost_usd=0.01)
    
    meta = get_meta(sid)
    assert meta.engine_usage.agentic == 2
    assert meta.engine_usage.fallback == 1
    assert meta.last_engine == "traditional_fallback"
    assert meta.total_cost_usd == pytest.approx(0.018)
    
    clear_meta(sid)


def test_get_session_stats_extended():
    from shared.session.manager import record_engine_use, get_session_stats_extended, clear_meta
    
    sid = "test_session_2"
    clear_meta(sid)
    
    record_engine_use(sid, "agentic", cost_usd=0.005)
    
    stats = get_session_stats_extended(sid)
    assert stats["engine_usage"]["agentic"] == 1
    assert stats["last_engine"] == "agentic"
    assert stats["total_cost_usd"] == 0.005
    
    clear_meta(sid)


def test_engine_usage_to_dict():
    from shared.session.models import EngineUsage
    
    usage = EngineUsage(agentic=5, traditional=2, fallback=1)
    d = usage.to_dict()
    assert d == {"agentic": 5, "traditional": 2, "fallback": 1}
```

- [ ] **Step 4: Run tests (expect pass)**

Run: `python -m pytest tests/test_session.py -v`
Expected: 3/3 PASS

- [ ] **Step 5: Commit**

```bash
git add shared/session/ tests/test_session.py
git commit -m "feat(unified-rag): unified session manager with engine tracking"
```

---

## Task 5: Gateway Models

**Files:**
- Create: `gateway/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write test**

```python
"""tests/test_models.py — Unified request/response model tests."""

import pytest


def test_query_request_minimal():
    from gateway.models import QueryRequest
    req = QueryRequest(query="What materials are used?", project_id=7325)
    assert req.query == "What materials are used?"
    assert req.project_id == 7325
    assert req.engine is None
    assert req.session_id is None
    assert req.set_id is None


def test_query_request_with_engine_override():
    from gateway.models import QueryRequest
    req = QueryRequest(query="test", project_id=1, engine="traditional")
    assert req.engine == "traditional"


def test_query_request_rejects_short_query():
    from gateway.models import QueryRequest
    with pytest.raises(Exception):
        QueryRequest(query="", project_id=1)


def test_unified_response_defaults():
    from gateway.models import UnifiedResponse
    resp = UnifiedResponse(answer="Test answer")
    assert resp.success is True
    assert resp.engine_used == "agentic"
    assert resp.fallback_used is False
    assert resp.follow_up_questions == []
    assert resp.cost_usd == 0.0


def test_unified_response_with_fallback():
    from gateway.models import UnifiedResponse
    resp = UnifiedResponse(
        answer="Found via FAISS",
        engine_used="traditional_fallback",
        fallback_used=True,
        agentic_confidence="low",
        confidence="high",
    )
    assert resp.fallback_used is True
    assert resp.agentic_confidence == "low"
    assert resp.confidence == "high"
```

- [ ] **Step 2: Run tests (expect fail)**

- [ ] **Step 3: Implement gateway/models.py**

```python
"""
gateway/models.py — Unified request/response schemas.

Backward compatible with old RAG request format.
New fields (engine, set_id, etc.) are all Optional with None defaults.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Unified query request — backward compatible with old RAG."""
    query: str = Field(..., min_length=1, max_length=2000)
    project_id: int = Field(..., ge=1, le=999999)
    session_id: Optional[str] = None
    search_mode: Optional[str] = None
    generate_document: bool = True
    filter_source_type: Optional[str] = None
    filter_drawing_name: Optional[str] = None
    set_id: Optional[int] = None
    conversation_history: Optional[list] = None
    engine: Optional[str] = Field(
        None,
        description="Force engine: 'agentic', 'traditional', or null (auto)",
    )


class UnifiedResponse(BaseModel):
    """Unified response — backward compatible + engine attribution."""
    success: bool = True
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: str = "high"
    session_id: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    
    engine_used: str = "agentic"
    fallback_used: bool = False
    agentic_confidence: Optional[str] = None
    
    cost_usd: float = 0.0
    elapsed_ms: int = 0
    total_steps: int = 0
    model: str = ""
```

- [ ] **Step 4: Run tests (expect pass)**

Run: `python -m pytest tests/test_models.py -v`
Expected: 5/5 PASS

- [ ] **Step 5: Commit**

```bash
git add gateway/models.py tests/test_models.py
git commit -m "feat(unified-rag): gateway models — backward-compatible request/response"
```

---

## Task 6: Orchestrator

**Files:**
- Create: `gateway/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write test**

```python
"""tests/test_orchestrator.py — Orchestrator fallback logic tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass


@dataclass
class MockAgenticResult:
    answer: str = ""
    sources: list = None
    confidence: str = "high"
    needs_escalation: bool = False
    cost_usd: float = 0.005
    elapsed_ms: int = 3000
    total_steps: int = 2
    model: str = "gpt-4.1"
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = [{"name": "M-101"}]


def test_should_fallback_on_low_confidence():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(confidence="low", answer="Some answer", sources=[{"x": 1}])
    assert _should_fallback(result) is True


def test_should_not_fallback_on_high_confidence():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(confidence="high", answer="Good answer", sources=[{"x": 1}])
    assert _should_fallback(result) is False


def test_should_fallback_on_empty_answer():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(answer="", sources=[{"x": 1}])
    assert _should_fallback(result) is True


def test_should_fallback_on_none():
    from gateway.orchestrator import _should_fallback
    assert _should_fallback(None) is True


def test_should_fallback_on_no_sources():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(answer="Answer without sources", sources=[])
    assert _should_fallback(result) is True


def test_should_fallback_on_escalation():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(answer="Partial answer", needs_escalation=True)
    assert _should_fallback(result) is True


def test_should_not_fallback_on_medium_confidence():
    from gateway.orchestrator import _should_fallback
    result = MockAgenticResult(confidence="medium", answer="Decent answer", sources=[{"x": 1}])
    assert _should_fallback(result) is False
```

- [ ] **Step 2: Run tests (expect fail)**

- [ ] **Step 3: Implement orchestrator.py**

```python
"""
gateway/orchestrator.py — Agentic-first with Traditional RAG fallback.

Flow:
  1. Run AgenticRAG (MongoDB tools, ReAct loop)
  2. Evaluate confidence, answer quality, sources
  3. If fails → lazy-load FAISS → run Traditional RAG
  4. Return best result with engine attribution
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("unified_rag.orchestrator")


def _should_fallback(result: Any) -> bool:
    """Determine if Traditional RAG should be tried."""
    if result is None:
        return True
    
    answer = getattr(result, "answer", "") or ""
    if len(answer.strip()) < 20:
        return True
    
    confidence = getattr(result, "confidence", "low")
    if confidence == "low":
        return True
    
    sources = getattr(result, "sources", []) or []
    if not sources:
        return True
    
    if getattr(result, "needs_escalation", False):
        return True
    
    return False


class AgenticEngine:
    """Wrapper around AgenticRAG's agent.run_agent()."""
    
    def __init__(self):
        self._initialized = False
    
    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            from agentic.core.db import get_client, ensure_indexes
            get_client()
            ensure_indexes()
            self._initialized = True
            logger.info("AgenticRAG engine initialized (MongoDB connected)")
        except Exception as e:
            logger.error("AgenticRAG init failed: %s", e)
            raise
    
    async def query(self, query: str, project_id: int,
                    set_id: Optional[int] = None,
                    conversation_history: Optional[list] = None) -> Any:
        """Run AgenticRAG agent."""
        from agentic.core.agent import run_agent
        return await asyncio.to_thread(
            run_agent, query, project_id=project_id,
            set_id=set_id, conversation_history=conversation_history,
        )


class TraditionalEngine:
    """Wrapper around Traditional RAG's generation pipeline."""
    
    def __init__(self):
        self._faiss_loaded = False
        self._lock = threading.Lock()
    
    @property
    def is_loaded(self) -> bool:
        return self._faiss_loaded
    
    def ensure_loaded(self, project_id: int) -> None:
        """Lazy-load FAISS index for a project (first call only)."""
        if self._faiss_loaded:
            return
        with self._lock:
            if self._faiss_loaded:
                return
            logger.info("Lazy-loading FAISS indexes (first fallback)...")
            start = time.monotonic()
            try:
                from traditional.rag.retrieval.loaders import _load_project
                from traditional.rag.retrieval.state import PROJECTS
                if project_id in PROJECTS:
                    config = PROJECTS[project_id]
                    if not config.loaded:
                        _load_project(config)
                self._faiss_loaded = True
                elapsed = int((time.monotonic() - start) * 1000)
                logger.info("FAISS loaded in %dms", elapsed)
            except Exception as e:
                logger.error("FAISS loading failed: %s", e)
                raise
    
    async def query(self, query: str, project_id: int,
                    session_id: Optional[str] = None,
                    **kwargs: Any) -> dict:
        """Run Traditional RAG pipeline."""
        self.ensure_loaded(project_id)
        
        from traditional.rag.api.generation_unified import generate_response
        result = await asyncio.to_thread(
            generate_response, query, project_id=project_id,
            session_id=session_id, **kwargs,
        )
        return result


class Orchestrator:
    """Routes queries: agentic-first → traditional fallback."""
    
    def __init__(self, fallback_enabled: bool = True,
                 fallback_timeout: int = 30):
        self.agentic = AgenticEngine()
        self.traditional = TraditionalEngine()
        self.fallback_enabled = fallback_enabled
        self.fallback_timeout = fallback_timeout
    
    async def query(self, query: str, project_id: int,
                    engine: Optional[str] = None,
                    session_id: Optional[str] = None,
                    set_id: Optional[int] = None,
                    conversation_history: Optional[list] = None,
                    **kwargs: Any) -> dict:
        """
        Execute query with engine routing.
        
        engine=None: agentic-first, fallback on failure
        engine="agentic": agentic only, no fallback
        engine="traditional": traditional only, skip agentic
        """
        start = time.monotonic()
        
        # Direct engine selection
        if engine == "traditional":
            return await self._run_traditional(
                query, project_id, session_id, start, **kwargs,
            )
        
        # Try agentic first
        agentic_result = None
        agentic_error = None
        try:
            self.agentic.ensure_initialized()
            agentic_result = await self.agentic.query(
                query, project_id, set_id, conversation_history,
            )
        except Exception as e:
            agentic_error = str(e)
            logger.warning("AgenticRAG failed: %s", e)
        
        # Evaluate: should we fallback?
        if engine == "agentic" or not self.fallback_enabled:
            # No fallback allowed
            elapsed = int((time.monotonic() - start) * 1000)
            return self._build_response(
                agentic_result, "agentic", elapsed,
                fallback_used=False, error=agentic_error,
            )
        
        if not _should_fallback(agentic_result):
            # Agentic succeeded
            elapsed = int((time.monotonic() - start) * 1000)
            return self._build_response(
                agentic_result, "agentic", elapsed, fallback_used=False,
            )
        
        # Fallback to traditional
        logger.info(
            "Falling back to Traditional RAG (agentic confidence=%s)",
            getattr(agentic_result, "confidence", "N/A"),
        )
        agentic_confidence = getattr(agentic_result, "confidence", None)
        
        try:
            trad_result = await asyncio.wait_for(
                self._run_traditional(
                    query, project_id, session_id, start, **kwargs,
                ),
                timeout=self.fallback_timeout,
            )
            trad_result["fallback_used"] = True
            trad_result["agentic_confidence"] = agentic_confidence
            trad_result["engine_used"] = "traditional_fallback"
            return trad_result
        except Exception as e:
            logger.error("Traditional RAG fallback also failed: %s", e)
            # Return whatever agentic had (even if low quality)
            elapsed = int((time.monotonic() - start) * 1000)
            return self._build_response(
                agentic_result, "agentic", elapsed,
                fallback_used=True,
                error=f"Fallback also failed: {e}",
            )
    
    async def _run_traditional(self, query, project_id, session_id,
                                start, **kwargs) -> dict:
        """Run traditional RAG and format response."""
        result = await self.traditional.query(
            query, project_id, session_id, **kwargs,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        
        if isinstance(result, dict):
            result["engine_used"] = "traditional"
            result["elapsed_ms"] = elapsed
            result["fallback_used"] = False
            return result
        
        return {
            "success": True,
            "answer": str(result),
            "engine_used": "traditional",
            "elapsed_ms": elapsed,
            "fallback_used": False,
        }
    
    @staticmethod
    def _build_response(result: Any, engine: str, elapsed_ms: int,
                        fallback_used: bool = False,
                        error: Optional[str] = None) -> dict:
        """Convert engine result to unified response dict."""
        if result is None:
            return {
                "success": False,
                "answer": error or "No answer available",
                "sources": [],
                "confidence": "low",
                "engine_used": engine,
                "fallback_used": fallback_used,
                "elapsed_ms": elapsed_ms,
                "error": error,
            }
        
        return {
            "success": True,
            "answer": getattr(result, "answer", str(result)),
            "sources": getattr(result, "sources", []) or [],
            "confidence": getattr(result, "confidence", "medium"),
            "engine_used": engine,
            "fallback_used": fallback_used,
            "cost_usd": getattr(result, "cost_usd", 0.0),
            "elapsed_ms": elapsed_ms,
            "total_steps": getattr(result, "total_steps", 0),
            "model": getattr(result, "model", ""),
        }
```

- [ ] **Step 4: Run tests (expect pass)**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: 7/7 PASS

- [ ] **Step 5: Commit**

```bash
git add gateway/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(unified-rag): orchestrator — agentic-first with fallback logic"
```

---

## Task 7: Gateway App + Router

**Files:**
- Create: `gateway/app.py`
- Create: `gateway/router.py`
- Create: `tests/test_gateway.py`

> This is the largest task. The router implements all 18 endpoints.
> Due to plan size constraints, the full router code is provided — key endpoints shown, pattern repeats for the rest.

- [ ] **Step 1: Write gateway test**

```python
"""tests/test_gateway.py — Gateway route registration tests."""

import pytest


def test_router_has_all_endpoints():
    from gateway.router import router
    paths = [r.path for r in router.routes]
    
    assert "/query" in paths
    assert "/query/stream" in paths
    assert "/quick-query" in paths
    assert "/web-search" in paths
    assert "/health" in paths
    assert "/config" in paths
    assert "/sessions/create" in paths
    assert "/sessions" in paths
    assert "/sessions/{session_id}/stats" in paths
    assert "/sessions/{session_id}/conversation" in paths
    assert "/sessions/{session_id}/update" in paths
    assert "/sessions/{session_id}" in paths
    assert "/metrics" in paths


def test_router_has_correct_methods():
    from gateway.router import router
    method_map = {}
    for route in router.routes:
        if hasattr(route, "methods"):
            method_map[route.path] = route.methods
    
    assert "POST" in method_map.get("/query", set())
    assert "POST" in method_map.get("/query/stream", set())
    assert "GET" in method_map.get("/health", set())
    assert "GET" in method_map.get("/sessions", set())
    assert "DELETE" in method_map.get("/sessions/{session_id}", set())
```

- [ ] **Step 2: Implement gateway/router.py**

Create the router with all 18 endpoints. The `/query` endpoint calls `orchestrator.query()`. Session endpoints delegate to the traditional MemoryManager (imported via `traditional.memory_manager`). Health checks both engines.

The router follows the same pattern as the old `rag/api/routes.py` but delegates to the orchestrator instead of calling generation directly.

- [ ] **Step 3: Implement gateway/app.py**

```python
"""
gateway/app.py — Unified RAG Agent FastAPI application.

Single entry point on port 8001. Replaces both old RAG and AgenticRAG.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure unified-rag-agent root is in path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv(str(_root / ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import get_config
from gateway.orchestrator import Orchestrator
from gateway.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("unified_rag")
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init agentic engine + verify MongoDB. Shutdown: cleanup."""
    logger.info("=" * 60)
    logger.info(" Unified RAG Agent — Starting up")
    logger.info("=" * 60)
    
    if not config.api_key:
        logger.warning(
            "API_KEY is empty — running WITHOUT authentication (dev mode only)"
        )
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        fallback_enabled=config.fallback_enabled,
        fallback_timeout=config.fallback_timeout_seconds,
    )
    
    # Init agentic engine (MongoDB connect + indexes)
    try:
        orchestrator.agentic.ensure_initialized()
        logger.info("AgenticRAG engine ready (model=%s)", config.agentic_model)
    except Exception as e:
        logger.error("AgenticRAG init failed: %s (will retry on first query)", e)
    
    # Traditional engine stays in standby (lazy FAISS)
    logger.info(
        "Traditional RAG engine: standby (FAISS lazy-load=%s)",
        config.faiss_lazy_load,
    )
    
    app.state.orchestrator = orchestrator
    app.state.config = config
    
    logger.info(
        "Unified RAG ready on %s:%d (fallback=%s)",
        config.host, config.port, config.fallback_enabled,
    )
    
    yield
    
    logger.info("Shutting down...")
    try:
        from agentic.core.db import close_db
        close_db()
    except Exception:
        pass
    logger.info("Shutdown complete")


app = FastAPI(
    title="Unified RAG Agent — Construction Document Q&A",
    description="AgenticRAG (primary) + Traditional RAG (fallback)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.ifieldsmart.com",
        "https://ai.ifieldsmart.com",
        "https://ai3.ifieldsmart.com",
        "https://ai5.ifieldsmart.com",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed")

# Mount router
app.include_router(router)


# Dev entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway.app:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level=config.log_level.lower(),
    )
```

- [ ] **Step 4: Run tests (expect pass)**

Run: `python -m pytest tests/test_gateway.py -v`
Expected: 2/2 PASS

- [ ] **Step 5: Verify app loads**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent"
python -c "from gateway.app import app; print(f'Routes: {len(app.routes)}'); print('App loads OK')"
```

- [ ] **Step 6: Commit**

```bash
git add gateway/app.py gateway/router.py tests/test_gateway.py
git commit -m "feat(unified-rag): gateway app + router — 18 endpoints, backward compatible"
```

---

## Task 8: Import Path Fixes

**Files:**
- Modify: various `__init__.py` and import statements in both engines

> Both engines have internal imports that assume they're at the repo root.
> The compat.py shims (Task 3) handle most cases via sys.path.
> This task fixes any remaining broken imports found during testing.

- [ ] **Step 1: Test agentic engine imports**

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent"
python -c "
import agentic
from agentic.core.agent import run_agent
from agentic.core.cache import get_tool_result, set_tool_result
from agentic.core.db import get_client, ensure_indexes
from agentic.core.audit import log_query
from agentic.core.text_reconstruction import reconstruct_drawing_text
from agentic.tools.registry import TOOL_DEFINITIONS, TOOL_FUNCTIONS
print('All agentic imports OK')
"
```

- [ ] **Step 2: Fix any broken agentic imports**

Common fixes needed:
- `agentic/config.py` may import from `config` (now needs `agentic.config` or relative import)
- `agentic/core/agent.py` may import `from config import ...` → fix to `from agentic.config import ...`
- `agentic/tools/*.py` may import `from config import ...` → fix similarly

For each broken import, update the import statement. The compat.py shim adds `agentic/` to sys.path, so `from config import ...` should work within the agentic package. If not, switch to explicit `from agentic.config import ...`.

- [ ] **Step 3: Test traditional engine imports**

```bash
python -c "
import traditional
from traditional.rag.retrieval.engine import retrieve_context
from traditional.rag.retrieval.state import PROJECTS
from traditional.rag.api.generation_unified import generate_response
from traditional.rag.api.intent import detect_intent
print('All traditional imports OK')
"
```

- [ ] **Step 4: Fix any broken traditional imports**

Common fixes:
- `traditional/rag/api/routes.py` imports `from memory_manager import ...` → `from traditional.memory_manager import ...`
- `traditional/rag/retrieval/loaders.py` imports `from s3_utils import ...` → redirected by compat.py to `shared.s3_utils`
- `traditional/memory_manager.py` imports `from s3_utils import ...` → redirected by compat.py

- [ ] **Step 5: Run existing agentic tests**

```bash
python -m pytest tests/agentic/ -v --tb=short
```
Expected: 57/57 PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fix(unified-rag): import path fixes for both engines in subdirectories"
```

---

## Task 9: Integration Test + CLAUDE.md

**Files:**
- Create: `tests/test_integration.py`
- Create: `CLAUDE.md`

- [ ] **Step 1: Write integration test**

```python
"""tests/test_integration.py — Full pipeline integration test."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_orchestrator_returns_agentic_result_on_success():
    """When AgenticRAG succeeds, return its result without fallback."""
    from gateway.orchestrator import Orchestrator
    
    orch = Orchestrator(fallback_enabled=True)
    
    mock_result = MagicMock()
    mock_result.answer = "The XVENT models specified are THEB-446 and 6SEB."
    mock_result.sources = [{"name": "M-401"}]
    mock_result.confidence = "high"
    mock_result.needs_escalation = False
    mock_result.cost_usd = 0.005
    mock_result.total_steps = 2
    mock_result.model = "gpt-4.1"
    
    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(return_value=mock_result)
    
    result = await orch.query("What XVENT models?", project_id=2361)
    
    assert result["success"] is True
    assert result["engine_used"] == "agentic"
    assert result["fallback_used"] is False
    assert "XVENT" in result["answer"]


@pytest.mark.asyncio
async def test_orchestrator_falls_back_on_low_confidence():
    """When AgenticRAG returns low confidence, fallback to traditional."""
    from gateway.orchestrator import Orchestrator
    
    orch = Orchestrator(fallback_enabled=True)
    
    # Agentic returns low confidence
    mock_agentic = MagicMock()
    mock_agentic.answer = "I'm not sure about this."
    mock_agentic.sources = []
    mock_agentic.confidence = "low"
    mock_agentic.needs_escalation = False
    
    orch.agentic.ensure_initialized = MagicMock()
    orch.agentic.query = AsyncMock(return_value=mock_agentic)
    
    # Traditional returns good result
    trad_result = {
        "success": True,
        "answer": "Based on the drawings, the panel is rated at 200A.",
        "sources": [{"text": "200A panel"}],
        "confidence": "high",
    }
    orch.traditional.query = AsyncMock(return_value=trad_result)
    orch.traditional._faiss_loaded = True  # skip lazy load in test
    
    result = await orch.query("What is the panel rating?", project_id=7325)
    
    assert result["fallback_used"] is True
    assert result["engine_used"] == "traditional_fallback"
    assert result["agentic_confidence"] == "low"
    assert "200A" in result["answer"]


@pytest.mark.asyncio
async def test_orchestrator_engine_override_traditional():
    """engine='traditional' skips agentic entirely."""
    from gateway.orchestrator import Orchestrator
    
    orch = Orchestrator(fallback_enabled=True)
    
    trad_result = {
        "success": True,
        "answer": "Direct traditional answer",
        "sources": [],
    }
    orch.traditional.query = AsyncMock(return_value=trad_result)
    orch.traditional._faiss_loaded = True
    
    result = await orch.query(
        "test query", project_id=7212, engine="traditional",
    )
    
    assert result["engine_used"] == "traditional"
    assert result["fallback_used"] is False
```

- [ ] **Step 2: Run integration tests**

```bash
python -m pytest tests/test_integration.py -v
```
Expected: 3/3 PASS

- [ ] **Step 3: Write CLAUDE.md**

Create a comprehensive CLAUDE.md documenting the unified agent: architecture, endpoints, config, engines, fallback logic, folder structure.

- [ ] **Step 4: Run ALL tests**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass (config + models + orchestrator + session + gateway + agentic + integration)

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py CLAUDE.md
git commit -m "feat(unified-rag): integration tests + CLAUDE.md documentation"
```

---

## Task 10: Deploy to Sandbox VM

**Files:** None created locally — deployment operations only.

**Sandbox:** `54.197.189.113` via SSH with PEM key at `c:\Users\ANIRUDDHA ASUS\Downloads\projects\VCS\ai_assistant_sandbox.pem`

- [ ] **Step 1: Upload unified-rag-agent to sandbox**

```bash
PEM="c:\\Users\\ANIRUDDHA ASUS\\Downloads\\projects\\VCS\\ai_assistant_sandbox.pem"
scp -i "$PEM" -o StrictHostKeyChecking=no -r \
  "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent" \
  ubuntu@54.197.189.113:/home/ubuntu/unified-rag-agent
```

- [ ] **Step 2: Install dependencies on sandbox**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  cd /home/ubuntu/unified-rag-agent &&
  /home/ubuntu/myenv/bin/pip install -r requirements.txt 2>&1 | tail -5
"
```

- [ ] **Step 3: Copy FAISS indexes to unified-rag-agent/index/**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  mkdir -p /home/ubuntu/unified-rag-agent/index &&
  cp /home/ubuntu/chatbot/aniruddha/vcsai/RAG/index/* /home/ubuntu/unified-rag-agent/index/ 2>/dev/null;
  ls -la /home/ubuntu/unified-rag-agent/index/ | head -10
"
```

- [ ] **Step 4: Merge .env with secrets from both existing systems**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  # Get secrets from old RAG
  grep OPENAI_API_KEY /home/ubuntu/chatbot/aniruddha/vcsai/RAG/.env >> /home/ubuntu/unified-rag-agent/.env
  grep MONGODB_URI /home/ubuntu/AgenticRAG/.env >> /home/ubuntu/unified-rag-agent/.env
  grep AWS_ /home/ubuntu/chatbot/aniruddha/vcsai/RAG/.env >> /home/ubuntu/unified-rag-agent/.env
  # Set port
  echo 'PORT=8001' >> /home/ubuntu/unified-rag-agent/.env
"
```

- [ ] **Step 5: Test import + app load on sandbox**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  cd /home/ubuntu/unified-rag-agent &&
  /home/ubuntu/myenv/bin/python -c 'from gateway.app import app; print(\"App loads OK\")'
"
```

- [ ] **Step 6: Update systemd service to point to unified agent**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  sudo tee /etc/systemd/system/rag-agent.service << 'EOF'
[Unit]
Description=VCS Unified RAG Agent (Port 8001)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/unified-rag-agent
EnvironmentFile=/home/ubuntu/unified-rag-agent/.env
ExecStart=/home/ubuntu/myenv/bin/uvicorn gateway.app:app --host 0.0.0.0 --port 8001 --workers 1
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rag-agent

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl restart rag-agent
  sleep 3
  systemctl is-active rag-agent
"
```

- [ ] **Step 7: Verify health endpoint**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  curl -s http://localhost:8001/health | python3 -m json.tool
"
```

- [ ] **Step 8: Commit deployment notes**

```bash
git commit --allow-empty -m "deploy(unified-rag): deployed to sandbox VM 54.197.189.113:8001"
```

---

## Task 11: Test on Sandbox + Push to GitHub

- [ ] **Step 1: Run a real query via agentic engine**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  curl -s -X POST http://localhost:8001/query \
    -H 'Content-Type: application/json' \
    -d '{\"query\": \"What XVENT models are in the mechanical drawings?\", \"project_id\": 2361}' \
    --max-time 120 | python3 -m json.tool | head -20
"
```

Expected: `engine_used: "agentic"`, confidence: "high", sources with drawing names

- [ ] **Step 2: Force traditional engine**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  curl -s -X POST http://localhost:8001/query \
    -H 'Content-Type: application/json' \
    -d '{\"query\": \"What plumbing fixtures are shown?\", \"project_id\": 7212, \"engine\": \"traditional\"}' \
    --max-time 120 | python3 -m json.tool | head -20
"
```

Expected: `engine_used: "traditional"`, FAISS-based answer

- [ ] **Step 3: Test fallback scenario**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  # Query a project that only exists in FAISS (not in MongoDB)
  curl -s -X POST http://localhost:8001/query \
    -H 'Content-Type: application/json' \
    -d '{\"query\": \"Show me the electrical plans\", \"project_id\": 7166}' \
    --max-time 120 | python3 -m json.tool | head -20
"
```

Expected: If agentic has no data for 7166 → `fallback_used: true`, `engine_used: "traditional_fallback"`

- [ ] **Step 4: Test via Nginx gateway**

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  curl -s http://localhost:8000/rag/health | python3 -m json.tool
"
```

- [ ] **Step 5: Push all changes to GitHub**

Push the following to GitHub:
1. `unified-rag-agent/` — the new merged agent
2. `construction-intelligence-agent/` — scope gap pipeline changes
3. Any VisionOCR changes if applicable

```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP"
git add unified-rag-agent/
git add construction-intelligence-agent/
git commit -m "feat: unified-rag-agent + scope-gap-pipeline — production deploy"
git push origin main
```

- [ ] **Step 6: Final verification**

Run complete test suite on sandbox:

```bash
ssh -i "$PEM" ubuntu@54.197.189.113 "
  cd /home/ubuntu/unified-rag-agent &&
  /home/ubuntu/myenv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
"
```

Expected: All tests pass. Service running on 8001. Both engines functional.
