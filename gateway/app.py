"""
Gateway App — FastAPI application with lifespan, CORS, and Prometheus.

Entrypoint: ``uvicorn gateway.app:app --host 0.0.0.0 --port 8001``
"""

from __future__ import annotations
from vcsai_tokens import autoinstrument, TokenMiddleware  # noqa: E402
autoinstrument(agent_name="drawing_agent_v31")


import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.orchestrator import Orchestrator
from gateway.router import router
from shared.config import get_config

# v3.3 Deep Dive — additive router (gated by DEEP_DIVE_ENABLED env flag).
# Import wrapped in try/except so a missing optional dependency
# (e.g. google-generativeai not yet installed in the venv) does NOT
# affect the main /query path.
# Phase 1 — feedback router (additive, env-gated by FEEDBACK_ENABLED).
try:
    from gateway.feedback_router import router as feedback_router  # type: ignore
except Exception as _feedback_import_exc:  # noqa: BLE001
    feedback_router = None  # type: ignore
    logging.getLogger(__name__).warning(
        "Feedback router unavailable: %s", _feedback_import_exc,
    )

try:
    from gateway.deep_dive_router import router as deep_dive_router  # type: ignore
except Exception as _deep_dive_import_exc:  # noqa: BLE001
    deep_dive_router = None  # type: ignore
    logging.getLogger(__name__).warning(
        "Deep Dive router unavailable: %s", _deep_dive_import_exc,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — init orchestrator, attach to app.state
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: create orchestrator, verify agentic engine.
    Shutdown: log teardown.
    """
    cfg = get_config()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting Unified RAG Agent on %s:%d", cfg.host, cfg.port)

    orchestrator = Orchestrator(
        fallback_enabled=cfg.fallback_enabled,
        fallback_timeout=cfg.fallback_timeout_seconds,
    )

    # Try to initialize agentic engine (MongoDB indexes)
    try:
        orchestrator.agentic.ensure_initialized()
        logger.info("Agentic engine initialized successfully")
    except Exception as exc:
        logger.warning(
            "Agentic engine initialization failed (will retry on first query): %s",
            exc,
        )

    app.state.orchestrator = orchestrator
    logger.info(
        "Orchestrator ready — fallback=%s, timeout=%ds",
        cfg.fallback_enabled,
        cfg.fallback_timeout_seconds,
    )

    yield

    logger.info("Shutting down Unified RAG Agent")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Unified RAG Agent",
    description="Construction Document Q&A — Agentic-first with Traditional RAG fallback",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(TokenMiddleware)


# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ifieldsmart.com",
        "https://ai5.ifieldsmart.com",
        "https://ifieldsmart.ai",
        "https://sandbox.ifieldsmart.ai",
        "http://localhost:3000",
        "http://localhost:4200",
        "http://localhost:8080",
    ],
    allow_origin_regex=r"https://.*\.ifieldsmart\.(com|ai)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus (optional) ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import-untyped]
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed, metrics disabled")

# --- Mount Router ---
app.include_router(router)

# v3.3 — Deep Dive router mount. Skipped when import failed (logged
# above). The endpoints themselves are gated by DEEP_DIVE_ENABLED flag,
# so even when mounted the feature stays dormant until ops flips it on.
if deep_dive_router is not None:
    app.include_router(deep_dive_router)
    logger.info("Deep Dive router mounted at /deep-dive (feature flag gated)")

# Phase 1 — feedback router mount (env-gated)
import os as _os_fb
if (
    feedback_router is not None
    and _os_fb.getenv("FEEDBACK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
):
    app.include_router(feedback_router)
    logger.info("Feedback router mounted (FEEDBACK_ENABLED=true)")
elif feedback_router is None:
    logger.info("Feedback router not loaded (import failed)")
else:
    logger.info("Feedback router available but FEEDBACK_ENABLED is off")



# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "gateway.app:app",
        host=cfg.host,
        port=cfg.port,
        reload=True,
    )
