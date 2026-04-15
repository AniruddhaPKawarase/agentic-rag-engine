"""
Gateway App — FastAPI application with lifespan, CORS, and Prometheus.

Entrypoint: ``uvicorn gateway.app:app --host 0.0.0.0 --port 8001``
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway.orchestrator import Orchestrator
from gateway.router import router
from shared.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiting (optional — degrades gracefully if slowapi not installed)
# ---------------------------------------------------------------------------

_limiter = None
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore[import-untyped]
    from slowapi.errors import RateLimitExceeded  # type: ignore[import-untyped]
    from slowapi.util import get_remote_address  # type: ignore[import-untyped]

    _limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
    logger.info("slowapi rate limiting loaded (20/min default)")
except ImportError:
    logger.info("slowapi not installed — rate limiting disabled")


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

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ifieldsmart.com",
        "https://*.ifieldsmart.com",
        "https://ai5.ifieldsmart.com",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limiting ---
if _limiter is not None:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled")

# --- Prometheus (optional) ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import-untyped]
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed, metrics disabled")

# --- Mount Router ---
app.include_router(router)


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
