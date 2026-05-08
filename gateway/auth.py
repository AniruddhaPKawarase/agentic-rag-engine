"""Gateway authentication — API-key verification with timing-safe comparison."""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from shared.config import get_config

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

PUBLIC_ENDPOINTS: set[str] = {"/", "/health", "/metrics"}


async def verify_api_key(
    request: Request,
    key: str | None = Depends(api_key_header),
) -> str | None:
    """Validate the caller-supplied API key against the configured secret.

    Public endpoints (health, root, metrics) skip authentication so that
    load-balancer health checks work without API keys.

    If no ``API_KEY`` is configured the check is skipped (dev mode).
    Uses ``hmac.compare_digest`` to prevent timing-based side-channels.
    """
    if request.url.path in PUBLIC_ENDPOINTS:
        return None

    cfg = get_config()

    if not cfg.api_key:
        logger.warning("API_KEY is not set — authentication disabled (dev mode)")
        return None

    if key is None or not hmac.compare_digest(key, cfg.api_key):
        raise HTTPException(status_code=403, detail="Forbidden")

    return key


async def auth_required(
    _key: str | None = Depends(verify_api_key),
) -> None:
    """Dependency suitable for ``dependencies=[Depends(auth_required)]``."""
