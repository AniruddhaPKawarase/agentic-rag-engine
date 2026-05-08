"""
agentic.memory.embeddings
=========================

Thin embedding helper for the v3.1 conversation-memory subsystem.

We deliberately reuse the same model the rest of the project uses
(``text-embedding-3-small``, 1536 dims) so vectors written by the Memory
Writer are comparable to the document vectors stored elsewhere if we ever
want to mix them.

Design choices
--------------
* OpenAI sync client cached at module level (no per-call construction).
* Two-attempt retry with exponential backoff on transient errors only
  (rate limits, network, server errors). Auth / bad-request failures are
  raised immediately so we don't burn retries on permanent problems.
* Hard input cap at 2000 chars before embedding — a turn that long is
  almost certainly a paste-bomb and is not worth $$ to embed in full.

Public API
----------
``embed_text(text: str) -> list[float]``
    Return a 1536-dim embedding for ``text``. Raises ``RuntimeError`` if
    both attempts fail.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

logger = logging.getLogger("agentic_rag.memory.embeddings")

# Module-level cached client + lock for first-time construction.
_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

# Tunables.
EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = 1536
MAX_INPUT_CHARS = 2000
MAX_ATTEMPTS = 2
BACKOFF_BASE_SECONDS = 0.5


def _get_client() -> OpenAI:
    """Return the cached OpenAI client, creating it on first call.

    Uses double-checked locking so concurrent first calls don't race.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                # Fall through to default constructor — OpenAI SDK will
                # raise a clearer error than we would. This keeps tests
                # that monkeypatch the SDK simple.
                _client = OpenAI()
            else:
                _client = OpenAI(api_key=api_key)
    return _client


def _truncate(text: str) -> str:
    """Cap embedding input at ``MAX_INPUT_CHARS`` to bound cost."""
    if not isinstance(text, str):
        raise TypeError(f"embed_text expects str, got {type(text).__name__}")
    if len(text) <= MAX_INPUT_CHARS:
        return text
    logger.debug(
        "Truncating embedding input from %d to %d chars", len(text), MAX_INPUT_CHARS
    )
    return text[:MAX_INPUT_CHARS]


def embed_text(text: str) -> List[float]:
    """Embed ``text`` with the configured OpenAI embedding model.

    Behaviour
    ---------
    * Empty / whitespace-only input returns a zero vector (no API call).
    * Input is truncated to ``MAX_INPUT_CHARS`` before sending.
    * Transient errors (rate limit, network, 5xx) are retried up to
      ``MAX_ATTEMPTS`` times with exponential backoff.
    * Permanent errors (4xx other than 429) are not retried.

    Returns
    -------
    list[float]
        1536-dimensional embedding.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    TypeError
        If ``text`` is not a string.
    """
    payload = _truncate(text or "")
    if not payload.strip():
        # Avoid a pointless paid round-trip for empty input.
        return [0.0] * EMBEDDING_DIMS

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            client = _get_client()
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=payload)
            vector = resp.data[0].embedding
            if len(vector) != EMBEDDING_DIMS:
                raise RuntimeError(
                    f"Unexpected embedding dim {len(vector)} (want {EMBEDDING_DIMS})"
                )
            return list(vector)
        except (RateLimitError, APIConnectionError) as exc:
            last_err = exc
            if attempt >= MAX_ATTEMPTS:
                break
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "embed_text transient error (%s) on attempt %d/%d — sleeping %.2fs",
                type(exc).__name__, attempt, MAX_ATTEMPTS, sleep_s,
            )
            time.sleep(sleep_s)
        except APIError as exc:
            # Server-side 5xx is worth one retry; client-side 4xx is not.
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                logger.error("embed_text permanent error (%s): %s", status, exc)
                raise
            last_err = exc
            if attempt >= MAX_ATTEMPTS:
                break
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "embed_text APIError on attempt %d/%d — sleeping %.2fs",
                attempt, MAX_ATTEMPTS, sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"embed_text failed after {MAX_ATTEMPTS} attempts: {last_err}")
