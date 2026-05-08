"""Unified LLM client for the v3.1 generation chain.

Routes calls to Anthropic Claude or OpenAI based on the model name prefix,
with automatic fallback on transient provider failures (5xx / 429 / connection
errors). Supports both blocking and streaming modes.

Usage:
    from agentic.generation.llm_client import generate

    text = generate(
        system_prompt="...",
        user_prompt="...",
        model="claude-haiku-4-5",
        fallback_model="gpt-4o-mini",
    )

    for chunk in generate(..., stream=True):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator, Optional, Union

logger = logging.getLogger("agentic_rag.generation.llm_client")

# ── Module-level cached clients ────────────────────────────────────────
_anthropic_client = None
_openai_client = None

# De-duplicate noisy warnings during eval runs. Each (provider, model, kind)
# tuple is warned about once per process; subsequent identical failures are
# logged at DEBUG level only.
_LOGGED_PRIMARY_FAILURES: set = set()
_LOGGED_FALLBACKS: set = set()

# ── Tunables ───────────────────────────────────────────────────────────
_MAX_ATTEMPTS = 2          # 2 attempts per provider (initial + 1 retry)
_BACKOFF_SECONDS = 0.75    # base backoff between retries

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ── Provider routing ───────────────────────────────────────────────────

def _provider_for_model(model: str) -> str:
    """Return 'anthropic' or 'openai' based on model name prefix."""
    name = (model or "").lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    # Default to OpenAI for unknown names (safest given existing infra).
    logger.warning("Unknown model prefix for %r; defaulting to openai", model)
    return "openai"


def _get_anthropic_client():
    """Lazily construct + cache a single Anthropic client.

    We pass an explicit httpx.Client so the anthropic SDK can't try to
    construct httpx with kwargs (like ``proxies=``) that newer httpx
    versions reject. This avoids the well-known
        TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
    that surfaces under thread-pool execution when the SDK auto-detects
    proxy settings differently per thread context.
    """
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # local import so test envs without SDK still load

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot use Claude models. "
                "Set the env var or switch to a gpt-* model."
            )
        try:
            import httpx  # noqa: WPS433 — local import to avoid hard dep at import time
            _anthropic_client = anthropic.Anthropic(
                api_key=api_key,
                http_client=httpx.Client(timeout=60.0, follow_redirects=True),
            )
        except TypeError:
            # SDK doesn't accept http_client kwarg on this version — fall back.
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_openai_client():
    """Lazily construct + cache a single OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # local import

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; cannot use GPT models."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ── Error classification ───────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """True if the exception looks transient (5xx / 429 / connection)."""
    # status_code attribute (anthropic + openai both expose this on APIError)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
        return True

    # Class-name heuristic — avoids hard import dependency on either SDK.
    name = type(exc).__name__.lower()
    transient_markers = (
        "ratelimit",
        "apiconnection",
        "apitimeout",
        "internalserver",
        "serviceunavailable",
        "overloaded",
    )
    return any(m in name for m in transient_markers)


# ── Anthropic call ─────────────────────────────────────────────────────

def _call_anthropic(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Union[str, Iterator[str]]:
    """Invoke Anthropic's Messages API. Streams via .stream() when requested."""
    client = _get_anthropic_client()

    messages = [{"role": "user", "content": user_prompt}]

    if stream:
        return _anthropic_stream(
            client=client,
            system_prompt=system_prompt,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Token accounting
    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info(
            "anthropic.tokens model=%s input=%s output=%s",
            model,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )

    content_blocks = getattr(response, "content", []) or []
    text_parts: list[str] = []
    for block in content_blocks:
        # SDK returns objects with .text on TextBlock; fall back to dict access.
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            text_parts.append(text)
    return "".join(text_parts)


def _anthropic_stream(
    *,
    client,
    system_prompt: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
    """Yield text chunks from an Anthropic streaming response."""
    with client.messages.stream(
        model=model,
        system=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    ) as stream:
        for text in stream.text_stream:
            if text:
                yield text
        # Usage available after stream completes.
        try:
            final = stream.get_final_message()
            usage = getattr(final, "usage", None)
            if usage is not None:
                logger.info(
                    "anthropic.tokens model=%s input=%s output=%s (stream)",
                    model,
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                )
        except Exception:  # noqa: BLE001 — usage is best-effort
            logger.debug("anthropic stream usage unavailable", exc_info=True)


# ── OpenAI call ────────────────────────────────────────────────────────

def _call_openai(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Union[str, Iterator[str]]:
    """Invoke OpenAI's Chat Completions API."""
    client = _get_openai_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if stream:
        return _openai_stream(
            client=client,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info(
            "openai.tokens model=%s input=%s output=%s",
            model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    msg = choices[0].message
    return getattr(msg, "content", "") or ""


def _openai_stream(
    *,
    client,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
    """Yield text chunks from an OpenAI streaming response."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for event in response:
        choices = getattr(event, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        text = getattr(delta, "content", None)
        if text:
            yield text


# ── Public API ─────────────────────────────────────────────────────────

def generate(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    fallback_model: Optional[str] = None,
    max_tokens: int = 400,
    temperature: float = 0.3,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    """Generate a response from the configured LLM provider.

    Args:
        system_prompt: System / role instructions.
        user_prompt: User-side prompt content.
        model: Primary model name. ``claude-*`` routes to Anthropic, ``gpt-*``
            (and ``o1-*`` / ``o3-*``) routes to OpenAI.
        fallback_model: Used on transient provider failure. Should be of the
            other provider for true redundancy.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        stream: When ``True``, yields chunks (str). When ``False``, returns the
            full string.

    Returns:
        Full response string when ``stream=False``; an iterator of string
        chunks when ``stream=True``.

    Raises:
        Re-raises the underlying provider exception when all retries and the
        fallback are exhausted.
    """
    primary_provider = _provider_for_model(model)
    last_exc: Optional[BaseException] = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return _dispatch(
                provider=primary_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
        except Exception as exc:  # noqa: BLE001 — classified below
            last_exc = exc
            retryable = _is_retryable(exc)
            # Throttle the "key not set" warning — log once per process per
            # (provider, model) pair to keep eval logs readable.
            err_str = str(exc)
            is_key_missing = "API_KEY is not set" in err_str
            warn_key = (primary_provider, model, "key_missing" if is_key_missing else "other")
            if warn_key not in _LOGGED_PRIMARY_FAILURES:
                logger.warning(
                    "primary llm call failed provider=%s model=%s attempt=%d retryable=%s err=%s",
                    primary_provider, model, attempt, retryable, exc,
                )
                _LOGGED_PRIMARY_FAILURES.add(warn_key)
            else:
                logger.debug(
                    "primary llm call failed (suppressed repeat) provider=%s model=%s",
                    primary_provider, model,
                )
            if not retryable:
                break  # 4xx (other than 429) — don't retry, try fallback.
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_SECONDS * attempt)

    # Primary exhausted. Try the fallback if configured.
    if fallback_model:
        fb_provider = _provider_for_model(fallback_model)
        fb_warn_key = (primary_provider, model, fb_provider, fallback_model)
        if fb_warn_key not in _LOGGED_FALLBACKS:
            logger.warning(
                "falling back to %s/%s after primary failure", fb_provider, fallback_model,
            )
            _LOGGED_FALLBACKS.add(fb_warn_key)
        else:
            logger.debug(
                "falling back to %s/%s (suppressed repeat)", fb_provider, fallback_model,
            )
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return _dispatch(
                    provider=fb_provider,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=fallback_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = _is_retryable(exc)
                logger.error(
                    "fallback llm call failed provider=%s model=%s attempt=%d err=%s",
                    fb_provider, fallback_model, attempt, exc,
                )
                if not retryable:
                    break
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_BACKOFF_SECONDS * attempt)

    assert last_exc is not None  # defensive — we only reach here on failure.
    raise last_exc


def _dispatch(
    *,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Union[str, Iterator[str]]:
    """Route to the chosen provider's call function."""
    if provider == "anthropic":
        return _call_anthropic(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
    return _call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )


# ── Test helpers ───────────────────────────────────────────────────────

def _reset_clients_for_tests() -> None:
    """Reset cached clients. Test-only — never call from production code."""
    global _anthropic_client, _openai_client
    _anthropic_client = None
    _openai_client = None
