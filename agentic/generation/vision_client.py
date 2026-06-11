"""
agentic.generation.vision_client
================================

Multi-provider vision client for the Deep Dive feature (v3.3).

Supports three providers behind a single ``generate_with_images`` API:

* **OpenAI**           — ``gpt-4.1``, ``gpt-4o``
* **Anthropic**        — ``claude-sonnet-4-20250514`` (vision)
* **Google Gemini**    — ``gemini-2.0-flash``, ``gemini-1.5-pro``

Why three providers
-------------------
Per product direction (2026-05-12 clarifications):
    "Whichever gives best and accurate response, no limit of cost — for
    fallback use one version lower models."

We want to A/B test vision OCR quality across providers on real
construction drawings, so the same payload (text + presigned PNG URLs)
must be routable to any provider without touching downstream code.

Fallback ladder (default)
-------------------------
``gpt-4.1`` → ``claude-sonnet-4`` → ``gemini-1.5-pro`` → ``gpt-4o``

Each rung is tried in order on rate-limit (429) or 5xx errors. The
primary + fallback list is fully configurable per call.

Design constraints
------------------
* **No fresh DB hits** — image URLs come in as presigned strings; we
  never re-fetch metadata from Mongo.
* **Stream-first** — every provider exposes a streaming interface.
* **Schema-stable** — return value is either ``Iterator[str]`` (streaming)
  or ``str`` (non-streaming). Downstream consumers don't care which
  provider answered.
* **Fail-safe** — single-provider failure must fall through to the next
  rung; full failure returns an explicit error string, never raises.

The client is intentionally stateless — caller passes the entire payload
on every call. No connection pooling beyond what the underlying SDKs
provide. Safe to import in async handlers.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

# Per-image HTTP fetch timeout (Anthropic/Gemini fetch each image before
# sending to the model). Env-configurable so slow-S3 regions can extend it.
IMAGE_FETCH_TIMEOUT_S = float(os.getenv("VISION_IMAGE_FETCH_TIMEOUT_S", "15.0"))

logger = logging.getLogger("agentic_rag.vision_client")

# Anthropic hard limit: no image dimension may exceed 8000px. We cap a little
# under that for safety margin.
_ANTHROPIC_MAX_DIM = 7800


def _downscale_b64_for_anthropic(media_type: str, b64_data: str) -> tuple[str, str]:
    """Return (media_type, b64) downscaled so max(width,height) <= 7800px.

    Anthropic returns HTTP 400 for images with any dimension > 8000px. We
    decode, and only if oversized, resize preserving aspect ratio and re-encode
    as PNG. Best-effort: on ANY failure (or if Pillow is unavailable) the
    original is returned unchanged so the call still proceeds.
    """
    try:
        import io
        from PIL import Image  # Pillow

        raw = base64.b64decode(b64_data)
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            if max(w, h) <= _ANTHROPIC_MAX_DIM:
                return media_type, b64_data
            scale = _ANTHROPIC_MAX_DIM / float(max(w, h))
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            im = im.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            new_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            logger.info(
                "vision_client: downscaled oversized image %dx%d -> %dx%d for Anthropic",
                w, h, new_size[0], new_size[1],
            )
            return "image/png", new_b64
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision_client: image downscale skipped (%s)", exc)
    return media_type, b64_data


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


class VisionProvider(str, Enum):
    """The set of vision-capable providers we wire."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass(frozen=True)
class VisionModel:
    """Static metadata describing one vision-capable model."""

    name: str                  # human-facing identifier passed by the caller
    provider: VisionProvider
    api_model_id: str          # exact ID the provider SDK expects
    max_images: int            # hard cap (we still also apply our own cap)
    supports_pdf: bool         # can ingest raw PDF bytes (skip the PNG rendering step)
    description: str = ""


# Registry of known models. The keys are the user-facing names we accept
# on the API surface; the values carry provider-specific dispatch info.
#
# To add a new model: drop another VisionModel into MODELS and update the
# dispatch table in _generate_*. No other consumers need to change.
MODELS: Dict[str, VisionModel] = {
    # ---------- OpenAI ----------
    "gpt-4.1": VisionModel(
        name="gpt-4.1",
        provider=VisionProvider.OPENAI,
        api_model_id="gpt-4.1",
        max_images=50,
        supports_pdf=True,
        description="OpenAI GPT-4.1 — top quality on dense construction drawings",
    ),
    "gpt-4o": VisionModel(
        name="gpt-4o",
        provider=VisionProvider.OPENAI,
        api_model_id="gpt-4o",
        max_images=50,
        supports_pdf=False,
        description="OpenAI GPT-4o — cheaper / faster fallback",
    ),
    # ---------- Anthropic ----------
    "claude-sonnet-4": VisionModel(
        name="claude-sonnet-4",
        provider=VisionProvider.ANTHROPIC,
        api_model_id="claude-sonnet-4-20250514",
        max_images=100,
        supports_pdf=False,
        description="Anthropic Claude Sonnet 4 — strong reasoning on tables",
    ),
    "claude-opus-4-7": VisionModel(
        name="claude-opus-4-7",
        provider=VisionProvider.ANTHROPIC,
        api_model_id="claude-opus-4-7",
        max_images=100,
        supports_pdf=False,
        description="Anthropic Claude Opus 4.7 — highest tier reasoning",
    ),
    # ---------- Google Gemini ----------
    "gemini-2.0-flash": VisionModel(
        name="gemini-2.0-flash",
        provider=VisionProvider.GEMINI,
        api_model_id="gemini-2.0-flash",
        max_images=25,
        supports_pdf=True,
        description="Google Gemini 2.0 Flash — fast multimodal, native PDF support",
    ),
    "gemini-1.5-pro": VisionModel(
        name="gemini-1.5-pro",
        provider=VisionProvider.GEMINI,
        api_model_id="gemini-1.5-pro",
        max_images=25,
        supports_pdf=True,
        description="Google Gemini 1.5 Pro — long-context, strong on schedules",
    ),
    # ----- Added 2026-05-28 (Phase 1.5+): latest Gemini models -----
    "gemini-2.5-pro": VisionModel(
        name="gemini-2.5-pro",
        provider=VisionProvider.GEMINI,
        api_model_id="gemini-2.5-pro",
        max_images=25,
        supports_pdf=True,
        description="Google Gemini 2.5 Pro — strong reasoning, native PDF, well-tested",
    ),
    "gemini-3-pro": VisionModel(
        name="gemini-3-pro",
        provider=VisionProvider.GEMINI,
        api_model_id="gemini-3-pro",
        max_images=25,
        supports_pdf=True,
        description="Google Gemini 3 Pro — latest (early 2026), top document grounding",
    ),
    "gemini-2.5-flash": VisionModel(
        name="gemini-2.5-flash",
        provider=VisionProvider.GEMINI,
        api_model_id="gemini-2.5-flash",
        max_images=25,
        supports_pdf=True,
        description="Google Gemini 2.5 Flash — fast, cheaper for OCR-style work",
    ),
}


# ---------------------------------------------------------------------------
# Defaults & config
# ---------------------------------------------------------------------------


def _env_default_primary() -> str:
    """Primary model from env, with a safe built-in fallback."""
    return os.getenv("DEEP_DIVE_PRIMARY_MODEL", "gpt-4.1").strip()


def _env_default_fallbacks() -> List[str]:
    """Fallback ladder from env (comma-separated), with built-in default."""
    raw = os.getenv(
        "DEEP_DIVE_FALLBACK_MODELS",
        "claude-sonnet-4,gemini-1.5-pro,gpt-4o",
    )
    return [m.strip() for m in raw.split(",") if m.strip()]


def _env_max_images() -> int:
    """Hard ceiling on images per call (per Q9 answer = 20, raise to 25 max)."""
    try:
        return min(25, max(1, int(os.getenv("DEEP_DIVE_MAX_IMAGES", "20"))))
    except (TypeError, ValueError):
        return 20


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------


@dataclass
class VisionResult:
    """Single non-streaming response."""

    text: str
    model_used: str
    provider: str
    fallback_used: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_with_images(
    *,
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    primary_model: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    stream: bool = True,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    timeout_s: int = 120,
    _attribution: Optional[Dict[str, Any]] = None,
) -> Union[Iterator[str], VisionResult]:
    """Run a vision-capable LLM call with automatic provider fallback.

    Parameters
    ----------
    system_prompt
        System message — sets agent persona / task framing.
    user_prompt
        The user-visible text portion. Image URLs are attached separately.
    image_urls
        Presigned PNG URLs (or HTTPS URLs the model can fetch directly).
        Capped at ``DEEP_DIVE_MAX_IMAGES`` (env, default 20, hard max 25).
    conversation_history
        Optional list of ``{"role": "user"|"assistant", "content": "..."}``
        messages to inject between the system prompt and the current
        question. Used by Deep Dive to feed prior chat context.
    primary_model
        Caller's preferred model. Falls back to env ``DEEP_DIVE_PRIMARY_MODEL``
        which defaults to ``gpt-4.1``.
    fallback_models
        Ordered list of models to try if the primary fails. Falls back to
        env ``DEEP_DIVE_FALLBACK_MODELS`` (comma-separated).
    stream
        ``True`` (default) returns a token iterator. ``False`` returns a
        completed ``VisionResult``.
    max_tokens, temperature, timeout_s
        Standard LLM knobs. Same meaning across providers.

    Returns
    -------
    Iterator[str] | VisionResult
        Streaming mode yields token strings until exhausted. Non-streaming
        mode returns a VisionResult carrying text + provider metadata.
        On total failure the result has ``error`` set and ``text == ""``
        — never raises.
    """
    primary = primary_model or _env_default_primary()
    ladder = [primary] + (fallback_models or _env_default_fallbacks())
    # Dedupe while preserving order
    seen: set = set()
    ladder = [m for m in ladder if not (m in seen or seen.add(m))]

    # Apply the global image cap
    cap = _env_max_images()
    if len(image_urls) > cap:
        logger.warning(
            "vision_client: %d images exceeds cap %d — truncating",
            len(image_urls), cap,
        )
        image_urls = image_urls[:cap]

    if stream:
        return _stream_with_fallback(
            ladder=ladder,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_urls=image_urls,
            conversation_history=conversation_history or [],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            attribution=_attribution,
        )

    # Non-streaming: collect tokens, fall through ladder on error
    attempts: List[Dict[str, Any]] = []
    for idx, model_name in enumerate(ladder):
        model = MODELS.get(model_name)
        if model is None:
            attempts.append({"model": model_name, "error": "unknown_model"})
            continue
        try:
            text, in_tok, out_tok = _generate_blocking(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_urls=image_urls,
                conversation_history=conversation_history or [],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            attempts.append({"model": model_name, "ok": True})
            return VisionResult(
                text=text,
                model_used=model_name,
                provider=model.provider.value,
                fallback_used=(idx > 0),
                input_tokens=in_tok,
                output_tokens=out_tok,
                attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vision_client: %s failed (%s) — trying next rung",
                model_name, exc,
            )
            attempts.append({"model": model_name, "error": str(exc)[:200]})

    # All rungs failed
    return VisionResult(
        text="",
        model_used="",
        provider="",
        fallback_used=True,
        error="all_providers_failed",
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Streaming dispatcher
# ---------------------------------------------------------------------------


def _stream_with_fallback(
    *,
    ladder: List[str],
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    attribution: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """Try each model in order; yield from the first one that opens a stream.

    Once tokens start flowing we DO NOT fall through — partial output is
    preserved. Failures only happen before the first token arrives.

    2026-06-02 (C1 fix): when ``attribution`` is provided (a mutable dict),
    populate it as soon as the winning model yields its first chunk. The
    caller can then read attribution["model_used"] / ["provider"] /
    ["fallback_used"] after the stream drains — accurate audit trail.
    """
    last_error: Optional[str] = None
    primary_name = ladder[0] if ladder else ""
    attempts: List[Dict[str, Any]] = []
    for model_name in ladder:
        model = MODELS.get(model_name)
        if model is None:
            last_error = f"unknown_model: {model_name}"
            attempts.append({"model": model_name, "error": "unknown_model"})
            continue
        try:
            first_chunk = True
            for chunk in _generate_stream(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_urls=image_urls,
                conversation_history=conversation_history,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            ):
                if first_chunk:
                    if attribution is not None:
                        attribution["model_used"] = model_name
                        attribution["provider"] = model.provider.value
                        attribution["fallback_used"] = (model_name != primary_name)
                        attribution["attempts"] = list(attempts) + [{"model": model_name, "ok": True}]
                    first_chunk = False
                yield chunk
            # Empty-but-not-error stream: still attribute to this model.
            if first_chunk and attribution is not None and "model_used" not in attribution:
                attribution["model_used"] = model_name
                attribution["provider"] = model.provider.value
                attribution["fallback_used"] = (model_name != primary_name)
                attribution["attempts"] = list(attempts) + [{"model": model_name, "ok": True, "empty": True}]
            return
        except Exception as exc:  # noqa: BLE001
            last_error = f"{model_name}: {exc}"
            attempts.append({"model": model_name, "error": str(exc)[:200]})
            logger.warning(
                "vision_client (stream): %s failed (%s) — trying next rung",
                model_name, exc,
            )
    # All rungs failed before yielding even one chunk
    if attribution is not None:
        attribution["model_used"] = ""
        attribution["provider"] = ""
        attribution["fallback_used"] = True
        attribution["attempts"] = attempts
        attribution["error"] = last_error or "all_providers_failed"
    yield (
        f"[Deep Dive Vision Error] All providers failed. "
        f"Last error: {last_error or 'unknown'}"
    )


# ---------------------------------------------------------------------------
# Provider-specific generators
# ---------------------------------------------------------------------------


def _generate_stream(
    *,
    model: VisionModel,
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Iterator[str]:
    """Streaming dispatcher — route to the right provider implementation."""
    if model.provider == VisionProvider.OPENAI:
        yield from _openai_stream(
            model, system_prompt, user_prompt, image_urls,
            conversation_history, max_tokens, temperature, timeout_s,
        )
    elif model.provider == VisionProvider.ANTHROPIC:
        yield from _anthropic_stream(
            model, system_prompt, user_prompt, image_urls,
            conversation_history, max_tokens, temperature, timeout_s,
        )
    elif model.provider == VisionProvider.GEMINI:
        yield from _gemini_stream(
            model, system_prompt, user_prompt, image_urls,
            conversation_history, max_tokens, temperature, timeout_s,
        )
    else:
        raise ValueError(f"Unhandled provider: {model.provider}")


def _generate_blocking(
    *,
    model: VisionModel,
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> tuple[str, int, int]:
    """Non-streaming dispatcher. Returns (text, input_tokens, output_tokens)."""
    # We implement non-streaming as "drain the stream" so we only maintain
    # one provider implementation per. Token counts come back as 0 because
    # streaming APIs don't always surface them — callers should treat 0 as
    # "unknown" and rely on provider-side cost dashboards.
    chunks: List[str] = []
    for c in _generate_stream(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt,
        image_urls=image_urls, conversation_history=conversation_history,
        max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s,
    ):
        chunks.append(c)
    return "".join(chunks), 0, 0


# --- OpenAI ----------------------------------------------------------------


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _split_urls_by_kind(image_urls: List[str]) -> tuple[List[str], List[str]]:
    """Partition URLs into (image_urls, pdf_urls) by extension.

    The chat-completions ``image_url`` content type only accepts
    PNG/JPEG/GIF/WebP. PDFs need a different transport (OpenAI Files API,
    Anthropic ``document`` content type, or Gemini's native PDF mime).
    Today we route only the image URLs via image_url and silently drop
    PDFs on providers that don't accept them via this path.

    2026-06-02: recognize ``data:`` URLs by their MIME prefix instead of
    treating them as unknown extension == image. data:application/pdf
    must route through the PDF list; data:image/* is the image list.
    """
    images: List[str] = []
    pdfs: List[str] = []
    for u in image_urls:
        if u.startswith("data:"):
            # Inspect the MIME prefix between "data:" and the first ";"
            mt = u[5:].split(";", 1)[0].strip().lower() if len(u) > 5 else ""
            if mt == "application/pdf":
                pdfs.append(u)
            else:
                images.append(u)
            continue
        # Strip presigned-URL query string before extension match
        path = (u.split("?", 1)[0] or "").lower()
        if path.endswith(".pdf"):
            pdfs.append(u)
        elif any(path.endswith(e) for e in _IMAGE_EXTS):
            images.append(u)
        else:
            # Unknown extension — assume image (most S3 presigned URLs
            # for our pipeline end in .png).
            images.append(u)
    return images, pdfs


def _openai_stream(
    model: VisionModel,
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Iterator[str]:
    """OpenAI Chat Completions streaming with multimodal content.

    PDF handling: the chat-completions ``image_url`` content type does
    not accept PDFs. We split the URL list and only feed image-shaped
    URLs through image_url. PDF URLs are silently dropped on this path
    — callers that want PDF parsing should use ``source_format="png"``
    so the orchestrator resolves the PNG sibling that already exists in
    S3 (per the 2026-05-12 ingestion confirmation that every PDF has a
    matching .png in the same folder).
    """
    from openai import OpenAI  # local import — keeps mocking simple

    images_only, pdfs_dropped = _split_urls_by_kind(image_urls)
    if pdfs_dropped:
        logger.info(
            "openai_stream: dropping %d PDF URLs (image_url accepts PNG/JPEG only). "
            "PNG siblings already in S3 are the canonical extraction path.",
            len(pdfs_dropped),
        )
    if not images_only:
        raise RuntimeError("openai: no image-formatted URLs after PDF filter")

    client = OpenAI(timeout=timeout_s)
    # Build the content blocks for the final user message. Image blocks
    # come first so the model "sees" them before reading the prompt — this
    # ordering empirically helps GPT-4.1 ground its OCR.
    user_content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": u, "detail": "high"}}
        for u in images_only
    ]
    user_content.append({"type": "text", "text": user_prompt})

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for turn in conversation_history:
        # Conversation history is text-only — image grounding only on the
        # current question so we don't blow the context window.
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})

    stream = client.chat.completions.create(
        model=model.api_model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for event in stream:
        try:
            delta = event.choices[0].delta
            chunk = getattr(delta, "content", None)
            if chunk:
                yield chunk
        except (AttributeError, IndexError):
            continue


# --- Anthropic -------------------------------------------------------------


def _anthropic_stream(
    model: VisionModel,
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Iterator[str]:
    """Anthropic Messages API streaming with multimodal content.

    Anthropic does not (as of this build) accept arbitrary HTTPS URLs as
    image refs — only base64-encoded bytes or specific URL hosts. We
    download the image bytes, b64-encode them, and send as a `source`
    of type `base64`. This costs one extra round trip per image but
    is the documented contract.
    """
    import httpx
    from anthropic import Anthropic  # local import — keeps mocking simple

    # Drop PDF URLs — Anthropic's image content block also requires
    # PNG/JPEG/GIF/WebP. PDFs need a different (document) content type
    # which we don't wire here. PNG siblings exist in S3 anyway.
    images_only, _pdfs_dropped = _split_urls_by_kind(image_urls)
    if not images_only:
        raise RuntimeError("anthropic: no image-formatted URLs after PDF filter")

    # Fetch each image. We use a short timeout per image — slow S3 is
    # likely a transient issue and we'd rather fall through to the next
    # provider than hang the deep-dive call.
    images_b64: List[tuple[str, str]] = []  # (media_type, b64_bytes)
    # 2026-06-02 fix: handle data: URLs inline (httpx cannot fetch them).
    # hires_render emits base64 data: URLs which previously caused
    # "no images successfully fetched" and tripped the whole ladder.
    http_urls: List[str] = []
    for u in images_only:
        if u.startswith("data:"):
            try:
                # Format: data:<mime>;base64,<payload>
                header, _, payload = u.partition(",")
                mt = "image/png"
                if header.startswith("data:") and ";" in header:
                    mt = header[5:].split(";", 1)[0].strip() or "image/png"
                # payload is already base64; re-validate it round-trips
                _ = base64.b64decode(payload, validate=True)
                images_b64.append((mt, payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("anthropic: data URL decode failed: %s", exc)
                continue
        else:
            http_urls.append(u)
    with httpx.Client(timeout=IMAGE_FETCH_TIMEOUT_S, follow_redirects=True) as http:
        for u in http_urls:
            try:
                r = http.get(u)
                r.raise_for_status()
                ct = r.headers.get("content-type", "image/png").split(";")[0].strip()
                images_b64.append((ct, base64.b64encode(r.content).decode("ascii")))
            except Exception as exc:  # noqa: BLE001
                logger.warning("anthropic: image fetch failed for %s: %s", u, exc)
                continue
    if not images_b64:
        # We have nothing to send — propagate so the fallback ladder kicks in.
        raise RuntimeError("anthropic: no images successfully fetched")

    # Anthropic rejects any image whose width OR height exceeds 8000px
    # ("image dimensions exceed max allowed size"). High-res drawing renders
    # routinely blow past that, which previously failed EVERY Claude rung and
    # wasted the whole ladder. Downscale oversized images in-place first.
    images_b64 = [_downscale_b64_for_anthropic(mt, b64) for (mt, b64) in images_b64]

    content: List[Dict[str, Any]] = []
    for media_type, b64 in images_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
    content.append({"type": "text", "text": user_prompt})

    msgs: List[Dict[str, Any]] = []
    for turn in conversation_history:
        role = turn.get("role", "user")
        text = turn.get("content", "")
        if role in {"user", "assistant"} and text:
            msgs.append({"role": role, "content": text})
    msgs.append({"role": "user", "content": content})

    client = Anthropic(timeout=timeout_s)
    # 2026-06-02 fix (revised L8): Anthropic deprecated `temperature` for the
    # Claude 4+ family. Use an ALLOWLIST of older model substrings that still
    # accept temperature — failing closed when a new model ships (better than
    # the prior denylist which would silently re-enable temperature for
    # claude-sonnet-5 / claude-opus-5 and re-trigger the bug).
    stream_kwargs = dict(
        model=model.api_model_id,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=msgs,
    )
    mid = (model.api_model_id or "").lower()
    TEMP_ACCEPTING_PREFIXES = ("claude-3", "claude-2", "claude-instant")
    if any(p in mid for p in TEMP_ACCEPTING_PREFIXES):
        stream_kwargs["temperature"] = temperature
    with client.messages.stream(**stream_kwargs) as stream:
        for chunk in stream.text_stream:
            if chunk:
                yield chunk


# --- Google Gemini ---------------------------------------------------------


def _gemini_stream(
    model: VisionModel,
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str],
    conversation_history: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Iterator[str]:
    """Google Gemini streaming with multimodal content.

    Uses the ``google-generativeai`` SDK. Reads the API key from the
    ``GEMINI_API_KEY`` env var (falls back to ``GOOGLE_API_KEY``).

    Like Anthropic, Gemini wants inline image bytes — we fetch and pass
    raw bytes (not base64 — the SDK handles that for us via Blob).
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"google-generativeai not installed: {exc}")
    import httpx

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not configured")
    genai.configure(api_key=api_key)

    # Fetch image AND/OR PDF bytes — Gemini accepts both as inline parts.
    # Unlike OpenAI/Anthropic chat-completion image_url, Gemini's
    # ``Blob`` part supports mime_type="application/pdf" natively. So
    # "both" source_format actually delivers a richer payload on Gemini.
    image_parts: List[Dict[str, Any]] = []
    # 2026-06-02 fix: handle data: URLs inline.
    gem_http_urls: List[str] = []
    for u in image_urls:
        if u.startswith("data:"):
            try:
                header, _, payload = u.partition(",")
                mt = "image/png"
                if header.startswith("data:") and ";" in header:
                    mt = header[5:].split(";", 1)[0].strip() or "image/png"
                raw = base64.b64decode(payload, validate=True)
                image_parts.append({"mime_type": mt, "data": raw})
            except Exception as exc:  # noqa: BLE001
                logger.warning("gemini: data URL decode failed: %s", exc)
                continue
        else:
            gem_http_urls.append(u)
    with httpx.Client(timeout=IMAGE_FETCH_TIMEOUT_S, follow_redirects=True) as http:
        for u in gem_http_urls:
            try:
                r = http.get(u)
                r.raise_for_status()
                # Sniff mime from response header; fall back to URL ext
                ct_header = r.headers.get("content-type", "").split(";")[0].strip()
                mime = ct_header or "image/png"
                path_lc = u.split("?", 1)[0].lower()
                if not ct_header:
                    if path_lc.endswith(".pdf"):
                        mime = "application/pdf"
                    elif path_lc.endswith((".jpg", ".jpeg")):
                        mime = "image/jpeg"
                    elif path_lc.endswith(".webp"):
                        mime = "image/webp"
                    elif path_lc.endswith(".gif"):
                        mime = "image/gif"
                    else:
                        mime = "image/png"
                image_parts.append({"mime_type": mime, "data": r.content})
            except Exception as exc:  # noqa: BLE001
                logger.warning("gemini: fetch failed for %s: %s", u, exc)
                continue
    if not image_parts:
        raise RuntimeError("gemini: no assets successfully fetched")

    # Gemini's chat history shape — alternating user/model.
    history: List[Dict[str, Any]] = []
    for turn in conversation_history:
        role = turn.get("role", "user")
        text = turn.get("content", "")
        if not text:
            continue
        gem_role = "user" if role == "user" else "model"
        history.append({"role": gem_role, "parts": [{"text": text}]})

    # Compose the system instruction. Gemini supports a top-level
    # `system_instruction` on the model — we use that path.
    model_inst = genai.GenerativeModel(
        model_name=model.api_model_id,
        system_instruction=system_prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    # If we have history, use a chat session; otherwise single-shot.
    final_parts: List[Any] = [*image_parts, {"text": user_prompt}]
    if history:
        chat = model_inst.start_chat(history=history)
        response = chat.send_message(final_parts, stream=True)
    else:
        response = model_inst.generate_content(final_parts, stream=True)

    for chunk in response:
        try:
            txt = getattr(chunk, "text", None) or ""
            if txt:
                yield txt
        except Exception as exc:  # noqa: BLE001
            # Some Gemini chunks are tool-call envelopes with no .text —
            # safe to skip.
            logger.debug("gemini chunk skip: %s", exc)
            continue


# ---------------------------------------------------------------------------
# Convenience helper — preflight check
# ---------------------------------------------------------------------------


def available_models() -> Dict[str, Dict[str, Any]]:
    """Return a dict of {name: metadata} for all configured-and-credentialed models.

    Used by the deep-dive router's /models endpoint so the UI / Postman
    can see which providers are actually usable in this environment.
    """
    out: Dict[str, Dict[str, Any]] = {}
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    gemini_ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    cred_map = {
        VisionProvider.OPENAI: openai_ok,
        VisionProvider.ANTHROPIC: anthropic_ok,
        VisionProvider.GEMINI: gemini_ok,
    }
    for name, m in MODELS.items():
        out[name] = {
            "provider": m.provider.value,
            "api_model_id": m.api_model_id,
            "max_images": m.max_images,
            "supports_pdf": m.supports_pdf,
            "description": m.description,
            "credentials_present": cred_map.get(m.provider, False),
        }
    return out
