"""
agentic.preprocessing.hires_render
==================================

Phase 1.5+ — On-demand 300 DPI re-rendering of selected PDFs (2026-05-28).

When `DEEP_DIVE_HIRES_DPI_ENABLED=true`, intercepts the image URL list and:
  1. For each selected document, derives the source PDF S3 URL
  2. Downloads the PDF
  3. Renders page 1 at HIRES_DPI (default 300) using pdf2image / poppler
  4. Encodes the result as a base64 data URL ready for vision LLM
  5. Substitutes that data URL in place of the original PNG sibling

Rationale: the existing PNG siblings in S3 were rendered at default DPI (~96-150).
At that resolution, ~6-8pt CAD text becomes unreadable to VLMs (and to OCR
engines). Re-rendering at 300 DPI gives the model ~4× the pixel data per
unit area without changing the prompt or model.

Tradeoffs:
  * Latency: +2-5 sec per selected doc (PDF download + render). Acceptable
    on the Deep Dive slow path.
  * Memory: each 300 DPI page can be 5-15 MB in PNG bytes; sent as base64
    inflates further. We cap MAX_PAGES_TO_RENDER (default 4) to bound this.
  * CPU: pdf2image runs CPU; minimal load.

Additive: default OFF. Failure mode: per-page render error → fall back to
the original PNG URL silently. Never raises.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger("agentic_rag.hires_render")

HIRES_DPI = int(os.getenv("HIRES_DPI", "300"))
HIRES_MAX_PAGES = int(os.getenv("HIRES_MAX_PAGES_TO_RENDER", "4"))
HIRES_FETCH_TIMEOUT_S = int(os.getenv("HIRES_FETCH_TIMEOUT_S", "20"))
HIRES_RENDER_TIMEOUT_S = int(os.getenv("HIRES_RENDER_TIMEOUT_S", "30"))

# PNG-bytes cache: saves the 25-30s per-page download + render cost on repeat calls.
HIRES_CACHE_DIR = os.getenv(
    "HIRES_CACHE_DIR",
    "/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.hires_cache",
)
HIRES_CACHE_ENABLED = os.getenv("HIRES_CACHE_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}


def _hires_cache_key(cache_url: str) -> str:
    import hashlib
    payload = "hires-v2|" + str(HIRES_DPI) + "|cap7800|" + (cache_url.split("?", 1)[0] if cache_url else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hires_cache_get(cache_url: str):
    if not HIRES_CACHE_ENABLED or not cache_url:
        return None
    try:
        path = os.path.join(HIRES_CACHE_DIR, _hires_cache_key(cache_url) + ".png")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return None
        logger.info(
            "hires_render(cache): HIT %d KB (key=%s)",
            len(data) // 1024, _hires_cache_key(cache_url)[:12],
        )
        return data
    except Exception as exc:
        logger.debug("hires_render(cache): get failed: %s", exc)
        return None


def _hires_cache_put(cache_url: str, png_bytes: bytes) -> None:
    if not HIRES_CACHE_ENABLED or not cache_url or not png_bytes:
        return
    try:
        os.makedirs(HIRES_CACHE_DIR, exist_ok=True)
        path = os.path.join(HIRES_CACHE_DIR, _hires_cache_key(cache_url) + ".png")
        with open(path, "wb") as f:
            f.write(png_bytes)
        logger.info(
            "hires_render(cache): WROTE %d KB (key=%s)",
            len(png_bytes) // 1024, _hires_cache_key(cache_url)[:12],
        )
    except Exception as exc:
        logger.warning("hires_render(cache): put failed: %s", exc)


def is_enabled() -> bool:
    return os.environ.get("DEEP_DIVE_HIRES_DPI_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# PDF source resolution
# ---------------------------------------------------------------------------

def _derive_pdf_url_from_doc(doc: Dict[str, Any]) -> Optional[str]:
    """Return a freshly presigned PDF URL from s3_path + pdf_name.

    NOTE (2026-05-29 fix): we DELIBERATELY ignore doc["download_url"] here.
    Upstream may cache stale presigned URLs that 403 (wrong bucket / expired).
    Always re-derive from s3_path so we get a fresh correctly-bucketed presign.
    """
    if not isinstance(doc, dict):
        return None
    s3_path = doc.get("s3_path") or ""
    pdf_name = doc.get("pdf_name") or doc.get("file_name") or ""
    if not (s3_path and pdf_name):
        # Last resort fallback only when we have nothing else
        return doc.get("download_url")
    try:
        url = _presign_pdf_sibling(s3_path, pdf_name)
        if url:
            return url
        return doc.get("download_url")
    except Exception as exc:
        logger.debug("hires_render: pdf url derivation failed for %s: %s", pdf_name, exc)
        return doc.get("download_url")


_S3 = None


def _s3_client():
    global _S3
    if _S3 is not None:
        return _S3
    try:
        from botocore.config import Config
        region = os.getenv("S3_REGION") or os.getenv("AWS_REGION") or "us-east-1"
        _S3 = boto3.client(
            "s3",
            region_name=region,
            config=Config(signature_version="s3v4", retries={"max_attempts": 2}),
        )
        return _S3
    except Exception as exc:
        logger.warning("hires_render: boto3 client init failed: %s", exc)
        return None


def _presign_pdf_sibling(s3_path: str, pdf_name: str, expires: int = 3600) -> Optional[str]:
    client = _s3_client()
    if not client:
        return None
    bucket, _, prefix = s3_path.partition("/")
    if not bucket:
        return None
    stem = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
    key = f"{prefix}/{stem}.pdf" if prefix else f"{stem}.pdf"
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except Exception as exc:
        logger.debug("hires_render: presign failed for %s/%s: %s", bucket, key, exc)
        return None


# ---------------------------------------------------------------------------
# PDF download + 300 DPI rasterize
# ---------------------------------------------------------------------------

def _fetch_pdf_bytes(pdf_url: str) -> Optional[bytes]:
    import httpx
    try:
        with httpx.Client(timeout=HIRES_FETCH_TIMEOUT_S, follow_redirects=True) as http:
            r = http.get(pdf_url)
            r.raise_for_status()
            return r.content
    except Exception as exc:
        logger.warning("hires_render: PDF fetch failed for %s: %s", pdf_url[:80], exc)
        return None


def _render_pdf_first_page_to_png_bytes(
    pdf_bytes: bytes, dpi: int, label: str = ""
) -> Optional[bytes]:
    """Render the FIRST page of a PDF at the given DPI to PNG bytes.

    Construction drawings in this system are typically 1 page per PDF (per
    Q1 clarification on the PNG sibling convention).
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        logger.warning("hires_render: pdf2image not installed")
        return None
    try:
        t0 = time.monotonic()
        images = convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            first_page=1,
            last_page=1,
            fmt="png",
            thread_count=2,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if not images:
            logger.warning("hires_render: 0 pages rendered for %s", label)
            return None
        img = images[0]
        # 2026-06-02 fix: Anthropic Vision rejects any dimension > 8000 px.
        # 200 DPI on a 30x42 architectural sheet = ~8400-11000 px wide. Cap
        # the long edge at 7800 px so the bytes are accepted by every
        # provider (OpenAI and Gemini tolerate larger but smaller helps cost).
        from PIL import Image  # already a transitive dep of pdf2image
        MAX_LONG_EDGE = 7800
        long_edge = max(img.width, img.height)
        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / float(long_edge)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            # M4 fix: Image.LANCZOS deprecated in Pillow 9.1; use namespaced
            # enum so a routine pip upgrade to Pillow 10+ doesn't break us.
            try:
                resample = Image.Resampling.LANCZOS  # Pillow 9.1+
            except AttributeError:
                resample = Image.LANCZOS  # pre-9.1 fallback
            img = img.resize((new_w, new_h), resample)
            logger.info(
                "hires_render: %s downscaled %dpx -> %dpx (Anthropic 8000 cap)",
                label or "(unlabeled)", long_edge, MAX_LONG_EDGE,
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        logger.info(
            "hires_render: %s rendered %d-DPI page (%dx%d), %d KB in %d ms",
            label or "(unlabeled)", dpi, img.width, img.height,
            len(png_bytes) // 1024, elapsed_ms,
        )
        return png_bytes
    except Exception as exc:
        logger.warning("hires_render: render failed for %s: %s", label, exc)
        return None


def _to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Public API: replace image URLs with hi-res data URLs
# ---------------------------------------------------------------------------

def upgrade_image_urls(
    image_urls: List[str],
    source_documents: List[Dict[str, Any]],
) -> List[str]:
    """Replace each image URL with a 300 DPI data URL when possible.

    Returns a NEW list of the same length as image_urls. Each entry is
    either the upgraded data URL (success) or the original URL (fallback).
    """
    if not image_urls:
        return image_urls
    if not is_enabled():
        return image_urls

    upgraded: List[str] = []
    rendered_count = 0
    for i, url in enumerate(image_urls):
        if rendered_count >= HIRES_MAX_PAGES:
            upgraded.append(url)
            continue
        doc = source_documents[i] if i < len(source_documents) else {}
        label = ""
        cache_url = None
        if isinstance(doc, dict):
            label = (doc.get("display_title") or doc.get("pdf_name")
                     or doc.get("file_name") or f"page_{i+1}")
            # Use the bare PNG sibling URL as the stable cache key
            try:
                from agentic.generation.deep_dive_agent import _derive_png_url
                cu = _derive_png_url(doc)
                if cu:
                    cache_url = cu.split("?", 1)[0]
            except Exception:
                cache_url = None

        # CACHE CHECK: skip PDF download + render entirely if we have it
        png_bytes = _hires_cache_get(cache_url) if cache_url else None
        if png_bytes:
            upgraded.append(_to_data_url(png_bytes))
            rendered_count += 1
            continue

        # CACHE MISS: full pipeline
        pdf_url = _derive_pdf_url_from_doc(doc)
        if not pdf_url:
            logger.debug("hires_render: no PDF URL derivable for %s, keeping original", label)
            upgraded.append(url)
            continue
        pdf_bytes = _fetch_pdf_bytes(pdf_url)
        if not pdf_bytes:
            upgraded.append(url)
            continue
        png_bytes = _render_pdf_first_page_to_png_bytes(pdf_bytes, HIRES_DPI, label=label)
        if not png_bytes:
            upgraded.append(url)
            continue
        # Write to cache for next time
        if cache_url:
            _hires_cache_put(cache_url, png_bytes)
        # Optional debug dump to disk for inspection
        try:
            dbg_dir = os.getenv("HIRES_DUMP_DIR", "")
            if dbg_dir:
                os.makedirs(dbg_dir, exist_ok=True)
                ts = time.strftime("%H%M%S")
                fp = os.path.join(dbg_dir, f"hires_{ts}_{label[:30].replace('/', '_')}.png")
                with open(fp, "wb") as f:
                    f.write(png_bytes)
        except Exception:
            pass
        upgraded.append(_to_data_url(png_bytes))
        rendered_count += 1

    logger.info(
        "hires_render: upgraded %d/%d image URLs to %d DPI data URLs",
        rendered_count, len(image_urls), HIRES_DPI,
    )
    return upgraded
