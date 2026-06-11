"""
agentic.preprocessing.eager_ocr
===============================

Phase 1.5 — Eager per-call OCR (2026-05-28).

v3 (2026-05-29): added Tesseract CPU provider. The breakthrough — gpt-4o-mini
hallucinated OCR and Textract is blocked on IAM. Tesseract (open-source, free,
CPU-only) achieves 100% coverage of ground-truth phrases on P-600 in pilot.

Provider matrix (EAGER_OCR_PROVIDER env):
  openai     — gpt-4o-mini (default; HALLUCINATES on dense engineering text)
  tesseract  — local CPU Tesseract via pytesseract (RECOMMENDED for drawings)
  gemini     — google.generativeai (needs valid GEMINI_API_KEY)

Activated by env: DEEP_DIVE_EAGER_OCR_ENABLED=true. Default OFF.
Failure mode: per-page OCR error → silently skipped → vision call still runs.
"""
from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_rag.eager_ocr")


OCR_PROVIDER = os.getenv("EAGER_OCR_PROVIDER", "openai").strip().lower()
OCR_TIMEOUT_S = int(os.getenv("EAGER_OCR_TIMEOUT_S", "30"))
OCR_MAX_TOKENS = int(os.getenv("EAGER_OCR_MAX_TOKENS", "2000"))
OCR_MAX_CONCURRENCY = int(os.getenv("EAGER_OCR_MAX_CONCURRENCY", "5"))
# M1 fix (2026-06-02): default points to v31 path (matches the deployed
# service location). The prior default pointed to a stale v32-deepdive dev
# path so cache writes went to /v32-... if env wasn't explicitly set.
OCR_CACHE_DIR = os.getenv(
    "EAGER_OCR_CACHE_DIR",
    "/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.ocr_cache",
)
OCR_CACHE_ENABLED = os.getenv("EAGER_OCR_CACHE_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}


def _cache_key(image_url: str) -> str:
    """Stable cache key from provider + model + stripped URL (query removed).

    For HTTPS URLs: strip query (presigned URL signature changes every hour).
    For data: URLs: would produce different keys per render. Callers should
    instead pass a STABLE cache_url via the explicit cache_url path in
    _ocr_one_page and extract_text_for_pages.
    """
    import hashlib
    bare_url = image_url.split("?", 1)[0] if image_url else ""
    payload = f"{OCR_PROVIDER}|{OCR_MODEL}|{bare_url}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(image_url: str) -> str:
    return os.path.join(OCR_CACHE_DIR, _cache_key(image_url) + ".txt")


def _cache_get(image_url: str) -> Optional[str]:
    if not OCR_CACHE_ENABLED:
        return None
    try:
        path = _cache_path(image_url)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            logger.info(
                "eager_ocr(cache): HIT %d chars (provider=%s key=%s)",
                len(text), OCR_PROVIDER, _cache_key(image_url)[:12],
            )
            return text
    except Exception as exc:
        logger.debug("eager_ocr(cache): get failed: %s", exc)
    return None


def _cache_put(image_url: str, text: str) -> None:
    if not OCR_CACHE_ENABLED or not text:
        return
    try:
        os.makedirs(OCR_CACHE_DIR, exist_ok=True)
        path = _cache_path(image_url)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(
            "eager_ocr(cache): WROTE %d chars (provider=%s key=%s)",
            len(text), OCR_PROVIDER, _cache_key(image_url)[:12],
        )
    except Exception as exc:
        logger.warning("eager_ocr(cache): put failed: %s", exc)


def _default_model() -> str:
    if OCR_PROVIDER == "gemini":
        return os.getenv("EAGER_OCR_MODEL", "gemini-2.5-pro")
    if OCR_PROVIDER == "tesseract":
        return os.getenv("EAGER_OCR_MODEL", "tesseract-4.1")
    return os.getenv("EAGER_OCR_MODEL", "gpt-4o-mini")


OCR_MODEL = _default_model()


def is_enabled() -> bool:
    return os.environ.get("DEEP_DIVE_EAGER_OCR_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


_OCR_SYSTEM_PROMPT = """You are a high-fidelity OCR system for construction and engineering drawings.

Your task: read EVERY piece of visible text on the attached image and return it as a structured transcription. This includes:
- Title blocks, sheet numbers, drawing names
- General notes blocks (numbered lists with verbatim text)
- Legend entries
- Dimension labels, elevation labels, callout labels
- Symbol annotations
- Keynotes and reference tags
- Tables and schedules (preserve row/column structure)

RULES:
1. Transcribe text VERBATIM - preserve exact wording, casing, punctuation, units, and symbols.
2. If text is rotated, transcribe it as it reads (do not transcribe gibberish).
3. If text is partially obscured or unreadable, write [UNREADABLE] in that position rather than guessing.
4. Group text by region of the page (e.g., "TITLE BLOCK", "GENERAL NOTES", "LEGEND", "DETAIL 1", etc.).
5. Do NOT summarize, interpret, or explain. Pure transcription only.
6. If the same text appears multiple times (e.g., repeated labels in a riser diagram), transcribe each occurrence.

Output format: plain text, grouped by region with clear headers. No JSON, no markdown formatting beyond region headers."""

_OCR_USER_PROMPT = (
    "Transcribe ALL visible text on this drawing page, grouped by region. "
    "Be exhaustive and verbatim. If a value or label appears, preserve it exactly as written."
)


# ---------------------------------------------------------------------------
# Helper: get image bytes from URL OR data URL
# ---------------------------------------------------------------------------

def _image_bytes_from_url(image_url: str) -> Optional[bytes]:
    if not image_url:
        return None
    if image_url.startswith("data:image/"):
        # data:image/png;base64,<payload>
        try:
            _, _, payload = image_url.partition(",")
            return base64.b64decode(payload)
        except Exception as exc:
            logger.warning("eager_ocr: data URL decode failed: %s", exc)
            return None
    # HTTPS URL
    import httpx
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as http:
            r = http.get(image_url)
            r.raise_for_status()
            return r.content
    except Exception as exc:
        logger.warning("eager_ocr: image fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# OpenAI OCR (kept for completeness — known to hallucinate on dense drawings)
# ---------------------------------------------------------------------------

def _ocr_openai(image_url: str, page_label: str = "") -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("eager_ocr(openai): SDK not installed")
        return None
    try:
        client = OpenAI(timeout=OCR_TIMEOUT_S)
        # For data URLs, OpenAI accepts them directly
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=OCR_MODEL,
            messages=[
                {"role": "system", "content": _OCR_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "text", "text": _OCR_USER_PROMPT},
                ]},
            ],
            max_tokens=OCR_MAX_TOKENS,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "eager_ocr(openai): %d chars in %dms model=%s page=%s",
            len(text), elapsed_ms, OCR_MODEL, page_label or "?",
        )
        return text.strip()
    except Exception as exc:
        logger.warning("eager_ocr(openai): failed for %s: %s", page_label, exc)
        return None


# ---------------------------------------------------------------------------
# Gemini OCR
# ---------------------------------------------------------------------------

def _ocr_gemini(image_url: str, page_label: str = "") -> Optional[str]:
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("eager_ocr(gemini): google-generativeai SDK not installed")
        return None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("eager_ocr(gemini): no API key")
        return None
    genai.configure(api_key=api_key)

    img_bytes = _image_bytes_from_url(image_url)
    if not img_bytes:
        return None

    try:
        model = genai.GenerativeModel(
            model_name=OCR_MODEL,
            system_instruction=_OCR_SYSTEM_PROMPT,
            generation_config={
                "max_output_tokens": OCR_MAX_TOKENS * 2,
                "temperature": 0.0,
            },
        )
        parts = [{"mime_type": "image/png", "data": img_bytes}, {"text": _OCR_USER_PROMPT}]
        t0 = time.monotonic()
        resp = model.generate_content(parts, stream=False)
        text = getattr(resp, "text", "") or ""
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "eager_ocr(gemini): %d chars in %dms model=%s page=%s",
            len(text), elapsed_ms, OCR_MODEL, page_label or "?",
        )
        return text.strip()
    except Exception as exc:
        logger.warning("eager_ocr(gemini): failed for %s: %s", page_label, exc)
        return None


# ---------------------------------------------------------------------------
# Tesseract OCR (CPU, free, no API)
# ---------------------------------------------------------------------------

def _ocr_tesseract(image_url: str, page_label: str = "") -> Optional[str]:
    """Local CPU OCR via Tesseract - dual PSM pass for full coverage.

    PSM 11 (sparse text) catches scattered labels/dimensions/callouts.
    PSM 6 (uniform block) catches notes blocks + structured legend regions.
    The two together give the model both verbatim values AND structural
    context needed to disambiguate when multiple values exist.

    Pilot proved this combo achieves 13/13 = 100% on P-600 ground truth.
    Total runtime on resized 5000px image: ~30-45s (cached after first call).
    """
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError as exc:
        logger.warning("eager_ocr(tesseract): missing dep: %s", exc)
        return None

    # M2 fix (2026-06-02): cap PIL decompression bombs LOCALLY per call
    # rather than disabling the safety globally for the process. Saved
    # value is restored after the call.
    _prior_pixel_cap = Image.MAX_IMAGE_PIXELS
    # Allow up to 200 MP — covers a 14000x14000 PDF render with headroom,
    # rejects anything pathological.
    Image.MAX_IMAGE_PIXELS = 200_000_000
    max_w = int(os.getenv("TESSERACT_MAX_WIDTH", "5000"))
    psm_primary = os.getenv("TESSERACT_PSM", "11")
    psm_secondary = os.getenv("TESSERACT_PSM_SECONDARY", "6")
    enable_secondary = os.getenv("TESSERACT_DUAL_PASS", "true").strip().lower() in {"1","true","yes","on"}
    # M3 fix: pass timeout to pytesseract so a hung Tesseract subprocess
    # gets killed rather than wedging the worker indefinitely.
    tess_timeout = OCR_TIMEOUT_S

    img_bytes = _image_bytes_from_url(image_url)
    if not img_bytes:
        Image.MAX_IMAGE_PIXELS = _prior_pixel_cap
        return None

    try:
        t0 = time.monotonic()
        img = Image.open(io.BytesIO(img_bytes))
        orig_size = img.size
        if img.size[0] > max_w:
            ratio = max_w / img.size[0]
            new_size = (max_w, int(img.size[1] * ratio))
            # M4 fix: Image.LANCZOS deprecated in Pillow 9.1+
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img = img.resize(new_size, resample)
        img_gray = ImageOps.grayscale(img)
        img_pre = ImageOps.autocontrast(img_gray, cutoff=2)

        cfg_primary = "--psm " + psm_primary
        text_sparse = pytesseract.image_to_string(
            img_pre, lang="eng", config=cfg_primary, timeout=tess_timeout,
        )
        t_sparse = int((time.monotonic() - t0) * 1000)

        text_block = ""
        t_block = 0
        if enable_secondary:
            t1 = time.monotonic()
            cfg_secondary = "--psm " + psm_secondary
            text_block = pytesseract.image_to_string(
                img_pre, lang="eng", config=cfg_secondary, timeout=tess_timeout,
            )
            t_block = int((time.monotonic() - t1) * 1000)

        seen = set()
        merged_lines = []
        for chunk in (text_block, text_sparse):
            for raw_line in chunk.splitlines():
                ln = raw_line.strip()
                if not ln:
                    continue
                key = ln.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged_lines.append(ln)
        merged = chr(10).join(merged_lines)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "eager_ocr(tesseract): %d chars merged (sparse=%d/%dms, block=%d/%dms) total=%dms size=%s->%s page=%s",
            len(merged), len(text_sparse), t_sparse, len(text_block), t_block, elapsed_ms,
            orig_size, img.size, page_label or "?",
        )
        return merged
    except RuntimeError as exc:
        # pytesseract raises RuntimeError on timeout
        logger.warning("eager_ocr(tesseract): timed out or failed for %s: %s", page_label, exc)
        return None
    except Exception as exc:
        logger.warning("eager_ocr(tesseract): failed for %s: %s", page_label, exc)
        return None
    finally:
        # M2 fix: restore prior PIL bomb cap so we don't leak the cap setting
        # to other PIL consumers in the same process.
        try:
            Image.MAX_IMAGE_PIXELS = _prior_pixel_cap
        except Exception:
            pass



# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _ocr_one_page(image_url: str, page_label: str = "", cache_url: Optional[str] = None) -> Optional[str]:
    if not image_url:
        return None
    # Skip raw PDFs (image OCR providers won't accept them)
    if image_url.startswith("http"):
        url_no_q = (image_url.split("?", 1)[0] or "").lower()
        if url_no_q.endswith(".pdf"):
            logger.debug("eager_ocr: skipping PDF URL %s", page_label)
            return None

    # Cache check: prefer the stable cache_url (caller-provided) over the
    # potentially-upgraded image_url (e.g. data URL whose bytes vary).
    cache_lookup_url = cache_url or image_url
    cached = _cache_get(cache_lookup_url)
    if cached:
        return cached

    # Performance-isolated testing: if EAGER_OCR_FORCE_TEXT_PATH points to a
    # readable file, return its content as the OCR result. Bypasses live OCR
    # entirely so we can validate the architectural hypothesis without CPU
    # contention from other workloads on the VM.
    force_path = os.getenv("EAGER_OCR_FORCE_TEXT_PATH", "")
    if force_path:
        try:
            with open(force_path, "r", encoding="utf-8") as _f:
                forced = _f.read().strip()
            if forced:
                logger.info(
                    "eager_ocr(force-inject): using %d chars from %s page=%s",
                    len(forced), force_path, page_label or "?",
                )
                return forced
        except FileNotFoundError:
            pass
        except Exception as _exc:
            logger.warning("eager_ocr(force-inject): failed to read %s: %s",
                           force_path, _exc)

    if OCR_PROVIDER == "tesseract":
        result = _ocr_tesseract(image_url, page_label)
    elif OCR_PROVIDER == "gemini":
        result = _ocr_gemini(image_url, page_label)
    else:
        result = _ocr_openai(image_url, page_label)
    if result:
        _cache_put(cache_lookup_url, result)
    return result


# ---------------------------------------------------------------------------
# Multi-page concurrent runner
# ---------------------------------------------------------------------------

def extract_text_for_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not pages:
        return []
    results: List[Dict[str, Any]] = [
        {"label": p.get("label", ""), "ocr_text": None, "ok": False} for p in pages
    ]

    def _do(idx: int, page: Dict[str, Any]) -> None:
        text = _ocr_one_page(
            page.get("image_url", ""),
            page.get("label", ""),
            cache_url=page.get("cache_url"),
        )
        results[idx]["ocr_text"] = text
        results[idx]["ok"] = bool(text)

    workers = min(OCR_MAX_CONCURRENCY, len(pages))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do, i, p) for i, p in enumerate(pages)]
        for _ in concurrent.futures.as_completed(futures):
            pass
    return results


def format_for_prompt(ocr_results: List[Dict[str, Any]]) -> Optional[str]:
    parts: List[str] = []
    any_ok = False
    for r in ocr_results:
        if not r.get("ok") or not r.get("ocr_text"):
            continue
        any_ok = True
        label = r.get("label") or "page"
        parts.append(f"=== OCR FROM PAGE: {label} ===")
        parts.append(r["ocr_text"])
        parts.append("")
    if not any_ok:
        return None
    # M5 fix (2026-06-02): trimmed from 500-char header to a tight one-liner.
    # The verbatim-quoting instruction is already in the system prompt; the
    # 400+ extra chars on every call were ~120 wasted prompt tokens.
    header = "OCR-extracted text from attached pages (verbatim — quote exact values):\n"
    return header + "\n".join(parts)
