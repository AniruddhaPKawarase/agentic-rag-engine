"""
agentic.preprocessing.crop_locator
==================================

Fix C — Crop-and-answer (2026-06-02).

When a Deep Dive question names a specific tag (a unit number like U-218, a
sheet reference like P-600, a keynote code like 03.05A, or any tag with the
[LETTERS]-[DIGITS] / [LETTERS][DIGITS] / DIGITS.DIGITS[LETTERS] shape) this
module:

  1. Extracts candidate tags from the query
  2. Runs Tesseract `image_to_data` over each page to get word + bbox tuples
  3. Locates each tag's pixel position on the page
  4. Crops a 1500x1500 px window around each location
  5. Returns those crops as base64 data URLs ready to PREPEND to image_urls

The vision model gets:
  - The new ZOOM crops first (high signal-to-noise — only values within
    300-400 px of the tag are visible)
  - Then the full sheet for context

The prompt instructs the model to PREFER the crops for value-binding
questions ("what is the SF of U-218?", "what is the DFU load of FAN-12?")
and use the full sheet for context only.

Why this fixes the U-218 → "413 SF" hallucination:
  The model was binding a wrong SF value (413 SF, located ~1800 px away on
  the same sheet) to U-218 because plain OCR text gives no spatial anchor.
  With a crop centered on U-218 the model only SEES values within the
  immediate neighborhood — 1153 SF — and binding succeeds.

Cache:
  Tesseract bbox extraction is expensive (15-30s per page). We cache the
  bbox list keyed by the same stable URL the OCR module uses. Re-runs
  on the same page reuse cached bboxes.

Additive: when DEEP_DIVE_CROP_ENABLED is off (default), this module is
never called. When on but no tags found in query, no crops are added.
When on and Tesseract fails, falls back to original image_urls silently.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentic_rag.crop_locator")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CROP_RADIUS_PX = int(os.getenv("DEEP_DIVE_CROP_RADIUS_PX", "750"))  # half-side
CROP_MAX_PER_PAGE = int(os.getenv("DEEP_DIVE_CROP_MAX_PER_PAGE", "2"))
CROP_TESSERACT_TIMEOUT = int(os.getenv("DEEP_DIVE_CROP_TESSERACT_TIMEOUT_S", "30"))
CROP_TESSERACT_MAX_WIDTH = int(os.getenv("DEEP_DIVE_CROP_TESSERACT_MAX_WIDTH", "5000"))
BBOX_CACHE_DIR = os.getenv(
    "DEEP_DIVE_BBOX_CACHE_DIR",
    "/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.bbox_cache",
)
BBOX_CACHE_ENABLED = os.getenv("DEEP_DIVE_BBOX_CACHE_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}


def is_enabled() -> bool:
    return os.environ.get("DEEP_DIVE_CROP_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Tag extraction from query
# ---------------------------------------------------------------------------

# Tag shapes we recognize:
#   U-218 / U218 / U 218          — unit number
#   P-600 / A-211 / A-320         — sheet ref / drawing tag
#   F7.0 / D10.3                  — unit type code (letter-digit-dot-digit)
#   03.05A / 22.04B               — keynote (CSI division-dot-section[-letter])
#   FAN-12 / RTU-3 / EF-2         — equipment tag
#   RFP-06 / RP-06                — riser tag
_TAG_PATTERNS = [
    # Letter+hyphen+digits (with optional letter suffix): U-218, P-600, RP-06
    r"\b([A-Z]{1,4})[- ]?(\d{2,4})([A-Z])?\b",
    # Letter+digit+dot+digit: F7.0, D10.3
    r"\b([A-Z]\d{1,2}\.\d{1,2}[A-Z]?)\b",
    # Keynote: 03.05A, 22.04B
    r"\b(\d{2}\.\d{2}[A-Z]?)\b",
]

# Words that look like tags but aren't (the regex would match them)
_TAG_STOPWORDS = {
    "level-01", "level-02", "level01", "level02", "page-1", "page-01",
    "sheet-1", "rev-01", "rev-02", "rev01", "rev02", "deep-dive",
}


def extract_tags(query: str) -> List[str]:
    """Pull candidate tag strings from a user query.

    Returns DEDUPED list, preserving first-occurrence order. Each returned
    tag is in CANONICAL form (uppercased, hyphenated where applicable) so
    the matcher can compare against OCR variants.
    """
    if not query:
        return []
    q = query.strip()
    tags: List[str] = []
    seen: set = set()

    for pat in _TAG_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            groups = m.groups()
            # Reconstruct canonical form depending on pattern
            if len(groups) == 3 and groups[1] is not None:
                # Letter-digits-(letter): U-218, U-218A
                core = f"{groups[0].upper()}-{groups[1]}"
                if groups[2]:
                    core += groups[2].upper()
            else:
                # Already a single group — keep as-is, uppercased
                core = (groups[0] or "").upper()
            if not core:
                continue
            key = core.lower()
            if key in seen or key in _TAG_STOPWORDS:
                continue
            seen.add(key)
            tags.append(core)
    return tags


# ---------------------------------------------------------------------------
# Bbox cache (Tesseract is slow — cache the raw bbox list per image)
# ---------------------------------------------------------------------------

def _bbox_cache_key(cache_url: str) -> str:
    payload = "bbox-v1|" + (cache_url.split("?", 1)[0] if cache_url else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bbox_cache_get(cache_url: str) -> Optional[List[Dict[str, Any]]]:
    if not BBOX_CACHE_ENABLED or not cache_url:
        return None
    try:
        path = os.path.join(BBOX_CACHE_DIR, _bbox_cache_key(cache_url) + ".json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.debug("bbox_cache get failed: %s", exc)
    return None


def _bbox_cache_put(cache_url: str, bboxes: List[Dict[str, Any]]) -> None:
    if not BBOX_CACHE_ENABLED or not cache_url or not bboxes:
        return
    try:
        os.makedirs(BBOX_CACHE_DIR, exist_ok=True)
        path = os.path.join(BBOX_CACHE_DIR, _bbox_cache_key(cache_url) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bboxes, f)
    except Exception as exc:
        logger.warning("bbox_cache put failed: %s", exc)


# ---------------------------------------------------------------------------
# Image fetch + Tesseract bbox extraction
# ---------------------------------------------------------------------------

def _image_bytes_from_url(url: str) -> Optional[bytes]:
    if not url:
        return None
    if url.startswith("data:"):
        try:
            _, _, payload = url.partition(",")
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            logger.warning("crop_locator: data URL decode failed: %s", exc)
            return None
    import httpx
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as http:
            r = http.get(url)
            r.raise_for_status()
            return r.content
    except Exception as exc:
        logger.warning("crop_locator: fetch failed: %s", exc)
        return None


def _tesseract_bboxes(image_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
    """Run Tesseract image_to_data; return list of {text, x, y, w, h, conf}.

    Scales down to CROP_TESSERACT_MAX_WIDTH first (same as eager_ocr). The
    returned bboxes are in the SCALED image coords — caller must scale back
    to the original image coords when cropping.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        logger.warning("crop_locator: missing dep: %s", exc)
        return None

    prior_cap = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 200_000_000
    try:
        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        scale_factor = 1.0
        if img.width > CROP_TESSERACT_MAX_WIDTH:
            scale_factor = CROP_TESSERACT_MAX_WIDTH / float(img.width)
            new_size = (CROP_TESSERACT_MAX_WIDTH, int(img.height * scale_factor))
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img = img.resize(new_size, resample)

        # PSM 11 (sparse) is best for finding scattered tags
        t0 = time.monotonic()
        data = pytesseract.image_to_data(
            img, lang="eng", config="--psm 11",
            output_type=pytesseract.Output.DICT,
            timeout=CROP_TESSERACT_TIMEOUT,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        bboxes: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = int(data["conf"][i])
            except (KeyError, ValueError, TypeError):
                conf = -1
            if conf >= 0 and conf < 30:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            # Scale bbox back to original image coords
            if scale_factor != 1.0:
                inv = 1.0 / scale_factor
                x = int(x * inv); y = int(y * inv)
                w = int(w * inv); h = int(h * inv)
            bboxes.append({"text": txt, "x": x, "y": y, "w": w, "h": h, "conf": conf})

        logger.info(
            "crop_locator: tesseract bbox extracted %d words in %dms (orig=%dx%d, scaled=%s)",
            len(bboxes), elapsed_ms, orig_w, orig_h, "yes" if scale_factor != 1.0 else "no",
        )
        return bboxes
    except RuntimeError as exc:
        logger.warning("crop_locator: tesseract timed out: %s", exc)
        return None
    except Exception as exc:
        logger.warning("crop_locator: tesseract failed: %s", exc)
        return None
    finally:
        try:
            Image.MAX_IMAGE_PIXELS = prior_cap
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tag matching against bboxes
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Strip non-alphanumeric and uppercase for fuzzy comparison."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _find_tag_locations(
    bboxes: List[Dict[str, Any]], tag: str,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) for every bbox matching the tag.

    Match is on normalized strings — so OCR variants "U-218", "U218",
    "U 218" all match the canonical "U-218". Cross-word matches (the OCR
    sometimes splits "U-218" into "U-" and "218") are also resolved by
    looking at the next-word bbox.
    """
    if not bboxes or not tag:
        return []
    target = _normalize(tag)
    hits: List[Tuple[int, int, int, int]] = []
    for i, b in enumerate(bboxes):
        t = _normalize(b.get("text", ""))
        if not t:
            continue
        if t == target:
            hits.append((b["x"], b["y"], b["w"], b["h"]))
            continue
        # Try concatenating with the next word (handles "U-" + "218")
        if i + 1 < len(bboxes):
            nxt = _normalize(bboxes[i + 1].get("text", ""))
            if t + nxt == target:
                # Merge bbox span
                x = min(b["x"], bboxes[i + 1]["x"])
                y = min(b["y"], bboxes[i + 1]["y"])
                xr = max(b["x"] + b["w"], bboxes[i + 1]["x"] + bboxes[i + 1]["w"])
                yb = max(b["y"] + b["h"], bboxes[i + 1]["y"] + bboxes[i + 1]["h"])
                hits.append((x, y, xr - x, yb - y))
    return hits


# ---------------------------------------------------------------------------
# Public API — crop a page for a query
# ---------------------------------------------------------------------------

def crop_for_pages(
    pages: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Build ZOOM crops for a query against a list of pages.

    Each input ``pages`` entry: {image_url, label, cache_url}
    Returns a list of crop entries: {image_url (data URL), label, source_idx}
    where source_idx is the index in the original ``pages`` list.

    No-op (returns []) when no tags are extractable from the query or
    Tesseract/PIL is unavailable.
    """
    if not is_enabled():
        return []
    tags = extract_tags(query)
    if not tags:
        logger.info("crop_locator: no tags extracted from query=%r", (query or "")[:80])
        return []
    logger.info("crop_locator: extracted tags=%s from query=%r", tags, (query or "")[:80])

    out_crops: List[Dict[str, Any]] = []
    for src_idx, p in enumerate(pages):
        image_url = p.get("image_url", "")
        label = p.get("label") or f"page_{src_idx+1}"
        cache_url = p.get("cache_url") or image_url
        if not image_url:
            continue

        # Get bbox data (cached if possible)
        bboxes = _bbox_cache_get(cache_url)
        if bboxes is None:
            img_bytes = _image_bytes_from_url(image_url)
            if not img_bytes:
                continue
            bboxes = _tesseract_bboxes(img_bytes)
            if bboxes:
                _bbox_cache_put(cache_url, bboxes)
            else:
                continue
        else:
            logger.info("crop_locator: bbox cache HIT for %s", label)

        # Find tag locations on this page
        page_crops = 0
        for tag in tags:
            if page_crops >= CROP_MAX_PER_PAGE:
                break
            locations = _find_tag_locations(bboxes, tag)
            if not locations:
                logger.debug("crop_locator: tag %s not found on %s", tag, label)
                continue
            logger.info(
                "crop_locator: tag %s found at %d location(s) on %s",
                tag, len(locations), label,
            )
            # Crop around each location (cap CROP_MAX_PER_PAGE total per page)
            img_bytes = _image_bytes_from_url(image_url)
            if not img_bytes:
                continue
            for (x, y, w, h) in locations:
                if page_crops >= CROP_MAX_PER_PAGE:
                    break
                crop_url = _make_crop_data_url(img_bytes, x, y, w, h)
                if crop_url:
                    out_crops.append({
                        "image_url": crop_url,
                        "label": f"[ZOOM: {tag} — {label}]",
                        "source_idx": src_idx,
                        "tag": tag,
                    })
                    page_crops += 1
    return out_crops


def _make_crop_data_url(
    image_bytes: bytes, x: int, y: int, w: int, h: int,
) -> Optional[str]:
    """Crop a CROP_RADIUS_PX square around (x+w/2, y+h/2) and return as data URL."""
    try:
        from PIL import Image
    except ImportError:
        return None
    prior_cap = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 200_000_000
    try:
        img = Image.open(io.BytesIO(image_bytes))
        cx, cy = x + w // 2, y + h // 2
        radius = CROP_RADIUS_PX
        left = max(0, cx - radius)
        top = max(0, cy - radius)
        right = min(img.width, cx + radius)
        bottom = min(img.height, cy + radius)
        if right - left < 200 or bottom - top < 200:
            return None
        crop = img.crop((left, top, right, bottom))
        # Cap crop dimensions at 7800 px to stay under Anthropic limit (defensive)
        if max(crop.width, crop.height) > 7800:
            scale = 7800 / float(max(crop.width, crop.height))
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                resample,
            )
        buf = io.BytesIO()
        crop.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        logger.info(
            "crop_locator: emitted crop %dx%d (centered on %d,%d), %d KB",
            crop.width, crop.height, cx, cy, len(buf.getvalue()) // 1024,
        )
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        logger.warning("crop_locator: crop failed: %s", exc)
        return None
    finally:
        try:
            Image.MAX_IMAGE_PIXELS = prior_cap
        except Exception:
            pass


CROP_PROMPT_DIRECTIVE = """
================================================================
ZOOM CROPS PRESENT — SPATIAL BINDING RULE
================================================================
Among the attached images, any image whose label begins with "[ZOOM:" is a
CROPPED, HIGH-DETAIL VIEW of the region around a specific tag mentioned in
the user's question.

When the question asks for a VALUE bound to a tag (e.g. "the SF of U-218",
"the DFU load of FAN-12", "the dimension at A-211"):
  1. Use the [ZOOM: ...] crop as the AUTHORITATIVE source for that value.
  2. Only quote numbers/labels that are visible INSIDE the crop window.
  3. The full sheet is provided ONLY for context (sheet name, surrounding
     sections, where the tag sits on the page). DO NOT use values from the
     full sheet to answer a tag-binding question — those values are too far
     from the tag and may bind to a different tag.
  4. If the value you need is not visible inside the [ZOOM:] crop, say so
     explicitly: "The crop centered on <tag> does not show <value> at this
     zoom level."
"""
