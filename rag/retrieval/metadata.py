"""Metadata field extraction and display helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

# All known key variants for the human-readable drawing title.
# "drawingTitle" is the primary production key; others are defensive fallbacks.
_TITLE_KEYS = (
    "drawingTitle",     # primary production key (camelCase)
    "drawing_title",    # snake_case variant
    "Drawing Title",    # spaced variant
    "drawing Title",
    "DrawingTitle",
    "DRAWING_TITLE",
    "section_title",    # specification section fallback
)

# Keys for the raw filename — used ONLY when no human-readable title exists.
_NAME_KEYS = ("pdfName", "pdf_name", "fileName", "filename", "drawing_name", "drawingName", "TradeName", "trade_name")


def get_drawing_title(meta: Dict) -> Optional[str]:
    """Return the human-readable drawing title from metadata, or None."""
    for key in _TITLE_KEYS:
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def get_display_title(meta: Dict) -> str:
    """
    Best human-readable title for a chunk.

    Priority:  drawing_title  →  pdfName/drawing_name  →  text snippet  →  id
    """
    title = get_drawing_title(meta)
    if title:
        return title

    for key in _NAME_KEYS:
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    text = (meta.get("text") or "").strip()
    if text:
        words = text.split()
        return " ".join(words[:6]) + ("…" if len(words) > 6 else "")

    return f"Document {meta.get('drawing_id') or meta.get('id') or '?'}"


def build_pdf_download_url(s3_path: str, pdf_name: str) -> Optional[str]:
    """
    Construct a public HTTPS download URL for a PDF.

    s3_path format (from metadata):
        "ifieldsmart/projectslug/Drawings/pdfhash"
        → bucket   = "ifieldsmart"
        → key_prefix = "projectslug/Drawings/pdfhash"

    Result:
        "https://ifieldsmart.s3.amazonaws.com/projectslug/Drawings/pdfhash/filename.pdf"

    Returns None when either argument is empty/None.
    """
    if not s3_path or not pdf_name:
        return None

    bucket, _, key_prefix = s3_path.partition("/")
    if not bucket:
        return None

    filename = pdf_name if pdf_name.lower().endswith(".pdf") else f"{pdf_name}.pdf"
    return f"https://{bucket}.s3.amazonaws.com/{key_prefix}/{filename}"


# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_field(meta: Dict, primary: str, *alternatives: str) -> Any:
    """Return the first non-None value from primary key or alternatives."""
    for key in (primary, *alternatives):
        val = meta.get(key)
        if val is not None:
            return val
    return None


def get_document_name(meta: Dict) -> str:
    """Raw document identifier (pdfName or similar), without drawing_title lookup."""
    for key in _NAME_KEYS:
        val = meta.get(key)
        if val:
            return str(val)
    text = (meta.get("text") or "").strip()
    if text:
        return " ".join(text.split()[:5]) + "…"
    return f"Document {meta.get('index', 'Unknown')}"


def get_page_number(meta: Dict) -> Optional[int]:
    """Extract page number with fallbacks."""
    for key in ("page", "page_number", "pagenum"):
        val = meta.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def get_s3_path(meta: Dict) -> Optional[str]:
    """Extract S3 path with fallbacks."""
    for key in ("s3_path", "s3BucketPath", "path", "location"):
        val = meta.get(key)
        if val:
            return str(val)
    return None
