"""Output text normalizer for the generation chain.

Strips Unicode characters that break in cp1252 viewers (Excel on Windows
opening UTF-8 CSVs with the wrong codec, legacy terminals, some PDF
exporters). The classic symptom is `â€¢` instead of `•` — that's UTF-8
bytes 0xE2 0x80 0xA2 being decoded as cp1252.

Two fixes are layered:
  1. Eval CSVs are written with `utf-8-sig` so Excel auto-detects UTF-8.
  2. This module strips/replaces the most common offenders before the
     answer leaves the chain, so even broken viewers render cleanly.

Public function:
    normalize_output(text: str) -> str
"""

from __future__ import annotations

# Replacement table — typographic Unicode → ASCII equivalents.
# Keep this list small and focused. We don't want to nuke legitimate
# non-English content (project names, addresses, etc.).
_REPLACEMENTS = {
    # Bullets and dashes
    "•": "-",   # • bullet
    "·": "-",   # · middle dot
    "‣": "-",   # ‣ triangular bullet
    "◦": "-",   # ◦ white bullet
    "⁃": "-",   # ⁃ hyphen bullet
    "–": "-",   # – en dash
    "—": "-",   # — em dash
    "−": "-",   # − minus sign
    # Smart quotes
    "‘": "'",   # ‘ left single quote
    "’": "'",   # ’ right single quote
    "‚": "'",   # ‚ low single quote
    "“": '"',   # “ left double quote
    "”": '"',   # ” right double quote
    "„": '"',   # „ low double quote
    # Ellipsis + non-breaking space
    "…": "...", # … ellipsis
    " ": " ",   # non-breaking space
    # Mojibake double-encoding (defensive: if we ever see â€¢ literally
    # in input, fix it on the way out)
    "â€¢": "-",
    "â€“": "-",  # â€“ -> -
    "â€”": "-",  # â€” -> -
    "â€˜": "'",  # â€˜ -> '
    "â€™": "'",  # â€™ -> '
    "â€œ": '"',  # â€œ -> "
    "â€": '"',  # â€\x9d -> "
}


def normalize_output(text: str) -> str:
    """Replace Unicode typographic characters with ASCII equivalents.

    Idempotent — running it twice produces the same result. Returns
    input unchanged if input is empty/None.

    Replacements are applied longest-needle-first to prevent prefix
    collisions (e.g. `â€"` and `â€\x9d` both start with `â€`, so the
    short one would mask the long one if applied first).
    """
    if not text:
        return text
    # Sort by needle length descending so multi-char mojibake patterns
    # win over their prefixes.
    for needle, replacement in sorted(
        _REPLACEMENTS.items(), key=lambda kv: -len(kv[0])
    ):
        if needle in text:
            text = text.replace(needle, replacement)
    return text


def normalize_chunks(chunks):
    """Generator wrapper for streaming chains.

    Streamed LLM tokens may split a multi-byte sequence across chunk
    boundaries, so we do best-effort per-chunk normalization. The
    resulting concatenation is identical to ``normalize_output`` on
    the joined string for our replacement table (all single-codepoint).
    """
    for chunk in chunks:
        yield normalize_output(chunk) if isinstance(chunk, str) else chunk
