"""
Text reconstruction for the legacy drawing collection.

The drawing collection stores 2.8M bounding-box-level OCR fragments.
Each fragment has (text, x, y, width, height, page, drawingId).

This module reconstructs readable text by:
1. Grouping fragments by (drawingId, page)
2. Sorting by y-position (top to bottom) then x-position (left to right)
3. Merging fragments on the same line (similar y-values)
4. Joining into coherent text blocks

This avoids loading raw fragments into the LLM context.
"""

import logging
from typing import Dict, List

logger = logging.getLogger("agentic_rag.reconstruct")

# Fragments within this y-distance are considered same line
LINE_MERGE_THRESHOLD = 15  # pixels


def reconstruct_drawing_text(
    fragments: List[Dict],
    max_chars: int = 50000,
    pre_sorted: bool = False,
) -> str:
    """Reconstruct readable text from OCR bounding-box fragments.

    Args:
        fragments: list of {text, x, y, page} dicts
        max_chars: maximum output characters
        pre_sorted: if True, skip Python sort (already sorted by MongoDB)

    Returns:
        Reconstructed text with lines joined spatially.
    """
    if not fragments:
        return ""

    if pre_sorted:
        sorted_frags = fragments
    else:
        sorted_frags = sorted(fragments, key=lambda f: (
            f.get("page", 0),
            f.get("y", 0),
            f.get("x", 0),
        ))

    lines: List[List[Dict]] = []
    current_line: List[Dict] = []
    current_y = -9999
    current_page = -1

    for frag in sorted_frags:
        text = (frag.get("text") or "").strip()
        if not text:
            continue

        page = frag.get("page", 0)
        y = frag.get("y", 0)

        # New page = new line
        if page != current_page:
            if current_line:
                lines.append(current_line)
            current_line = [frag]
            current_y = y
            current_page = page
            continue

        # Same line if y-distance is small
        if abs(y - current_y) <= LINE_MERGE_THRESHOLD:
            current_line.append(frag)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [frag]
            current_y = y

    if current_line:
        lines.append(current_line)

    # Build text from lines
    result_parts = []
    total_chars = 0

    for line_frags in lines:
        # Sort fragments within line by x-position
        line_frags.sort(key=lambda f: f.get("x", 0))
        line_text = " ".join(
            (f.get("text") or "").strip()
            for f in line_frags
            if (f.get("text") or "").strip()
        )

        if not line_text:
            continue

        if total_chars + len(line_text) > max_chars:
            result_parts.append("[... truncated ...]")
            break

        result_parts.append(line_text)
        total_chars += len(line_text)

    return "\n".join(result_parts)


def reconstruct_by_page(fragments: List[Dict]) -> Dict[int, str]:
    """Reconstruct text grouped by page number.

    Returns: {page_num: reconstructed_text}
    """
    by_page: Dict[int, List[Dict]] = {}
    for f in fragments:
        page = f.get("page", 1)
        by_page.setdefault(page, []).append(f)

    result = {}
    for page, frags in sorted(by_page.items()):
        result[page] = reconstruct_drawing_text(frags, max_chars=20000)

    return result
