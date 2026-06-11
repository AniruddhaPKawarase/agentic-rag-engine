"""
citation_aware_sort.py
======================

Reorder ``source_documents`` so the docs the answer actually cites bubble to
the top.

Why this exists
---------------
The previous post-generation reranker (gateway/reranker.py) was running an
LLM-as-judge over TITLES only and reordering the user-facing source list AFTER
the answer was already written. That broke the answer<->citation mapping: the
answer would be grounded in doc A but the UI displayed doc B as the primary
reference because doc B's title looked more like the query.

This module fixes that by reading the actual answer text and matching
citation tokens (sheet numbers, section numbers, file names, drawing IDs)
against the candidate ``source_documents``. Docs the answer cites bubble to
the top; docs it doesn't cite preserve their existing relative order.

Strict invariants
-----------------
- ``answer`` text is NEVER modified.
- Set of items in ``source_documents`` is NEVER reduced (cannot drop a citation).
- When no citations are detected in the answer, returns the input list
  UNCHANGED (preserves whatever upstream ordering chose to do — quality
  cannot regress vs. that ordering).
- Idempotent: applying twice yields the same result.
- Pure function (no I/O, no LLM call, no network).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation extraction patterns
# ---------------------------------------------------------------------------
# Sheet number — letter prefix + dash + 2-4 digits. Matches A-211, M-401, E-322,
# P-600, S-503, and the longer A-2102 style.
_SHEET_NUMBER_RE = re.compile(r"\b([A-Z])-(\d{2,4}[A-Z]?)\b")

# Six-digit CSI MasterFormat section (e.g. "Section 220553", "Section 07 92 00",
# inline "23 81 26"). Capture both the compact and spaced variants.
_SECTION_NUMBER_RE = re.compile(
    r"\b(?:(?:Section|Specification|Spec)\s+)?(\d{2}\s?\d{2}\s?\d{2}[A-Z]?)\b",
    re.IGNORECASE,
)

# Explicit bracketed citations like [Source: A-211], [Sheet A-211], [Drawing M-401]
_BRACKET_CITE_RE = re.compile(
    r"\[(?:source|sheet|drawing|spec|specification|section|ref|reference)[:\s]+([^\]]{2,80})\]",
    re.IGNORECASE,
)

# Inline citations in prose: "per sheet A-211", "see section 23 81 26"
_INLINE_REF_RE = re.compile(
    r"\b(?:per|see|refer\s+to|on|in|from|sheet|drawing|spec(?:ification)?|section)\s+"
    r"([A-Z]-?\d{2,4}[A-Z]?|\d{2}\s?\d{2}\s?\d{2}[A-Z]?)\b",
    re.IGNORECASE,
)

# Construction-doc filename heuristic (PDF mentions like "23 81 26 - CRAC.pdf")
_PDF_FILENAME_RE = re.compile(r"\b([^\s\"']+\.pdf)\b", re.IGNORECASE)

# Discipline-prefix sheet numbers (less strict — covers AD-211, ME1-401)
_LOOSE_SHEET_RE = re.compile(r"\b([A-Z]{1,3}\d?-\d{2,4}[A-Z]?)\b")


# ---------------------------------------------------------------------------
# Doc-side matchable fields
# ---------------------------------------------------------------------------
_DOC_ID_FIELDS = (
    "sheet_number", "sheetNumber",
    "drawing_id", "drawingId", "drawingID",
    "drawing_name", "drawingName",
    "drawing_title", "drawingTitle",
    "specification_number", "specificationNumber",
    "section_number", "sectionNumber", "section",
    "section_title", "sectionTitle",
    "display_title", "title",
    "pdf_name", "pdfName", "file_name", "fileName",
    "csi_division", "csiDivision",
    "id", "doc_id", "docId",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def citation_aware_sort(
    query: str,
    answer: str,
    source_documents: Sequence[Dict[str, Any]],
    *,
    extra_text: str = "",
) -> List[Dict[str, Any]]:
    """Return source_documents reordered so docs cited in the answer come first.

    Parameters
    ----------
    query : str
        User question. Used as secondary signal when answer text lacks citations.
    answer : str
        The generated answer text. Primary signal.
    source_documents : list of dict
        Candidate sources as returned by the orchestrator. NOT mutated.
    extra_text : str, optional
        Any additional text to scan for citations (e.g. follow-up questions,
        debug_info). Concatenated with answer for citation extraction.

    Returns
    -------
    list of dict
        New list with cited docs bubbled to top, non-cited preserving order.
        Same dict references — no copying — so downstream mutation behaviour
        is unchanged.
    """
    if not isinstance(source_documents, list) or len(source_documents) <= 1:
        return list(source_documents) if isinstance(source_documents, list) else []
    if not isinstance(answer, str) or not answer.strip():
        return list(source_documents)

    # Extract citation tokens from the answer (and any extra text).
    full_text = "\n".join(t for t in (answer, extra_text or "") if t)
    citation_tokens = _extract_citation_tokens(full_text)
    if not citation_tokens:
        # No citations to align to — preserve upstream order. Strictly safe.
        return list(source_documents)

    # Score each doc by how many citation tokens it matches.
    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for original_index, sd in enumerate(source_documents):
        if not isinstance(sd, dict):
            scored.append((0, original_index, sd))
            continue
        match_count = _count_doc_matches(sd, citation_tokens)
        scored.append((match_count, original_index, sd))

    # Stable sort: matched docs first (by match_count desc), ties keep original
    # order; non-matched docs trail in original order.
    scored.sort(key=lambda t: (-t[0], t[1]))
    reordered = [sd for _, _, sd in scored]

    cited_count = sum(1 for s, _, _ in scored if s > 0)
    if cited_count:
        logger.info(
            "citation_aware_sort: %d/%d sources matched %d citation tokens",
            cited_count, len(source_documents), len(citation_tokens),
        )
    return reordered


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _extract_citation_tokens(text: str) -> Set[str]:
    """Return the set of normalised citation tokens found in ``text``."""
    tokens: Set[str] = set()
    if not text:
        return tokens

    # Sheet numbers (A-211 style)
    for letter, num in _SHEET_NUMBER_RE.findall(text):
        tokens.add(_normalize_sheet(f"{letter}-{num}"))

    # Loose sheet variants (AD-211, ME1-401)
    for raw in _LOOSE_SHEET_RE.findall(text):
        tokens.add(_normalize_sheet(raw))

    # Section numbers (with / without "Section" prefix, with / without spaces)
    for raw in _SECTION_NUMBER_RE.findall(text):
        normalised = re.sub(r"\s+", "", raw)
        # Only accept six-digit canonical section IDs
        if re.fullmatch(r"\d{6}[A-Z]?", normalised):
            tokens.add(normalised)

    # Bracket citations — extract whatever's inside
    for raw in _BRACKET_CITE_RE.findall(text):
        inner = raw.strip()
        if inner:
            tokens.add(inner.lower())
            # Also try to re-extract sheet / section IDs from the inner text
            inner_tokens = _extract_citation_tokens(inner)
            tokens.update(inner_tokens)

    # Inline references after "per / see / sheet / section"
    for raw in _INLINE_REF_RE.findall(text):
        compact = re.sub(r"\s+", "", raw)
        if re.fullmatch(r"[A-Z]-?\d{2,4}[A-Z]?", compact, re.IGNORECASE):
            tokens.add(_normalize_sheet(compact))
        elif re.fullmatch(r"\d{6}[A-Z]?", compact):
            tokens.add(compact)

    # Explicit PDF filenames
    for raw in _PDF_FILENAME_RE.findall(text):
        tokens.add(raw.lower().strip())

    return tokens


def _normalize_sheet(raw: str) -> str:
    """Strip whitespace, uppercase, ensure dash between prefix and number.

    Examples
    --------
    >>> _normalize_sheet("a-211")
    'A-211'
    >>> _normalize_sheet("A 211")
    'A-211'
    >>> _normalize_sheet("ME1-401")
    'ME1-401'
    """
    s = raw.strip().upper().replace(" ", "")
    if "-" not in s:
        # Split at the boundary between letters (the discipline prefix) and
        # the canonical 2-4 digit sheet number. Letters-only prefix is the
        # common case; ME1-style sub-discipline prefixes already carry a
        # dash so they don't reach this branch.
        m = re.match(r"^([A-Z]+)(\d{2,4}[A-Z]?)$", s)
        if m:
            s = f"{m.group(1)}-{m.group(2)}"
    return s


def _count_doc_matches(doc: Dict[str, Any], tokens: Iterable[str]) -> int:
    """Return how many citation tokens ``doc`` plausibly contains.

    Uses word-boundary matching so that token ``P-100`` does NOT match a
    haystack containing ``GP-100`` (different sheet code). Without word
    boundaries the substring check would produce false positives — e.g.
    grading-plumbing ``GP-100`` getting credit for a ``P-100`` citation.
    """
    haystack_parts: List[str] = []
    for field in _DOC_ID_FIELDS:
        v = doc.get(field)
        if v is None:
            continue
        haystack_parts.append(str(v))
    # Also include the s3 path / URL because filenames live there
    for field in ("s3_path", "s3BucketPath", "sourceFile", "download_url"):
        v = doc.get(field)
        if v:
            haystack_parts.append(str(v))
    haystack = " ".join(haystack_parts)
    if not haystack:
        return 0

    haystack_upper = haystack.upper()
    haystack_lower = haystack.lower()
    haystack_compact = re.sub(r"\s+", "", haystack_upper)

    count = 0
    for tok in tokens:
        if not tok:
            continue
        tok_upper = tok.upper()
        # Sheet-number style — word-boundary match prevents G/P-100 vs P-100
        # false positives. \b treats '-' as a boundary; the negative
        # lookbehind/lookahead with [A-Z0-9] enforce that the token is not
        # adjacent to other alphanumerics either side.
        if re.fullmatch(r"[A-Z]+\d?-\d{2,4}[A-Z]?", tok_upper):
            pattern = re.compile(
                r"(?<![A-Z0-9])" + re.escape(tok_upper) + r"(?![A-Z0-9])",
                re.IGNORECASE,
            )
            if pattern.search(haystack_upper):
                count += 1
                continue
        # Section-number style — six-digit CSI MasterFormat. Match either
        # compact form (220553) or spaced form (22 05 53), word-bounded.
        if re.fullmatch(r"\d{6}[A-Z]?", tok_upper):
            spaced = " ".join([tok_upper[0:2], tok_upper[2:4], tok_upper[4:6]])
            pat_compact = re.compile(
                r"(?<!\d)" + re.escape(tok_upper) + r"(?!\d)"
            )
            pat_spaced = re.compile(
                r"(?<!\d)" + re.escape(spaced) + r"(?!\d)"
            )
            if pat_compact.search(haystack_compact) or pat_spaced.search(haystack_upper):
                count += 1
                continue
        # PDF / generic fallback — use word boundaries to avoid false subs.
        # PDF filenames always end with .pdf so the trailing boundary is real.
        if tok.lower().endswith(".pdf") and tok.lower() in haystack_lower:
            count += 1
            continue
        # Generic tokens longer than 3 chars — word-bounded match.
        if len(tok) > 3:
            pat = re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            if pat.search(haystack):
                count += 1
    return count
