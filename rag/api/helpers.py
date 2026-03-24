"""Context formatting, token budgeting, trade filtering, and validation helpers.

Key additions for v2:
  - budget_context_window(): keeps total context under a token cap
  - detect_trade_from_query(): lightweight regex trade extractor
  - filter_chunks_by_trade(): narrows results to the user's trade
  - parse_follow_up_questions(): extracts follow-up Qs from LLM output
  - compute_confidence_score(): maps retrieval quality to 0-1 confidence
"""
import re
from typing import Any, Dict, List, Optional, Tuple

# ── Approximate token estimator (1 token ≈ 4 chars for English) ───────────────
def estimate_tokens(text: str) -> int:
    """Fast token estimate without tiktoken dependency."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Trade detection (zero-cost, regex-only) ───────────────────────────────────

_TRADES = [
    "electrical", "mechanical", "plumbing", "hvac", "fire protection",
    "fire alarm", "structural", "architectural", "civil", "landscape",
    "roofing", "glazing", "masonry", "concrete", "steel", "demolition",
    "earthwork", "sitework", "painting", "flooring", "ceiling",
    "insulation", "waterproofing", "elevator", "conveying",
    "telecommunications", "security", "audio visual", "low voltage",
]
_TRADE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _TRADES) + r")\b",
    re.IGNORECASE,
)


def detect_trade_from_query(query: str) -> Optional[str]:
    """Extract the first recognised trade from a user query. Returns None if
    no trade keyword is found."""
    m = _TRADE_PATTERN.search(query)
    return m.group(1).lower() if m else None


def filter_chunks_by_trade(chunks: List[Dict], trade: str) -> List[Dict]:
    """Keep only chunks whose trade_name or text mentions the given trade.
    Falls back to the full list if filtering eliminates everything."""
    if not trade or not chunks:
        return chunks
    trade_lower = trade.lower()
    filtered = [
        c for c in chunks
        if trade_lower in (c.get("trade_name") or "").lower()
        or trade_lower in (c.get("text") or "").lower()
    ]
    # Fall back to unfiltered if trade filter is too aggressive
    return filtered if filtered else chunks


# ── Context window budgeting ──────────────────────────────────────────────────

def budget_context_window(
    chunks: List[Dict],
    max_context_tokens: int = 4000,
    max_chunks: int = 10,
) -> List[Dict]:
    """Select top chunks that fit within the token budget, highest similarity
    first.  This prevents blowing the LLM context window when projects have
    hundreds of thousands of records.

    Returns a (possibly shorter) list of chunks sorted by similarity desc.
    """
    if not chunks:
        return []

    # Sort by similarity descending (best first)
    sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True)

    selected: List[Dict] = []
    running_tokens = 0
    for chunk in sorted_chunks:
        if len(selected) >= max_chunks:
            break
        text = chunk.get("text", "")
        chunk_tokens = estimate_tokens(text)
        if running_tokens + chunk_tokens > max_context_tokens:
            # Try to fit at least 3 chunks even if over budget
            if len(selected) < 3:
                selected.append(chunk)
                running_tokens += chunk_tokens
            break
        selected.append(chunk)
        running_tokens += chunk_tokens

    return selected


# ── Source name extraction ────────────────────────────────────────────────────

def extract_source_name(chunk: Dict) -> str:
    """Extract the best source name from a chunk based on source_type."""
    source_type = chunk.get("source_type", "unknown").lower()
    pdf_name = chunk.get("pdf_name")
    drawing_name = chunk.get("drawing_name")
    section_title = chunk.get("section_title")
    trade_name = chunk.get("trade_name")

    doc_name = None
    if source_type == "drawing":
        doc_name = pdf_name or drawing_name
        if not doc_name:
            return f"Drawing {chunk.get('drawing_id', 'Unknown')}"
    elif source_type == "specification":
        doc_name = pdf_name or section_title or drawing_name
        if not doc_name:
            return f"Specification {chunk.get('drawing_id', 'Unknown')}"
    elif source_type == "sql":
        doc_name = trade_name or drawing_name
        if not doc_name:
            return f"SQL Data {chunk.get('drawing_id', 'Unknown')}"
    else:
        text = chunk.get("text", "")
        return text[:50] + "..." if len(text) > 50 else (text or f"Source {chunk.get('index', 'Unknown')}")

    if doc_name:
        return format_filename_for_display(doc_name)
    return f"Unknown {source_type} document"


def format_filename_for_display(filename: str) -> str:
    """Format filename for display, ensuring it has proper extension."""
    if not filename:
        return "Unknown Document"
    lower_name = filename.lower()
    common_extensions = [
        ".pdf", ".dwg", ".dxf", ".doc", ".docx", ".xls", ".xlsx",
        ".png", ".jpg", ".jpeg", ".gif", ".tiff",
    ]
    if not any(lower_name.endswith(ext) for ext in common_extensions):
        return f"{filename}.pdf"
    return filename


# ── Context text builder ─────────────────────────────────────────────────────

def build_context_text(context_chunks: List[Dict], include_citations: bool = True) -> str:
    """Build formatted context text for LLM prompt."""
    if not context_chunks:
        return "No relevant technical documentation found for this query."

    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source_type = chunk.get("source_type", "unknown").upper()
        source_name = extract_source_name(chunk)
        similarity = chunk.get("similarity", 0)
        text = chunk.get("text", "")
        if not text or not text.strip():
            continue

        page_info = ""
        page = chunk.get("page")
        if page:
            page_info = f", Page: {page}"

        if include_citations:
            header = (
                f"RELEVANT EXCERPT [{i+1}] (Source: {source_type}, Document: {source_name}{page_info}, Relevance: {similarity:.2f}):"
            )
        else:
            header = (
                f"DOCUMENT (Source: {source_type}, Document: {source_name}{page_info}):"
            )

        context_parts.append(f"{header}\n{text}")

    if not context_parts:
        return "No relevant technical documentation found for this query."

    return "\n\n" + "\n\n".join(context_parts)


# ── Follow-up question parser ────────────────────────────────────────────────

def parse_follow_up_questions(llm_output: str) -> Tuple[str, List[str]]:
    """Split LLM output into (answer, follow_up_questions).

    The LLM is instructed to write "---FOLLOW_UP---" as a separator, followed
    by lines starting with "- ".  If the separator is missing the full output
    is treated as the answer and an empty list is returned.
    """
    separator = "---FOLLOW_UP---"
    if separator not in llm_output:
        return llm_output.strip(), []

    parts = llm_output.split(separator, 1)
    answer = parts[0].strip()
    raw_questions = parts[1].strip()

    questions: List[str] = []
    for line in raw_questions.splitlines():
        line = line.strip()
        if line.startswith("- "):
            q = line[2:].strip()
            if q:
                questions.append(q)
    # Ensure exactly 3 (pad or trim)
    questions = questions[:3]
    return answer, questions


# ── Confidence / quality scoring ──────────────────────────────────────────────

def compute_confidence_score(chunks: List[Dict], search_mode: str = "rag") -> float:
    """Compute a 0-1 confidence score from retrieval quality.

    For web/hybrid modes the base confidence is higher because web search
    always returns *something*.
    """
    if search_mode == "web":
        return 0.65  # web answers always have moderate baseline

    if not chunks:
        return 0.0

    similarities = [c.get("similarity", 0) for c in chunks]
    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)

    # Weighted: 60% avg + 40% max
    raw = 0.6 * avg_sim + 0.4 * max_sim

    if search_mode == "hybrid":
        raw = min(1.0, raw + 0.1)  # slight boost for having both sources

    return round(min(1.0, raw), 4)


# ── Context quality validation ────────────────────────────────────────────────

def validate_context_quality(context_chunks: List[Dict]) -> Dict[str, Any]:
    """Validate the quality of retrieved context."""
    if not context_chunks:
        return {
            "has_context": False,
            "average_similarity": 0,
            "text_quality": "poor",
            "recommendation": "No context retrieved",
        }

    similarities = [chunk.get("similarity", 0) for chunk in context_chunks]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    text_lengths = [len(chunk.get("text", "")) for chunk in context_chunks]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    if avg_similarity > 0.7 and avg_length > 50:
        quality = "excellent"
    elif avg_similarity > 0.5 and avg_length > 30:
        quality = "good"
    elif avg_similarity > 0.3:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "has_context": True,
        "average_similarity": avg_similarity,
        "average_text_length": avg_length,
        "text_quality": quality,
        "recommendation": f"Context quality is {quality} for answering",
    }
