"""Query enhancement — improved queries and tips when no results found.

When the agent cannot answer and document discovery shows available docs,
this module generates domain-aware rephrased queries and actionable tips
to help the user refine their search.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Generic tips used as fallback when LLM is unavailable
_GENERIC_TIPS: list[str] = [
    "Try specifying a trade (e.g., HVAC, Electrical, Plumbing)",
    "Include a drawing number or name (e.g., M-101, E-201)",
    "Add a specification section number (e.g., 23 31 00)",
    "Specify the project stage (design, construction, closeout)",
    "Use more specific terminology from the construction documents",
]

_GENERIC_IMPROVED: list[str] = [
    "Search for specific equipment or materials by name",
    "Ask about a specific drawing or specification section",
    "Try narrowing to a trade or discipline",
]


def generate_query_enhancement(
    original_query: str,
    available_documents: list[dict],
    project_id: int,
) -> dict:
    """Generate improved queries and tips based on the failed query.

    Returns {
        "improved_queries": ["...", "..."],
        "query_tips": ["...", "..."],
    }
    """
    try:
        return _llm_enhance(original_query, available_documents, project_id)
    except Exception as exc:
        logger.warning("LLM query enhancement failed, using generic: %s", exc)
        return _generic_enhance(original_query, available_documents)


def _llm_enhance(
    original_query: str,
    available_documents: list[dict],
    project_id: int,
) -> dict:
    """Use LLM to generate domain-aware improved queries."""
    from openai import OpenAI
    from shared.config import get_config

    cfg = get_config()
    if not cfg.openai_api_key:
        return _generic_enhance(original_query, available_documents)

    # Summarize available docs for context
    doc_summary = _summarize_docs(available_documents, max_items=10)

    client = OpenAI(api_key=cfg.openai_api_key)
    response = client.responses.create(
        model=cfg.agentic_model_fallback,  # Use cheaper model
        input=[{
            "role": "system",
            "content": (
                "You are a construction document search assistant. "
                "The user's query returned no results. Based on the query "
                "and the available documents listed below, generate:\n"
                "1. Exactly 3 improved/rephrased versions of the query\n"
                "2. Exactly 3 specific tips to refine the search\n\n"
                "Available documents:\n" + doc_summary + "\n\n"
                "Output format (exactly):\n"
                "IMPROVED:\n- query 1\n- query 2\n- query 3\n"
                "TIPS:\n- tip 1\n- tip 2\n- tip 3"
            ),
        }, {
            "role": "user",
            "content": original_query,
        }],
        temperature=0.3,
        max_output_tokens=300,
    )

    return _parse_enhancement(response.output_text)


def _generic_enhance(
    original_query: str,
    available_documents: list[dict],
) -> dict:
    """Fallback: generate generic suggestions based on available documents."""
    improved: list[str] = []

    # Extract trades from available documents
    trades = set()
    for doc in available_documents:
        trade = doc.get("trade", "")
        if trade:
            trades.add(trade)

    if trades:
        trade_list = ", ".join(sorted(trades)[:3])
        improved.append(
            f"{original_query} (try specifying a trade: {trade_list})"
        )

    # Suggest a specific drawing title
    for doc in available_documents[:2]:
        title = doc.get("drawing_title") or doc.get("section_title", "")
        if title:
            improved.append(f"Search for '{original_query}' in {title}")

    if not improved:
        improved = _GENERIC_IMPROVED[:3]

    return {
        "improved_queries": improved[:3],
        "query_tips": _GENERIC_TIPS[:3],
    }


def _summarize_docs(docs: list[dict], max_items: int = 10) -> str:
    """Create a compact summary of available documents for LLM context."""
    lines = []
    for doc in docs[:max_items]:
        doc_type = doc.get("type", "document")
        if doc_type == "drawing":
            lines.append(
                f"- Drawing: {doc.get('drawing_title', '?')} "
                f"({doc.get('drawing_name', '?')}, trade: {doc.get('trade', '?')})"
            )
        elif doc_type == "specification":
            lines.append(
                f"- Spec: {doc.get('section_title', '?')} "
                f"(PDF: {doc.get('pdf_name', '?')})"
            )
    if len(docs) > max_items:
        lines.append(f"... and {len(docs) - max_items} more")
    return "\n".join(lines) if lines else "No documents available"


def _parse_enhancement(text: str) -> dict:
    """Parse the LLM output into improved_queries and query_tips."""
    improved: list[str] = []
    tips: list[str] = []
    current: Optional[list] = None

    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("IMPROVED"):
            current = improved
        elif line.upper().startswith("TIPS") or line.upper().startswith("TIP"):
            current = tips
        elif line.startswith("- ") and current is not None:
            item = line[2:].strip()
            if item:
                current.append(item)

    return {
        "improved_queries": improved[:3] or _GENERIC_IMPROVED[:3],
        "query_tips": tips[:3] or _GENERIC_TIPS[:3],
    }
