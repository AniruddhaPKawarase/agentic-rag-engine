"""
Intent classifier for RAG-to-DocQA agent switching.

Lightweight keyword + heuristic classifier — no LLM call needed.
Determines whether a user query should go to RAG, DocQA, or suggest switching.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Signals that the user wants project-wide information (→ RAG agent)
PROJECT_WIDE_SIGNALS = [
    "across project", "across the project", "across all",
    "all drawings", "all specifications", "all trades",
    "project-wide", "project wide", "entire project",
    "how many", "list all", "show all", "compare all",
    "missing scope", "scope gap", "total count",
    "project summary", "project overview",
    "every floor", "all floors", "all mechanical", "all electrical",
    "all plumbing", "different drawings", "which drawings",
    "generate scope", "generate report",
]

# Signals that the user wants document-specific information (→ DocQA agent)
DOCUMENT_SCOPED_SIGNALS = [
    "this document", "this drawing", "this spec", "this specification",
    "this page", "this section", "this file", "this pdf",
    "in this", "on this", "from this", "of this",
    "the selected", "the current", "the uploaded",
    "current document", "selected document",
    "what does it say", "what is mentioned",
    "page number", "which page", "on page",
    "section number", "in section",
]

# Signals that are exit commands from DocQA mode
EXIT_DOCQA_SIGNALS = [
    "back to search", "back to project", "exit document",
    "stop chatting with", "return to rag", "go back",
    "search project", "project search",
]


def classify_intent(
    query: str,
    active_agent: str = "rag",
) -> str:
    """Classify user intent for agent routing.

    Parameters
    ----------
    query : str
        The user's query text.
    active_agent : str
        Currently active agent: "rag" or "docqa".

    Returns
    -------
    str
        One of:
        - "rag" — route to RAG agent
        - "docqa" — route to DocQA agent
        - "suggest_switch_to_rag" — suggest switching back to RAG (show prompt)
        - "exit_docqa" — user explicitly wants to leave DocQA mode
    """
    query_lower = query.lower().strip()

    if active_agent == "docqa":
        # Check for explicit exit commands
        if any(sig in query_lower for sig in EXIT_DOCQA_SIGNALS):
            return "exit_docqa"

        # Check for project-wide signals → suggest switching
        if any(sig in query_lower for sig in PROJECT_WIDE_SIGNALS):
            logger.info(
                "Intent: suggest_switch_to_rag (project-wide signal in docqa mode)"
            )
            return "suggest_switch_to_rag"

        # Default: stay in DocQA mode
        return "docqa"

    # active_agent == "rag" — default behavior
    return "rag"
