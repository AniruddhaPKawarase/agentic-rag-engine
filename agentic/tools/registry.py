"""
Unified tool registry for 2 collections (drawing + specification).
Provides TOOL_DEFINITIONS for OpenAI function calling and
TOOL_FUNCTIONS map for execution.
"""

from tools.drawing_tools import (
    list_project_drawings as legacy_list_drawings,
    get_drawing_text as legacy_get_text,
    search_drawing_text as legacy_search_text,
    search_drawings_by_trade as legacy_search_trade,
)
from tools.specification_tools import (
    list_specifications as spec_list,
    search_specification_text as spec_search,
    get_specification_section as spec_get_section,
)


TOOL_DEFINITIONS = [
    # ── Legacy drawing tools (2.8M fragments, broad coverage) ──────
    {
        "type": "function",
        "function": {
            "name": "legacy_list_drawings",
            "description": "List ALL drawings for a project from the OCR collection (2.8M docs). Returns unique drawings with drawingTitle, drawingName, trade, and fragment counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "set_id": {"type": "integer"},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "legacy_get_text",
            "description": "Get reconstructed text for a specific legacy drawing. Assembles OCR fragments into readable text by spatial position. Use drawingId from legacy_list_drawings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "drawing_id": {"type": "integer"},
                    "page": {"type": "integer", "description": "Optional page number"},
                },
                "required": ["project_id", "drawing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "legacy_search_text",
            "description": "Search legacy drawing fragments for specific text. Returns drawings containing the search term, grouped by drawingId. Use for finding specific content across all project drawings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "search_text": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["project_id", "search_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "legacy_search_trade",
            "description": "Search legacy drawings by trade (Electrical, Mechanical, Plumbing, etc.). Returns drawings grouped by drawingId.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "trade": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["project_id", "trade"],
            },
        },
    },

    # ── Specification tools (80K docs, rich text) ──────────────────
    {
        "type": "function",
        "function": {
            "name": "spec_list",
            "description": "List available specifications for a project. Specifications contain material requirements, standards, submittals, and warranties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spec_search",
            "description": "Search specification content by keywords. Finds materials, standards, CSI sections, submittals, and warranty requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "search_text": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["project_id", "search_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spec_get_section",
            "description": "Get full text of a specific specification section. Use sectionTitle or pdfName from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "section_title": {"type": "string"},
                    "pdf_name": {"type": "string"},
                },
                "required": ["project_id"],
            },
        },
    },
]


TOOL_FUNCTIONS = {
    # Legacy (drawing)
    "legacy_list_drawings": legacy_list_drawings,
    "legacy_get_text": legacy_get_text,
    "legacy_search_text": legacy_search_text,
    "legacy_search_trade": legacy_search_trade,
    # Specification
    "spec_list": spec_list,
    "spec_search": spec_search,
    "spec_get_section": spec_get_section,
}
