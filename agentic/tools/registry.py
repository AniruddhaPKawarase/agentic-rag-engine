"""
Unified tool registry for all 3 collections.
Provides TOOL_DEFINITIONS for OpenAI function calling and
TOOL_FUNCTIONS map for execution.
"""

from tools.mongodb_tools import (
    list_drawings as vision_list_drawings,
    search_by_text as vision_search_text,
    search_by_filters as vision_search_filters,
    get_drawing_content as vision_get_content,
)
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
    get_full_specification_text as spec_get_full_text,
)
from tools.aggregation_tools import (
    count_equipment_tags as agg_count_equipment,
    find_typical_levels as agg_find_typical_levels,
    list_schedule_entries as agg_list_schedule,
)


TOOL_DEFINITIONS = [
    # ── drawingVision tools (highest quality data) ──────────────────
    {
        "type": "function",
        "function": {
            "name": "vision_list_drawings",
            "description": "List all VisionOCR-extracted drawings for a project. These have the richest data: page summaries, key notes, general notes, and structured elements. Use this FIRST for content questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "set_id": {"type": "integer", "description": "Optional set filter"},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vision_search_text",
            "description": "Full-text search across VisionOCR drawing content (summaries, notes, text blocks). Best for finding specific terms, materials, specs, equipment.",
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
            "name": "vision_get_content",
            "description": "Get detailed content from a specific VisionOCR drawing. Accepts EITHER the full sourceFile name OR the sheet number (e.g. 'M-101A', 'M-200'). Use after finding a drawing via list or search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "source_file": {"type": "string", "description": "Full sourceFile name OR sheet number (e.g. 'M-101A')"},
                    "content_type": {"type": "string", "enum": ["all", "notes", "elements", "summary"], "default": "all"},
                },
                "required": ["project_id", "source_file"],
            },
        },
    },

    # ── Legacy drawing tools (2.8M fragments, broad coverage) ──────
    {
        "type": "function",
        "function": {
            "name": "legacy_list_drawings",
            "description": "List ALL drawings for a project from the legacy OCR collection (2.8M docs). Returns unique drawings with metadata. Use when VisionOCR doesn't have the project or you need a complete inventory.",
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
            "description": "Get fragments of a specific specification section (capped at 50, 2000 chars each). Use sectionTitle or pdfName from search results. For a consolidated whole-section summary, prefer spec_get_full_text.",
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
    {
        "type": "function",
        "function": {
            "name": "spec_get_full_text",
            "description": (
                "Return CONSOLIDATED full text of one or more specification sections, with all "
                "fragments concatenated in page order. Use when you need to summarise an entire "
                "section (e.g. 'insulation requirements for plumbing') instead of skimming "
                "individual fragments. Filter by section_title, pdf_name, or specification_number; "
                "combine filters for precision. Returns up to 3 sections by default, ordered by "
                "fragment-match density."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "section_title": {"type": "string", "description": "Section title or keyword phrase (regex match)"},
                    "pdf_name": {"type": "string", "description": "PDF filename anchor (regex match)"},
                    "specification_number": {"type": "string", "description": "CSI number like 220700"},
                    "max_sections": {"type": "integer", "default": 3, "description": "1-5 parent sections"},
                },
                "required": ["project_id"],
            },
        },
    },
]


TOOL_FUNCTIONS = {
    # Vision (drawingVision)
    "vision_list_drawings": vision_list_drawings,
    "vision_search_text": vision_search_text,
    "vision_get_content": vision_get_content,
    # Legacy (drawing)
    "legacy_list_drawings": legacy_list_drawings,
    "legacy_get_text": legacy_get_text,
    "legacy_search_text": legacy_search_text,
    "legacy_search_trade": legacy_search_trade,
    # Specification
    "spec_list": spec_list,
    "spec_search": spec_search,
    "spec_get_section": spec_get_section,
    "spec_get_full_text": spec_get_full_text,
    # Aggregation (Fix #3)
    "agg_count_equipment": agg_count_equipment,
    "agg_find_typical_levels": agg_find_typical_levels,
    "agg_list_schedule": agg_list_schedule,
}


# Append aggregation tool definitions (Fix #3). Registered AFTER the existing
# list so the agent sees them as additional options; the system prompt below
# (agent.py) now prefers them for counting questions.
TOOL_DEFINITIONS.extend([
    {
        "type": "function",
        "function": {
            "name": "agg_count_equipment",
            "description": (
                "DETERMINISTIC count of equipment tags (DOAS-1, AHU-3, VAV-201, "
                "etc.) across a project's drawings. Use this for ANY 'how many', "
                "'count', 'number of', or 'total' question. Returns exact unique "
                "tag list + per-drawing + per-level breakdown. Always prefer this "
                "over counting from prose. Provide at least one keyword that "
                "matches the tag PREFIX (e.g. 'DOAS', 'AHU', 'VAV', 'VALVE')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Prefix keywords for the equipment family (e.g. ['DOAS','Dedicated Outdoor Air']).",
                    },
                    "level_filter": {"type": "integer", "description": "Optional: only count tags on drawings for this level."},
                    "drawing_title_filter": {"type": "string", "description": "Optional regex substring of drawingTitle."},
                },
                "required": ["project_id", "keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agg_find_typical_levels",
            "description": (
                "Cluster drawings by normalised title to determine which levels are "
                "'typical' (share a floor plan) vs unique. Use this for ANY question "
                "about 'typical levels', 'typical floors', 'which levels repeat'. "
                "Also parses explicit hints like '3 THRU 6' from drawing text. "
                "Returns typical_groups, standalone_levels, and explicit_typical_hints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "set_id": {"type": "integer"},
                    "min_cluster_size": {"type": "integer", "default": 2},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agg_list_schedule",
            "description": (
                "Extract structured schedule rows for a given equipment type "
                "(doas, ahu, vav, rtu, fan, pump, valve, plumbing_fixture, "
                "panelboard, chiller). Uses VisionOCR vision_elements first, "
                "falls back to OCR text tag extraction. Use when the user asks "
                "for a schedule listing, e.g. 'show me all DOAS in the schedule'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "schedule_type": {"type": "string"},
                    "level_filter": {"type": "integer"},
                    "max_rows": {"type": "integer", "default": 50},
                },
                "required": ["project_id", "schedule_type"],
            },
        },
    },
])
