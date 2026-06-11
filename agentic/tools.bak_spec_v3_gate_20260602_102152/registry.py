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
# v3 tools (drawings_v3 / specifications_v3) — preferred for projects that
# have v3 collection coverage. Falls back gracefully (empty list) on projects
# without v3 data, so existing v1 tools stay the path-of-record for those.
from tools.drawing_tools_v3 import (
    search_drawings_v3,
    get_drawing_by_sheet as v3_get_drawing_by_sheet,
    list_drawings_v3,
    get_drawing_schedules as v3_get_drawing_schedules,
    get_drawing_symbols as v3_get_drawing_symbols,
    get_drawing_full_text as v3_get_drawing_full_text,
    find_drawing_with_most_symbols as v3_find_drawing_with_most_symbols,
    find_drawings_with_schedules as v3_find_drawings_with_schedules,
)
from tools.drawing_blocks_tool import search_drawing_blocks_v3  # Layer 2 (env-gated below)
from tools.specification_tools_v3 import (
    search_specifications_v3,
    get_spec_by_csi as v3_get_spec_by_csi,
    get_full_spec_section as v3_get_full_spec_section,
    list_specifications_v3,
    get_spec_submittals as v3_get_spec_submittals,
    get_spec_warranties as v3_get_spec_warranties,
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
    # v3 — drawings_v3
    "search_drawings_v3":              search_drawings_v3,
    "v3_get_drawing_by_sheet":         v3_get_drawing_by_sheet,
    "list_drawings_v3":                list_drawings_v3,
    "v3_get_drawing_schedules":        v3_get_drawing_schedules,
    "v3_get_drawing_symbols":          v3_get_drawing_symbols,
    "v3_get_drawing_full_text":        v3_get_drawing_full_text,
    "v3_find_drawing_with_most_symbols": v3_find_drawing_with_most_symbols,
    "v3_find_drawings_with_schedules": v3_find_drawings_with_schedules,
    # v3 — specifications_v3
    "search_specifications_v3":        search_specifications_v3,
    "v3_get_spec_by_csi":              v3_get_spec_by_csi,
    "v3_get_full_spec_section":        v3_get_full_spec_section,
    "list_specifications_v3":          list_specifications_v3,
    "v3_get_spec_submittals":          v3_get_spec_submittals,
    "v3_get_spec_warranties":          v3_get_spec_warranties,
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



# ===========================================================================
# v3 tool definitions (drawings_v3 + specifications_v3)
# ---------------------------------------------------------------------------
# Appended AFTER the existing list. Agent sees these as additional options.
# The agent's SYSTEM_PROMPT (agent.py) instructs it to prefer v3 tools on
# projects that have v3 collection coverage; v3 tools naturally return an
# empty list for projects without v3 data, so legacy tools remain the path
# of record for those.
# ===========================================================================

TOOL_DEFINITIONS.extend([
    {
        "type": "function",
        "function": {
            "name": "search_drawings_v3",
            "description": (
                "PREFERRED for drawing-content questions on projects with v3 data. "
                "Semantic search over per-PAGE drawing docs with vector embeddings + "
                "structured fields. Each result is ONE FULL PAGE (not a fragment). "
                "Use BEFORE legacy_search_text whenever possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "search_text": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "discipline": {"type": "string"},
                    "drawing_type": {"type": "string"},
                    "set_id": {"type": "integer"}
                },
                "required": ["project_id", "search_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_drawing_by_sheet",
            "description": "PREFERRED for sheet-specific lookups on v3 projects. Returns the WHOLE PAGE doc with fullText, textBlocks, schedules, symbols, titleBlock, scale, revision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "sheet_number": {"type": "string"}
                },
                "required": ["project_id", "sheet_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_drawings_v3",
            "description": "List v3 drawings, optionally filtered by discipline or drawingType.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "discipline": {"type": "string"},
                    "drawing_type": {"type": "string"},
                    "limit": {"type": "integer", "default": 50}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_drawing_schedules",
            "description": "STRUCTURED schedule extraction - returns parsed schedules[] with headers/rows/rowCount/scheduleKind. USE THIS for any schedule question instead of OCR-parsing text. v3 only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "sheet_number": {"type": "string"}
                },
                "required": ["project_id", "sheet_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_drawing_symbols",
            "description": "STRUCTURED symbol list - parsed symbols[] for a drawing, optionally filtered by kind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "sheet_number": {"type": "string"},
                    "kind": {"type": "string"}
                },
                "required": ["project_id", "sheet_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_drawing_full_text",
            "description": "Return fullText + reconstructedText + pageSummary + textBlocks for one drawing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "sheet_number": {"type": "string"}
                },
                "required": ["project_id", "sheet_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_find_drawing_with_most_symbols",
            "description": "DETERMINISTIC answer for which drawing has the most symbols - sorts v3 docs by symbolsCount DESC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_find_drawings_with_schedules",
            "description": "List drawings containing schedules, optional type hint (door / window / panel / duct).",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "schedule_type_hint": {"type": "string"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_specifications_v3",
            "description": "PREFERRED for spec content questions on v3 projects. Semantic search across CONSOLIDATED spec sections. Each result is one full section with fullText ~13K chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "search_text": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "csi_division": {"type": "string"},
                    "set_id": {"type": "integer"}
                },
                "required": ["project_id", "search_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_spec_by_csi",
            "description": "Exact CSI lookup - accepts 23 07 19, 230719, or 23-07-19. Returns FULL section with consolidated fullText.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "csi_number": {"type": "string"}
                },
                "required": ["project_id", "csi_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_full_spec_section",
            "description": "Return FULL consolidated spec section text - use csi OR section_title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "section_title": {"type": "string"},
                    "csi": {"type": "string"}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_specifications_v3",
            "description": "Enumerate v3 spec sections, optional CSI division filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "csi_division": {"type": "string"},
                    "set_id": {"type": "integer"},
                    "limit": {"type": "integer", "default": 50}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_spec_submittals",
            "description": "Return structured submittals data for a v3 spec section by CSI number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "csi": {"type": "string"}
                },
                "required": ["project_id", "csi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "v3_get_spec_warranties",
            "description": "Return structured warranties data for a v3 spec section by CSI number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "csi": {"type": "string"}
                },
                "required": ["project_id", "csi"]
            }
        }
    }
])

# === LAYER 2 BLOCK SEARCH (env-gated; ENABLE_BLOCK_RETRIEVAL) ===
import os as _os_block
if _os_block.getenv("ENABLE_BLOCK_RETRIEVAL", "false").strip().lower() in ("1","true","yes","on"):
    TOOL_FUNCTIONS["search_drawing_blocks_v3"] = search_drawing_blocks_v3
    TOOL_DEFINITIONS.append({
        "type": "function",
        "function": {
            "name": "search_drawing_blocks_v3",
            "description": (
                "BLOCK-LEVEL drawing search. Use when the answer is likely a single "
                "note or callout buried on a busy drawing (e.g. specific pipe size, slope, "
                "annotation, dimension). Returns the matched block text + parent drawing's "
                "sheet/name so the LLM can cite the source. Complements search_drawings_v3 "
                "(whole-page granularity)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "search_text": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "discipline": {"type": "string"},
                    "sheet_number": {"type": "string"},
                    "block_kind": {"type": "string", "enum": ["textBlock", "note"]},
                    "set_id": {"type": "integer"}
                },
                "required": ["project_id", "search_text"]
            }
        }
    })



# === TIER B2 — SYMBOL COUNT TOOLS (env-gated; ENABLE_SYMBOL_COUNT_V3) ===
# Deterministic count / inventory queries via drawings_v3.symbolsByKind
# (96-100% populated per audit 2026-05-27). Bypasses RAG entirely for
# count questions where the answer is a Mongo integer, not a text excerpt.
import os as _os_b2
if _os_b2.getenv("ENABLE_SYMBOL_COUNT_V3", "true").strip().lower() in ("1", "true", "yes", "on"):
    from tools.symbol_count_v3 import (
        count_symbols_by_kind_v3,
        sum_symbols_by_kind_v3,
        find_drawings_by_symbol_kind_v3,
        list_symbol_kinds_v3,
    )
    TOOL_FUNCTIONS["count_symbols_by_kind_v3"] = count_symbols_by_kind_v3
    TOOL_FUNCTIONS["sum_symbols_by_kind_v3"] = sum_symbols_by_kind_v3
    TOOL_FUNCTIONS["find_drawings_by_symbol_kind_v3"] = find_drawings_by_symbol_kind_v3
    TOOL_FUNCTIONS["list_symbol_kinds_v3"] = list_symbol_kinds_v3
    TOOL_DEFINITIONS.extend([
        {
            "type": "function",
            "function": {
                "name": "count_symbols_by_kind_v3",
                "description": (
                    "PREFERRED for count/inventory questions on v3 projects. "
                    "Returns DETERMINISTIC per-drawing symbol counts from the "
                    "structured symbolsByKind field. Examples: 'how many fixtures "
                    "on sheet P-101', 'all symbols on A-101'. If kind is given, "
                    "returns {drawingName, count, matched_kinds}; otherwise returns "
                    "the full symbolsByKind dict per drawing so you can pick the "
                    "right category. Use BEFORE search_drawings_v3 for any 'how many "
                    "X' question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "sheet_number": {"type": "string", "description": "Optional: limit to one sheet (e.g. 'P-101', 'A-101')"},
                        "drawing_name": {"type": "string", "description": "Optional: limit to one drawing by name"},
                        "kind": {"type": "string", "description": "Optional symbol category — substring match. e.g. 'pipe', 'fixture', 'door', 'duct', 'detail_callout'"},
                        "discipline": {"type": "string", "description": "Optional discipline filter — e.g. 'Plumbing', 'Architecture', 'Mechanical'"},
                        "level": {"type": "string", "description": "Optional level filter — e.g. 'Level 01'"}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "sum_symbols_by_kind_v3",
                "description": (
                    "DETERMINISTIC portfolio-level totals. Sums one symbol kind "
                    "across all drawings matching the filters. Use for 'total X "
                    "across the project' questions, e.g. 'how many roof drains "
                    "across all level 1 plumbing plans', 'total fixtures in "
                    "architectural set'. Returns {total, drawings_contributing, "
                    "matched_kinds, top_drawings}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "kind": {"type": "string", "description": "Symbol category — substring match"},
                        "discipline": {"type": "string"},
                        "level": {"type": "string"},
                        "drawing_type": {"type": "string"}
                    },
                    "required": ["project_id", "kind"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_drawings_by_symbol_kind_v3",
                "description": (
                    "Returns top-N drawings sorted by count of a given symbol kind. "
                    "Use for 'which drawing has the most X' questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "kind": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                        "discipline": {"type": "string"},
                        "level": {"type": "string"}
                    },
                    "required": ["project_id", "kind"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_symbol_kinds_v3",
                "description": (
                    "Discovery helper — lists all distinct symbol kinds detected "
                    "in this project (or discipline) with aggregate counts. "
                    "Use when you don't know what kinds were extracted, e.g. "
                    "'what kinds of symbols exist in architectural drawings?'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "discipline": {"type": "string"},
                        "top_n": {"type": "integer", "default": 30}
                    },
                    "required": ["project_id"]
                }
            }
        }
    ])


# === TIER B4 — TITLEBLOCK LOOKUP TOOLS (env-gated; ENABLE_TITLEBLOCK_V3) ===
# Deterministic titleblock factoid queries via drawings_v3.titleBlock
# (96-100% populated per audit 2026-05-27). Bypasses RAG entirely for
# "scale of X-101", "who is the architect", "revision date".
import os as _os_b4
if _os_b4.getenv("ENABLE_TITLEBLOCK_V3", "true").strip().lower() in ("1", "true", "yes", "on"):
    from tools.titleblock_v3 import (
        lookup_titleblock_v3,
        get_project_titleblock_info_v3,
    )
    TOOL_FUNCTIONS["lookup_titleblock_v3"] = lookup_titleblock_v3
    TOOL_FUNCTIONS["get_project_titleblock_info_v3"] = get_project_titleblock_info_v3
    TOOL_DEFINITIONS.extend([
        {
            "type": "function",
            "function": {
                "name": "lookup_titleblock_v3",
                "description": (
                    "PREFERRED for titleblock factoids on v3 projects. Returns "
                    "DETERMINISTIC titleblock fields (scale, date, revision, "
                    "architect, sheet_title, project_name, sheet_number) for one "
                    "or more drawings — bypasses retrieval entirely. Examples: "
                    "'what is the scale on A-101', 'revision date of sheet "
                    "GS-100', 'sheet title of M-301'. Pass a `field` to get just "
                    "one value; omit it to get the full titleBlock dict."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "sheet_number": {"type": "string", "description": "Limit to one sheet, e.g. 'A-101'"},
                        "drawing_name": {"type": "string"},
                        "field": {"type": "string", "description": "One of: scale, date, revision, architect, sheet_title, project_name, sheet_number. Omit to return full titleBlock."},
                        "discipline": {"type": "string"}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_project_titleblock_info_v3",
                "description": (
                    "Project-level aggregate of titleblock fields. Returns top "
                    "architects, project names, scale distribution, revision/issue "
                    "date ranges across all drawings in the project. Use for "
                    "'who is the architect on this project', 'what scales are "
                    "used', 'when was this project last revised'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"}
                    },
                    "required": ["project_id"]
                }
            }
        }
    ])


# === TIER D1 — DRAWING<->SPEC CROSS-REFERENCE (env-gated; ENABLE_CROSS_REF_V3) ===
# Deterministic CSI MasterFormat join between drawings_v3.csiDivisions
# and specifications_v3.csiDivision. Bypasses RAG for cross-ref questions.
import os as _os_d1
if _os_d1.getenv("ENABLE_CROSS_REF_V3", "true").strip().lower() in ("1", "true", "yes", "on"):
    from tools.cross_ref_v3 import (
        get_specs_for_drawing,
        get_drawings_for_spec,
        get_drawings_for_csi_division,
        get_specs_for_csi_division,
        list_csi_divisions_v3,
    )
    TOOL_FUNCTIONS["get_specs_for_drawing"] = get_specs_for_drawing
    TOOL_FUNCTIONS["get_drawings_for_spec"] = get_drawings_for_spec
    TOOL_FUNCTIONS["get_drawings_for_csi_division"] = get_drawings_for_csi_division
    TOOL_FUNCTIONS["get_specs_for_csi_division"] = get_specs_for_csi_division
    TOOL_FUNCTIONS["list_csi_divisions_v3"] = list_csi_divisions_v3
    TOOL_DEFINITIONS.extend([
        {
            "type": "function",
            "function": {
                "name": "get_specs_for_drawing",
                "description": (
                    "Returns the spec sections that govern a given drawing, joined via "
                    "CSI MasterFormat division. Use for 'which spec sections apply to "
                    "sheet A-210', 'what specs govern this drawing'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "sheet_number": {"type": "string"},
                        "drawing_name": {"type": "string"},
                        "limit": {"type": "integer", "default": 50}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_drawings_for_spec",
                "description": (
                    "Returns drawings referenced by a given spec section. Accepts "
                    "specification_number ('230593'), csi_division ('22'), or "
                    "section_title ('Plumbing'). Use for 'which drawings does "
                    "section 230593 govern'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "specification_number": {"type": "string"},
                        "csi_division": {"type": "string"},
                        "section_title": {"type": "string"},
                        "limit": {"type": "integer", "default": 50}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_drawings_for_csi_division",
                "description": (
                    "All drawings tagged with a given CSI division. Accepts '22', "
                    "'Division 22', '22 - Plumbing'. Use for 'show me all plumbing "
                    "drawings' or 'drawings under Division 22'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "csi_division": {"type": "string"},
                        "limit": {"type": "integer", "default": 50}
                    },
                    "required": ["project_id", "csi_division"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_specs_for_csi_division",
                "description": (
                    "All spec sections under a given CSI division. Use for "
                    "'list all Division 22 spec sections'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "csi_division": {"type": "string"},
                        "limit": {"type": "integer", "default": 50}
                    },
                    "required": ["project_id", "csi_division"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_csi_divisions_v3",
                "description": (
                    "Discovery — list every CSI division present in this project, "
                    "with counts of drawings AND specs per division. Use to scope "
                    "follow-up questions or check which divisions have coverage."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"}
                    },
                    "required": ["project_id"]
                }
            }
        }
    ])


# === TIER B3 — PAGESUMMARY SEMANTIC SEARCH (env-gated; ENABLE_PAGESUMMARY_SEARCH_V3) ===
# Atlas Vector Search over drawings_v3.pageSummary_embedding — a parallel
# retrieval channel using LLM-generated page descriptions (~96-100% populated).
# Complements existing search_drawings_v3 (which embeds textBlocks-derived
# content). Useful for high-level / topic queries.
import os as _os_b3
if _os_b3.getenv("ENABLE_PAGESUMMARY_SEARCH_V3", "true").strip().lower() in ("1", "true", "yes", "on"):
    from tools.pagesummary_search_v3 import search_drawings_by_summary_v3
    TOOL_FUNCTIONS["search_drawings_by_summary_v3"] = search_drawings_by_summary_v3
    TOOL_DEFINITIONS.extend([
        {
            "type": "function",
            "function": {
                "name": "search_drawings_by_summary_v3",
                "description": (
                    "Semantic search drawings via pageSummary embedding "
                    "(LLM-generated page descriptions). USE FOR HIGH-LEVEL "
                    "OR TOPIC queries like 'which drawings show the demolition "
                    "plan', 'find architectural floor plan drawings', "
                    "'drawings about HVAC layout'. For specific in-text "
                    "needle queries, prefer search_drawings_v3 or "
                    "search_drawing_blocks_v3 (which embed textBlocks). "
                    "Returns the same shape as search_drawings_v3."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "search_text": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                        "discipline": {"type": "string"}
                    },
                    "required": ["project_id", "search_text"]
                }
            }
        }
    ])


# === TIER E — EXTRACTED DATA TOOLS (NEW 2026-05-28; env-gated ENABLE_EXTRACTED_DATA_V3) ===
# Queries the v3 extraction fields populated by master_extraction_v3.py:
#   weak_symbol_labels, cfm_callouts, duct_sizes, unit_tags_mined,
#   unit_types_mined, sf_callouts, cross_sheet_refs, vlm_element_labels, keynote_index
#
# Use these for symbol-count enumeration, equipment-tag listing, spatial pairing
# (CFM→unit, duct trunk reduction), keynote retrieval, cross-sheet topology.
import os as _os_e
if _os_e.getenv("ENABLE_EXTRACTED_DATA_V3", "true").strip().lower() in ("1", "true", "yes", "on"):
    from tools.extracted_data_tools_v3 import (
        enumerate_equipment_tags,
        list_cfm_callouts,
        list_duct_sizes,
        get_keynotes,
        pair_textblocks_by_proximity,
        list_unit_inventory,
        get_vlm_element_labels,
        get_cross_sheet_refs,
    )
    TOOL_FUNCTIONS["enumerate_equipment_tags"] = enumerate_equipment_tags
    TOOL_FUNCTIONS["list_cfm_callouts"] = list_cfm_callouts
    TOOL_FUNCTIONS["list_duct_sizes"] = list_duct_sizes
    TOOL_FUNCTIONS["get_keynotes"] = get_keynotes
    TOOL_FUNCTIONS["pair_textblocks_by_proximity"] = pair_textblocks_by_proximity
    TOOL_FUNCTIONS["list_unit_inventory"] = list_unit_inventory
    TOOL_FUNCTIONS["get_vlm_element_labels"] = get_vlm_element_labels
    TOOL_FUNCTIONS["get_cross_sheet_refs"] = get_cross_sheet_refs

    TOOL_DEFINITIONS.extend([
        {
            "type": "function",
            "function": {
                "name": "enumerate_equipment_tags",
                "description": (
                    "Enumerate EVERY equipment tag on a drawing or across a project. "
                    "Returns full list with bbox. USE THIS FIRST for symbol-count / "
                    "list-all-X / what-X-serves-Y questions. Examples: "
                    "'what FCUs serve common areas' (kind=FCU, tag_pattern='^FCU-XC-'), "
                    "'how many fan coil units' (kind=FCU), 'list all WCs' (kind=WC). "
                    "Returns count + by_kind + drawings_covered + full tag list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string", "description": "e.g. 'M-232' — optional"},
                        "kind": {"type": "string", "description": "FCU/FSD/WC/SD/RG/AC/DOAS/EF/door/window/etc."},
                        "tag_pattern": {"type": "string", "description": "regex applied to tag text, e.g. '^FCU-XC-' for common-area FCUs"},
                        "limit": {"type": "integer", "default": 200}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_cfm_callouts",
                "description": (
                    "List every CFM callout (e.g. '70 CFM OA', '85 CFM OA') with bbox. "
                    "Returns by_value and by_modifier histograms. Use for OA/RA/SA "
                    "volume questions and unit-by-unit CFM enumeration."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"},
                        "modifier": {"type": "string", "description": "OA/RA/SA/EA"},
                        "value": {"type": "integer", "description": "filter to specific CFM value"},
                        "limit": {"type": "integer", "default": 200}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_duct_sizes",
                "description": (
                    "List every duct-size callout (e.g. '26x8 OA', '20x10 OA'). "
                    "Returns distinct_sizes set + full list with bbox. Use for "
                    "trunk-duct-reduction-sequence questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"},
                        "modifier": {"type": "string", "description": "OA/RA/SA/EA"},
                        "limit": {"type": "integer", "default": 300}
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_keynotes",
                "description": (
                    "Enumerate keynotes and sheet-notes on a drawing. Each note has "
                    "id, kind (key/general), verbatim text, and bbox. Use for "
                    "'what do the keynotes say', 'what are general notes', "
                    "or to look up a specific keynote ID."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"},
                        "kind": {"type": "string", "description": "'key' or 'general'"},
                        "keynote_id": {"type": "string", "description": "e.g. '100', '7' — to fetch one note"}
                    },
                    "required": ["project_id", "drawing_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "pair_textblocks_by_proximity",
                "description": (
                    "Spatial-join: for each anchor (e.g. each CFM callout), find the "
                    "K nearest target items (e.g. nearest unit-type code or unit-tag) "
                    "within radius_pt. Use for 'what unit gets 85 CFM', 'which FCU is "
                    "near grid 4/H', 'what value is at this symbol'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"},
                        "anchor_filter": {
                            "type": "object",
                            "description": "e.g. {field:'cfm_callouts', value:85} or {field:'weak_symbol_labels', kind:'FCU'}"
                        },
                        "target_field": {
                            "type": "string",
                            "description": "e.g. 'unit_types_mined', 'sf_callouts', 'unit_tags_mined', 'weak_symbol_labels'"
                        },
                        "radius_pt": {"type": "number", "default": 150.0},
                        "limit_pairs": {"type": "integer", "default": 30}
                    },
                    "required": ["project_id", "drawing_name", "anchor_filter", "target_field"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_unit_inventory",
                "description": (
                    "Build a per-unit inventory on a floor-plan drawing by spatially "
                    "joining U-### tags + unit-type codes (e.g. C2.1, H14.0) + SF callouts. "
                    "Returns ordered list of units each with unit_id + unit_type + area_sf. "
                    "Use for 'what residential units are shown and what are their sizes' or "
                    "'what are the largest units on this floor'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"}
                    },
                    "required": ["project_id", "drawing_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_vlm_element_labels",
                "description": (
                    "Returns VLM-classified elements recovered from PostgreSQL CKG "
                    "(includes equipment labels, schedule entries, structural members, "
                    "room labels, dimensions). Filter by elementType, trade, or "
                    "regex label_pattern. Returns count + by_elementType + full labels."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"},
                        "element_type": {"type": "string", "description": "e.g. 'fixture', 'equipment', 'schedule_entry'"},
                        "trade": {"type": "string"},
                        "label_pattern": {"type": "string", "description": "regex on label"},
                        "limit": {"type": "integer", "default": 200}
                    },
                    "required": ["project_id", "drawing_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_cross_sheet_refs",
                "description": (
                    "Returns all 'Detail X/Y-###' and 'Refer to M-###' cross-sheet "
                    "references on a drawing, grouped by target_sheet. Use for "
                    "topology / 'where is detail X' / 'this sheet references' questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "integer"},
                        "drawing_name": {"type": "string"}
                    },
                    "required": ["project_id", "drawing_name"]
                }
            }
        },
    ])
