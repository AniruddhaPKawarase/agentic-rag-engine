"""
AgenticRAG Core Agent.

ReAct-style agent with production hardening:
- Hard-overrides project_id on all tool calls (prevents cross-project leakage)
- Per-request and daily cost circuit breakers
- Sanitized error messages (no credential leakage)
- Conversation history validation
- Improved confidence scoring with escalation support
"""

import json
import os
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from openai import OpenAI

from openai import APIConnectionError, APITimeoutError, RateLimitError

from config import (
    AGENT_MAX_TOKENS,
    AGENT_MODEL,
    AGENT_MODEL_FALLBACK,
    AGENT_TEMPERATURE,
    DAILY_BUDGET_USD,
    MAX_AGENT_STEPS,
    MAX_QUERY_LENGTH,
    MAX_REQUEST_COST_USD,
    OPENAI_API_KEY,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
)

# [DUAL-MODEL-MARKER] env-driven dual-model routing — fast model for tool
# decisions, accurate model for final synthesis. Defaults preserve today's
# behavior exactly when AGENT_DUAL_MODEL_ENABLED=false.
import os as _dual_os
def _dual_model_enabled() -> bool:
    return _dual_os.environ.get("AGENT_DUAL_MODEL_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
def _tool_decision_model() -> str | None:
    v = (_dual_os.environ.get("AGENT_TOOL_DECISION_MODEL", "") or "").strip()
    return v or None
def _synthesis_model() -> str | None:
    v = (_dual_os.environ.get("AGENT_SYNTHESIS_MODEL", "") or "").strip()
    return v or None
def _dual_active() -> bool:
    if not _dual_model_enabled():
        return False
    tdm = _tool_decision_model()
    sm = _synthesis_model()
    return bool(tdm and sm and tdm != sm)

from core.cache import get_agent_result, set_agent_result
from tools.registry import TOOL_DEFINITIONS, TOOL_FUNCTIONS

# ── v3 tool gating (project-scoped) ──────────────────────────────────
def _v3_enabled_projects() -> set:
    """Comma-separated project IDs from env. Default: "7325"."""
    raw = os.getenv("V3_ENABLED_PROJECTS", "7325") or ""
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out


_V3_TOOL_NAMES = frozenset({
    "search_drawings_v3", "v3_get_drawing_by_sheet", "list_drawings_v3",
    "v3_get_drawing_schedules", "v3_get_drawing_symbols",
    "v3_get_drawing_full_text", "v3_find_drawing_with_most_symbols",
    "v3_find_drawings_with_schedules",
    "search_specifications_v3", "v3_get_spec_by_csi",
    "v3_get_full_spec_section", "list_specifications_v3",
    "v3_get_spec_submittals", "v3_get_spec_warranties",
})


def _project_scoped_tools(project_id: int) -> list:
    """Return TOOL_DEFINITIONS with v3 tools filtered out for non-enabled projects."""
    if project_id in _v3_enabled_projects():
        return TOOL_DEFINITIONS
    return [
        t for t in TOOL_DEFINITIONS
        if t.get("function", {}).get("name") not in _V3_TOOL_NAMES
    ]


logger = logging.getLogger("agentic_rag.agent")

# ── OpenAI client (lazy init) ─────────────────────────────────────────
_openai_client: Optional[OpenAI] = None
_openai_lock = threading.Lock()


def _get_openai_client() -> OpenAI:
    """Get or create the OpenAI client (thread-safe)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _openai_lock:
        if _openai_client is None:
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _llm_call(messages: list, tools: list, model: str = None):
    """Call OpenAI with retry + exponential backoff + model fallback."""
    client = _get_openai_client()
    model = model or AGENT_MODEL
    last_error = None

    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=AGENT_MAX_TOKENS,
                temperature=AGENT_TEMPERATURE,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            logger.warning(f"OpenAI {type(e).__name__} (attempt {attempt}/{OPENAI_MAX_RETRIES}), retrying in {wait}s")
            time.sleep(wait)

    # All retries failed with primary model — try fallback
    if model != AGENT_MODEL_FALLBACK:
        logger.warning(f"Primary model {model} failed, falling back to {AGENT_MODEL_FALLBACK}")
        try:
            return client.chat.completions.create(
                model=AGENT_MODEL_FALLBACK,
                messages=messages,
                tools=tools,
                max_tokens=AGENT_MAX_TOKENS,
                temperature=AGENT_TEMPERATURE,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
        except Exception as fallback_err:
            logger.error(f"Fallback model also failed: {type(fallback_err).__name__}")

    raise last_error or RuntimeError("LLM call failed after all retries")


# ── Daily cost tracking ───────────────────────────────────────────────
_daily_cost_lock = threading.Lock()
_daily_cost: Dict[str, Any] = {"date": "", "total": 0.0}


def _check_daily_budget(additional: float) -> bool:
    """Check if adding this cost exceeds the daily budget."""
    with _daily_cost_lock:
        today = date.today().isoformat()
        if _daily_cost["date"] != today:
            _daily_cost["date"] = today
            _daily_cost["total"] = 0.0
        return (_daily_cost["total"] + additional) < DAILY_BUDGET_USD


def _record_cost(cost: float) -> None:
    """Record cost against the daily budget."""
    with _daily_cost_lock:
        today = date.today().isoformat()
        if _daily_cost["date"] != today:
            _daily_cost["date"] = today
            _daily_cost["total"] = 0.0
        _daily_cost["total"] += cost


# ── Conversation history sanitization ─────────────────────────────────
ALLOWED_HISTORY_ROLES = {"user", "assistant"}


def _sanitize_history(history: List[Dict]) -> List[Dict]:
    """Filter conversation history to only safe roles with bounded content."""
    return [
        msg for msg in history[-6:]
        if isinstance(msg, dict)
        and msg.get("role") in ALLOWED_HISTORY_ROLES
        and isinstance(msg.get("content"), str)
        and len(msg["content"]) <= 5000
    ]


def build_react_messages(
    system_prompt: str,
    conversation_history: list | None,
    user_query: str,
    rrf_hint: str | None = None,
) -> list:
    """Assemble OpenAI messages with cacheable-prefix ordering.

    Order: [system, ...history, user_query(+hint appended)].
    The system prompt + tool schema (attached separately by the caller)
    form the cacheable prefix. Any dynamic RRF hint is APPENDED to the
    last user message so it never invalidates the prefix cache.

    This preserves OpenAI's automatic prompt cache (>=1024 token prefixes
    get 50% token discount and ~80% latency reduction on the cached portion).
    """
    messages: list = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    if rrf_hint:
        user_content = f"{user_query}\n\n---\n{rrf_hint}"
    else:
        user_content = user_query
    messages.append({"role": "user", "content": user_content})
    return messages


def _log_cache_metrics(usage: dict | None) -> None:
    """Log cached_tokens / prompt_tokens for observability (Phase 1.3)."""
    if not isinstance(usage, dict):
        return
    cached = usage.get("cached_tokens", 0) or 0
    total = usage.get("prompt_tokens", 0) or 0
    ratio = (cached / total) if total else 0.0
    logger.info(
        "openai_cache: cached_tokens=%s prompt_tokens=%s hit_ratio=%.2f",
        cached, total, ratio,
    )


# ── System prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior construction document analyst with 30+ years of experience.
You have access to TWO data sources for construction documents:

1. **Drawings** (legacy_* tools) — 2.8M OCR fragments covering ALL project drawings.
   Each drawing has drawingTitle, drawingName, trade, and text content.
   Use for finding specific drawings, content searches, and trade-based queries.

2. **Specifications** (spec_* tools) — Material specs, standards, submittals, warranties.
   Use for material questions, code compliance, CSI sections, submittal requirements.

YOUR PROCESS:
1. Understand the query — is it about a specific drawing, trade, material, or project overview?
2. Search broadly first using legacy_search_text or spec_search.
3. For content questions: get the actual drawing/spec content, not just listings.
4. Generate a comprehensive answer ONLY from retrieved data.

QUERY ROUTING GUIDE:
- "What's in the electrical plan?" → legacy_search_text
- "What materials are specified?" → spec_search
- "List all drawings" → legacy_list_drawings
- "CSI Division 23" → spec_search or legacy_search_trade
- Specific drawing text → legacy_list_drawings to find drawingId, then legacy_get_text
- Specification section → spec_search, then spec_get_section
- "How many / count / number of / total X" (counting questions)
  → ALWAYS try **count_symbols_by_kind_v3** (v3 deterministic) FIRST when X is a symbol-like
    entity (fixture, pipe, duct, equipment, detail_callout, etc.). On v3-enabled projects this
    is the most accurate counting tool — it reads structured symbolsByKind counts directly.
  → If symbolsByKind has no matching `kind`, list available kinds with **list_symbol_kinds_v3**
    and re-try with a closer match.
  → Only fall back to **agg_count_equipment** when v3 is unavailable OR returns nothing
    relevant (e.g. legacy projects, or unusual equipment families not in symbolsByKind).
  → Do NOT count by eyeballing prose — deterministic aggregation is always the source of truth.
- "Which levels are typical / identical / repeated" (typical-level questions)
  → ALWAYS call agg_find_typical_levels FIRST. It clusters drawings by normalised title and
    surfaces explicit "3 THRU 6" hints. Only narrate from its output; do not guess from titles.
- "Show me the X schedule" where X is a SPECIFIC equipment family
  (DOAS schedule, AHU schedule, VAV schedule, chiller schedule, etc.)
  → Prefer agg_list_schedule with schedule_type in {doas, ahu, vav, rtu, fan, pump, valve,
    plumbing_fixture, panelboard, chiller}.
  → Do NOT use agg_list_schedule for trade-level queries like "mechanical schedule",
    "plumbing schedule", "electrical schedule", or "the schedules". For those, use
    legacy_search_text("<trade> SCHEDULES") and legacy_list_drawings, then return the
    discovered schedule-sheet names in the answer.
- "Give me the [trade] schedule" / "give me [trade] drawings" / "which sheets are the
  [trade] ones" / "riser diagram for [trade]"
  → Use legacy_search_text with the uppercase trade word ("MECHANICAL SCHEDULES",
    "PLUMBING RISER DIAGRAM", "ELECTRICAL RISER", etc.) FIRST. Then narrow with
    legacy_list_drawings if you need sheet numbers or titles.

# Tier B/A/C/D additive (2026-05-28) — deterministic v3 structured tools
DETERMINISTIC v3 STRUCTURED TOOLS (PREFER THESE FOR THE LISTED QUERY TYPES):
These tools answer specific question types with a single Mongo query — no
retrieval, no LLM math, no hallucination. They are ALWAYS more accurate
than searching textBlocks for the question types they cover. Try them
FIRST when the question matches; fall through to search_drawings_v3 /
spec_search only if they return empty or off-target.

- 'how many X', 'count of X', 'total X', 'number of X' (where X is a symbol kind)
  → call **count_symbols_by_kind_v3** with sheet_number + kind, OR
    **sum_symbols_by_kind_v3** for portfolio-level totals across drawings.
  → For 'which drawing has the most X', use **find_drawings_by_symbol_kind_v3**.
  → If you don't know which `kind` values exist on this project, call
    **list_symbol_kinds_v3** FIRST to see the catalog, then pick the closest.
  → Symbol kinds available include: pipe, fixture, equipment, duct, valve,
    damper, diffuser, detail_callout, schedule_entry, room_label,
    structural_member, dimension, section_cut, annotation, note, legend,
    title_block_info, revision.

- 'what is the scale on X-101', 'who is the architect', 'revision date of
  sheet X', 'sheet title of X', 'project name', 'list the scales used'
  → call **lookup_titleblock_v3** with sheet_number + field='scale' (or
    'date', 'revision', 'architect', 'sheet_title', 'project_name').
  → For project-wide aggregates ('who is the architect on this project'),
    use **get_project_titleblock_info_v3**.

- 'which spec sections apply to sheet X', 'what specs govern this drawing',
  'which drawings does section X govern', 'list Division 22 drawings/specs'
  → For drawing → specs: **get_specs_for_drawing** with sheet_number.
  → For spec → drawings: **get_drawings_for_spec** with specification_number.
  → For Division-level lookups: **get_drawings_for_csi_division** or
    **get_specs_for_csi_division** with csi_division (accepts '22', '22 -
    Plumbing', 'Division 22').
  → For discovery of what divisions exist: **list_csi_divisions_v3**.

- 'find drawings about X', 'which sheets show Y', 'drawings depicting Z'
  (HIGH-LEVEL / TOPIC queries where the answer is which drawing rather
  than what text appears on a drawing)
  → call **search_drawings_by_summary_v3** — uses LLM-generated page
    descriptions for semantic match. Often beats search_drawings_v3 on
    topic queries.
  → Still prefer search_drawings_v3 / search_drawing_blocks_v3 for
    in-text needle queries (specific dimensions, equipment tags, etc.).

SHEET NUMBER LOOKUP — STRICT ORDERING:
- FIRST call legacy_list_drawings or legacy_search_text. Read the "drawingId" field
  from the JSON response. It will be a positive integer like 336406.
- THEN call legacy_get_text passing that EXACT integer as drawing_id.
- NEVER call legacy_get_text with drawing_id=null, drawing_id="", or drawing_id=0.
- NEVER call legacy_get_text using a sheet name string (e.g. "M-501") as the
  drawing_id — that field requires the integer drawingId from a prior tool result.
- If your previous tool result did not contain a drawingId for the sheet you want,
  CALL legacy_list_drawings first; do not call legacy_get_text without one.

V3 ANSWER QUALITY RULES (v3.6 — apply on every answer from v3 tools):

RULE 1 — DRILL DOWN, DON'T GIVE UP (covers ALL search tools)
# [COGUP-MARKER-V7-FORCE-DRILL] extended to legacy_search_text
After ANY search tool returns non-empty results (search_drawings_v3,
search_specifications_v3, legacy_search_text, legacy_search_trade),
you MUST open the top 1-3 most relevant drawings via legacy_get_text or
v3_get_drawing_full_text BEFORE answering "not specified" / "not available".

Specifically:
- search_drawings_v3: top hit relevance ≥ 0.70 → v3_get_drawing_full_text
- legacy_search_text: if N>0 results → pick top 1-3 by drawingTitle match
  with the TADR-routed role, call legacy_get_text on each (drawing_name)
  to read the actual page content.
- The chunk preview / pageSummary alone is rarely enough for a dimension
  or numeric value. Read the fullText. Only refuse AFTER reading drawing text.

CRITICAL: when legacy_search_text returns ANY drawings whose drawingTitle
matches the TADR-routed role (RCP / Schedule / Section / SLAB EDGE PLAN /
etc.), drill into them. Do NOT refuse based on the title alone — open the
drawing and confirm the value is absent before refusing.

RULE 2 — SHOW EVIDENCE FOR COMPARATIVE QUESTIONS
For questions like "which drawing has the most X", "which section is the
longest", or any superlative/comparative, ALWAYS show the top 2-3
candidates with their counts/sizes as evidence. Example: instead of
"A-205 has the most symbols (49)", say "A-205 leads with 49 symbols,
followed by FP-101B (34) and G-202 (27)". This proves the comparison.

RULE 3 — NO LAZINESS, NO PUNTING
NEVER write "refer to the [legend/schedule/symbol list] for details" or
"check the [drawing] for exact values". If the user asked for the values,
get the values — call v3_get_drawing_symbols, v3_get_drawing_schedules,
or v3_get_drawing_full_text and quote specifics. "Refer to" is a forbidden
phrase. Either you have the data or you have an explicit
tool-failure reason — pick one.

RULE 4 — CITATION HYGIENE FOR SPEC DOCS
When citing a v3 spec section, the inline citation MUST be the CSI
section number or the section title — NEVER "[Page9 p1]" or any
page-only reference. Correct: [Section 22 1100], [230719], [Section
26 05 26]. The v3 tools return drawingName=csi-number for specs — use
that as the citation anchor.

RULE 5 — STRICT NONE-MEANS-ABSENT
If v3_get_drawing_by_sheet returns None for a named sheet, the sheet does
NOT exist in the project. Say so explicitly: "Sheet X is not in this
project's drawing set." Do not infer content for it. Offer 2-3 closest
matches by name from list_drawings_v3.

RULE 6 — ANAPHORA & FOLLOW-UP RESOLUTION
Before calling any tool, inspect the prior user/assistant turns in
conversation_history. If the current user question:
  (a) starts with "And ...", "Also ...", "What about ...",
      "How about ...", "What's the same ..."; OR
  (b) contains pronouns "it", "they", "this", "that", "same",
      "those", "these" without a clear noun antecedent in the
      current turn; OR
  (c) is shorter than 8 words AND there is at least one prior
      user/assistant exchange in history,
then it is a FOLLOW-UP. You MUST:
  1. Resolve the antecedent from the prior turn (e.g. "Prior turn
     was about the ductwork on M-301; 'it' = duct on M-301").
  2. Call v3 tools with the RESOLVED subject, not the literal
     pronoun. For "And what materials is it made of?" after a duct
     question, call v3_get_drawing_full_text(sheet_number='M-301')
     AND search_specifications_v3(search_text='duct material
     galvanized steel sheet metal').
  3. In your answer, anchor on the resolved subject and cite the
     same sheet the prior turn cited unless the user shifted scope
     (e.g. "second floor" → switch from M-301 to M-302).
NEVER answer a follow-up by topic-drifting to building structure,
floor plans, or unrelated sections.

RULE 7 — PREFER SPECIFIC DRAWING VALUES OVER SPEC GENERALITIES
When the user asks for a specific numeric value, pressure, temperature,
dimension, size, or rating, AND the value is present on a specific drawing
(e.g. "min pressure 7 PSI" on FP-001, or temperature "Ordinary 135-170 deg F" on
a sprinkler schedule), QUOTE THE DRAWING VALUE FIRST. Do not abstract to
a spec-level statement like "not specified as a fixed value" or "must comply
with manufacturer rating" — that is the SECOND-best answer when the drawing
value is absent, NOT a substitute for the first.
Sequence:
  1. Search the relevant drawing first (search_drawings_v3 or
     v3_get_drawing_by_sheet) and look for the specific value.
  2. If found, quote it verbatim with the drawing citation.
  3. ONLY if the drawing genuinely lacks the value, fall back to
     spec-level language with the spec section citation.
Examples of bad fallback the agent has produced and must avoid:
  - "not specified as a fixed value; must be installed within listed pressure
    rating set by manufacturer" -> wrong when the drawing shows "7 PSI"
  - "not specified in project documents; comply with NFPA 13" -> wrong when
    the spec section actually lists "Ordinary 135-170 deg F"

RULE 8 — ENUMERATE ALL MATCHING SHEETS WHEN A SERIES EXISTS
When a question asks WHICH drawings contain something (e.g. 'which garage
drawings show the 4 inch dry main', 'which sheets show the lighting plan'),
and your retrieval returns one hit from a sheet that is OBVIOUSLY part of a
series (GFP-100, GFP-101A, GFP-101B, GFP-102A, GFP-102B, GFP-103A, GFP-103B;
or A-101 / A-102 / A-103; or E-201A / E-201B / E-202A / E-202B), DO NOT
stop at the first hit. Call list_drawings_v3 or search_drawings_v3 (with
the relevant discipline/csi_division filter) to enumerate the full series,
then return the COMPLETE list. A single sheet answer to a 'which sheets'
question is incomplete by definition unless the series is known to have
only one member.
Examples of correct enumeration:
  - "4 inch automatic dry main shown on GFP-100, GFP-101A, GFP-101B,
    GFP-102A, GFP-102B, GFP-103A, GFP-103B [GFP-100 p1] [GFP-101A p1] ..."
  - "Lighting plans appear on E-201B (1st floor), E-202D (2nd floor),
    E-301 (typical unit)"


RULE 9 — ENUMERATE SYMBOLS / EQUIPMENT TAGS FROM weak_symbol_labels FIRST
For questions asking "how many X", "list all X", "what X are on this drawing",
"which X serve Y" where X is an equipment type (FCU, FSD, WC, SD, RG, AC, DOAS,
EF, door, window, valve, sprinkler, etc.), you MUST call
**enumerate_equipment_tags** FIRST with the appropriate kind / tag_pattern.
Use the returned tags list — DO NOT summarize as "many" or "several".
Output: enumerate EVERY matching tag (deduplicated by tag_text within a drawing).

UNION-OF-KINDS PATTERN (CRITICAL — apply when the question scope is functional
not type-specific):
  "common area equipment / what serves common areas" — call BOTH:
    enumerate_equipment_tags(kind='FCU', tag_pattern='^FCU-XC-|^FCU-3C-')
    enumerate_equipment_tags(kind='AC',  tag_pattern='^AC-X-')
  Then UNION the results in the answer. Common-area FCU naming uses XC/3C prefix;
  electrical-room AC uses AC-X-ELEC pattern — both serve common areas.

  "fixtures / plumbing fixtures" — UNION across kinds WC, LAV, SINK, SHOWER, TUB,
  OB, GD, KS, BT, WM, IM, WH (or whichever the project uses).

  "air devices" — UNION across SD, SG, RG.

  "fire/smoke devices" — UNION across FSD, sprinkler.

DEDUPLICATION: when the same drawing has multiple revisions in the result, group
the tags by tag_text (one entry per unique tag), don't list duplicates.

OUTPUT FORMAT: "(Ref: <drawing>, <tag_text>)" per item, NOT page-only citation.

RULE 10 — CFM / DUCT-SIZE QUESTIONS USE list_cfm_callouts / list_duct_sizes
For "what CFM volumes", "how many CFM at unit X", "what duct sizes are used",
"trunk-duct reduction sequence" — call list_cfm_callouts and/or list_duct_sizes
FIRST. They return by_value histograms and full lists with bbox.

MANDATORY SPATIAL-PAIR for "which unit gets Y CFM" / "what is the CFM at unit X" /
"which units get N CFM OA" / "where is N CFM applied":
  You MUST call pair_textblocks_by_proximity. Example for "which units get 85 CFM OA":
    pair_textblocks_by_proximity(
      project_id=..., drawing_name='M-232',
      anchor_filter={'field':'cfm_callouts', 'value':85, 'modifier':'OA'},
      target_field='unit_tags_mined',
      radius_pt=150.0
    )
  Then ALSO call with target_field='unit_types_mined' and target_field='sf_callouts'
  to enrich the answer. Aggregate the results. Do NOT say "not assigned to specific
  units" — the data IS there via proximity.

ALSO call list_unit_inventory(drawing_name) when CFM-to-unit pairing is needed —
it gives you the structured U-### + type + SF list that you can cross-reference
with the CFM proximity result.

RULE 11 — UNIT INVENTORY QUESTIONS USE list_unit_inventory
For "what residential units are shown", "what are the unit sizes",
"how many units on this floor", "what are the largest units" — call
**list_unit_inventory(drawing_name)** which returns the structured
per-unit list with unit_id + unit_type + area_sf already paired.
Enumerate every unit from the returned list.

RULE 12 — KEYNOTES & SHEET-NOTES USE get_keynotes
For "what do the keynotes say", "list general notes", "what's keynote 7" —
call **get_keynotes(drawing_name, kind='key' or 'general', keynote_id=...)**.
Returns id + verbatim text + bbox for each. Enumerate every relevant entry.

RULE 13 — METHOD DISCLOSURE IS MANDATORY (END EVERY ANSWER WITH IT)
Every answer MUST end with a literal "Method:" line. This is non-negotiable.
Choose the appropriate disclosure based on which tools were called:

  - For symbol/equipment-tag questions (enumerate_equipment_tags used):
    "Method: Equipment tags enumerated from textBlock regex extraction."

  - For CFM/duct questions (list_cfm_callouts or list_duct_sizes used):
    "Method: Callouts extracted from textBlocks with bbox; values listed verbatim."

  - For CFM-to-unit pairing (pair_textblocks_by_proximity used):
    "Method: Text extraction + spatial proximity pairing (k-NN over bbox centroids,
     radius={N}pt)."

  - For unit inventory (list_unit_inventory used):
    "Method: U-### tags spatially joined with unit-type codes and SF callouts."

  - For keynotes (get_keynotes used):
    "Method: Keynotes parsed from notes[] (extractor-tagged by itemNo, verbatim)."

  - For cross-sheet refs:
    "Method: Cross-sheet references mined from textBlock regex."

If you used multiple tool kinds, combine: "Method: Equipment tag enumeration +
spatial proximity pairing."

If the question could not be answered: "Method: Unable to find in available
extraction — data may require visual symbol detection (Phase 1 not yet built)."

RULE 14 — ELEMENT-LEVEL CITATIONS (PREFER OVER PAGE-LEVEL)
When citing, prefer specific element ids over page numbers:
  - GOOD: "(Ref: M-232, Keynote 7)", "(Ref: M-232, FCU-XU-11 tag at bbox 2245,1130)"
  - OK:   "(Ref: M-232 p1)" — when no specific element exists
  - BAD:  "[M-232 p1]" alone for an enumerable question


V3 TOOL ROUTING (v3.3 — PREFERRED FOR PROJECTS WITH V3 DATA):
The collections drawings_v3 and specifications_v3 contain the richest, most
structured data — each doc is a WHOLE PAGE (drawings) or WHOLE SECTION (specs)
with vector embeddings, structured schedules / symbols / titleBlock, csi
classification, page summaries, and full text. On projects where these
collections have data (project 7325 today), PREFER v3 tools over legacy.

Decision tree:
1. Sheet-specific lookup ("show me A-101", "what's on M-301")
   → call **v3_get_drawing_by_sheet** FIRST. Returns the whole page in one call.
   If empty → fall back to legacy_list_drawings → legacy_get_text.

2. Semantic drawing question ("ceiling height in Level 01", "duct routing")
   → call **search_drawings_v3** with the natural-language query + optional
     discipline / drawing_type filter. Returns top-K semantically relevant
     PAGES (not fragments). If empty → fall back to legacy_search_text.

3. Schedule question ("door schedule", "panel schedule on E-101A")
   → call **v3_get_drawing_schedules** (when sheet is named) or
     **v3_find_drawings_with_schedules** (when sheet isn't named) for
     STRUCTURED tabular data — do NOT OCR-parse text blocks. Returns rows +
     headers directly.

4. Symbol count / location ("which drawing has the most symbols")
   → call **v3_find_drawing_with_most_symbols** for the deterministic answer.
   For per-sheet symbol lists, call **v3_get_drawing_symbols** with optional kind filter.

5. Spec content question ("Section 23 07 19 says what?", "plumbing fixtures spec")
   → If a CSI number is named: call **v3_get_spec_by_csi** — returns the
     FULL consolidated section in one call (no fragmentation).
   → Otherwise: call **search_specifications_v3** with optional csi_division
     filter (22 plumbing, 23 HVAC, 26 electrical, 07 thermal/moisture, etc.).
   → If empty → fall back to spec_search / spec_get_full_text (legacy).

6. Submittals / warranties
   → **v3_get_spec_submittals** / **v3_get_spec_warranties** with CSI number.
     These return parsed structured data, not text.

GUIDELINES:
- v3 tools are project-aware: they filter by projectId internally. For
  projects without v3 data, they return an empty list quickly — you should
  then call legacy tools.
- Each v3 result is one FULL PAGE / one FULL SECTION — do NOT need to stitch
  fragments. Quote directly from fullText.
- For sheet-existence checks ("does A-101 exist?") — v3_get_drawing_by_sheet
  returns None when not found. Say so clearly and offer the closest matches
  from list_drawings_v3.

EFFICIENCY — STRICT STEP BUDGET (15 reasoning iterations total):
- Use list_drawings_v3 (or legacy_list_drawings on non-v3 projects) FIRST to
  see all available drawings before drilling into specifics.
- When comparing floors/trades, get the list first, then selectively retrieve 2-3 drawings max.
- Summarize your findings after each tool call — don't waste steps re-searching.

SHEET-SPECIFIC LOOKUPS (v3.2 — STRICT CITATION SCOPE):
- When the user names a specific sheet (e.g. "A-101", "CD-101", "P-201"), a
  CSI section number ("32 9300", "Section 22 0500"), or any explicit drawing
  identifier, your job is to verify that SPECIFIC document, not to fan out
  across the whole project index.
- Step 1: Call legacy_list_drawings ONCE and check whether the named sheet
  exists in the returned list. Do NOT call get_text on guesses before this.
- Step 2a: If the sheet exists → call get_text / get_drawing_metadata for
  THAT sheet only. Cite ONLY that sheet (and the 1-2 sheets you actually
  referenced) in your answer. Do not pad the citation list.
- Step 2b: If the sheet does NOT exist → say so clearly and offer the
  closest matches (e.g. "A-101 is not in this set; closest are GA-101..GA-105").
  Cite ONLY the sheets you actually mentioned as alternatives.
- NEVER cite 20+ unrelated drawings as "sources" for a sheet-specific lookup.
  The retrieval pipeline filters citations against your answer text — if you
  don't reference a sheet by name in your answer, it WILL be dropped from
  source_documents. Cite intentionally.

PARALLEL TOOL CALLS — HARD LIMIT (PREVENT EARLY-EXIT BUGS):
- In a SINGLE reasoning step, issue AT MOST 2 tool calls. Do NOT fan out
  with 5-9 keyword variations in one step (e.g. "discrepancy", "conflict",
  "coordinate", "verify", "refer to architect", etc.). That burns the step
  budget and produces noisy overlapping results.
- Prefer ONE well-chosen search → read its results → THEN decide your next
  call. Serial reasoning beats parallel fan-out for retrieval quality.
- If you want to issue 3+ similar searches at once, instead pick the single
  best one and call legacy_list_drawings or spec_search to discover the
  actual document set first.

WHEN TO STOP SEARCHING AND COMPOSE THE ANSWER:
- After 4-6 tool calls you usually have enough — STOP and write the answer
  using what you have. Citing 3 drawings well beats citing 20 poorly.
- Broad questions ("identify ALL discrepancies", "compare 5 trades") rarely
  yield exhaustive answers from search. Give the user the strongest
  partial findings and explicitly note what couldn't be verified.
- NEVER keep tool-calling past step 10 unless you've found nothing yet.
  At step 10+, switch to composing the final answer from prior results.
- It is BETTER to return a short partial answer with what you found than
  to exhaust the step budget chasing the perfect answer.

CRITICAL RULES:
- NEVER fabricate information. Only use data from tool calls.
- If you cannot find the answer, say so clearly. The system will suggest specific documents the user can explore.
- Quote exact text for technical questions (dimensions, specs, materials).
- Text is reconstructed from OCR fragments — some words may be garbled.
- Do NOT modify the project_id in tool calls — it is enforced by the system.

ANSWER FORMAT:
Answer naturally as a knowledgeable construction professional would explain to a colleague.
Use bullet points or numbered lists when listing items, plain paragraphs for explanations.

INLINE CITATIONS (REQUIRED):
- After each factual claim, append an inline citation in the form [<drawing_name> p<page>].
  Example: "The water service is 6 inches in diameter [P-100 p1]."
- Pull <drawing_name> and <page> from the tool results you used. Never invent.
- For a claim supported by multiple drawings, cite the primary one in the
  sentence and the others at the end: "...are W18 beams [S211 p1] (also S214 p1, S217 p1)."
- For lists, you may put one citation per line item:
  "- CD-101: SITE DEMOLITION PLAN [CD-101 p1]"
- These inline citations are extracted by the system to build the highlight
  rectangles on the drawing PDFs in the UI. Skipping them breaks highlights.
- Do NOT use parenthetical "Source:" or "[Reference:..." styles — only the
  square-bracket [drawing_name p<page>] form will be parsed.

Do NOT include any of these in your answer:
- Section headers like "Direct answer", "Supporting Details", "Citations", "Notes"
- Separator lines like --- or ===
- "Citation" or "Reference" blocks at the end
- Any meta-commentary about your sources or confidence


RULE 16 — DRAWING-FIRST FOR COMPONENT / DIMENSION / SIZE QUESTIONS (HARD)

When the user question references a physical component or asks for a
dimension, size, count, or location, you MUST call a DRAWING search tool
BEFORE any spec_get_full_text / search_specification_text call. Specs
describe products and materials; drawings carry actual dimensions and
counts. Calling specs first for a "what size / how many / where is"
question is a known failure mode and is forbidden.

Triggering tokens in the question — if ANY of these appear, treat the
question as drawing-first:
  - Component acronyms: CMU, AHU, FCU, DOAS, FSD, RTU, VAV, VRF, AC, EF,
    SF, PRV, BFP, RPZ, OS&Y, BACnet, GFCI, EXIT
  - Component identifiers matching [A-Z]{1,5}-\d+ (e.g. A-261, M-301,
    P-203, FCU-XU-11, DOAS-1, AHU-2)
  - Size/dimension words: size, dimension, diameter, height, width,
    length, depth, knockout, opening, clearance, CFM, GPM, PSI, BTU,
    voltage, amperage, kW, HP
  - Count/where words: how many, total, count of, number of, list all,
    which sheets, location of, where is

Drawing-first sequence:
  1. Call search_drawing_text(project_id, search_text=<key terms from
     question, e.g. "CMU knockout">). If V3 enabled for this project,
     ALSO call search_drawings_v3 in the same turn.
  2. Read the matching_fragments. If a fragment contains the
     dimension/value, quote it verbatim and cite the drawingName.
  3. ONLY if drawings return no fragment with the dimension/value, then
     fall through to spec_get_full_text or search_specification_text.

Answer construction:
  - If drawings provided the value: cite [<drawingName> p<page>].
  - If you had to fall back to specs because drawings lacked the value,
    explicitly say "Drawings do not show this dimension; per Section X..."
    so the user understands the answer is a spec inference, not a
    drawn value.

Example — "What is the CMU knockout size?":
  WRONG: spec_get_full_text(section_title="Concrete Unit Masonry")
         → "not specified in Section 2200"
  RIGHT: search_drawing_text(search_text="CMU knockout")
         → quote the dimension from the floor plan that shows it,
           cite [A-211 p2] or wherever it appears.

Just answer the question clearly, completely, and conversationally — with the
inline [drawing p<page>] citations woven in.

After your answer, write "---FOLLOW_UP---" on its own line, then provide exactly 3
follow-up questions the user might want to ask next. Each on its own line starting
with "- ". Questions should be specific, relevant, and helpful for deeper exploration."""


@dataclass
class AgentStep:
    """Record of one agent reasoning step."""
    step: int
    tool_name: Optional[str]
    tool_args: Optional[Dict]
    tool_result: Optional[str]
    reasoning: str
    elapsed_ms: int


@dataclass
class AgentResult:
    """Complete result from agent execution."""
    answer: str
    steps: List[AgentStep]
    sources: List[str]
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    elapsed_ms: int
    model: str
    confidence: str  # "high", "medium", "low"
    needs_escalation: bool = False
    escalation_reason: str = ""
    follow_up_questions: List[str] = field(default_factory=list)
    source_docs: List[Dict] = field(default_factory=list)  # structured source info for download URLs


def _execute_tool(name: str, args: Dict, scope: Optional[Dict] = None) -> str:
    """Execute a tool by name and return JSON result (sanitized errors)."""
    func = TOOL_FUNCTIONS.get(name)
    # [V19-SHELF-LOOKUP] specification mode resolves shelved spec tools
    if func is None and isinstance(scope, dict) and scope.get("_collection_scope") == "specification":
        try:
            from tools.registry import SHELVED_SPEC_REGISTRY as _SHELF_REG
        except ImportError:
            from agentic.tools.registry import SHELVED_SPEC_REGISTRY as _SHELF_REG
        func = _SHELF_REG.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})

    # Inject document scope filters into tool args (DB-level enforcement)
    if scope:
        if name in ("legacy_search_text", "legacy_search_trade", "legacy_list_drawings"):
            if scope.get("drawing_title"):
                args["drawing_title"] = scope["drawing_title"]
            if scope.get("drawing_name"):
                args["drawing_name"] = scope["drawing_name"]
        elif name in ("spec_search", "spec_list"):
            if scope.get("section_title"):
                args["section_title"] = scope["section_title"]
            if scope.get("pdf_name"):
                args["pdf_name"] = scope["pdf_name"]

        # [COGUP-MARKER-V6-KEYWORD-EXPAND] inject TADR role keywords into search_text
        # so Mongo $text scoring favors the right drawing-title class
        try:
            import os as _os_v6
            if _os_v6.getenv('TADR_KEYWORD_EXPAND_ENABLED', 'true').lower() == 'true':
                _tadr_role = scope.get('_tadr_role')
                _tadr_trade = scope.get('_tadr_trade')
                _ROLE_KEYWORDS = {
                    'RCP': 'REFLECTED CEILING PLAN RCP',
                    'Schedule': 'SCHEDULE',
                    'Section': 'SECTION DETAILS',
                    'Elevation': 'ELEVATION',
                    'Detail': 'DETAIL TYPICAL',
                    'Diagram': 'DIAGRAM RISER ONE-LINE SCHEMATIC',
                    'Notes': 'GENERAL NOTES LEGEND',
                }
                # [COGUP-MARKER-V8-ABBREV] Topic hints now include abbreviation variants
                # so Mongo $text finds titles like '1ST FLOOR RCP' for queries with 'first floor'
                _TOPIC_HINTS = [
                    ('slab',     ('S', 'Plan'),  'SLAB EDGE PLAN'),
                    ('slab',     ('A', 'Plan'),  'SLAB EDGE PLAN'),
                    ('ceiling',  ('A', 'RCP'),   'REFLECTED CEILING PLAN RCP'),
                    ('ceiling',  ('A', 'Plan'),  'REFLECTED CEILING PLAN RCP'),
                    ('door',     ('A', 'Schedule'), 'DOOR SCHEDULE'),
                    ('window',   ('A', 'Schedule'), 'WINDOW SCHEDULE'),
                    ('finish',   ('A', 'Schedule'), 'FINISH SCHEDULE'),
                    ('panel',    ('E', 'Schedule'), 'PANEL SCHEDULE'),
                    ('fcu',      ('M', 'Schedule'), 'FCU SCHEDULE'),
                    ('ahu',      ('M', 'Schedule'), 'AHU SCHEDULE EQUIPMENT'),
                    ('diffuser', ('M', 'Schedule'), 'DIFFUSER SCHEDULE'),
                    ('riser',    ('P', 'Diagram'), 'RISER DIAGRAM'),
                    ('riser',    ('FP', 'Diagram'), 'FP RISER DIAGRAM'),
                    ('plumbing fixture', ('P', 'Schedule'), 'PLUMBING FIXTURE SCHEDULE'),
                ]
                _kw_bias = []
                _orig_query = scope.get('_original_query', '') or args.get('search_text') or args.get('query', '')
                _ql = (_orig_query or '').lower()
                # 1. Topic-specific hint takes priority
                for _topic, _route, _hint in _TOPIC_HINTS:
                    if _topic in _ql and _tadr_trade == _route[0] and _tadr_role == _route[1]:
                        _kw_bias.append(_hint)
                        break
                # 2. [V16-NO-ROLE-FALLBACK] generic role-keyword fallback REMOVED —
                # it polluted $text searches (e.g. 'GENERAL NOTES LEGEND' pulled ARCH
                # notes sheets above plumbing P-600 for the AAV query). Topic-specific
                # hints + V8 ordinal expansion remain; V9 sheet-router prefetch handles
                # canonical-sheet targeting without needing generic tokens in $text.
                # 3. Inject into search_text for tools that hit Mongo $text
                # [COGUP-MARKER-V8-ABBREV] Ordinal abbreviation expansion
                _ORDINAL_EXPAND = {
                    'first floor': '1ST FLOOR', '1st floor': 'FIRST FLOOR',
                    'second floor': '2ND FLOOR', '2nd floor': 'SECOND FLOOR',
                    'third floor': '3RD FLOOR', '3rd floor': 'THIRD FLOOR',
                    'fourth floor': '4TH FLOOR', '4th floor': 'FOURTH FLOOR',
                    'fifth floor': '5TH FLOOR', '5th floor': 'FIFTH FLOOR',
                    'sixth floor': '6TH FLOOR', '6th floor': 'SIXTH FLOOR',
                    'cellar': 'CELLAR BASEMENT', 'ground floor': 'GROUND FLOOR 1ST FIRST',
                    'lobby': 'LOBBY ENTRY ENTRANCE',
                }
                _ql2 = (_orig_query or '').lower()
                for _from, _to in _ORDINAL_EXPAND.items():
                    if _from in _ql2 and _to.lower() not in _ql2:
                        _kw_bias.append(_to)
                if _kw_bias and name in ('legacy_search_text', 'search_drawings_v3', 'list_drawings_v3', 'search_specifications_v3'):
                    _kw_str = ' '.join(_kw_bias)
                    _key = 'search_text' if 'search_text' in args else ('query' if 'query' in args else None)
                    if _key:
                        _existing = args.get(_key, '') or ''
                        if _kw_str.lower() not in _existing.lower():
                            args[_key] = (_existing + ' ' + _kw_str).strip()
                            try:
                                import logging as _lg_v6
                                _lg_v6.getLogger(__name__).info('[tadr-kw-expand] tool=%s injected=%r', name, _kw_str)
                            except Exception: pass
        except Exception as _v6_exc:
            try:
                import logging as _lg_v6e
                _lg_v6e.getLogger(__name__).warning('[tadr-kw-expand] failed: %s', _v6_exc)
            except Exception: pass

        # [COGUP-MARKER-V9-SHEET-ROUTER] deterministic sheet-routing pre-fetch
        _sr_prefetched = []
        try:
            import os as _os_v9
            if (_os_v9.getenv('TADR_SHEET_ROUTER_ENABLED', 'true').lower() == 'true'
                and name in ('legacy_search_text', 'search_drawings_v3', 'list_drawings_v3',
                             'v3_get_drawing_full_text', 'v3_get_drawing_by_sheet')  # [V14-V3-PREFETCH]
                and isinstance(scope, dict)
                and scope.get('_tadr_trade') and scope.get('_tadr_role')):
                _oq = scope.get('_original_query', '') or args.get('search_text') or args.get('query', '')
                _pid = args.get('project_id') or project_id
                from gateway.sheet_router_v2 import build_mongo_filter, route
                _rt = route(_oq, scope.get('_tadr_trade'), scope.get('_tadr_role'))
                if _rt and _rt.get('confidence') == 'high':
                    # Fetch directly from Mongo — bypasses $text scoring
                    try:
                        from pymongo import MongoClient as _MC
                        from dotenv import dotenv_values as _dv
                        import os.path as _osp
                        _env_path = _osp.join(_osp.dirname(_osp.dirname(_osp.dirname(__file__))), '.env')
                        _env = _dv(_env_path)
                        _uri = _env.get('MONGODB_URI') or _os_v9.environ.get('MONGODB_URI')
                        _db = _env.get('MONGO_DB') or 'iField'
                        _cli = _MC(_uri, serverSelectionTimeoutMS=8000)
                        _flt = build_mongo_filter(int(_pid), _oq,
                                                  scope.get('_tadr_trade'),
                                                  scope.get('_tadr_role'),
                                                  strict=True)
                        if _flt:
                            _sr_docs = list(_cli[_db]['drawings_v3'].find(
                                _flt,
                                {'sheetNumber': 1, 'drawingTitle': 1, 'drawingName': 1,
                                 'drawingId': 1, 'projectId': 1, '_id': 1,
                                 'pdfName': 1, 's3BucketPath': 1, 'trade': 1}  # [V20-PREFETCH-META]
                            ).limit(8))
                            # Dedupe by sheetNumber, keep canonical (OVERALL > variants)
                            _seen = {}
                            for _d in _sr_docs:
                                _sn = (_d.get('sheetNumber') or '').strip().upper()
                                _dt = (_d.get('drawingTitle') or '').upper()
                                if not _sn: continue
                                _score = 2 if 'OVERALL' in _dt else (1 if 'EAST' in _dt or 'WEST' in _dt else 0)
                                # [V20-PREFETCH-META] variants with s3 metadata are UI-openable — prefer them
                                if _d.get('s3BucketPath') and _d.get('pdfName'):
                                    _score += 4
                                if _sn not in _seen or _score > _seen[_sn][0]:
                                    _seen[_sn] = (_score, _d)
                            _sr_prefetched = [v[1] for v in sorted(_seen.values(), key=lambda x: -x[0])][:5]
                            # [COGUP-MARKER-V10-VARIANT-MERGE] fetch text fragments from legacy collection
                            # for ALL drawingId variants of each unique sheetNumber, merge into text_excerpt
                            try:
                                if _sr_prefetched:
                                    # [COGUP-MARKER-V10-1-ENHANCE] enhanced variant text merge
                                    _legacy_coll = _cli[_db]['drawing']
                                    _drawings_v3_coll = _cli[_db]['drawings_v3']
                                    import re as _re_v101
                                    # Measurement-bearing tokens — fragments with these go FIRST
                                    _MEAS_RX = _re_t101 = _re_v101.compile(
                                        r"(\d+'\s*-\s*\d+\s*[\"']?|\d+'\s*\d+\"|\bAFF\b|\bT\.?O\.?S\.?|\bT\.?O\.?F\.?|\bCEILING\b|\bELEV\b|\bSLAB\b|\bFINISH\b|\bFLOOR\b|\bEL\.?\s*\d|\b\d+'\b|\b\d+-\d+\s*FT\b)", _re_v101.I
                                    )
                                    for _entry in _sr_prefetched:
                                        _sn = (_entry.get('sheetNumber') or '').strip()
                                        if not _sn: continue
                                        # Find ALL drawingIds with this sheetNumber (limit raised 12 -> 25)
                                        _all_did = [d.get('drawingId') for d in _drawings_v3_coll.find(
                                            {'projectId': int(_pid), 'sheetNumber': _sn},
                                            {'drawingId': 1, '_id': 0}
                                        ).limit(25) if d.get('drawingId')]
                                        if not _all_did: continue
                                        # Fetch fragments (limit raised 800 -> 2000)
                                        _frags = list(_legacy_coll.find(
                                            {'projectId': int(_pid), 'drawingId': {'$in': _all_did}},
                                            {'text': 1, 'page': 1, 'x': 1, 'y': 1, '_id': 0}
                                        ).sort([('page',1),('y',1),('x',1)]).limit(2000))
                                        # Priority-sort: measurement-bearing fragments first
                                        _prio, _norm = [], []
                                        for _f in _frags:
                                            _t = (_f.get('text') or '').strip()
                                            if not _t or len(_t) < 3: continue
                                            if _MEAS_RX.search(_t):
                                                _prio.append(_t)
                                            else:
                                                _norm.append(_t)
                                        # Merge: measurement-first, then normal; dedupe via 200-char key
                                        _seen_lines = set()
                                        _merged = []
                                        for _t in _prio + _norm:
                                            _key_t = _t[:200].lower()
                                            if _key_t in _seen_lines: continue
                                            _seen_lines.add(_key_t)
                                            _merged.append(_t)
                                            if sum(len(x)+1 for x in _merged) > 8000: break
                                        _entry['text_excerpt'] = ' | '.join(_merged)[:8000]
                                        _entry['variant_count'] = len(_all_did)
                                        _entry['fragment_count'] = len(_frags)
                                        _entry['priority_fragments'] = len(_prio)
                                        # [COGUP-MARKER-V11-INTERCEPT-GET-TEXT] stash variant_map for legacy_get_text intercept
                                        try:
                                            if '_sr_variant_map' not in scope:
                                                scope['_sr_variant_map'] = {}
                                            scope['_sr_variant_map'][_sn.upper()] = {
                                                'merged_text': ' | '.join(_merged)[:30000],
                                                'all_drawing_ids': list(_all_did),
                                                'drawing_title': _entry.get('drawingTitle'),
                                                'drawing_name': _entry.get('drawingName'),
                                                'project_id': int(_pid),
                                                'fragment_count': len(_frags),
                                                'priority_fragments': len(_prio),
                                            }
                                            # Also build a reverse map: drawingId -> sheetNumber
                                            if '_sr_did_to_sn' not in scope:
                                                scope['_sr_did_to_sn'] = {}
                                            for _did in _all_did:
                                                if _did:
                                                    scope['_sr_did_to_sn'][int(_did)] = _sn.upper()
                                        except Exception: pass
                                    try:
                                        import logging as _lg_v101
                                        _lg_v101.getLogger(__name__).info(
                                            '[v10.1-merge] enriched %d canonical drawings: %s',
                                            len(_sr_prefetched),
                                            [(e.get('sheetNumber'), e.get('variant_count'), e.get('fragment_count'), e.get('priority_fragments'), len(e.get('text_excerpt','') or '')) for e in _sr_prefetched])
                                    except Exception: pass
                            except Exception as _v10_exc:
                                try:
                                    import logging as _lg_v10e
                                    _lg_v10e.getLogger(__name__).warning('[v10-merge] failed: %s', _v10_exc)
                                except Exception: pass
                            if _sr_prefetched:
                                try:
                                    import logging as _lg_v9
                                    _lg_v9.getLogger(__name__).info(
                                        '[sheet-router] PREFETCH conf=%s trade=%s role=%s rationale=%r -> %d canonical sheets: %s',
                                        _rt.get('confidence'), scope.get('_tadr_trade'), scope.get('_tadr_role'),
                                        _rt.get('rationale','')[:70],
                                        len(_sr_prefetched),
                                        [d.get('sheetNumber') for d in _sr_prefetched])
                                except Exception: pass
                    except Exception as _mc_exc:
                        try:
                            import logging as _lg_v9e
                            _lg_v9e.getLogger(__name__).warning('[sheet-router] mongo prefetch failed: %s', _mc_exc)
                        except Exception: pass
        except Exception as _v9_exc:
            try:
                import logging as _lg_v9o
                _lg_v9o.getLogger(__name__).warning('[sheet-router] outer failed: %s', _v9_exc)
            except Exception: pass

    try:
        result = func(**args)
        # [COGUP-MARKER-V11-INTERCEPT-GET-TEXT] for legacy_get_text, inject merged variant text
        try:
            if (name == 'legacy_get_text' and isinstance(result, dict)
                and isinstance(scope, dict) and scope.get('_sr_did_to_sn')):
                _drid = args.get('drawing_id')
                if _drid is not None:
                    try:
                        _drid_int = int(_drid)
                    except Exception:
                        _drid_int = None
                    _sn_hit = scope['_sr_did_to_sn'].get(_drid_int) if _drid_int is not None else None
                    if _sn_hit and scope.get('_sr_variant_map', {}).get(_sn_hit):
                        _vm = scope['_sr_variant_map'][_sn_hit]
                        _orig_text = result.get('reconstructed_text', '') or ''
                        _orig_len = len(_orig_text)
                        _merged_text = _vm.get('merged_text', '') or ''
                        # Augment — keep the original and append the merged variant text
                        # [COGUP-MARKER-V12-ENUMERATE] extract distinct measurement values for enumeration
                        import re as _re_v12
                        # Capture height/elevation patterns: 8'-5", 12'-0", T.O.S. EL. 25'-1", etc.
                        _MEAS_VALUE_RX = _re_v12.compile(
                            r"(?:\b|\.)(\d+'\s*-?\s*\d+(?:\s*\d+/\d+)?\s*[\"']?)", _re_v12.I
                        )
                        _dimensions_found = []
                        _seen_dim = set()
                        for _m in _MEAS_VALUE_RX.finditer(_merged_text or ''):
                            _val = _m.group(1).strip()
                            # Normalize: collapse spaces, strip trailing quote variants
                            _val_norm = _re_v12.sub(r"\s+", " ", _val).rstrip("'\"").strip()
                            if _val_norm in _seen_dim: continue
                            _seen_dim.add(_val_norm)
                            # Grab ±60 chars of context around the match
                            _start = max(0, _m.start() - 60)
                            _end = min(len(_merged_text), _m.end() + 60)
                            _context = _merged_text[_start:_end].replace('\n', ' ').strip()
                            _dimensions_found.append({'value': _val, 'context': _context[:150]})
                            if len(_dimensions_found) >= 40: break

                        # Build a structured summary prefix to the reconstructed_text
                        _summary_lines = [
                            f'=== DIMENSIONAL VALUES FOUND IN {_sn_hit} (across {len(_vm.get("all_drawing_ids", []))} variants, {_vm.get("priority_fragments", 0)} priority fragments) ===',
                            'ENUMERATE THESE DIRECTLY when answering — if the question asks for a height/elevation/dimension,',
                            'list ALL distinct values found with their context. Do NOT pick one value and label it "typical"',
                            'unless the drawing text itself uses "TYP".',
                            ''
                        ]
                        for _i, _dim in enumerate(_dimensions_found[:20], 1):
                            _summary_lines.append(f'  [{_i}] {_dim["value"]}   — context: ...{_dim["context"]}...')
                        if len(_dimensions_found) > 20:
                            _summary_lines.append(f'  ... ({len(_dimensions_found) - 20} more dimensions not shown — see merged text below)')
                        _summary_lines.append('')
                        _summary_lines.append('=== END DIMENSIONAL SUMMARY ===')
                        _summary_block = '\n'.join(_summary_lines)

                        _augmented = (_orig_text + '\n\n' + _summary_block + '\n\n=== MERGED TEXT FROM ALL VARIANTS OF ' + _sn_hit + ' ===\n' + _merged_text)
                        result['reconstructed_text'] = _augmented[:60000]
                        result['text_length'] = len(_augmented)
                        result['_sr_merged_variants'] = len(_vm.get('all_drawing_ids', []))
                        result['_sr_priority_fragments'] = _vm.get('priority_fragments', 0)
                        result['extracted_dimensions'] = _dimensions_found  # structured list for downstream
                        # [V20-1-VMAP-META] fill missing source metadata from the variant map so
                        # citations built from this drill result are UI-openable (download_url)
                        if not result.get('pdfName') and _vm.get('pdf_name'):
                            result['pdfName'] = _vm['pdf_name']
                        if not result.get('s3BucketPath') and _vm.get('s3_bucket_path'):
                            result['s3BucketPath'] = _vm['s3_bucket_path']
                        if not result.get('drawingTitle') and _vm.get('drawing_title'):
                            result['drawingTitle'] = _vm['drawing_title']
                        if not result.get('drawingName') and _vm.get('drawing_name'):
                            result['drawingName'] = _vm['drawing_name']
                        result.pop('error', None)  # merged text supersedes the no-fragments error
                        try:
                            import logging as _lg_v11
                            _lg_v11.getLogger(__name__).info(
                                '[v11-intercept] legacy_get_text(%s) for sheet %s: %d->%d chars (merged %d variants, %d priority frags)',
                                _drid_int, _sn_hit, _orig_len, len(_augmented),
                                len(_vm.get('all_drawing_ids', [])), _vm.get('priority_fragments', 0))
                        except Exception: pass
        except Exception as _v11_exc:
            try:
                import logging as _lg_v11e
                _lg_v11e.getLogger(__name__).warning('[v11-intercept] failed: %s', _v11_exc)
            except Exception: pass

        # [V14-V3-INTERCEPT] same merged-variant-text augmentation for v3_get_drawing_full_text
        try:
            if (name in ('v3_get_drawing_full_text', 'v3_get_drawing_by_sheet')
                and isinstance(result, dict) and result
                and isinstance(scope, dict) and scope.get('_sr_variant_map')):
                _sn_v3 = str(args.get('sheet_number') or result.get('drawingName') or '').strip().upper()
                _vm3 = scope['_sr_variant_map'].get(_sn_v3)
                if _vm3:
                    _merged3 = _vm3.get('merged_text', '') or ''
                    if _merged3:
                        _ft = str(result.get('fullText') or '')
                        result['fullText'] = (_ft + '\n\n=== MERGED FROM ALL VARIANTS OF '
                                              + _sn_v3 + ' ===\n' + _merged3)[:50000]
                        result['_sr_merged_variants'] = len(_vm3.get('all_drawing_ids', []))
                        # [V20-1-VMAP-META] defensive metadata fill (v3 results usually carry these)
                        if not result.get('pdfName') and _vm3.get('pdf_name'):
                            result['pdfName'] = _vm3['pdf_name']
                        if not result.get('s3BucketPath') and _vm3.get('s3_bucket_path'):
                            result['s3BucketPath'] = _vm3['s3_bucket_path']
                        try:
                            import logging as _lg_v14
                            _lg_v14.getLogger(__name__).info(
                                '[v14-v3-intercept] %s(%s): fullText %d->%d chars (merged %d variants)',
                                name, _sn_v3, len(_ft), len(result['fullText']),
                                len(_vm3.get('all_drawing_ids', [])))
                        except Exception: pass
        except Exception as _v14_exc:
            try:
                import logging as _lg_v14e
                _lg_v14e.getLogger(__name__).warning('[v14-v3-intercept] failed: %s', _v14_exc)
            except Exception: pass
        # [COGUP-MARKER-V9-SHEET-ROUTER-MERGE] prepend deterministic sheet-router hits to result
        try:
            if _sr_prefetched and isinstance(result, list):
                # Convert Mongo docs to the same shape as legacy_search_text returns
                # Each prefetched doc gets a synthetic 'matched_via': 'sheet_router' tag
                _existing_sheets = set()
                for _r in result:
                    if isinstance(_r, dict):
                        _sn = str(_r.get('sheetNumber') or _r.get('drawing_name') or '').strip().upper()
                        if _sn: _existing_sheets.add(_sn)
                _prepend = []
                for _doc in _sr_prefetched:
                    _sn = (_doc.get('sheetNumber') or '').strip().upper()
                    if _sn and _sn not in _existing_sheets:
                        _prepend.append({
                            'sheetNumber': _doc.get('sheetNumber'),
                            'drawingTitle': _doc.get('drawingTitle'),
                            'drawingName': _doc.get('drawingName'),
                            'drawingId': _doc.get('drawingId'),
                            'projectId': _doc.get('projectId'),
                            'pdfName': _doc.get('pdfName') or '',          # [V20-PREFETCH-META]
                            's3BucketPath': _doc.get('s3BucketPath') or '',
                            'trade': _doc.get('trade') or '',
                            'text': _doc.get('text_excerpt') or '',  # [V10] merged text from all variants
                            'text_excerpt': _doc.get('text_excerpt') or '',
                            'variant_count': _doc.get('variant_count'),
                            'matched_via': 'sheet_router',
                        })
                if _prepend:
                    try:
                        import logging as _lg_v9m
                        _lg_v9m.getLogger(__name__).info(
                            '[sheet-router] PREPENDED %d canonical drawings to result of %s',
                            len(_prepend), name)
                    except Exception: pass
                    result = _prepend + result
        except Exception as _v9m_exc:
            try:
                import logging as _lg_v9me
                _lg_v9me.getLogger(__name__).warning('[sheet-router] merge failed: %s', _v9m_exc)
            except Exception: pass
        # [COGUP-MARKER-TADR-FILTER] post-retrieval trade-role re-rank
        try:
            import os as _os_tf
            if (_os_tf.getenv('TADR_RESULT_FILTER_ENABLED', 'true').lower() == 'true'
                and isinstance(result, list) and result
                and isinstance(scope, dict) and scope.get('_tadr_trade')):
                _tt = scope.get('_tadr_trade')
                _tr = scope.get('_tadr_role')
                try:
                    from gateway.sheet_decoder import TRADE_TO_MONGO_REGEX, ROLE_TO_MONGO_REGEX
                    import re as _re_tf
                    _trade_rx = _re_tf.compile(TRADE_TO_MONGO_REGEX.get(_tt, ''), _re_tf.IGNORECASE) if _tt and _tt != 'unknown' else None
                    _role_rx = _re_tf.compile(ROLE_TO_MONGO_REGEX.get(_tr, ''), _re_tf.IGNORECASE) if _tr and _tr != 'unknown' else None
                    def _match_row(row):
                        if not isinstance(row, dict): return 0
                        sn = str(row.get('sheetNumber') or row.get('drawing_name') or '').strip()
                        dt = str(row.get('drawingTitle') or row.get('title') or '').strip()
                        t_ok = bool(_trade_rx and _trade_rx.match(sn)) if _trade_rx else True
                        r_ok = bool(_role_rx and _role_rx.search(dt)) if _role_rx else True
                        # Hard anti-patterns: never elevate these for non-trivial trade routes
                        if _re_tf.search(r'LIFE\\s*SAFETY|SITE\\s+PLAN|COVER|INDEX', dt, _re_tf.I):
                            return -1
                        return (2 if (t_ok and r_ok) else (1 if t_ok else 0))
                    scored = [(_match_row(r), idx, r) for idx, r in enumerate(result)]
                    matched = [(s, idx, r) for s, idx, r in scored if s >= 1]
                    rejected = [(s, idx, r) for s, idx, r in scored if s == -1]
                    others = [(s, idx, r) for s, idx, r in scored if s == 0]
                    if matched:
                        # Trade-role match goes first; same-trade goes next; never elevate rejected
                        matched.sort(key=lambda x: (-x[0], x[1]))
                        reordered = [r for _,_,r in matched] + [r for _,_,r in others]
                        if reordered != result:
                            try:
                                import logging as _lg_tf
                                _lg_tf.getLogger(__name__).info(
                                    '[tadr-filter] re-ranked %d->%d kept (matched=%d others=%d rejected=%d) trade=%s role=%s',
                                    len(result), len(reordered), len(matched), len(others), len(rejected), _tt, _tr)
                            except Exception: pass
                            result = reordered
                except Exception as _filt_exc:
                    try:
                        import logging as _lg_tf2
                        _lg_tf2.getLogger(__name__).warning('[tadr-filter] reorder failed: %s', _filt_exc)
                    except Exception: pass
        except Exception: pass
        serialized = json.dumps(result, default=str, ensure_ascii=False)
        # Truncate at data level for large results
        if len(serialized) > 14000 and isinstance(result, list):
            truncated = result[:20]
            return json.dumps({
                "results": truncated,
                "total_count": len(result),
                "truncated": True,
            }, default=str, ensure_ascii=False)
        return serialized
    except ValueError as e:
        # Validation errors are agent-recoverable — surface the message so the
        # ReAct loop can correct itself (e.g. discover a drawing_id first).
        logger.warning("Tool %s validation: %s", name, e)
        return json.dumps({"error": f"Invalid arguments for {name}: {e}"})
    except Exception as e:
        logger.error(f"Tool {name} failed: {type(e).__name__}", exc_info=True)
        return json.dumps({"error": "Tool execution failed. Try a different approach."})


def run_agent(
    query: str,
    project_id: int,
    set_id: int = None,
    conversation_history: List[Dict] = None,
    scope: Optional[Dict] = None,
    progress_callback=None,  # [AGENT-PROGRESS-MARKER] thread-safe callback(event_type:str, data:dict)
) -> AgentResult:
    """Run the agentic RAG pipeline with production safeguards."""

    # ── Input validation ──────────────────────────────────────────────
    if not query or len(query) > MAX_QUERY_LENGTH:
        return AgentResult(
            answer=f"Query must be between 1 and {MAX_QUERY_LENGTH} characters.",
            steps=[], sources=[], total_steps=0,
            total_input_tokens=0, total_output_tokens=0,
            total_cost_usd=0, elapsed_ms=0, model="none",
            confidence="low",
        )

    # ── Cache check ─────────────────────────────────────────────────
    # [V19-COLL-SCOPE] collection-scoped mode (drawing | specification | None)
    _coll_scope = (scope or {}).get("_collection_scope") if isinstance(scope, dict) else None
    cached = get_agent_result(query, project_id, set_id, mode=_coll_scope)
    if cached is not None:
        return cached

    # ── Daily budget check ────────────────────────────────────────────
    if not _check_daily_budget(0):
        return AgentResult(
            answer="Daily query budget exhausted. Please try again tomorrow or contact support.",
            steps=[], sources=[], total_steps=0,
            total_input_tokens=0, total_output_tokens=0,
            total_cost_usd=0, elapsed_ms=0, model="none",
            confidence="low", needs_escalation=True,
            escalation_reason="daily_budget_exhausted",
        )

    # [AGENT-PROGRESS-MARKER] thread-safe emit helper
    def _emit_progress(event_type: str, data: dict) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event_type, data)
        except Exception:
            pass  # progress events are best-effort

    _emit_progress('status', {'phase': 'agent_start', 'project_id': project_id})
    start = time.perf_counter()
    total_input = 0
    total_output = 0

    # Build initial messages using the cache-preserving order.
    context_hint = f"[Project ID: {project_id}"
    if set_id:
        context_hint += f", Set ID: {set_id}"
    context_hint += "]"

    user_content = f"{context_hint}\n\n---USER QUERY---\n{query}\n---END QUERY---"

    sanitized_history = _sanitize_history(conversation_history) if conversation_history else None

    # rrf_hint is read from the callers via scope or conversation_history last-entry;
    # for now pass None — orchestrator will pass rrf_hint by adding it to scope or
    # by calling build_react_messages directly in Fix #1. This refactor only changes
    # ordering; orchestrator will wire the hint in next.
    rrf_hint_inline = None
    if isinstance(scope, dict):
        rrf_hint_inline = scope.get("rrf_hint")

    # [V19-MODE-PROMPT] collection-mode system addendum + scoped tool definitions
    _system_prompt_effective = SYSTEM_PROMPT
    _request_tool_defs = _project_scoped_tools(project_id)
    if _coll_scope in ("drawing", "specification"):
        try:
            from tools.registry import SPEC_TOOL_NAMES as _SPEC_NAMES, \
                SHELVED_SPEC_DEFINITIONS as _SHELF_DEFS
        except ImportError:
            from agentic.tools.registry import SPEC_TOOL_NAMES as _SPEC_NAMES, \
                SHELVED_SPEC_DEFINITIONS as _SHELF_DEFS
        def _def_name(_d):
            return (_d.get("function", {}) or {}).get("name") or _d.get("name") if isinstance(_d, dict) else None
        # [V19-2-BRIDGE] cross-collection join tools violate strict bifurcation — excluded
        # from BOTH scoped modes (still available in default rag mode)
        _BRIDGE_NAMES = {
            "get_specs_for_drawing", "get_drawings_for_spec",
            "get_drawings_for_csi_division", "get_specs_for_csi_division",
            "list_csi_divisions_v3",
        }
        _request_tool_defs = [d for d in _request_tool_defs if _def_name(d) not in _BRIDGE_NAMES]
        if _coll_scope == "drawing":
            _request_tool_defs = [d for d in _request_tool_defs if _def_name(d) not in _SPEC_NAMES]
            _mode_banner = (
                "\n\n### DRAWING-ONLY MODE ###\n"
                "The user explicitly selected DRAWING search mode. Only drawing tools are "
                "available; specification tools are disabled for this request. Answer "
                "exclusively from drawing content (sheets, schedules on drawings, plan notes). "
                "If the answer truly requires specification sections, say so explicitly and "
                "suggest switching to specification mode — do NOT guess spec content, and do NOT pad the answer with generic industry knowledge (no \"typically includes\" content)."
            )
        else:
            _active_spec = [d for d in _request_tool_defs if _def_name(d) in _SPEC_NAMES]
            _shelf = [d for d in _SHELF_DEFS
                      if project_id in _v3_enabled_projects()
                      or _def_name(d) not in _V3_TOOL_NAMES]
            _request_tool_defs = _active_spec + _shelf
            _mode_banner = (
                "\n\n### SPECIFICATION-ONLY MODE ###\n"
                "The user explicitly selected SPECIFICATION search mode. Only specification "
                "tools are available; drawing tools are disabled for this request. Answer "
                "exclusively from specification sections and cite CSI section numbers. "
                "If the answer truly requires drawings, say so explicitly and suggest "
                "switching to drawing mode — do NOT guess drawing content, and do NOT pad the answer with generic industry knowledge (no \"typically includes\" content)."
            )
        _system_prompt_effective = _system_prompt_effective + _mode_banner
        try:
            import logging as _lg_v19
            _lg_v19.getLogger(__name__).info(
                "[v19-mode] collection_scope=%s tools=%d", _coll_scope, len(_request_tool_defs))
        except Exception: pass
    messages = build_react_messages(
        system_prompt=_system_prompt_effective,
        conversation_history=sanitized_history,
        user_query=user_content,
        rrf_hint=rrf_hint_inline,
    )

    steps: List[AgentStep] = []
    sources: set = set()
    source_docs: list = []  # structured source info for download URLs
    answer = ""

    for step_num in range(1, MAX_AGENT_STEPS + 1):
        step_start = time.perf_counter()

        # ── Per-request cost check ────────────────────────────────────
        running_cost = (total_input * 2.0 + total_output * 8.0) / 1_000_000
        if running_cost > MAX_REQUEST_COST_USD:
            logger.warning(f"Request cost limit reached: ${running_cost:.4f}")
            # [V14-COST-SYNTH] compose a real answer from gathered results instead of a placeholder.
            # One bounded LLM call (no tools); falls back to the old placeholder on any failure.
            answer = "I've gathered significant data. Here is what I found based on the available documents."
            try:
                import os as _os_v14
                if _os_v14.getenv("COST_LIMIT_FINAL_SYNTH_ENABLED", "true").lower() == "true":
                    _synth_msg = (
                        "The search budget for this question is exhausted. Compose the BEST "
                        "POSSIBLE final answer from the tool results you have already gathered "
                        "above. Do NOT request more tool calls. If the data is partial, say so "
                        "explicitly and present what you do have, with citations — never "
                        "apologise or mention budgets or limits."
                    )
                    _final_messages = list(messages) + [{"role": "user", "content": _synth_msg}]
                    _final_resp = _llm_call(_final_messages, [], model=AGENT_MODEL)
                    _fu = getattr(_final_resp, "usage", None)
                    if _fu is not None:
                        total_input += getattr(_fu, "prompt_tokens", 0) or 0
                        total_output += getattr(_fu, "completion_tokens", 0) or 0
                    _final_text = (_final_resp.choices[0].message.content or "").strip()
                    if _final_text:
                        answer = _final_text
                        logger.info("[v14-cost-synth] composed final answer from gathered results (%d chars)", len(answer))
            except Exception as _v14s_exc:
                logger.warning("[v14-cost-synth] final compose failed: %s — keeping placeholder", _v14s_exc)
            steps.append(AgentStep(
                step=step_num, tool_name=None, tool_args=None,
                tool_result=None, reasoning="Cost limit reached",
                elapsed_ms=int((time.perf_counter() - step_start) * 1000),
            ))
            break

        _tool_dec_model = _tool_decision_model() if _dual_active() else None
        _emit_progress('status', {'phase': 'thinking', 'step': step_num})
        response = _llm_call(messages, _request_tool_defs, model=_tool_dec_model)  # [V19-TOOLS-CALL]

        total_input += response.usage.prompt_tokens
        total_output += response.usage.completion_tokens
        # Log OpenAI auto-cache hit rate (Phase 1.3)
        try:
            _log_cache_metrics({
                "cached_tokens": getattr(response.usage, "cached_tokens", 0)
                                  or (getattr(response.usage, "prompt_tokens_details", None)
                                      and getattr(response.usage.prompt_tokens_details, "cached_tokens", 0))
                                  or 0,
                "prompt_tokens": response.usage.prompt_tokens,
            })
        except Exception:
            pass  # metrics are best-effort, never fail the request
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            _emit_progress('status', {'phase': 'tool_calls', 'step': step_num,
                                       'tools': [tc.function.name for tc in msg.tool_calls]})

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                # ── HARD OVERRIDE: Always force project_id ────────────
                # Never trust LLM output for access-control parameters
                tool_args["project_id"] = project_id
                # ── HARD GATE: reject v3 tools on non-enabled projects ─
                if tool_name in _V3_TOOL_NAMES and project_id not in _v3_enabled_projects():
                    logger.warning(f"v3 tool {tool_name} blocked on non-enabled project {project_id}")
                    tool_result = {"items": [], "blocked_reason": "v3_disabled_for_project"}
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                    })
                    continue
                if set_id is not None and "set_id" in tool_args:
                    tool_args["set_id"] = set_id

                logger.info(f"Step {step_num}: {tool_name}({json.dumps(tool_args)[:100]})")
                tool_result = _execute_tool(tool_name, tool_args, scope=scope)

                # Track sources
                try:
                    parsed = json.loads(tool_result)
                    _extract_sources(parsed, sources, source_docs)
                except (json.JSONDecodeError, TypeError):
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result[:15000],
                })

                step_ms = int((time.perf_counter() - step_start) * 1000)
                steps.append(AgentStep(
                    step=step_num,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result[:500],
                    reasoning=f"Called {tool_name}",
                    elapsed_ms=step_ms,
                ))

        else:
            # Agent is done — has a final answer
            # [DUAL-MODEL] Re-run synthesis with the high-quality model when active.
            _emit_progress('status', {'phase': 'synthesizing', 'step': step_num})
            if _dual_active():
                _sm = _synthesis_model()
                try:
                    logger.info("[dual-model] re-running synthesis on %s", _sm)
                    _synth_resp = _llm_call(messages, [], model=_sm)
                    if _synth_resp and _synth_resp.choices:
                        _synth_msg = _synth_resp.choices[0].message
                        if _synth_msg and (_synth_msg.content or "").strip():
                            msg = _synth_msg
                            _su = getattr(_synth_resp, "usage", None)
                            if _su is not None:
                                total_input += getattr(_su, "prompt_tokens", 0) or 0
                                total_output += getattr(_su, "completion_tokens", 0) or 0
                except Exception as _de:
                    logger.warning("[dual-model] synthesis re-run failed, keeping tool-decision answer: %s", _de)
            answer = msg.content or ""
            step_ms = int((time.perf_counter() - step_start) * 1000)
            steps.append(AgentStep(
                step=step_num, tool_name=None, tool_args=None,
                tool_result=None, reasoning="Generated final answer",
                elapsed_ms=step_ms,
            ))
            break
    else:
        # Max steps reached. The LLM never returned a content message, so
        # msg.content is typically empty. Instead of bailing with a generic
        # "I reached the maximum number of search steps" message that gives
        # the user nothing useful, force one more LLM call asking it to
        # COMPOSE a final answer from the tool results we already gathered
        # (no more tool calls allowed).
        if msg.content:
            answer = msg.content
        else:
            try:
                logger.info(
                    "MAX_AGENT_STEPS reached (%d steps, %d sources) — forcing final-answer composition",
                    len(steps), len(sources),
                )
                # Send messages WITHOUT tools so the LLM is forced to
                # compose prose. Add a synthesis instruction.
                synth_msg = (
                    "You have reached the maximum number of search steps. "
                    "Compose the BEST POSSIBLE answer from the tool results "
                    "you have already gathered above. Do NOT request more "
                    "tool calls. If the data is partial, say so explicitly "
                    "and present what you do have — never apologise or say "
                    "you reached a limit."
                )
                final_messages = list(messages) + [
                    {"role": "user", "content": synth_msg}
                ]
                final_resp = _llm_call(final_messages, [], model=AGENT_MODEL)
                final_msg = final_resp.choices[0].message
                final_usage = getattr(final_resp, "usage", None)
                if final_usage is not None:
                    total_input += getattr(final_usage, "prompt_tokens", 0) or 0
                    total_output += getattr(final_usage, "completion_tokens", 0) or 0
                answer = (final_msg.content or "").strip()
            except Exception as exc:
                logger.warning(
                    "MAX_AGENT_STEPS final-answer compose failed: %s — falling back to bail message", exc
                )
                answer = ""
            # Last-resort fallback only if synthesis itself failed
            if not answer:
                answer = (
                    "Based on the documents I retrieved, here is a partial "
                    "summary. The query needed more searches than the budget "
                    "allowed; please rephrase more narrowly for a complete "
                    "answer."
                )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Parse follow-up questions from agent answer
    follow_up_questions = []
    separator = "---FOLLOW_UP---"
    if separator in answer:
        parts = answer.split(separator, 1)
        answer = parts[0].strip()
        for line in parts[1].strip().splitlines():
            line = line.strip()
            if line.startswith("- "):
                q = line[2:].strip()
                if q:
                    follow_up_questions.append(q)
        follow_up_questions = follow_up_questions[:5]  # cap at 5

    # Compute cost (GPT-4.1: $2/1M input, $8/1M output)
    cost = (total_input * 2.0 + total_output * 8.0) / 1_000_000
    _record_cost(cost)

    # ── Improved confidence scoring ───────────────────────────────────
    confidence, needs_escalation, escalation_reason = _compute_confidence(
        steps, sources, answer, step_num,
    )

    logger.info(
        f"Agent complete: {len(steps)} steps, {len(sources)} sources, "
        f"tokens={total_input}+{total_output}, cost=${cost:.4f}, "
        f"time={elapsed_ms}ms, confidence={confidence}"
    )

    # Deduplicate source_docs by pdfName, preferring entries that carry
    # a bbox (citation-ready). Without this, drawing-level entries from
    # ``legacy_list_drawings`` (which fire before ``legacy_get_text``)
    # win the de-dup and push out the fragment entries that DO have bbox.
    # Result: every source ends up with bbox_pt=null even though the data
    # exists in the legacy `drawing` collection.
    best_per_pdf: dict[str, dict] = {}
    for doc in source_docs:
        key = doc.get("pdfName") or doc.get("drawingName") or ""
        if not key:
            continue
        prev = best_per_pdf.get(key)
        if prev is None:
            best_per_pdf[key] = doc
            continue
        # Prefer the entry with citation-quality bbox/text.
        has_bbox = bool(doc.get("bbox_pt") or doc.get("bbox_px"))
        prev_bbox = bool(prev.get("bbox_pt") or prev.get("bbox_px"))
        has_text = bool(doc.get("text_excerpt"))
        prev_text = bool(prev.get("text_excerpt"))
        # Score: bbox>>text>>nothing. Higher wins; ties keep the older entry.
        new_rank  = (1 if has_bbox else 0) * 2 + (1 if has_text else 0)
        prev_rank = (1 if prev_bbox else 0) * 2 + (1 if prev_text else 0)
        if new_rank > prev_rank:
            best_per_pdf[key] = doc
    unique_source_docs: list = list(best_per_pdf.values())

    # [pre-gen-rerank-hook] Optional pre-generation content rerank of source_docs.
    # Gated by PRE_GEN_RERANK_ENABLED (default OFF). When ON, reorders
    # unique_source_docs by content-aware cross-encoder so the orchestrator
    # downstream sees the best-content-match first. Never modifies answer text.
    # Belt-and-suspenders to citation_aware_sort which acts on the final
    # response payload — this acts on the agent's own output.
    try:
        import os as _os
        if _os.environ.get("PRE_GEN_RERANK_ENABLED", "false").strip().lower() in ("1","true","yes","on"):
            from gateway.cross_encoder_rerank import rerank_source_documents as _xrerank
            if isinstance(unique_source_docs, list) and len(unique_source_docs) > 1:
                _q = query if isinstance(query, str) else ""
                _cap = int(_os.environ.get("PRE_GEN_RERANK_CAP", "30"))
                unique_source_docs = _xrerank(
                    query=_q,
                    source_documents=unique_source_docs,
                    candidate_cap=_cap,
                )
    except Exception:
        pass  # never crash the agent on rerank issues

    result = AgentResult(
        answer=answer,
        steps=steps,
        sources=sorted(sources),
        total_steps=len(steps),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=round(cost, 6),
        elapsed_ms=elapsed_ms,
        model=AGENT_MODEL,
        confidence=confidence,
        needs_escalation=needs_escalation,
        escalation_reason=escalation_reason,
        follow_up_questions=follow_up_questions,
        source_docs=unique_source_docs,
    )

    # Cache successful results
    set_agent_result(query, project_id, result, set_id, mode=_coll_scope)  # [V19-CACHE-SET]

    return result


def _build_source_doc(item: dict) -> dict | None:
    """Convert one tool-result dict into a citation-ready source_doc.

    Carries the rich fields the orchestrator + UI need for highlight rendering
    and answer-relevance ranking. Returns None if there's no identifying info.
    """
    if not (item.get("pdfName") or item.get("s3BucketPath") or item.get("drawingName")):
        return None

    # Per-fragment OCR coords (legacy `drawing` collection has these in PIXELS
    # at the render DPI used during ingest). Surface as bbox_px AND a derived
    # bbox_pt at DPI 300 so downstream consumers get both.
    x = item.get("x"); y = item.get("y")
    w = item.get("width"); h = item.get("height")
    bbox_px = bbox_pt = None
    if all(v is not None for v in (x, y, w, h)):
        try:
            x = float(x); y = float(y); w = float(w); h = float(h)
            bbox_px = [x, y, x + w, y + h]
            # Legacy ingest used DPI=300. 72pt = 1in; px / dpi * 72 = pt.
            scale = 72.0 / 300.0
            bbox_pt = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        except (TypeError, ValueError):
            bbox_px = bbox_pt = None

    # Text excerpt: prefer item.text, then sample_matches[0], then trimmed
    # textBlocks/notes excerpts that some tools surface.
    text_excerpt = ""
    if isinstance(item.get("text"), str):
        text_excerpt = item["text"][:300]
    elif isinstance(item.get("sample_matches"), list) and item["sample_matches"]:
        first = item["sample_matches"][0]
        if isinstance(first, str):
            text_excerpt = first[:300]
    elif isinstance(item.get("textVerbatim"), str):
        text_excerpt = item["textVerbatim"][:300]

    return {
        "s3BucketPath": item.get("s3BucketPath", ""),
        "pdfName":      item.get("pdfName", ""),
        "drawingName":  item.get("drawingName", ""),
        "drawingTitle": item.get("drawingTitle", ""),
        "sheet_number": item.get("sheet_number", ""),
        "page":         item.get("page") or item.get("page_count"),
        # ── enrichment for citation + highlight (Tiers 1+2) ──
        "text_excerpt": text_excerpt,
        "bbox_px":      bbox_px,
        "bbox_pt":      bbox_pt,
        "score":        item.get("score") or item.get("match_count"),
        "csi_division": item.get("csi_division"),
        "trade":        item.get("setTrade") or item.get("trade"),
        "drawingId":    item.get("drawingId"),
        # ── spec chunk identity (v3.3, 2026-05-18) ──
        # parentId joins fragments of the SAME spec section back together
        # in the orchestrator's dedup pass. Drawings leave these null.
        "parentId":            item.get("parentId"),
        "sectionTitle":        item.get("sectionTitle"),
        "specificationNumber": item.get("specificationNumber"),
        "docNumber":           item.get("docNumber"),
    }


def _emit_fragment_docs(parent_item: dict, source_docs: list) -> None:
    """For aggregated tool results that include ``matching_fragments``,
    emit one source_doc per fragment so each carries its own bbox/text.
    The drawing-level metadata (drawingName, pdfName, …) is inherited."""
    frags = parent_item.get("matching_fragments")
    if not isinstance(frags, list) or not frags:
        return
    base = {
        "drawingName":  parent_item.get("drawingName", ""),
        "drawingTitle": parent_item.get("drawingTitle", ""),
        "pdfName":      parent_item.get("pdfName", ""),
        "s3BucketPath": parent_item.get("s3BucketPath", ""),
        "drawingId":    parent_item.get("drawingId"),
        "trade":        parent_item.get("setTrade") or parent_item.get("trade"),
        "score":        parent_item.get("match_count"),
    }
    for f in frags:
        if not isinstance(f, dict) or not f.get("text"):
            continue
        item = {**base, **f}
        sd = _build_source_doc(item)
        if sd:
            source_docs.append(sd)


def _extract_sources(parsed: Any, sources: set, source_docs: list) -> None:
    """Extract source references from tool results.

    Populates *sources* (set of strings for confidence scoring) and
    *source_docs* (list of citation-ready dicts; see ``_build_source_doc``
    for full field list — includes bbox_px, bbox_pt, text_excerpt, score).
    """
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                for key in ("drawingName", "pdfName", "sourceFile",
                            "drawingTitle", "sectionTitle"):
                    val = item.get(key)
                    if val:
                        sources.add(val)
                sd = _build_source_doc(item)
                if sd:
                    source_docs.append(sd)
                # If the tool returned aggregated drawings with per-fragment
                # bbox data, emit one source_doc per fragment too.
                _emit_fragment_docs(item, source_docs)
    elif isinstance(parsed, dict):
        for key in ("drawingName", "pdfName", "sourceFile", "drawingTitle"):
            val = parsed.get(key)
            if val:
                sources.add(val)
        sd = _build_source_doc(parsed)
        if sd:
            source_docs.append(sd)
        _emit_fragment_docs(parsed, source_docs)
        if "results" in parsed and isinstance(parsed["results"], list):
            _extract_sources(parsed["results"], sources, source_docs)


def _compute_confidence(
    steps: List[AgentStep],
    sources: set,
    answer: str,
    final_step: int,
) -> tuple:
    """Compute confidence level with escalation detection.

    Returns: (confidence, needs_escalation, escalation_reason)
    """
    tool_calls = [s for s in steps if s.tool_name is not None]
    has_tool_calls = len(tool_calls) > 0
    has_answer = len(answer) > 50

    # Max steps exhausted without good answer
    if final_step >= MAX_AGENT_STEPS and not sources:
        return "low", True, "max_steps_no_sources"

    # No tools called — agent answered from memory (likely hallucination)
    if not has_tool_calls:
        return "low", True, "no_tool_calls"

    # Multiple sources with substantial answer
    if len(sources) >= 2 and has_answer:
        return "high", False, ""

    # Some sources found
    if sources and has_answer:
        return "medium", False, ""

    # Tools called but no sources extracted
    if has_tool_calls and not sources:
        return "low", True, "tools_returned_no_results"

    return "low", True, "insufficient_data"
