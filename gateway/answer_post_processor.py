"""
Post-processor for final agent answers — runs AFTER stylize.

Two transforms (each preserves answer if conditions don't apply):
  1. inject_method_line(answer, agent_result):
       Append a one-line `Method:` disclosure based on which Tier-E tools were called.
       Skipped if answer already contains `Method:`.

  2. rewrite_citations(answer, agent_result):
       Convert page-level [M-232 p1] citations to (Ref: M-232) form. If the answer
       mentions a specific element (tag, keynote id, CFM value) right before the
       citation, attempt to enrich: (Ref: M-232, FCU-XU-11).

Hook from gateway/generation_chain.py just before _build_response_dict.
"""
from __future__ import annotations
import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger("agentic_rag.post_processor")


TIER_E_METHOD_LABELS = {
    "enumerate_equipment_tags": "equipment-tag enumeration",
    "list_cfm_callouts": "CFM callout extraction",
    "list_duct_sizes": "duct-size extraction",
    "get_keynotes": "verbatim keynote parsing",
    "pair_textblocks_by_proximity": "spatial proximity pairing (k-NN bbox)",
    "list_unit_inventory": "unit inventory (U-ID + type + SF spatial join)",
    "get_vlm_element_labels": "VLM element label retrieval from CKG",
    "get_cross_sheet_refs": "cross-sheet reference mining",
}

# Existing v3 tools we also want to disclose
V3_METHOD_LABELS = {
    "count_symbols_by_kind_v3": "symbols-by-kind count",
    "sum_symbols_by_kind_v3": "symbols-by-kind aggregation",
    "search_drawings_v3": "v3 hybrid retrieval",
    "search_drawings_by_summary_v3": "v3 pageSummary vector search",
    "v3_get_drawing_by_sheet": "v3 sheet lookup",
    "v3_get_drawing_schedules": "v3 schedule extraction",
    "get_specs_for_drawing": "drawing↔spec CSI cross-reference",
    "get_drawings_for_csi_division": "CSI division lookup",
    "lookup_titleblock_v3": "titleblock lookup",
    "search_drawing_blocks_v3": "block-level retrieval",
}

ALL_METHOD_LABELS = {**TIER_E_METHOD_LABELS, **V3_METHOD_LABELS}


def _safe_iter_steps(agent_result: Any):
    """Yield AgentStep objects from agent_result, gracefully if shape varies."""
    if agent_result is None:
        return
    steps = getattr(agent_result, "steps", None)
    if not steps:
        return
    for s in steps:
        yield s


def _tools_called(agent_result: Any) -> list:
    out = []
    for s in _safe_iter_steps(agent_result):
        name = getattr(s, "tool_name", None)
        if name:
            out.append(name)
    return out


def inject_method_line(final_answer: str, agent_result: Any) -> str:
    """Append a `Method:` line if not already present, based on tools called."""
    if not final_answer or not isinstance(final_answer, str):
        return final_answer
    if "Method:" in final_answer or "method:" in final_answer.lower()[-300:]:
        return final_answer

    tools = _tools_called(agent_result)
    if not tools:
        return final_answer

    # Pick labels for known tools, deduped + sorted for stability
    labels = sorted({ALL_METHOD_LABELS[t] for t in tools if t in ALL_METHOD_LABELS})
    if not labels:
        return final_answer

    if len(labels) == 1:
        method = f"Method: {labels[0]}."
    elif len(labels) == 2:
        method = f"Method: {labels[0]} + {labels[1]}."
    else:
        method = f"Method: {', '.join(labels[:-1])}, and {labels[-1]}."

    return final_answer.rstrip() + "\n\n" + method


# ── Citation rewriter ──────────────────────────────────────────────────

# Match page-level cites: [M-232 p1], [A-901 p1], [GFP-100 p1]
_PAGE_CITE = re.compile(r"\[([A-Z]{1,5}[-\.]?\d+[A-Z]?(?:[-\.]\d+)?)(\s*p\s*\d+)?\]")


def _collect_drawing_elements(agent_result: Any) -> dict:
    """Walk tool_results — collect drawing → [element refs] seen.

    Returns: { 'M-232': set of element_refs (tag_text or 'Keynote N' or 'CFM N OA') }
    """
    out: dict = {}
    for s in _safe_iter_steps(agent_result):
        tr = getattr(s, "tool_result", None)
        if not tr:
            continue
        try:
            data = json.loads(tr) if isinstance(tr, str) else tr
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue

        # enumerate_equipment_tags result shape
        for t in (data.get("tags") or []):
            d = t.get("drawing"); ref = t.get("tag")
            if d and ref:
                out.setdefault(d, set()).add(ref)

        # list_cfm_callouts shape
        for c in (data.get("callouts") or []):
            d = c.get("drawing"); ref = c.get("raw")
            if d and ref:
                out.setdefault(d, set()).add(ref)

        # list_duct_sizes shape
        for it in (data.get("items") or []):
            d = it.get("drawing"); ref = it.get("raw")
            if d and ref:
                out.setdefault(d, set()).add(ref)

        # get_keynotes shape
        for k in (data.get("keynotes") or []):
            d = k.get("drawing"); kid = k.get("id"); kind = k.get("kind", "key")
            if d and kid:
                tag = ("Keynote " if kind == "key" else "Note ") + str(kid)
                out.setdefault(d, set()).add(tag)

        # list_unit_inventory shape (single drawing)
        if data.get("units") and data.get("drawing"):
            d = data["drawing"]
            for u in data["units"]:
                uid = u.get("unit_id")
                if uid:
                    out.setdefault(d, set()).add(uid)

        # spatial pair result
        for p in (data.get("pairs") or []):
            d = data.get("drawing")
            a = p.get("anchor") or {}
            ref = a.get("raw") or a.get("tag_text")
            if d and ref:
                out.setdefault(d, set()).add(ref)

        # cross-sheet refs result
        if data.get("drawing") and data.get("by_target"):
            d = data["drawing"]
            for tgt, refs in (data["by_target"] or {}).items():
                for r in refs:
                    raw = r.get("raw")
                    if raw:
                        out.setdefault(d, set()).add(raw)

    return out


def rewrite_citations(final_answer: str, agent_result: Any) -> str:
    """Rewrite page-level [DRAWING pN] → (Ref: DRAWING) form.

    Element-level enrichment (e.g. adding "FCU-XU-11" inside the Ref) is
    intentionally NOT attempted post-hoc — without per-citation provenance
    in the answer, guessing which tag a specific [DRAWING pN] refers to is
    unreliable. The page-→Ref normalization alone matches the manager's
    expected format closer.
    """
    if not final_answer or not isinstance(final_answer, str):
        return final_answer

    # Collect drawing-element registry (for optional future enrichment)
    _collect_drawing_elements(agent_result)  # currently unused — placeholder

    def _repl(m):
        drawing = m.group(1)
        return f"(Ref: {drawing})"

    return _PAGE_CITE.sub(_repl, final_answer)


def post_process(final_answer: str, agent_result: Any) -> str:
    """Run all post-processors in order. Returns new final_answer."""
    try:
        out = final_answer
        out = rewrite_citations(out, agent_result)
        out = inject_method_line(out, agent_result)
        return out
    except Exception as e:
        logger.exception("post_process failed (returning original): %s", e)
        return final_answer
