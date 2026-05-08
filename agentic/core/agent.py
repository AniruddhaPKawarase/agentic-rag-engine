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
from core.cache import get_agent_result, set_agent_result
from tools.registry import TOOL_DEFINITIONS, TOOL_FUNCTIONS

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
  → ALWAYS call agg_count_equipment FIRST with the entity keyword (e.g. "DOAS", "AHU", "VALVE").
    Do NOT attempt to count by eyeballing prose — the deterministic aggregation is the source of truth.
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

SHEET NUMBER LOOKUP — STRICT ORDERING:
- FIRST call legacy_list_drawings or legacy_search_text. Read the "drawingId" field
  from the JSON response. It will be a positive integer like 336406.
- THEN call legacy_get_text passing that EXACT integer as drawing_id.
- NEVER call legacy_get_text with drawing_id=null, drawing_id="", or drawing_id=0.
- NEVER call legacy_get_text using a sheet name string (e.g. "M-501") as the
  drawing_id — that field requires the integer drawingId from a prior tool result.
- If your previous tool result did not contain a drawingId for the sheet you want,
  CALL legacy_list_drawings first; do not call legacy_get_text without one.

EFFICIENCY — STRICT STEP BUDGET (15 reasoning iterations total):
- Use legacy_list_drawings FIRST to see all available drawings before drilling into specifics.
- When comparing floors/trades, get the list first, then selectively retrieve 2-3 drawings max.
- Summarize your findings after each tool call — don't waste steps re-searching.

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

    try:
        result = func(**args)
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
    cached = get_agent_result(query, project_id, set_id)
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

    messages = build_react_messages(
        system_prompt=SYSTEM_PROMPT,
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
            answer = "I've gathered significant data. Here is what I found based on the available documents."
            steps.append(AgentStep(
                step=step_num, tool_name=None, tool_args=None,
                tool_result=None, reasoning="Cost limit reached",
                elapsed_ms=int((time.perf_counter() - step_start) * 1000),
            ))
            break

        response = _llm_call(messages, TOOL_DEFINITIONS)

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

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                # ── HARD OVERRIDE: Always force project_id ────────────
                # Never trust LLM output for access-control parameters
                tool_args["project_id"] = project_id
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
    set_agent_result(query, project_id, result, set_id)

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
                for key in ("drawingName", "pdfName", "sourceFile", "drawingTitle"):
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
