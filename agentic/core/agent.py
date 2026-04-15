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
5. Cite sources: [Source: drawingName / drawingTitle] for every fact.

QUERY ROUTING GUIDE:
- "What's in the electrical plan?" → legacy_search_text
- "What materials are specified?" → spec_search
- "List all drawings" → legacy_list_drawings
- "CSI Division 23" → spec_search or legacy_search_trade
- Specific drawing text → legacy_list_drawings to find drawingId, then legacy_get_text
- Specification section → spec_search, then spec_get_section

SHEET NUMBER LOOKUP:
- FIRST use legacy_list_drawings to find the drawingId for a sheet number.
- THEN use legacy_get_text with that drawingId to get the full text.
- ALWAYS search or list BEFORE trying to get content. Don't guess IDs.

EFFICIENCY:
- Use legacy_list_drawings FIRST to see all available drawings before drilling into specifics.
- When comparing floors/trades, get the list first, then selectively retrieve 2-3 drawings max.
- Summarize your findings after each tool call — don't waste steps re-searching.

CRITICAL RULES:
- NEVER fabricate information. Only use data from tool calls.
- If you cannot find the answer, say so clearly. The system will suggest specific documents the user can explore.
- Always cite which drawing/spec your information comes from.
- Quote exact text for technical questions (dimensions, specs, materials).
- Text is reconstructed from OCR fragments — some words may be garbled.
- Do NOT modify the project_id in tool calls — it is enforced by the system.

ANSWER FORMAT:
- Direct answer first
- Supporting details with exact quotes where relevant
- [Source: drawingName / drawingTitle] citations

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

    # Build initial messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(_sanitize_history(conversation_history))

    # Inject project context
    context_hint = f"[Project ID: {project_id}"
    if set_id:
        context_hint += f", Set ID: {set_id}"
    context_hint += "]"

    messages.append({
        "role": "user",
        "content": f"{context_hint}\n\n---USER QUERY---\n{query}\n---END QUERY---",
    })

    steps: List[AgentStep] = []
    sources: set = set()
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
                    _extract_sources(parsed, sources)
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
        # Max steps reached
        answer = msg.content or "I reached the maximum number of search steps. Here is what I found so far based on the available drawings."

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
    )

    # Cache successful results
    set_agent_result(query, project_id, result, set_id)

    return result


def _extract_sources(parsed: Any, sources: set) -> None:
    """Extract source file references from tool results."""
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                for key in ("drawingName", "pdfName", "sourceFile", "drawingTitle"):
                    val = item.get(key)
                    if val:
                        sources.add(val)
    elif isinstance(parsed, dict):
        for key in ("drawingName", "pdfName", "sourceFile", "drawingTitle"):
            val = parsed.get(key)
            if val:
                sources.add(val)
        if "results" in parsed and isinstance(parsed["results"], list):
            _extract_sources(parsed["results"], sources)


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
