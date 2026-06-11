"""
Pillar 1 -- L1 Anaphora Resolver (minimal tactical-patch variant).

Per HYBRID_RAG_v32_ROADMAP.md section 5.4 Pillar 1. The FULL 4-layer rewriter
is P80 (v32). This tactical-patch variant is JUST the L1 anaphora resolver:
one Haiku call BEFORE retrieval that takes (current query, conversation_history)
and returns a self-contained query with pronouns and implicit references
resolved.

Examples (input -> output):
    history=[{role:user,content:"Show me FCU-101"}, {role:assistant,content:"FCU-101..."}]
    query="What's its CFM rating?"
    -> "What is the CFM rating of FCU-101?"

    history=[{role:user,content:"What's specified on sheet M-201?"}]
    query="And on the next sheet?"
    -> "What is specified on sheet M-202?"

When conversation_history is empty or only one turn deep, the resolver
short-circuits (returns the original query unchanged).

Cost: one Haiku 4.5 call per non-first-turn query, ~50-100 input tokens,
~30 output tokens => ~$0.0001 per call (essentially free).
Latency overhead: ~80-150ms p95.
Reversible: HG_PILLAR_1=false -> resolver no-ops.
"""

import logging
import os
from typing import Iterable, List, Optional

from .flags import is_pillar_enabled

logger = logging.getLogger(__name__)

PILLAR_NUMBER = 1

# The L1 resolver is constrained to a SMALL number of history turns.
# More than this and we cap to most-recent N (the further-back turns rarely
# carry coreference signal worth the latency).
MAX_HISTORY_TURNS = 6

# Resolver prompt -- short, focused, no chain-of-thought. Forces the model
# to return ONLY the rewritten query (no preamble, no explanation).
ANAPHORA_PROMPT_TEMPLATE = """You rewrite a follow-up user query into a self-contained query.

Conversation so far (most recent first):
{history_block}

Latest user query:
{query}

Rules:
1. Resolve pronouns and references (its, that, the next one, same, those, etc.) using the conversation history.
2. Preserve the user's intent verbatim where there is no anaphora.
3. Do NOT answer the query. Do NOT add explanation. Output ONLY the rewritten query.
4. If the query is already self-contained, output it unchanged.
5. Keep the rewritten query under 300 characters.

Rewritten query:"""


def _format_history_block(history: List[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Render the most-recent N turns as a compact block for the prompt."""
    if not history:
        return "(no prior turns)"
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        role = turn.get("role", "?")
        content = (turn.get("content") or "").strip()
        if len(content) > 240:
            content = content[:240] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "..."


def _has_anaphora_signal(query: str) -> bool:
    """Cheap signal: is there a pronoun or implicit reference?

    If not, the resolver short-circuits (saves a Haiku call). False
    negatives are fine -- the resolver will still no-op when the query
    is already self-contained.
    """
    q = query.lower()
    anaphora_markers = (
        " it ", " its ", "it.", "it?", "it,",
        " they ", " them ", " their ",
        " that ", " this ", " those ", " these ",
        " same ", " next ", " previous ", " prior ",
        " above ", " below ",
        " its ", " his ", " hers ",
        " also ", " too ",
        " what about ", " how about ", " and on ", " and the ",
    )
    return any(m in q for m in anaphora_markers) or query.endswith("?") and len(query) < 60


def resolve_anaphora(query: str,
                     history: Optional[Iterable[dict]] = None,
                     anthropic_client=None,
                     force_enable: Optional[bool] = None) -> str:
    """Rewrite a follow-up query into a self-contained one.

    Args:
        query: The current turn's user query.
        history: List of {role, content} dicts from prior turns. If None or
            empty, returns the original query (no anaphora possible).
        anthropic_client: Inject an Anthropic client for testing. If None,
            creates one lazily from ANTHROPIC_API_KEY env var.
        force_enable: For unit tests; bypasses flag check.

    Returns:
        The self-contained query (or the original if resolver short-circuits
        or fails). NEVER returns None or raises -- this is a best-effort
        resolver and must not break the /query path.
    """
    enabled = is_pillar_enabled(PILLAR_NUMBER) if force_enable is None else bool(force_enable)
    if not enabled:
        return query

    if not query or not query.strip():
        return query

    history_list = list(history) if history else []
    if len(history_list) < 1:
        # No prior turn -> no anaphora possible
        return query

    # Cheap signal check -- skip the Haiku call if no anaphora markers
    if not _has_anaphora_signal(query):
        logger.debug("Pillar 1: no anaphora signal in query; skip resolver")
        return query

    # Lazy-init Anthropic client
    if anthropic_client is None:
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning("Pillar 1: ANTHROPIC_API_KEY not set; skipping resolver")
                return query
            anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("Pillar 1: anthropic SDK not available; skipping resolver")
            return query
        except Exception as exc:
            logger.warning("Pillar 1: client init failed: %s; skipping resolver", exc)
            return query

    history_block = _format_history_block(history_list)
    prompt = ANAPHORA_PROMPT_TEMPLATE.format(
        history_block=_truncate(history_block, 1500),
        query=_truncate(query, 500),
    )

    try:
        msg = anthropic_client.messages.create(
            model=os.environ.get("HG_PILLAR_1_MODEL", "claude-haiku-4-5"),
            max_tokens=120,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic SDK >=0.30: content is a list of TextBlock; older: content[0].text
        rewritten = ""
        if msg.content and len(msg.content) > 0:
            block = msg.content[0]
            rewritten = getattr(block, "text", "") or ""
        rewritten = rewritten.strip()

        # Defensive: refuse empty or obviously-degenerate rewrites
        if not rewritten or len(rewritten) < 3:
            logger.warning("Pillar 1: empty rewrite; falling back to original")
            return query
        if len(rewritten) > 600:
            logger.warning("Pillar 1: rewrite too long (%d chars); falling back", len(rewritten))
            return query

        # Strip common LLM preamble that slips through despite the prompt rule
        for prefix in ("Rewritten query:", "Rewritten:", "Query:", "Output:"):
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()

        # If nothing meaningful changed, log and return original
        if rewritten.lower() == query.lower():
            logger.debug("Pillar 1: rewrite identical to original")
            return query

        logger.info(
            "[hg-p1] anaphora resolved: '%s' -> '%s'",
            _truncate(query, 80), _truncate(rewritten, 80),
        )
        return rewritten

    except Exception as exc:
        # NEVER break the query path -- log and fall back
        logger.warning("Pillar 1 resolver failed (%s); using original query", exc)
        return query


def applied_metadata(was_rewritten: bool = False, original: str = "", rewritten: str = "") -> dict:
    """Metadata for response.verification_meta.pillar_1."""
    return {
        "pillar": PILLAR_NUMBER,
        "name": "anaphora_resolver",
        "applied": is_pillar_enabled(PILLAR_NUMBER),
        "version": "v1.0-L1-only",
        "rewritten": was_rewritten,
        "original_preview": _truncate(original, 80) if was_rewritten else None,
        "rewritten_preview": _truncate(rewritten, 80) if was_rewritten else None,
    }
