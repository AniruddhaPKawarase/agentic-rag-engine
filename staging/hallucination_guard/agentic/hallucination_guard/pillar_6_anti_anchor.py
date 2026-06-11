"""
Pillar 6 — Anti-Anchor System Prompt + Drift Detector (lightweight)

Adds a 6-rule system prompt prepend to synthesizer calls when enabled.
Designed for 8001 PROD as a near-zero-risk additive change.

From HYBRID_RAG_v32_ROADMAP.md §5.7:
  1. Every factual claim must be supported by THIS turn's retrieval.
  2. If asked to elaborate but retrieval lacks new supporting detail,
     say "I don't have additional information on that." Do not invent.
  3. If user pushes back ("are you sure?"), do NOT change the answer
     unless retrieval evidence has changed. Hold the line.
  4. If asked about an entity referenced earlier that does not appear
     in current retrieval, say "Let me re-check that" and re-retrieve.
  5. When entity reference is ambiguous, ask a clarifying question.
     Cite specific entity IDs in every answer.
  6. Cite every factual claim with its source. Claims without
     citations are forbidden.

Cost: ~150 prompt tokens added per synthesis call. Marginal latency: <5ms.
Reversible: HG_PILLAR_6=false → no prompt change.
Compatible: with ENABLE_SPECIFICATIONS=false (drawings-only) mode.
"""

import logging
from typing import Optional

from .flags import is_pillar_enabled

logger = logging.getLogger(__name__)

PILLAR_NUMBER = 6

# The 6-rule doctrine (verbatim per §5.7). Kept as a constant so version
# diffs against the roadmap are clean.
ANTI_ANCHOR_RULES = """\
You are answering questions about construction documents. The following six rules are NON-NEGOTIABLE and apply to every answer you produce:

1. Every factual claim must be supported by THIS turn's retrieval. Do not carry forward facts from prior turns unless current retrieval confirms them.

2. If asked to elaborate but retrieval lacks new supporting detail, say "I don't have additional information on that." Do not invent.

3. If user pushes back ("are you sure?", "I think it's different", etc.), do NOT change the answer unless retrieval evidence has changed. Hold the line.

4. If asked about an entity referenced earlier that does not appear in current retrieval, say "Let me re-check that" — do not answer from memory.

5. When entity reference is ambiguous, ask a clarifying question. Cite specific entity IDs (e.g., FCU-101, A-211, spec §23 81 26) in every answer.

6. Cite every factual claim with its source. Claims without citations are forbidden."""


def get_anti_anchor_prompt() -> str:
    """Return the anti-anchor 6-rule prompt block (constant).

    Caller is responsible for checking is_pillar_enabled(6) before
    calling this; this function does NOT check. (Allows test
    inspection of the prompt without setting env vars.)
    """
    return ANTI_ANCHOR_RULES


def compose_system_prompt(base_system_prompt: str,
                          force_enable: Optional[bool] = None) -> str:
    """Prepend the anti-anchor rules to an existing system prompt.

    Args:
        base_system_prompt: The synthesizer's current system prompt
            (whatever the production code already builds).
        force_enable: For unit tests — pass True to force-enable
            regardless of flag state; None reads the flag.

    Returns:
        A new system prompt string. If Pillar 6 is disabled, returns
        base_system_prompt unchanged (no-op).

    The prepended block is separated from the base by a clear marker
    so it's identifiable in logs / Langfuse traces.
    """
    if force_enable is None:
        enabled = is_pillar_enabled(PILLAR_NUMBER)
    else:
        enabled = bool(force_enable)

    if not enabled:
        return base_system_prompt

    block = (
        f"{ANTI_ANCHOR_RULES}\n"
        f"\n"
        f"---\n"
        f"[Project-specific instructions below]\n"
        f"---\n"
        f"\n"
        f"{base_system_prompt}"
    )

    logger.debug("Pillar 6 anti-anchor prompt prepended (+%d chars)",
                 len(ANTI_ANCHOR_RULES))
    return block


def applied_metadata() -> dict:
    """Return metadata for inclusion in response.verification_meta.

    Helps downstream consumers (eval grader, dashboards) confirm
    which pillars were active for a given response.
    """
    return {
        "pillar": PILLAR_NUMBER,
        "name": "anti_anchor",
        "applied": is_pillar_enabled(PILLAR_NUMBER),
        "version": "v1.0",
        "rules_count": 6,
    }


# Convenience: a tiny drift signal helper (the full UMAP drift detector
# is part of Pillar 6 in §5.7 but is OUT OF SCOPE for the tactical
# patch v2 — that lives in P90 later). This stub keeps the import
# surface stable for when the full detector lands.

def is_drift_detected(*args, **kwargs) -> bool:
    """Placeholder. Full UMAP cluster-shift detector lives in P90 (full v32).

    Returns False (no drift) as a safe default for the tactical patch.
    """
    return False
