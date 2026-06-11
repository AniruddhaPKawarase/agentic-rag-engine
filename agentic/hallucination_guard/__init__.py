"""
Hallucination Guard — additive, flag-gated defenses for 8001 PROD.

This package implements the Tactical Patch v2 from
HYBRID_RAG_v32_ROADMAP.md §5.21. Each pillar is a separate module,
independently flag-gated via env vars.

Master flag:  HALLUCINATION_GUARD_ENABLED (default: false)
Per-pillar:   HG_PILLAR_1 .. HG_PILLAR_11

When the master flag is false, ALL pillars are no-ops regardless of
sub-flag values. This is the kill-switch.

Pillars implemented in this package (Tactical Patch v2):
    Pillar 1  — L1 anaphora resolver         (pillar_1_anaphora.py)        [pending]
    Pillar 4  — Per-claim Lynx-style verifier (pillar_4_verifier.py)        [pending]
    Pillar 6  — Anti-anchor system prompt    (pillar_6_anti_anchor.py)     [shipped]
    Pillar 7  — Citation-forcing schema       (pillar_7_citation.py)        [pending]
    Pillar 11 — Sycophancy + injection       (pillar_11_adversarial.py)   [shipped]

Each pillar exposes a single entry point and is composable in the
synthesis pipeline. See docs/HALLUCINATION_GUARD_DEPLOYMENT.md for the
wire-up sequence.

Doctrine reminders (from §10.3 HARD RULES):
  - Every factual claim must be supported by THIS turn's retrieval.
  - Refusals are clean, never hedged.
  - Citation-forcing schema validation failure → re-synth (cap 3), then refuse.
  - Master flag is the kill-switch; reversible in 30 seconds.
"""

from .flags import (
    is_enabled,
    is_pillar_enabled,
    get_active_pillars,
    describe_state,
)

__all__ = [
    "is_enabled",
    "is_pillar_enabled",
    "get_active_pillars",
    "describe_state",
]

__version__ = "0.2.0"  # 0.1.0 = Pillar 6; 0.2.0 = Pillar 6 + Pillar 11
