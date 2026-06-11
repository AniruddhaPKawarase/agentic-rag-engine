"""
Hallucination Guard — Flag plumbing.

Reads env vars at module import time and on every call (for live reload).
Each pillar checks both the master flag and its own sub-flag.

Convention:
  - All flags default to "false".
  - "true", "1", "yes", "on" (case-insensitive) → enabled.
  - Anything else → disabled.

Master flag:  HALLUCINATION_GUARD_ENABLED
Sub-flags:    HG_PILLAR_1, HG_PILLAR_4, HG_PILLAR_6, HG_PILLAR_7, HG_PILLAR_11

Audit/observability:
  Every flag check is cheap (env read + string compare). If your hot path
  cares, call get_active_pillars() once per request and pass the set.
"""

import os
from typing import Set

MASTER_FLAG = "HALLUCINATION_GUARD_ENABLED"
PILLAR_FLAG_TEMPLATE = "HG_PILLAR_{}"

# Known pillars in the Tactical Patch v2 scope
KNOWN_PILLARS = (1, 4, 6, 7, 11)

_TRUTHY = {"true", "1", "yes", "on", "enabled"}


def _read_flag(name: str, default: bool = False) -> bool:
    """Read an env flag; return True if value is truthy."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in _TRUTHY


def is_enabled() -> bool:
    """Master flag. If false, no pillar runs."""
    return _read_flag(MASTER_FLAG, default=False)


def is_pillar_enabled(pillar_num: int) -> bool:
    """Check whether a specific pillar is enabled.

    Returns True only if:
      1. Master flag is enabled
      2. The pillar's sub-flag is enabled
    """
    if not is_enabled():
        return False
    flag_name = PILLAR_FLAG_TEMPLATE.format(pillar_num)
    return _read_flag(flag_name, default=False)


def get_active_pillars() -> Set[int]:
    """Return the set of currently-enabled pillar numbers.

    Useful at request entry to capture the active configuration once
    and avoid re-reading env vars throughout the pipeline.
    """
    if not is_enabled():
        return set()
    return {p for p in KNOWN_PILLARS if is_pillar_enabled(p)}


def describe_state() -> dict:
    """Return a dict snapshot of all HG flag values (for /health, logs, audit)."""
    return {
        "master": is_enabled(),
        "pillars": {p: is_pillar_enabled(p) for p in KNOWN_PILLARS},
        "active_set": sorted(get_active_pillars()),
    }
