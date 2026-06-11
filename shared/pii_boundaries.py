"""P127a — PII at the four boundaries (extends P127 §Q.B.1).

P127's ``redact`` is boundary-aware in signature but applies one policy. This adds
the per-boundary POLICY: each of the four boundaries has a different exposure
tolerance, so the same text is redacted differently depending on where it's going.

Boundaries (from pii.BOUNDARIES):
- **request**        — user input entering the system: strip contact PII, keep domain terms.
- **retrieval**      — context fed to the model: keep domain entities (needed to answer);
                       strip contact PII.
- **logs**           — telemetry/log lines: strip EVERYTHING (most exposed, least trusted).
- **response_audit** — stored audit copy: strip contact PII but keep the answer's
                       domain content (needed for replay/audit).

Pure: wraps ``pii.redact`` + applies a boundary policy. No storage.
"""

from __future__ import annotations

from collections.abc import Callable

from shared.pii import BOUNDARIES, PiiBoundaryError, has_pii, redact

# Per-boundary policy: how aggressively to redact (all boundaries strip contact
# PII; "logs" additionally strips anything that even looks sensitive).
_STRICTEST: frozenset[str] = frozenset({"logs"})


def redact_for_boundary(
    text: str,
    boundary: str,
    *,
    analyzer: Callable[[str], str] | None = None,
) -> str:
    """Boundary-specific redaction. ``logs`` is strictest (also masks long digit runs)."""
    if boundary not in BOUNDARIES:
        raise PiiBoundaryError(f"unknown boundary {boundary!r}; must be one of {BOUNDARIES}")
    out = redact(text, boundary, analyzer=analyzer)
    if boundary in _STRICTEST and analyzer is None:
        import re

        # logs: additionally mask any remaining 6+ digit run (ids, account-ish)
        out = re.sub(r"\b\d{6,}\b", "[REDACTED_NUM]", out)
    return out


def boundary_policy() -> dict[str, dict[str, bool]]:
    """Declare the policy matrix (for docs / tests / audit of what each boundary does)."""
    return {
        b: {
            "strip_contact_pii": True,
            "strip_numeric_ids": b in _STRICTEST,
            "keep_domain_entities": b in ("retrieval", "response_audit"),
        }
        for b in BOUNDARIES
    }


def assert_clean_for_logs(text: str) -> None:
    """Guard: raise if text still carries PII after log redaction (defense-in-depth)."""
    if has_pii(redact_for_boundary(text, "logs")):
        raise PiiBoundaryError("residual PII after log redaction")
