"""P127 — PII redaction at the 4 boundaries (request / retrieval / logs / response+audit).

Removes PII (emails, phones, SSNs; names/orgs via an injected Presidio analyzer)
before text is logged or written to the audit manifest (HARD RULE #4 + privacy),
without altering the working query used for retrieval. Pure regex by default;
``analyzer`` is injectable for Presidio-grade NER.
"""

from __future__ import annotations

import re
from collections.abc import Callable

BOUNDARIES: tuple[str, ...] = ("request", "retrieval", "logs", "response_audit")

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b(?:\+?\d[\s.-]?){9,14}\d\b")


class PiiBoundaryError(ValueError):
    """Raised for an unknown redaction boundary."""


def has_pii(text: str) -> bool:
    """True if *text* contains a recognised PII pattern (regex tier)."""
    t = text or ""
    return bool(_EMAIL.search(t) or _SSN.search(t) or _PHONE.search(t))


def redact(
    text: str,
    boundary: str = "logs",
    *,
    analyzer: Callable[[str], str] | None = None,
) -> str:
    """Return *text* with PII redacted for the given *boundary*.

    With an ``analyzer`` (Presidio), defer to it (NER-grade). Otherwise apply the
    regex tier: emails → ``[REDACTED_EMAIL]``, SSNs → ``[REDACTED_SSN]``,
    phones → ``[REDACTED_PHONE]``. SSN is redacted before phone so the SSN shape
    isn't swallowed by the looser phone matcher.
    """
    if boundary not in BOUNDARIES:
        raise PiiBoundaryError(f"unknown boundary {boundary!r}; must be one of {BOUNDARIES}")
    if not text:
        return text or ""
    if analyzer is not None:
        return analyzer(text)
    out = _EMAIL.sub("[REDACTED_EMAIL]", text)
    out = _SSN.sub("[REDACTED_SSN]", out)
    out = _PHONE.sub("[REDACTED_PHONE]", out)
    return out
