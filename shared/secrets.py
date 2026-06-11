"""P124 — Secrets management + plaintext-secret scanning.

Addresses the leaked-Atlas-credential class: secrets are read from a provider
(Vault in prod, env in dev) — NEVER hardcoded — and ``scan_for_secrets`` blocks
plaintext secrets from being committed/logged. The pre-commit guard (A/P1) reuses
``scan_for_secrets``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

# Patterns for high-confidence plaintext secrets (the leak class).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"sk-ant-[A-Za-z0-9_-]{20,}",                      # Anthropic key
        r"sk-[A-Za-z0-9]{20,}",                            # OpenAI-style key
        r"AKIA[0-9A-Z]{16}",                               # AWS access key id
        r"mongodb(\+srv)?://[^\s:]+:[^\s@]+@",             # Mongo URI with inline creds
        r"postgres(?:ql)?://[^\s:]+:[^\s@]+@",             # Postgres URI with inline creds
        r"(?i)(password|passwd|secret|api[_-]?key)\s*[=:]\s*['\"][^'\"]{6,}['\"]",
    )
)


class SecretNotFoundError(KeyError):
    """Raised when a required secret is absent from the provider."""


def scan_for_secrets(text: str) -> list[str]:
    """Return the list of plaintext-secret matches in *text* (empty = clean)."""
    found: list[str] = []
    for pat in _SECRET_PATTERNS:
        found.extend(m.group(0) for m in pat.finditer(text or ""))
    return found


def contains_secret(text: str) -> bool:
    return bool(scan_for_secrets(text))


def get_secret(name: str, *, provider: Callable[[str], str | None] | None = None) -> str:
    """Read secret *name* from the provider (Vault) or env fallback (dev).

    NEVER reads from source. Raises :class:`SecretNotFoundError` if absent.
    """
    value = provider(name) if provider is not None else os.environ.get(name)
    if value is None or value == "":
        raise SecretNotFoundError(f"secret {name!r} not found in provider/env")
    return value
