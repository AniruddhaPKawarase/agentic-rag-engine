"""P126 — Correlation IDs (W3C trace context).

One id ties together the log line, Langfuse trace, audit manifest, and
downstream model calls. The gateway extracts an incoming ``traceparent`` (reusing
its trace-id) or mints a new one; that id becomes ``Meta.request_id``.
Pure helpers — the middleware wires them.
"""

from __future__ import annotations

import re
import uuid

_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")


def parse_traceparent(header: str | None) -> str | None:
    """Return the 32-hex trace-id from a valid W3C traceparent, else None."""
    if not header:
        return None
    m = _TRACEPARENT.match(header.strip())
    return m.group(1) if m else None


def new_trace_id() -> str:
    """Mint a new 32-hex trace-id."""
    return uuid.uuid4().hex


def build_traceparent(trace_id: str) -> str:
    """Build a W3C traceparent header for *trace_id* with a fresh span id."""
    span_id = uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{span_id}-01"


def correlation_id(traceparent_header: str | None) -> str:
    """Return the correlation id for a request: reuse an incoming valid trace-id,
    else mint a new one. This id flows to logs, Langfuse, audit, and model calls.
    """
    return parse_traceparent(traceparent_header) or new_trace_id()
