"""
Structured audit logging for compliance and monitoring.
Logs who queried what, when, and what was returned.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

audit_logger = logging.getLogger("agentic_rag.audit")


def log_query(
    user_id: str,
    project_id: int,
    query: str,
    sources: List[str],
    confidence: str,
    cost_usd: float,
    elapsed_ms: int,
    steps: int,
    needs_escalation: bool = False,
    client_ip: str = "",
) -> None:
    """Log a completed query for audit trail."""
    audit_logger.info(json.dumps({
        "event": "query_completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "client_ip": client_ip,
        "project_id": project_id,
        "query": query[:500],
        "sources_count": len(sources),
        "sources": sources[:10],
        "confidence": confidence,
        "cost_usd": cost_usd,
        "elapsed_ms": elapsed_ms,
        "steps": steps,
        "needs_escalation": needs_escalation,
    }))


def log_auth_failure(
    client_ip: str,
    reason: str,
) -> None:
    """Log authentication failures."""
    audit_logger.warning(json.dumps({
        "event": "auth_failure",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": client_ip,
        "reason": reason,
    }))


def log_rate_limit(
    client_ip: str,
    user_id: str = "",
) -> None:
    """Log rate limit hits."""
    audit_logger.warning(json.dumps({
        "event": "rate_limited",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": client_ip,
        "user_id": user_id,
    }))
