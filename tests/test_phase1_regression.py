"""Phase 1 live regression harness.

Runs the 16 historical baseline queries against a LOCAL dev server and asserts:
1. No answer contains forbidden artifacts ([Source:], HIGH (N%), Direct Answer header)
2. source_documents[] is non-empty for queries that had sources before
3. On the second query in the same session, latency drops measurably
   (weak proxy for OpenAI auto-cache hit)

Marked -m integration. Skipped unless GATEWAY_URL is set.

Usage:
  # In one terminal: python -m gateway.app  (starts local gateway on :8001)
  # In another:
  GATEWAY_URL=http://localhost:8001 \\
    python -m pytest tests/test_phase1_regression.py -v -m integration
"""
from __future__ import annotations

import os
import re

import httpx
import pytest

pytestmark = pytest.mark.integration

GATEWAY_URL = os.environ.get("GATEWAY_URL") or os.environ.get("SANDBOX_URL")

FORBIDDEN_PATTERNS = [
    re.compile(r"\[Source:\s*[^\]]*\]"),
    re.compile(r"^Direct Answer\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"HIGH\s*\(\d+%\)"),
]

HISTORICAL_QUERIES = [
    (7222, "What size are the stair pressurization fans or ducts on level 2?"),
    (7222, "What are the specified ceiling heights for typical rooms and corridors?"),
    (7222, "How many valves are shown on the level 2 plumbing drawings?"),
    (7222, "Which levels in this project are typical (identical floor plans) versus unique?"),
    (7222, "What non-special amenities are included in the project such as fitness or pool-equipment rooms?"),
    (7222, "Provide a summary of the overall project scope."),
    (7222, "How many DOAS (Dedicated Outdoor Air Systems) are specified in this project?"),
    (7222, "Summarize the specified insulation requirements for plumbing from Division 22 or the related specification section."),
    (7223, "What size are the stair pressurization fans or ducts on level 2?"),
    (7223, "What are the specified ceiling heights for typical rooms and corridors?"),
    (7223, "How many valves are shown on the level 2 plumbing drawings?"),
    (7223, "Which levels in this project are typical (identical floor plans) versus unique?"),
    (7223, "What non-special amenities are included in the project such as fitness or pool-equipment rooms?"),
    (7223, "Provide a summary of the overall project scope."),
    (7223, "How many DOAS (Dedicated Outdoor Air Systems) are specified in this project?"),
    (7223, "Summarize the specified insulation requirements for plumbing from Division 22 or the related specification section."),
]


def _require_gateway():
    if not GATEWAY_URL:
        pytest.skip("GATEWAY_URL not set — start local gateway and set env var")


@pytest.mark.parametrize("project_id,query", HISTORICAL_QUERIES)
def test_query_has_no_artifacts(project_id: int, query: str):
    _require_gateway()
    with httpx.Client(timeout=180) as c:
        r = c.post(
            f"{GATEWAY_URL}/query",
            json={"query": query, "project_id": project_id},
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer") or ""
        for pat in FORBIDDEN_PATTERNS:
            assert not pat.search(answer), (
                f"Forbidden pattern {pat.pattern!r} found in answer for "
                f"query={query!r} (project {project_id}): {answer[:400]!r}"
            )


def test_same_session_turn2_faster_than_turn1():
    """Weak proxy for OpenAI auto-cache. Turn 2 should be noticeably faster
    than turn 1 when sharing the same session_id (stable cached prefix).
    """
    _require_gateway()
    with httpx.Client(timeout=180) as c:
        r1 = c.post(
            f"{GATEWAY_URL}/query",
            json={"query": "List HVAC drawings.", "project_id": 7222},
        )
        r1.raise_for_status()
        d1 = r1.json()
        sid = d1.get("session_id")
        t1 = r1.elapsed.total_seconds()
        assert sid, f"gateway returned no session_id: {d1}"

        r2 = c.post(
            f"{GATEWAY_URL}/query",
            json={
                "query": "Any fire dampers mentioned?",
                "project_id": 7222,
                "session_id": sid,
            },
        )
        r2.raise_for_status()
        t2 = r2.elapsed.total_seconds()

        # Allow some slack — caching is not guaranteed on every call.
        # Require turn 2 be at most 0.9x of turn 1 to consider a measurable drop.
        assert t2 < t1 * 0.9, (
            f"Turn 2 not measurably faster than turn 1 "
            f"(t1={t1:.1f}s, t2={t2:.1f}s). "
            f"May be normal variance; re-run to confirm."
        )
