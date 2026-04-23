"""Phase 6: regression of v2.0-docqa-extension against v1.2-hybrid-ship baseline.

Runs the 16 historical queries and asserts:
  1. URL integrity — every download_url in every source_document is reachable (200/206/302)
  2. Source coverage — ≥15 of 16 queries return at least one source_document
  3. Avg latency — p50 ≤ 22.0s (baseline v1.2 was 19.0s; allow +3s slack for
     new classifier pass + any docqa branching overhead)
  4. Answer format — no [Source:], Direct Answer, HIGH (N%) artifacts

Skipped by default. Engineer runs:
  GATEWAY_URL=http://localhost:8001 \\
    python -m pytest tests/test_phase6_regression_vs_v12.py -v -m regression

Artifacts saved to test_results/phase6_v2_results_<timestamp>.json
"""
from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.regression

GATEWAY_URL = os.environ.get("GATEWAY_URL") or os.environ.get("SANDBOX_URL")

QUERIES = [
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

FORBIDDEN_PATTERNS = [
    re.compile(r"\[Source:\s*[^\]]*\]"),
    re.compile(r"^Direct Answer\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"HIGH\s*\(\d+%\)"),
]

P50_LATENCY_THRESHOLD_S = 22.0  # baseline v1.2 was 19s, allow +3s slack


def _require_gateway():
    if not GATEWAY_URL:
        pytest.skip("GATEWAY_URL not set — start gateway and export env var")


def _save_artifact(results: list[dict]) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("test_results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"phase6_v2_results_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def test_phase6_regression():
    """Single test that runs all 16 queries and saves aggregate results."""
    _require_gateway()
    results: list[dict] = []
    artifacts_hits: list[dict] = []
    broken_urls: list[dict] = []

    with httpx.Client(timeout=180) as c:
        for project_id, query in QUERIES:
            r = c.post(
                f"{GATEWAY_URL}/query",
                json={"query": query, "project_id": project_id},
            )
            r.raise_for_status()
            d = r.json()
            srcs = d.get("source_documents") or []
            elapsed = r.elapsed.total_seconds()
            answer = d.get("answer") or ""

            # Artifact check
            for pat in FORBIDDEN_PATTERNS:
                if pat.search(answer):
                    artifacts_hits.append({
                        "query": query, "pattern": pat.pattern,
                        "snippet": answer[:200],
                    })

            # URL integrity — HEAD check each download_url
            url_results: list[dict] = []
            for s in srcs:
                url = s.get("download_url")
                if not url:
                    continue
                try:
                    head = c.head(url, timeout=10, follow_redirects=True)
                    url_results.append({"url_prefix": url.split("?")[0],
                                         "status": head.status_code})
                    if head.status_code not in (200, 206, 302):
                        broken_urls.append({"query": query, "url": url.split("?")[0],
                                            "status": head.status_code})
                except Exception as exc:
                    broken_urls.append({"query": query, "url": url.split("?")[0],
                                        "error": type(exc).__name__})

            results.append({
                "project_id": project_id,
                "query": query,
                "elapsed_s": elapsed,
                "sources_count": len(srcs),
                "confidence": d.get("confidence"),
                "active_agent": d.get("active_agent"),
                "engine_used": d.get("engine_used"),
                "url_checks": url_results,
            })

    path = _save_artifact(results)
    print(f"\n\nPhase 6 regression artifact: {path}")

    # Aggregate assertions
    latencies = [r["elapsed_s"] for r in results]
    p50 = statistics.median(latencies)
    avg = statistics.mean(latencies)
    zero_src = [r for r in results if r["sources_count"] == 0]

    print(f"avg latency: {avg:.1f}s   p50: {p50:.1f}s   "
          f"(baseline v1.2: 19.0s; threshold: {P50_LATENCY_THRESHOLD_S}s)")
    print(f"zero-source queries: {len(zero_src)}/16")
    print(f"broken URLs: {len(broken_urls)}")
    print(f"artifact hits: {len(artifacts_hits)}")

    assert not broken_urls, f"Broken URLs: {broken_urls[:3]}"
    assert len(zero_src) <= 1, (
        f"More than 1 query returned zero sources: {[r['query'] for r in zero_src]}"
    )
    assert not artifacts_hits, f"Answer-format artifacts found: {artifacts_hits[:3]}"
    assert p50 <= P50_LATENCY_THRESHOLD_S, (
        f"p50 latency {p50:.1f}s exceeds {P50_LATENCY_THRESHOLD_S}s threshold"
    )
