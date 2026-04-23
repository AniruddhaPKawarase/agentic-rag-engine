"""Local end-to-end harness for validating Fix #8 + Fix #5 without any VM deploy.

Runs the full Orchestrator.query() pipeline in-process against real MongoDB and
S3 (using .env creds). For every question, captures:

- engine_used, fallback_used, agentic_confidence, confidence
- source_documents[] including download_url
- Probe: does every download_url return 206/200 on ranged GET?
- Token usage + timing

Designed so we can A/B-compare against sandbox (before_fixes) to prove the
local fixes improve the traced failures before we ever deploy.

Usage:
    python PROD_SETUP/unified-rag-agent/tests/e2e/local_fix_harness.py \
        --project-ids 7222 7223 --label after_fixes_local
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Path bootstrap so we can import `gateway.*` and `agentic.*` from repo layout
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agentic"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from gateway.orchestrator import Orchestrator  # noqa: E402
# Reuse the same manager question set
sys.path.insert(0, str(REPO_ROOT / "tests" / "e2e"))
from reproduce_manager_failures import MANAGER_QUESTIONS  # noqa: E402


def probe_url(session: requests.Session, url: str, timeout: int = 10) -> dict[str, Any]:
    if not url:
        return {"ok": False, "status": None, "reason": "empty url"}
    try:
        r = session.get(url, headers={"Range": "bytes=0-1023"}, allow_redirects=True, timeout=timeout)
        return {"ok": r.status_code in (200, 206), "status": r.status_code}
    except requests.RequestException as exc:
        return {"ok": False, "status": None, "reason": str(exc)[:120]}


async def run_question(orch: Orchestrator, project_id: int, q: dict[str, str]) -> dict[str, Any]:
    start = time.monotonic()
    try:
        resp = await orch.query(query=q["question"], project_id=project_id)
    except Exception as exc:  # defensive: capture orchestrator errors
        return {
            "project_id": project_id,
            "question_id": q["id"],
            "question": q["question"],
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - start, 2),
        }

    elapsed = round(time.monotonic() - start, 2)
    sds = resp.get("source_documents") or []
    return {
        "project_id": project_id,
        "question_id": q["id"],
        "question": q["question"],
        "engine_used": resp.get("engine_used"),
        "fallback_used": resp.get("fallback_used"),
        "agentic_confidence": resp.get("agentic_confidence"),
        "confidence": resp.get("confidence"),
        "retrieval_count": resp.get("retrieval_count") or len(sds),
        "source_documents_total": len(sds),
        "source_documents_with_s3_path": sum(1 for s in sds if s.get("s3_path")),
        "source_documents_with_signed_url": sum(
            1 for s in sds if "X-Amz-Signature" in (s.get("download_url") or "")
        ),
        "answer_length": len(resp.get("answer") or ""),
        "answer": resp.get("answer") or "",
        "sample_download_urls": [s.get("download_url") for s in sds[:3] if s.get("download_url")],
        "elapsed_s": elapsed,
    }


async def main_async(project_ids: list[int], label: str, probe_urls: bool) -> int:
    orch = Orchestrator(fallback_enabled=True, fallback_timeout=30)
    http_session = requests.Session()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "test_results" / f"local_{label}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for pid in project_ids:
        print(f"=== project {pid} ===", flush=True)
        for q in MANAGER_QUESTIONS:
            print(f"  [{q['id']}] {q['question'][:70]}...", flush=True)
            rec = await run_question(orch, pid, q)
            if probe_urls:
                urls = rec.get("sample_download_urls") or []
                probes = [probe_url(http_session, u) for u in urls]
                rec["url_probe_sample"] = probes
                rec["url_probe_ok_ratio"] = (
                    sum(1 for p in probes if p.get("ok")) / len(probes) if probes else None
                )
            all_records.append(rec)

    agg = {
        "timestamp": datetime.now().isoformat(),
        "project_ids": project_ids,
        "label": label,
        "num_queries": len(all_records),
        "avg_elapsed_s": round(
            sum(r.get("elapsed_s") or 0 for r in all_records) / max(len(all_records), 1), 2
        ),
        "engine_distribution": {
            e: sum(1 for r in all_records if r.get("engine_used") == e)
            for e in {r.get("engine_used") for r in all_records if r.get("engine_used")}
        },
        "fallback_triggered": sum(1 for r in all_records if r.get("fallback_used")),
        "queries_with_zero_sources": sum(
            1 for r in all_records if (r.get("source_documents_total") or 0) == 0
        ),
        "queries_with_any_unsigned_url": sum(
            1
            for r in all_records
            if (r.get("source_documents_total") or 0)
            > (r.get("source_documents_with_signed_url") or 0)
        ),
        "queries_with_any_broken_url_GET": sum(
            1
            for r in all_records
            if (r.get("url_probe_ok_ratio") is not None and r["url_probe_ok_ratio"] < 1.0)
        ) if probe_urls else None,
    }

    (out_dir / "records.json").write_text(
        json.dumps(all_records, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(agg, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"\nArtifacts: {out_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-ids", nargs="+", type=int, default=[7222, 7223])
    p.add_argument("--label", default="after_fixes_local")
    p.add_argument("--no-probe", action="store_true", help="Skip HTTP URL probes")
    args = p.parse_args()
    return asyncio.run(main_async(args.project_ids, args.label, not args.no_probe))


if __name__ == "__main__":
    raise SystemExit(main())
