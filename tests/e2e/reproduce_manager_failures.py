"""Reproduce the 8 manager-reported failure questions on projects 7222 + 7223.

Captures:
- Full agent response (answer, engine, confidence, source_documents, s3_paths).
- Download URL validity for every source (HEAD probe).
- Timing + token usage.

Also runs against ANY list of project IDs, so the same script validates
fixes across projects later.

Usage:

  # Run reproduction against current sandbox state (pre-fix baseline)
  python PROD_SETUP/unified-rag-agent/tests/e2e/reproduce_manager_failures.py \
      --base-url https://ai6.ifieldsmart.com/rag \
      --project-ids 7222 7223 \
      --label before_fixes

  # Re-run with fixed local agent (once fixes deployed locally or elsewhere)
  python PROD_SETUP/unified-rag-agent/tests/e2e/reproduce_manager_failures.py \
      --base-url http://localhost:8001 \
      --project-ids 7222 7223 7166 \
      --label after_fixes
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


# 8 questions paraphrased from the manager's session. Wording kept close to
# how an end-user would actually type them (so we exercise the live routing).
MANAGER_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "Q1",
        "label": "stair_pressurization_size_L2",
        "question": "What size are the stair pressurization fans or ducts on level 2?",
        "expected_failure": "Agent returned electrical drawings instead of mechanical/plumbing; drawing links showed 'invalid document data'.",
    },
    {
        "id": "Q2",
        "label": "ceiling_heights",
        "question": "What are the specified ceiling heights for typical rooms and corridors?",
        "expected_failure": "WORKED CORRECTLY in manager's session \u2014 kept as regression check.",
    },
    {
        "id": "Q3",
        "label": "valve_count_L2",
        "question": "How many valves are shown on the level 2 plumbing drawings?",
        "expected_failure": "Broad, vague answer; directed user to drawings; pulled small/illegible drawing.",
    },
    {
        "id": "Q4",
        "label": "typical_levels",
        "question": "Which levels in this project are typical (identical floor plans) versus unique?",
        "expected_failure": "Said ALL levels were typical \u2014 factually incorrect.",
    },
    {
        "id": "Q5",
        "label": "non_special_amenities",
        "question": "What non-special amenities are included in the project such as fitness or pool-equipment rooms?",
        "expected_failure": "WORKED CORRECTLY in manager's session \u2014 kept as regression check.",
    },
    {
        "id": "Q6",
        "label": "overall_scope",
        "question": "Provide a summary of the overall project scope.",
        "expected_failure": "Vague answer; unusual drawing names in sources; several drawing links broken.",
    },
    {
        "id": "Q7",
        "label": "doas_count",
        "question": "How many DOAS (Dedicated Outdoor Air Systems) are specified in this project?",
        "expected_failure": "Named 1 DOAS \u2014 there are actually 2. Agent did not check Pan B.",
    },
    {
        "id": "Q8",
        "label": "plumbing_insulation_summary",
        "question": "Summarize the specified insulation requirements for plumbing from Division 22 or the related specification sections.",
        "expected_failure": "Said 'the full text of this section was not directly available' despite spec data being clean.",
    },
]


def run_query(session: requests.Session, base_url: str, project_id: int, question: str, timeout: int = 240) -> dict[str, Any]:
    start = time.monotonic()
    status = 0
    body: dict[str, Any] = {}
    err: str | None = None
    try:
        r = session.post(
            f"{base_url.rstrip('/')}/query",
            json={"query": question, "project_id": project_id},
            timeout=timeout,
        )
        status = r.status_code
        body = r.json() if r.content else {}
    except requests.Timeout:
        err = f"timeout after {timeout}s"
    except requests.RequestException as exc:
        err = f"request failed: {exc}"
    except ValueError as exc:
        err = f"non-json response: {exc}"
    elapsed = round(time.monotonic() - start, 2)
    body["_http_status"] = status
    body["_elapsed_s"] = elapsed
    body["_client_error"] = err
    return body


def validate_download_url(session: requests.Session, url: str, timeout: int = 10) -> dict[str, Any]:
    """Probe a download URL with a ranged GET (HEAD is commonly 403 on S3 even when GET is allowed).

    Returns status, content-type, and the first-byte-range length we received.
    """
    if not url:
        return {"ok": False, "status": None, "reason": "empty url"}
    try:
        r = session.get(
            url,
            headers={"Range": "bytes=0-1023"},
            allow_redirects=True,
            timeout=timeout,
            stream=False,
        )
        ok = r.status_code in (200, 206)
        return {
            "ok": ok,
            "status": r.status_code,
            "content_length_header": r.headers.get("Content-Length"),
            "content_range_header": r.headers.get("Content-Range"),
            "content_type": r.headers.get("Content-Type"),
            "final_url_host": requests.utils.urlparse(r.url).netloc,
            "body_sample": (r.text[:160] if not ok and r.text else None),
        }
    except requests.RequestException as exc:
        return {"ok": False, "status": None, "reason": str(exc)[:120]}


def summarise_response(body: dict[str, Any]) -> dict[str, Any]:
    sds = body.get("source_documents") or []
    s3_paths_populated = sum(1 for s in sds if s.get("s3_path"))
    url_populated = sum(1 for s in sds if s.get("download_url"))
    token_usage = body.get("token_usage") or {}
    return {
        "engine_used": body.get("engine_used"),
        "model_used": body.get("model_used"),
        "fallback_used": body.get("fallback_used"),
        "confidence": body.get("confidence"),
        "confidence_score": body.get("confidence_score"),
        "agentic_confidence": body.get("agentic_confidence"),
        "processing_time_ms": body.get("processing_time_ms"),
        "elapsed_s_client": body.get("_elapsed_s"),
        "http_status": body.get("_http_status"),
        "retrieval_count": body.get("retrieval_count") or len(sds),
        "source_documents_total": len(sds),
        "source_documents_with_s3_path": s3_paths_populated,
        "source_documents_with_download_url": url_populated,
        "answer_length": len(body.get("answer") or ""),
        "needs_document_selection": body.get("needs_document_selection"),
        "error": body.get("error") or body.get("_client_error"),
        "total_tokens": token_usage.get("total_tokens"),
    }


def probe_urls_parallel(
    session: requests.Session, urls: list[str], max_workers: int = 8
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not urls:
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        fs = {pool.submit(validate_download_url, session, u): u for u in urls}
        for f in concurrent.futures.as_completed(fs):
            out = f.result()
            out["url"] = fs[f]
            results.append(out)
    return results


def run_for_project(session: requests.Session, base_url: str, project_id: int) -> list[dict[str, Any]]:
    """Run all 8 questions for a single project, with URL probes."""
    per_q: list[dict[str, Any]] = []
    for q in MANAGER_QUESTIONS:
        print(f"  [{q['id']}] project {project_id}: {q['question'][:70]}...", flush=True)
        body = run_query(session, base_url, project_id, q["question"])
        sd = body.get("source_documents") or []
        urls = [s.get("download_url") for s in sd if s.get("download_url")]
        url_probe_sample = probe_urls_parallel(session, urls[:5])
        rec = {
            "project_id": project_id,
            "question_id": q["id"],
            "question_label": q["label"],
            "question": q["question"],
            "expected_failure": q["expected_failure"],
            "summary": summarise_response(body),
            "url_probe_sample": url_probe_sample,
            "url_probe_ok_ratio": (
                sum(1 for u in url_probe_sample if u.get("ok")) / len(url_probe_sample)
                if url_probe_sample else None
            ),
            "answer": body.get("answer", ""),
            "source_documents_sample": sd[:5],
            "top_level_s3_paths_sample": (body.get("s3_paths") or [])[:5],
        }
        per_q.append(rec)
    return per_q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://ai6.ifieldsmart.com/rag")
    parser.add_argument("--project-ids", nargs="+", type=int, default=[7222, 7223])
    parser.add_argument("--label", default="before_fixes")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parents[2] / "test_results" / f"manager_{args.label}_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    all_records: list[dict[str, Any]] = []
    for pid in args.project_ids:
        print(f"=== project {pid} ===", flush=True)
        all_records.extend(run_for_project(session, args.base_url, pid))

    # Aggregate stats
    agg = {
        "timestamp": datetime.now().isoformat(),
        "base_url": args.base_url,
        "label": args.label,
        "project_ids": args.project_ids,
        "num_queries": len(all_records),
        "avg_elapsed_s": round(
            sum((r["summary"].get("elapsed_s_client") or 0) for r in all_records)
            / max(len(all_records), 1),
            2,
        ),
        "total_tokens": sum((r["summary"].get("total_tokens") or 0) for r in all_records),
        "engine_distribution": {
            e: sum(1 for r in all_records if r["summary"].get("engine_used") == e)
            for e in {r["summary"].get("engine_used") for r in all_records}
            if e is not None
        },
        "fallback_triggered": sum(1 for r in all_records if r["summary"].get("fallback_used")),
        "queries_with_zero_sources": sum(
            1 for r in all_records if (r["summary"].get("source_documents_total") or 0) == 0
        ),
        "queries_with_any_null_s3_path": sum(
            1
            for r in all_records
            if (r["summary"].get("source_documents_with_s3_path") or 0)
            < (r["summary"].get("source_documents_total") or 0)
        ),
        "queries_with_any_broken_url": sum(
            1
            for r in all_records
            if (r["url_probe_ok_ratio"] is not None and r["url_probe_ok_ratio"] < 1.0)
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(
        json.dumps(all_records, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(agg, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
