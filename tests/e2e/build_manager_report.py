"""Package a harness run into a single manager-ready JSON.

Combines ``records.json`` + ``summary.json`` from a harness run into one file
with metadata, aggregate metrics, per-question records, and known-limitation
notes. Schema matches what we've been using in
``local_fix_harness.py`` so the manager sees the same field names we've used
in every prior deliverable.

Usage:

    python PROD_SETUP/unified-rag-agent/tests/e2e/build_manager_report.py \
        --run-dir PROD_SETUP/unified-rag-agent/test_results/local_v1_2_manager_final_<ts>/ \
        --version v1.2-hybrid-ship \
        --flags "MULTI_QUERY_RRF_ENABLED=true,RERANKER_ENABLED=true,SELF_RAG_ENABLED=false,RERANKER_INCLUDE_SCORE=false"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = json.loads((run_dir / "records.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return records, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--version", default="v1.2-hybrid-ship")
    p.add_argument("--flags", default="")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    records, summary = load(args.run_dir)

    flags = {}
    for pair in (args.flags or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        flags[k.strip()] = v.strip()

    # Per-question slice that keeps the fields from the harness records
    # (same schema the manager has seen in prior reports)
    per_q = []
    for r in records:
        per_q.append({
            "project_id": r.get("project_id"),
            "question_id": r.get("question_id"),
            "question": r.get("question"),
            "engine_used": r.get("engine_used"),
            "fallback_used": r.get("fallback_used"),
            "confidence": r.get("confidence"),
            "agentic_confidence": r.get("agentic_confidence"),
            "retrieval_count": r.get("retrieval_count"),
            "source_documents_total": r.get("source_documents_total"),
            "source_documents_with_s3_path": r.get("source_documents_with_s3_path"),
            "source_documents_with_signed_url": r.get("source_documents_with_signed_url"),
            "url_probe_ok_ratio": r.get("url_probe_ok_ratio"),
            "answer_length": r.get("answer_length"),
            "answer": r.get("answer"),
            "elapsed_s": r.get("elapsed_s"),
            "sample_download_urls": r.get("sample_download_urls") or [],
        })

    report = {
        "report_generated_at": datetime.now().isoformat(),
        "version": args.version,
        "environment": {
            "base_url": "local in-process (no VM deploy)",
            "flags_active": flags,
            "backend_agent": "AgenticRAG (gpt-4.1) with Fix #1-#5 + #8 active",
            "database": "MongoDB Atlas (production data read-only)",
            "s3_bucket": "ifieldsmart (signed URL via SigV4)",
        },
        "ui_contract": {
            "top_level_response_keys": 39,
            "source_documents_fields": [
                "s3_path", "file_name", "display_title", "download_url",
                "pdf_name", "drawing_name", "drawing_title", "page",
            ],
            "optional_extras_in_response": "none (SELF_RAG_ENABLED=false, RERANKER_INCLUDE_SCORE=false)",
            "schema_delta_vs_previous_version": "none — identical to v1.0 baseline",
        },
        "aggregate_metrics": {
            "num_queries": summary.get("num_queries"),
            "avg_latency_seconds": summary.get("avg_elapsed_s"),
            "engine_distribution": summary.get("engine_distribution"),
            "fallback_triggered_count": summary.get("fallback_triggered"),
            "queries_with_zero_sources": summary.get("queries_with_zero_sources"),
            "queries_with_any_unsigned_url": summary.get("queries_with_any_unsigned_url"),
            "queries_with_any_broken_url_get": summary.get("queries_with_any_broken_url_GET"),
            "project_ids_tested": summary.get("project_ids"),
        },
        "per_question_results": per_q,
        "known_limitations": [
            {
                "scope": "valve-count questions (Q3 across projects 7222 and 7223)",
                "symptom": "agg_count_equipment returns 0 tags → agent answers 'no valves shown'",
                "root_cause": (
                    "default VALVE keyword list (VALVE, V-, BV-, GV-, CV-) "
                    "does not match how plumbing drawings on these projects label "
                    "valves (e.g. 'CS-1-1/4-125' Thermomegatech callouts, or "
                    "unlabeled schematic symbols)"
                ),
                "impact": "factually correct 'no matching tags' answer, but not useful",
                "planned_fix": "broaden VALVE keyword coverage per trade in next dev cycle",
            },
        ],
        "how_to_reproduce": [
            "1. rsync -a PROD_SETUP/unified-rag-agent/_versions/v1.2-hybrid-ship/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md",
            "2. export MULTI_QUERY_RRF_ENABLED=true RERANKER_ENABLED=true SELF_RAG_ENABLED=false RERANKER_INCLUDE_SCORE=false",
            "3. python PROD_SETUP/unified-rag-agent/tests/e2e/local_fix_harness.py --project-ids 7222 7223 --label rerun",
        ],
    }

    out_path = args.out or (args.run_dir / "manager_report.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manager report written: {out_path}")
    print(f"  num_queries: {report['aggregate_metrics']['num_queries']}")
    print(f"  avg_latency: {report['aggregate_metrics']['avg_latency_seconds']} s")
    print(f"  broken_urls: {report['aggregate_metrics']['queries_with_any_broken_url_get']}")
    print(f"  zero_source: {report['aggregate_metrics']['queries_with_zero_sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
