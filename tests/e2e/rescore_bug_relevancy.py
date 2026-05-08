"""Re-score an existing bug_relevancy run with a cleaner relevance definition.

Instead of hard-coding which sheets "should" answer each question (biased and
fragile), we count how many sources are obviously universal-boilerplate
drawings that appear on every project and are never a useful reference for a
specific user question. Everything else is treated as potentially relevant.

Boilerplate regex (matches "always noise" titles):
    cover sheet | area plan | site photograph | list of drawings |
    general notes | ADA requirements | building code data |
    life safety plan | slab edge plan | link beam schedule

Relevance = 1 - boilerplate_count / total_sources
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BOILERPLATE = re.compile(
    r"cover\s*sheet|area\s*plan|site\s*photograph|list\s*of\s*drawings|"
    r"general\s*note|ADA\s*requirement|building\s*code\s*data|"
    r"life\s*safety\s*plan|slab\s*edge|link\s*beam\s*schedule",
    re.IGNORECASE,
)


def classify(title: str) -> str:
    if not title:
        return "neutral"
    if BOILERPLATE.search(title):
        return "boilerplate_noise"
    return "potentially_relevant"


def rescore(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rescored = []
    for r in report["results"]:
        all_titles = (
            (r.get("relevant_titles_sample") or [])
            + (r.get("noise_titles_sample") or [])
            + (r.get("neutral_titles_sample") or [])
        )
        total = r.get("total_sources") or len(all_titles)
        boiler = sum(1 for t in all_titles if classify(t) == "boilerplate_noise")
        # Extrapolate across full list using sample ratio
        if all_titles:
            boiler_ratio = boiler / len(all_titles)
            est_boiler = round(boiler_ratio * total)
        else:
            est_boiler = 0
        relevance = (total - est_boiler) / total * 100.0 if total else 0.0
        rescored.append({
            "id": r["id"],
            "question": r["question"],
            "total_sources": total,
            "boilerplate_sources_in_sample": boiler,
            "sample_size": len(all_titles),
            "estimated_boilerplate_total": est_boiler,
            "relevance_pct": round(relevance, 1),
            "passes_90pct": relevance >= 90.0,
        })
    avg = round(sum(x["relevance_pct"] for x in rescored) / max(len(rescored), 1), 1)
    passes = sum(1 for x in rescored if x["passes_90pct"])
    return {
        "source_report": str(report_path),
        "flags": report.get("flags"),
        "num_questions": len(rescored),
        "average_relevance_pct": avg,
        "questions_passing_90pct": passes,
        "overall_pass": passes == len(rescored),
        "results": rescored,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    aggregate = []
    for rp in args.report:
        aggregate.append(rescore(rp))

    out = args.out or (args.report[0].parent.parent / f"rescored_{args.report[0].parent.name}.json")
    out.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rescored report written: {out}")
    for a in aggregate:
        print(f"\n{a['source_report']}")
        print(f"  rerank={a['flags'].get('RERANKER_ENABLED','?')}  "
              f"thresh={a['flags'].get('RERANKER_SCORE_THRESHOLD','n/a')}")
        print(f"  avg_relevance={a['average_relevance_pct']}%  passes_90={a['questions_passing_90pct']}/{a['num_questions']}")
        for r in a["results"]:
            tag = "PASS" if r["passes_90pct"] else "FAIL"
            print(f"  [{r['id']}] total={r['total_sources']:>3}  boilerplate={r['estimated_boilerplate_total']:>3}  rel%={r['relevance_pct']:>5}  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
