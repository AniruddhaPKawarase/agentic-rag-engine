"""Test the 4 manager-reported bug questions on project 7201 and measure
source-relevance % against the expected domain keywords per question.

Reported bugs (from ``project Q&A bugs.docx``):
- Q1 "give me schedule mechanical" → sources contaminated with life-safety /
  cover / ADA drawings instead of mechanical schedules.
- Q2 "give me RCP for first floor" → correct sheets present, but padded with
  unrelated plans (cover, notes, cellar life safety, etc.).
- Q3 "where is the fcu located" → answer cites Z-017.00 but reference list
  shows unrelated floor plans.
- Q4 "what is the size of sanitary pipe on first floor" → NO plumbing
  drawings in the reference list at all; only cover / ADA / life safety.

For each question we compute:
  relevance = (# of sources whose display title matches the expected-domain
               regex) / (total source count)

Target: ≥ 0.9 on all four.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from gateway.orchestrator import Orchestrator  # noqa: E402


PROJECT_ID = 7201

# Each question declares which sources are "on-topic" and which are "noise".
# A matching source counts as relevant. Relevance = relevant / total.
BUG_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "Q1",
        "question": "give me schedule mechanical",
        "expected_domain": "mechanical schedules / mechanical drawings",
        "relevant_regex": re.compile(
            r"mechanical|M-\d|HVAC|schedule|VAV|AHU|DOAS|fan coil|air handling",
            re.IGNORECASE,
        ),
        "noise_regex": re.compile(
            r"cover sheet|area plan|site photograph|list of drawings|general note|"
            r"ADA requirement|building code|life safety|slab edge|link beam|"
            r"bathroom plans|elevators - plans|typical section",
            re.IGNORECASE,
        ),
    },
    {
        "id": "Q2",
        "question": "give me RCP for first floor",
        "expected_domain": "1st floor reflected ceiling plan (RCP)",
        "relevant_regex": re.compile(
            r"1ST\s*FLOOR.*RCP|REFLECTED\s*CEILING.*(1ST|FIRST|LEGEND)|"
            r"A-725|A-750|RCP\s*(1ST|FIRST|LEGEND|SCHEDULE)",
            re.IGNORECASE,
        ),
        "noise_regex": re.compile(
            r"cover sheet|area plan|site photograph|list of drawings|general note|"
            r"ADA requirement|building code|life safety|slab edge|8TH FLOOR RCP|"
            r"21ST FLOOR RCP|SECOND FLOOR|3RD FLOOR T\.O\.S|2ND FLOOR PLAN",
            re.IGNORECASE,
        ),
    },
    {
        "id": "Q3",
        "question": "where is the fcu located",
        "expected_domain": "mechanical / FCU / corridor deduction sheets",
        "relevant_regex": re.compile(
            r"FCU|fan coil|mechanical|corridor deduction|Z-017|HVAC|M-\d",
            re.IGNORECASE,
        ),
        "noise_regex": re.compile(
            r"cover sheet|life safety|ADA|general note|area plan|building code|"
            r"slab edge|8TH FLOOR RCP|11TH FLOOR|12TH FLOOR LIFE SAFETY",
            re.IGNORECASE,
        ),
    },
    {
        "id": "Q4",
        "question": "what is the size of sanitary pipe on first floor",
        "expected_domain": "plumbing / sanitary / P-prefix sheets",
        "relevant_regex": re.compile(
            r"plumb|sanitary|waste|vent|P-\d|drain|riser|first floor plumb|"
            r"1st floor plumb|drainage",
            re.IGNORECASE,
        ),
        "noise_regex": re.compile(
            r"cover sheet|area plan|site photograph|list of drawings|general note|"
            r"ADA requirement|building code|life safety",
            re.IGNORECASE,
        ),
    },
]


def classify_source(src: dict[str, Any], relevant: re.Pattern, noise: re.Pattern) -> str:
    """Return 'relevant', 'noise', or 'neutral' for one source dict."""
    title = " ".join(str(src.get(k) or "") for k in (
        "display_title", "drawing_title", "drawingTitle", "pdf_name", "pdfName",
        "file_name", "drawing_name", "drawingName",
    )).strip()
    if not title:
        return "neutral"
    if relevant.search(title):
        return "relevant"
    if noise.search(title):
        return "noise"
    return "neutral"


async def run_one(orch: Orchestrator, q: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()
    try:
        resp = await orch.query(query=q["question"], project_id=PROJECT_ID)
    except Exception as exc:
        return {
            "id": q["id"],
            "question": q["question"],
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - start, 2),
        }
    elapsed = round(time.monotonic() - start, 2)

    sds = resp.get("source_documents") or []
    buckets = {"relevant": [], "noise": [], "neutral": []}
    for sd in sds:
        cls = classify_source(sd, q["relevant_regex"], q["noise_regex"])
        title = (
            sd.get("display_title") or sd.get("drawing_title") or sd.get("drawingTitle")
            or sd.get("pdf_name") or sd.get("pdfName") or sd.get("file_name") or ""
        )
        buckets[cls].append(title[:120])

    total = len(sds)
    relevant = len(buckets["relevant"])
    noise = len(buckets["noise"])
    relevance_pct = (relevant / total * 100.0) if total else 0.0
    noise_pct = (noise / total * 100.0) if total else 0.0

    return {
        "id": q["id"],
        "question": q["question"],
        "expected_domain": q["expected_domain"],
        "engine_used": resp.get("engine_used"),
        "fallback_used": resp.get("fallback_used"),
        "confidence": resp.get("confidence"),
        "answer_length": len(resp.get("answer") or ""),
        "answer_preview": (resp.get("answer") or "")[:500],
        "total_sources": total,
        "relevant_sources": relevant,
        "noise_sources": noise,
        "neutral_sources": total - relevant - noise,
        "relevance_pct": round(relevance_pct, 1),
        "noise_pct": round(noise_pct, 1),
        "passes_90pct": relevance_pct >= 90.0,
        "relevant_titles_sample": buckets["relevant"][:8],
        "noise_titles_sample": buckets["noise"][:8],
        "neutral_titles_sample": buckets["neutral"][:8],
        "elapsed_s": elapsed,
    }


async def main_async() -> int:
    orch = Orchestrator(fallback_enabled=True, fallback_timeout=30)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "test_results" / f"bug_relevancy_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for q in BUG_QUESTIONS:
        print(f"[{q['id']}] {q['question']}", flush=True)
        rec = await run_one(orch, q)
        results.append(rec)
        rel = rec.get("relevance_pct")
        print(
            f"  -> {rec.get('total_sources')} sources  "
            f"relevant={rec.get('relevant_sources')}  noise={rec.get('noise_sources')}  "
            f"relevance={rel}%  {'PASS' if rec.get('passes_90pct') else 'FAIL'}",
            flush=True,
        )

    avg_rel = round(sum(r.get("relevance_pct") or 0 for r in results) / len(results), 1)
    passes = sum(1 for r in results if r.get("passes_90pct"))
    overall = {
        "timestamp": datetime.now().isoformat(),
        "project_id": PROJECT_ID,
        "version_tested": "v1.2-hybrid-ship",
        "flags": {
            "MULTI_QUERY_RRF_ENABLED": "true",
            "RERANKER_ENABLED": "true",
            "SELF_RAG_ENABLED": "false",
            "RERANKER_INCLUDE_SCORE": "false",
        },
        "num_questions": len(results),
        "average_relevance_pct": avg_rel,
        "questions_passing_90pct": passes,
        "target_pct": 90.0,
        "overall_pass": passes == len(results),
        "results": results,
    }
    (out_dir / "bug_relevancy_report.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nAverage relevance: {avg_rel}%")
    print(f"Questions ≥90%:    {passes}/{len(results)}")
    print(f"Artifacts:         {out_dir}")
    print(f"Pass overall:      {overall['overall_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
