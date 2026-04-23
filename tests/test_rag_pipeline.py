#!/usr/bin/env python3
"""
Unified RAG Agent -- Async Bulk Testing Pipeline
=================================================
Reads ~600 questions from an xlsx file and tests them against the
sandbox VM's unified RAG API for every project across all collections.

Uses async aiohttp for maximum throughput -- all projects tested in
parallel with configurable concurrency per project.

Output: One CSV per project + a JSON summary.

Usage:
    # Sample test (5 questions, 1 project)
    python test_rag_pipeline.py --sample 5 --projects 7325

    # Full run (all questions, all projects)
    python test_rag_pipeline.py

    # High concurrency (10 in-flight per project, all projects parallel)
    python test_rag_pipeline.py --max-workers 10

    # Sequential projects (lower VM load)
    python test_rag_pipeline.py --sequential
"""

import os
import sys
import re
import csv
import json
import time
import asyncio
import argparse
from datetime import datetime

import aiohttp
import openpyxl
from tqdm import tqdm


# --- Configuration -----------------------------------------------------------

SANDBOX_VM = "http://54.197.189.113:8001"

DEFAULT_QUESTIONS_PATH = r"C:\Users\ANIRUDDHA ASUS\Downloads\testing_questions for rag.xlsx"
DEFAULT_OUTPUT_DIR = r"C:\Users\ANIRUDDHA ASUS\Downloads\projects\VCS\VCS\PROD_SETUP\unified-rag-agent\test_results"

KNOWN_PROJECTS = [7166, 7201, 7212, 7222, 7223, 7277, 7292, 7325]

CONFIG = {
    "timeout": 120,
    "vm_url": SANDBOX_VM,
}


# --- Question Parser ---------------------------------------------------------

def read_questions(xlsx_path: str) -> list[str]:
    """Read questions from xlsx, skipping category headers."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Sheet1"]
    questions = []

    for row in ws.iter_rows(min_row=1, values_only=True):
        raw = str(row[0] if row[0] else "").strip()
        if not raw:
            continue
        if not raw[0].isdigit():
            continue
        clean = re.sub(r"^\d+[\xa0\s]*", "", raw).strip()
        if clean:
            questions.append(clean)

    wb.close()
    return questions


# --- Async API Caller --------------------------------------------------------

def _parse_response(question: str, data: dict, latency: float) -> dict:
    """Extract flat result dict from API response."""
    source_docs = data.get("source_documents", [])
    source_titles = "; ".join(
        d.get("display_title", d.get("file_name", "N/A")) for d in source_docs[:5]
    )
    pdf_names = "; ".join(
        d.get("file_name", "N/A") for d in source_docs[:5]
    )
    s3_paths = "; ".join(data.get("s3_paths", [])[:5])

    token_info = data.get("token_usage") or {}
    total_tokens = token_info.get("total_tokens", 0)

    answer_text = (data.get("answer") or "")[:5000]
    error_text = data.get("error", "") or ""
    # success field can be True, False, or null — treat as success if answer exists and no error
    is_success = bool(answer_text.strip()) and not bool(error_text)

    return {
        "question": question,
        "answer": answer_text,
        "confidence_score": data.get("confidence_score", 0),
        "source_document": source_titles,
        "pdf_name": pdf_names,
        "s3_path": s3_paths,
        "latency_seconds": latency,
        "server_latency_ms": data.get("processing_time_ms", 0),
        "tokens_used": total_tokens,
        "engine_used": data.get("engine_used", ""),
        "model_used": data.get("model_used", ""),
        "retrieval_count": data.get("retrieval_count", 0),
        "success": is_success,
        "error": error_text,
    }


def _error_result(question: str, latency: float, error: str) -> dict:
    return {
        "question": question,
        "answer": "",
        "confidence_score": 0,
        "source_document": "",
        "pdf_name": "",
        "s3_path": "",
        "latency_seconds": latency,
        "server_latency_ms": 0,
        "tokens_used": 0,
        "engine_used": "",
        "model_used": "",
        "retrieval_count": 0,
        "success": False,
        "error": error,
    }


async def query_rag_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    question: str,
    project_id: int,
    collection_filter: str | None,
    pbar: tqdm,
) -> dict:
    """Send one question via async HTTP, respecting the semaphore."""
    payload = {"query": question, "project_id": project_id}
    if collection_filter and collection_filter != "all":
        payload["filter_source_type"] = collection_filter

    url = f"{CONFIG['vm_url']}/query"
    timeout = aiohttp.ClientTimeout(total=CONFIG["timeout"])

    async with semaphore:
        start = time.time()
        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                latency = round(time.time() - start, 2)
                if resp.status != 200:
                    text = await resp.text()
                    result = _error_result(question, latency, f"HTTP {resp.status}: {text[:300]}")
                else:
                    data = await resp.json()
                    result = _parse_response(question, data, latency)
        except asyncio.TimeoutError:
            result = _error_result(question, round(time.time() - start, 2), "TIMEOUT")
        except aiohttp.ClientError as e:
            result = _error_result(question, round(time.time() - start, 2), f"CONNECTION: {e}")
        except Exception as e:
            result = _error_result(question, round(time.time() - start, 2), str(e)[:300])

    pbar.update(1)
    return result


# --- Per-Project Runner (async) ----------------------------------------------

CSV_FIELDS = [
    "question", "answer", "confidence_score", "source_document",
    "pdf_name", "s3_path", "latency_seconds", "server_latency_ms",
    "tokens_used", "engine_used", "model_used", "retrieval_count",
    "success", "error",
]


async def test_project_async(
    session: aiohttp.ClientSession,
    questions: list[str],
    project_id: int,
    output_dir: str,
    max_workers: int,
    collection_filter: str | None,
) -> tuple[list[dict], str, dict]:
    """Run all questions for one project asynchronously."""
    semaphore = asyncio.Semaphore(max_workers)

    desc = f"Project {project_id}"
    if collection_filter:
        desc += f" [{collection_filter}]"

    pbar = tqdm(
        total=len(questions),
        desc=desc,
        unit="q",
        leave=True,
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    tasks = [
        query_rag_async(session, semaphore, q, project_id, collection_filter, pbar)
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    pbar.close()

    # Write CSV
    suffix = f"_{collection_filter}" if collection_filter else ""
    csv_path = os.path.join(output_dir, f"project_{project_id}{suffix}_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Stats
    successful = [r for r in results if r["success"]]
    avg_conf = (
        sum(r["confidence_score"] for r in successful) / len(successful)
        if successful else 0
    )
    avg_lat = (
        sum(r["latency_seconds"] for r in successful) / len(successful)
        if successful else 0
    )
    total_tokens = sum(r["tokens_used"] for r in results)

    summary = {
        "project_id": project_id,
        "total_questions": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_confidence": round(avg_conf, 3),
        "avg_latency_s": round(avg_lat, 2),
        "total_tokens": total_tokens,
        "csv_file": csv_path,
    }

    print(f"  Project {project_id}: {len(successful)}/{len(results)} ok, "
          f"conf={avg_conf:.3f}, lat={avg_lat:.1f}s, tokens={total_tokens}")

    return results, csv_path, summary


# --- Health Check (sync -- runs once) ----------------------------------------

def check_vm_health(vm_url: str) -> bool:
    import requests
    try:
        r = requests.get(f"{vm_url}/health", timeout=10)
        data = r.json()
        print(f"  VM Status   : {data.get('status', 'unknown')}")
        engines = data.get("engines", {})
        print(f"  Agentic     : {'OK' if engines.get('agentic', {}).get('initialized') else 'UNAVAILABLE'}")
        print(f"  Traditional : {'OK' if engines.get('traditional', {}).get('faiss_loaded') else 'UNAVAILABLE'}")
        return data.get("status") == "healthy"
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# --- Project Discovery (async) -----------------------------------------------

async def discover_projects_async(vm_url: str, project_ids: list[int]) -> list[int]:
    """Quick-check which projects return data, all in parallel."""
    active = []
    print("\nDiscovering active projects (parallel)...")

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def check_one(pid: int) -> tuple[int, bool, str]:
            try:
                async with session.post(
                    f"{vm_url}/query",
                    json={"query": "List the main trades on this project", "project_id": pid},
                    timeout=timeout,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        has_answer = bool(data.get("answer") and len(str(data["answer"]).strip()) > 20)
                        has_error = bool(data.get("error"))
                        if has_answer and not has_error:
                            engine = data.get("engine_used", "unknown")
                            conf = data.get("confidence_score", 0)
                            return pid, True, f"engine={engine}, conf={conf}"
                        return pid, False, data.get("error", "empty/short answer")[:80]
                    return pid, False, f"HTTP {resp.status}"
            except Exception as e:
                return pid, False, str(e)[:80]

        tasks = [check_one(pid) for pid in project_ids]
        for coro in asyncio.as_completed(tasks):
            pid, ok, msg = await coro
            status = "ACTIVE" if ok else "SKIP"
            print(f"  Project {pid}: {status} -- {msg}")
            if ok:
                active.append(pid)

    active.sort()
    return active


# --- Main (async) ------------------------------------------------------------

async def run(args: argparse.Namespace):
    CONFIG["timeout"] = args.timeout
    CONFIG["vm_url"] = args.vm_url

    print("=" * 65)
    print("  Unified RAG Agent -- Async Bulk Testing Pipeline")
    print("=" * 65)

    # 1. Health check
    print("\n[1/4] Checking VM health...")
    if not check_vm_health(args.vm_url):
        print("ERROR: Sandbox VM is not healthy. Aborting.")
        sys.exit(1)

    # 2. Read questions
    print(f"\n[2/4] Reading questions from: {args.questions}")
    if not os.path.exists(args.questions):
        print(f"ERROR: File not found: {args.questions}")
        sys.exit(1)

    questions = read_questions(args.questions)
    print(f"  Parsed {len(questions)} questions")

    if args.sample > 0:
        questions = questions[: args.sample]
        print(f"  Using sample: first {len(questions)} questions")

    # 3. Discover projects
    project_ids = args.projects or KNOWN_PROJECTS
    if not args.skip_discovery and not args.projects:
        print(f"\n[3/4] Discovering active projects from {len(project_ids)} candidates...")
        active_projects = await discover_projects_async(args.vm_url, project_ids)
        if not active_projects:
            print("ERROR: No active projects found. Aborting.")
            sys.exit(1)
        print(f"  Active projects: {active_projects}")
    else:
        active_projects = project_ids
        print(f"\n[3/4] Using specified projects: {active_projects}")

    # 4. Run tests
    os.makedirs(args.output_dir, exist_ok=True)
    total_calls = len(questions) * len(active_projects)
    collection_filter = args.collection if args.collection != "all" else None

    # Estimate: with async, effective parallelism is max_workers * num_projects
    effective_parallel = args.max_workers * (1 if args.sequential else len(active_projects))
    est_seconds = (total_calls * 10) / effective_parallel
    est_minutes = est_seconds / 60

    mode = "SEQUENTIAL" if args.sequential else f"PARALLEL ({len(active_projects)} projects)"

    print(f"\n[4/4] Running tests -- async aiohttp")
    print(f"  Questions        : {len(questions)}")
    print(f"  Projects         : {len(active_projects)} {active_projects}")
    print(f"  Collection       : {args.collection}")
    print(f"  Concurrency/proj : {args.max_workers}")
    print(f"  Mode             : {mode}")
    print(f"  Total API calls  : {total_calls}")
    print(f"  Est. time        : ~{est_minutes:.0f} min")
    print(f"  Output dir       : {args.output_dir}")
    print()

    all_summaries = {}
    overall_start = time.time()

    # Connection pool: limit total connections to avoid overwhelming the VM
    total_limit = args.max_workers * len(active_projects) if not args.sequential else args.max_workers
    total_limit = min(total_limit, 30)  # cap at 30 to be safe
    connector = aiohttp.TCPConnector(limit=total_limit, keepalive_timeout=30)

    async with aiohttp.ClientSession(connector=connector) as session:
        if args.sequential:
            # One project at a time
            for i, pid in enumerate(active_projects):
                print(f"\n-- Project {pid}  ({i + 1}/{len(active_projects)}) --")
                _, csv_path, summary = await test_project_async(
                    session, questions, pid, args.output_dir,
                    args.max_workers, collection_filter,
                )
                all_summaries[pid] = summary
        else:
            # ALL projects in parallel -- massive speedup
            print("  Launching all projects in parallel...\n")
            project_tasks = [
                test_project_async(
                    session, questions, pid, args.output_dir,
                    args.max_workers, collection_filter,
                )
                for pid in active_projects
            ]
            project_results = await asyncio.gather(*project_tasks)
            for _, _, summary in project_results:
                all_summaries[summary["project_id"]] = summary

    # Overall summary
    elapsed = round(time.time() - overall_start, 1)
    summary_path = os.path.join(args.output_dir, "test_summary.json")
    overall = {
        "test_date": datetime.now().isoformat(),
        "total_questions": len(questions),
        "projects_tested": active_projects,
        "collection_filter": args.collection,
        "vm_url": args.vm_url,
        "mode": "sequential" if args.sequential else "parallel",
        "concurrency_per_project": args.max_workers,
        "total_elapsed_seconds": elapsed,
        "per_project": {str(k): v for k, v in all_summaries.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  COMPLETE -- {elapsed:.0f}s elapsed")
    print(f"  Summary : {summary_path}")
    print(f"  Results : {args.output_dir}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified RAG Agent -- Async Bulk Testing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questions", default=DEFAULT_QUESTIONS_PATH,
        help="Path to questions xlsx file",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-project CSV results",
    )
    parser.add_argument(
        "--projects", nargs="+", type=int, default=None,
        help="Project IDs to test (default: auto-discover from known list)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Concurrent requests per project (default: 10)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Test only first N questions (0 = all)",
    )
    parser.add_argument(
        "--collection", choices=["all", "drawing", "specification"],
        default="all",
        help="Filter by collection type (default: all)",
    )
    parser.add_argument(
        "--vm-url", default=SANDBOX_VM,
        help=f"Sandbox VM URL (default: {SANDBOX_VM})",
    )
    parser.add_argument(
        "--skip-discovery", action="store_true",
        help="Skip project discovery and test all listed projects",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Test projects one at a time (lower VM load, slower)",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
