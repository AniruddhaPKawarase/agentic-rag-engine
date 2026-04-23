"""
End-to-end test runner for the Unified RAG Agent.

Tests all 24 features we've developed, including:
- Core RAG (health, query, streaming, web, sessions)
- Answer format (natural prose, no citations)
- Deduplication (source_documents, available_documents)
- DocQA bridge (search_mode="docqa", S3 download)
- Intent classification (suggest_switch, exit_docqa)
- Multi-turn flows

Outputs:
    test_results/e2e_log_{timestamp}.jsonl  — Full request/response per test
    test_results/e2e_summary_{timestamp}.md — Human-readable summary
    test_results/e2e_summary_{timestamp}.json — Machine-readable summary

Usage:
    python run_full_e2e_tests.py [--base-url URL] [--project-id ID]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

# ===== Configuration =====
DEFAULT_BASE_URL = "http://54.197.189.113:8001"
DEFAULT_PROJECT_ID = 2361
REAL_PDF_PATH = (
    "10-998eckingtonyards0601201909072795/Contract Drawings/"
    "Mechanical/Pages/17514130232M401-ENLARGED-MECH-UNIT-PLAN---"
    "3RD-FLR-WEST-Rev.0.pdf"
)
REAL_PDF_NAME = "17514130232M401-ENLARGED-MECH-UNIT-PLAN---3RD-FLR-WEST-Rev.0.pdf"

# ===== Results Storage =====
RESULTS_DIR = Path(__file__).parent.parent.parent / "test_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = RESULTS_DIR / f"e2e_log_{TIMESTAMP}.jsonl"
SUMMARY_MD = RESULTS_DIR / f"e2e_summary_{TIMESTAMP}.md"
SUMMARY_JSON = RESULTS_DIR / f"e2e_summary_{TIMESTAMP}.json"


class TestRunner:
    def __init__(self, base_url: str, project_id: int):
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.results: list[dict] = []
        self.client = httpx.Client(timeout=180)
        self.session_id: Optional[str] = None
        self.docqa_session_id: Optional[str] = None

    def log(self, test: dict) -> None:
        """Append test result to JSONL log."""
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(test, default=str) + "\n")
        self.results.append(test)

    def run(
        self,
        test_id: str,
        description: str,
        method: str,
        path: str,
        body: Optional[dict] = None,
        assertions: Optional[list] = None,
        timeout: int = 60,
    ) -> dict:
        """Execute a single test and record result."""
        url = f"{self.base_url}{path}"
        started = time.time()

        test_result: dict[str, Any] = {
            "test_id": test_id,
            "description": description,
            "method": method,
            "url": url,
            "request_body": body,
            "started_at": datetime.now().isoformat(),
        }

        try:
            if method == "GET":
                response = self.client.get(url, timeout=timeout)
            elif method == "DELETE":
                response = self.client.delete(url, timeout=timeout)
            else:
                response = self.client.post(url, json=body, timeout=timeout)

            elapsed_ms = int((time.time() - started) * 1000)
            test_result["status_code"] = response.status_code
            test_result["elapsed_ms"] = elapsed_ms

            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text[:1000]}
            test_result["response_body"] = data

            # Run assertions
            passed = response.status_code < 400
            assertion_results = []
            for assertion in assertions or []:
                name = assertion.get("name", "unnamed")
                check = assertion.get("check")
                try:
                    result = check(data)
                    assertion_results.append({"name": name, "passed": bool(result)})
                    if not result:
                        passed = False
                except Exception as exc:
                    assertion_results.append(
                        {"name": name, "passed": False, "error": str(exc)}
                    )
                    passed = False

            test_result["assertions"] = assertion_results
            test_result["passed"] = passed
            test_result["status"] = "PASS" if passed else "FAIL"

        except httpx.TimeoutException:
            test_result["status"] = "TIMEOUT"
            test_result["passed"] = False
            test_result["elapsed_ms"] = int((time.time() - started) * 1000)
            test_result["error"] = f"Request timed out after {timeout}s"
        except Exception as exc:
            test_result["status"] = "ERROR"
            test_result["passed"] = False
            test_result["elapsed_ms"] = int((time.time() - started) * 1000)
            test_result["error"] = f"{type(exc).__name__}: {exc}"

        self.log(test_result)
        status_icon = "OK" if test_result.get("passed") else "XX"
        print(
            f"  [{status_icon}] {test_id} ({test_result.get('elapsed_ms', 0)}ms) — {description}"
        )
        if not test_result.get("passed"):
            if test_result.get("error"):
                print(f"      Error: {test_result['error']}")
            failed_assertions = [
                a for a in assertion_results if not a.get("passed")
            ] if 'assertion_results' in locals() else []
            for a in failed_assertions:
                print(f"      Failed: {a['name']}")

        return test_result


# ============================================================
# Test Suite
# ============================================================

def run_all_tests(base_url: str, project_id: int) -> TestRunner:
    runner = TestRunner(base_url, project_id)

    print(f"\n{'=' * 70}")
    print(f"Unified RAG Agent — End-to-End Test Suite")
    print(f"Base URL: {base_url}")
    print(f"Project ID: {project_id}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"{'=' * 70}\n")

    # ───────────────────────────────────────────────────────────
    # SECTION 1: Infrastructure & Health
    # ───────────────────────────────────────────────────────────
    print("[SECTION 1] Infrastructure & Health")

    runner.run(
        "T001", "API root endpoint returns service info",
        "GET", "/",
        assertions=[
            {"name": "has service name", "check": lambda d: "service" in d},
            {"name": "has version", "check": lambda d: "version" in d},
        ],
    )

    runner.run(
        "T002", "Health check returns healthy status",
        "GET", "/health",
        assertions=[
            {"name": "status is healthy", "check": lambda d: d.get("status") == "healthy"},
            {"name": "agentic engine initialized",
             "check": lambda d: d.get("engines", {}).get("agentic", {}).get("initialized")},
            {"name": "fallback enabled", "check": lambda d: d.get("fallback_enabled")},
        ],
    )

    runner.run(
        "T003", "Config endpoint returns runtime config",
        "GET", "/config",
        assertions=[
            {"name": "has agentic_model", "check": lambda d: "agentic_model" in d},
        ],
    )

    # ───────────────────────────────────────────────────────────
    # SECTION 2: Session Management
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 2] Session Management")

    sess_result = runner.run(
        "T004", "Create new session",
        "POST", "/sessions/create",
        body={"project_id": project_id},
        assertions=[
            {"name": "has session_id",
             "check": lambda d: isinstance(d.get("session_id"), str) and len(d["session_id"]) > 5},
        ],
    )
    if sess_result.get("passed"):
        runner.session_id = sess_result["response_body"]["session_id"]
        print(f"      -> Captured session_id: {runner.session_id}")

    runner.run(
        "T005", "List all sessions",
        "GET", "/sessions",
        assertions=[
            {"name": "has sessions array",
             "check": lambda d: isinstance(d.get("sessions"), list)},
            {"name": "our session is in list",
             "check": lambda d: runner.session_id in [s.get("session_id") for s in d.get("sessions", [])]},
        ],
    )

    if runner.session_id:
        runner.run(
            "T006", "Get session stats",
            "GET", f"/sessions/{runner.session_id}/stats",
        )

        runner.run(
            "T007", "Get conversation history (empty)",
            "GET", f"/sessions/{runner.session_id}/conversation",
            assertions=[
                {"name": "has conversation array",
                 "check": lambda d: isinstance(d.get("conversation"), list)},
            ],
        )

    # ───────────────────────────────────────────────────────────
    # SECTION 3: RAG Query — Answer Format
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 3] RAG Query — Answer Format (Phase 0 Fix)")

    def no_inline_citations(d):
        ans = d.get("answer", "")
        return all(marker not in ans for marker in ["[Source:", "---", "Citation:", "Direct answer"])

    q1_result = runner.run(
        "T008", "Basic RAG query returns natural prose (no inline citations)",
        "POST", "/query",
        body={
            "query": "What XVENT models are specified?",
            "project_id": project_id,
            "session_id": runner.session_id,
        },
        timeout=120,
        assertions=[
            {"name": "success true", "check": lambda d: d.get("success")},
            {"name": "has answer", "check": lambda d: len(d.get("answer", "")) > 20},
            {"name": "NO inline citations", "check": no_inline_citations},
        ],
    )

    runner.run(
        "T009", "Follow-up query uses session context",
        "POST", "/query",
        body={
            "query": "Tell me more about these models",
            "project_id": project_id,
            "session_id": runner.session_id,
        },
        timeout=120,
        assertions=[
            {"name": "success true", "check": lambda d: d.get("success")},
        ],
    )

    # ───────────────────────────────────────────────────────────
    # SECTION 4: Deduplication
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 4] Deduplication (Phase 1)")

    def no_duplicate_sources(d):
        sources = d.get("source_documents", [])
        titles = [s.get("display_title") or s.get("file_name") or s.get("s3_path", "") for s in sources]
        titles_clean = [t.lower().strip() for t in titles if t]
        return len(titles_clean) == len(set(titles_clean))

    def no_duplicate_available(d):
        docs = d.get("available_documents", [])
        titles = [doc.get("drawing_title", "") for doc in docs]
        titles_clean = [t.lower().strip() for t in titles if t]
        return len(titles_clean) == len(set(titles_clean))

    def no_generic_available(d):
        docs = d.get("available_documents", [])
        titles = [doc.get("drawing_title", "").lower().strip() for doc in docs]
        return not any(t in ("specification", "drawing", "unknown", "") for t in titles)

    runner.run(
        "T010", "Query that triggers document discovery — no duplicates",
        "POST", "/query",
        body={
            "query": "unrelated random query that wont match anything xyzabc123",
            "project_id": project_id,
        },
        timeout=60,
        assertions=[
            {"name": "available_documents deduplicated", "check": no_duplicate_available},
            {"name": "no generic titles in available_documents", "check": no_generic_available},
        ],
    )

    runner.run(
        "T011", "Source documents in normal query — no duplicates",
        "POST", "/query",
        body={
            "query": "What notes are on the floor plans?",
            "project_id": project_id,
        },
        timeout=120,
        assertions=[
            {"name": "source_documents deduplicated", "check": no_duplicate_sources},
        ],
    )

    # ───────────────────────────────────────────────────────────
    # SECTION 5: Quick Query
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 5] Quick Query Endpoint")

    runner.run(
        "T012", "Quick query returns simplified response",
        "POST", "/quick-query",
        body={
            "query": "List electrical drawings",
            "project_id": project_id,
        },
        timeout=120,
        assertions=[
            {"name": "has answer", "check": lambda d: "answer" in d},
            {"name": "has confidence", "check": lambda d: "confidence" in d},
            {"name": "has engine_used", "check": lambda d: "engine_used" in d},
        ],
    )

    # ───────────────────────────────────────────────────────────
    # SECTION 6: Web Search
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 6] Web Search")

    runner.run(
        "T013", "Web-only search",
        "POST", "/web-search",
        body={
            "query": "ASHRAE 90.1 requirements for HVAC",
            "project_id": project_id,
        },
        timeout=60,
        assertions=[
            {"name": "response returned",
             "check": lambda d: "result" in d or "answer" in d or "error" in d},
        ],
    )

    runner.run(
        "T014", "Hybrid search (RAG + Web)",
        "POST", "/query",
        body={
            "query": "What are industry standards for ductwork?",
            "project_id": project_id,
            "search_mode": "hybrid",
        },
        timeout=120,
        assertions=[
            {"name": "has answer", "check": lambda d: len(d.get("answer", "")) > 20},
        ],
    )

    # ───────────────────────────────────────────────────────────
    # SECTION 7: Pin Document (Legacy)
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 7] Pin Document (Legacy Endpoints)")

    if runner.session_id:
        runner.run(
            "T015", "Pin document to session",
            "POST", f"/sessions/{runner.session_id}/pin-document",
            body={"document_ids": ["test_doc_1"]},
            assertions=[
                {"name": "success response", "check": lambda d: d.get("success") is True or "success" in d},
            ],
        )

        runner.run(
            "T016", "Unpin document from session",
            "DELETE", f"/sessions/{runner.session_id}/pin-document",
            body={"document_ids": ["test_doc_1"]},
            assertions=[
                {"name": "success response", "check": lambda d: d.get("success") is True or "success" in d},
            ],
        )

    # ───────────────────────────────────────────────────────────
    # SECTION 8: DocQA Bridge — Error Cases (No Document)
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 8] DocQA Bridge — Error & Edge Cases")

    if runner.session_id:
        runner.run(
            "T017", "DocQA mode without document selected",
            "POST", "/query",
            body={
                "query": "What does this document say?",
                "project_id": project_id,
                "session_id": runner.session_id,
                "search_mode": "docqa",
            },
            timeout=30,
            assertions=[
                {"name": "asks for document selection",
                 "check": lambda d: d.get("needs_document_selection")
                 or "select" in d.get("answer", "").lower()},
            ],
        )

    # ───────────────────────────────────────────────────────────
    # SECTION 9: Intent Classification (Phase 2)
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 9] Intent Classification — Agent Switching")

    if runner.session_id:
        runner.run(
            "T018", "Intent: project-wide query in DocQA mode → suggest switch",
            "POST", "/query",
            body={
                "query": "Show all HVAC equipment across the project",
                "project_id": project_id,
                "session_id": runner.session_id,
                "search_mode": "docqa",
            },
            timeout=30,
            assertions=[
                {"name": "suggest_switch is rag",
                 "check": lambda d: d.get("suggest_switch") == "rag"},
            ],
        )

        runner.run(
            "T019", "Intent: 'back to project search' exits DocQA mode",
            "POST", "/query",
            body={
                "query": "back to project search",
                "project_id": project_id,
                "session_id": runner.session_id,
                "search_mode": "docqa",
            },
            timeout=30,
            assertions=[
                {"name": "active_agent is rag",
                 "check": lambda d: d.get("active_agent") == "rag"},
            ],
        )

    # ───────────────────────────────────────────────────────────
    # SECTION 10: DocQA Full Flow (S3 + DocQA Agent)
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 10] DocQA Full Flow (S3 Download + DocQA Upload)")

    docqa_session = runner.run(
        "T020", "Create fresh session for DocQA flow",
        "POST", "/sessions/create",
        body={"project_id": project_id},
        assertions=[
            {"name": "has session_id", "check": lambda d: isinstance(d.get("session_id"), str)},
        ],
    )
    docqa_sess_id = None
    if docqa_session.get("passed"):
        docqa_sess_id = docqa_session["response_body"]["session_id"]

    if docqa_sess_id:
        def docqa_answered(d):
            ans = d.get("answer", "")
            return d.get("success") and len(ans) > 50 and "Could not download" not in ans

        runner.run(
            "T021", "DocQA upload + initial query (real S3 PDF)",
            "POST", "/query",
            body={
                "query": "Summarize what this mechanical drawing contains in 3 sentences",
                "project_id": project_id,
                "session_id": docqa_sess_id,
                "search_mode": "docqa",
                "docqa_document": {
                    "s3_path": REAL_PDF_PATH,
                    "file_name": REAL_PDF_NAME,
                },
            },
            timeout=180,
            assertions=[
                {"name": "successful answer", "check": docqa_answered},
                {"name": "engine is docqa", "check": lambda d: d.get("engine_used") == "docqa"},
                {"name": "active_agent is docqa", "check": lambda d: d.get("active_agent") == "docqa"},
            ],
        )

        runner.run(
            "T022", "DocQA follow-up query (same session, no re-upload)",
            "POST", "/query",
            body={
                "query": "What heat pump equipment is shown?",
                "project_id": project_id,
                "session_id": docqa_sess_id,
                "search_mode": "docqa",
            },
            timeout=60,
            assertions=[
                {"name": "successful answer", "check": lambda d: d.get("success") and len(d.get("answer", "")) > 20},
                {"name": "engine is docqa", "check": lambda d: d.get("engine_used") == "docqa"},
            ],
        )

    # ───────────────────────────────────────────────────────────
    # SECTION 11: Session Cleanup
    # ───────────────────────────────────────────────────────────
    print("\n[SECTION 11] Session Cleanup")

    if runner.session_id:
        runner.run(
            "T023", "Delete session",
            "DELETE", f"/sessions/{runner.session_id}",
            assertions=[
                {"name": "success response", "check": lambda d: d.get("success") is True or "success" in d},
            ],
        )

    if docqa_sess_id:
        runner.run(
            "T024", "Delete DocQA session",
            "DELETE", f"/sessions/{docqa_sess_id}",
            assertions=[
                {"name": "success response", "check": lambda d: d.get("success") is True or "success" in d},
            ],
        )

    return runner


# ============================================================
# Report Generation
# ============================================================

def generate_reports(runner: TestRunner) -> dict:
    total = len(runner.results)
    passed = sum(1 for r in runner.results if r.get("passed"))
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0

    # Group by section
    sections: dict[str, list] = {}
    for r in runner.results:
        tid = r.get("test_id", "")
        num = int(tid[1:]) if tid.startswith("T") and tid[1:].isdigit() else 0
        section_map = {
            (1, 3): "1. Infrastructure & Health",
            (4, 7): "2. Session Management",
            (8, 9): "3. RAG Query — Answer Format",
            (10, 11): "4. Deduplication",
            (12, 12): "5. Quick Query",
            (13, 14): "6. Web Search & Hybrid",
            (15, 16): "7. Pin Document (Legacy)",
            (17, 17): "8. DocQA Edge Cases",
            (18, 19): "9. Intent Classification",
            (20, 22): "10. DocQA Full Flow",
            (23, 24): "11. Session Cleanup",
        }
        section = "Misc"
        for (lo, hi), name in section_map.items():
            if lo <= num <= hi:
                section = name
                break
        sections.setdefault(section, []).append(r)

    summary = {
        "timestamp": TIMESTAMP,
        "base_url": runner.base_url,
        "project_id": runner.project_id,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 1),
        "total_elapsed_ms": sum(r.get("elapsed_ms", 0) for r in runner.results),
        "log_file": str(LOG_FILE.name),
        "results_by_section": {
            section: {
                "passed": sum(1 for r in items if r.get("passed")),
                "total": len(items),
                "tests": [
                    {
                        "test_id": r.get("test_id"),
                        "description": r.get("description"),
                        "status": r.get("status"),
                        "elapsed_ms": r.get("elapsed_ms"),
                        "status_code": r.get("status_code"),
                    }
                    for r in items
                ],
            }
            for section, items in sections.items()
        },
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown report
    md_lines = [
        f"# Unified RAG Agent — E2E Test Report",
        f"",
        f"**Timestamp:** {TIMESTAMP}",
        f"**Base URL:** {runner.base_url}",
        f"**Project ID:** {runner.project_id}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tests | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Pass rate | {pass_rate:.1f}% |",
        f"| Total latency | {summary['total_elapsed_ms'] / 1000:.1f}s |",
        f"",
    ]

    for section_name, items in sections.items():
        sec_passed = sum(1 for r in items if r.get("passed"))
        md_lines.append(f"## {section_name} ({sec_passed}/{len(items)} passed)")
        md_lines.append("")
        md_lines.append("| Test ID | Description | Status | Code | Time (ms) |")
        md_lines.append("|---------|-------------|--------|------|-----------|")
        for r in items:
            status = r.get("status", "?")
            icon = "PASS" if r.get("passed") else "FAIL"
            md_lines.append(
                f"| {r.get('test_id')} | {r.get('description')} | {icon} | {r.get('status_code', '-')} | {r.get('elapsed_ms', 0)} |"
            )
        md_lines.append("")

        # Show failures with details
        failures = [r for r in items if not r.get("passed")]
        if failures:
            md_lines.append(f"### Failures in {section_name}")
            md_lines.append("")
            for r in failures:
                md_lines.append(f"**{r.get('test_id')}: {r.get('description')}**")
                if r.get("error"):
                    md_lines.append(f"- Error: `{r.get('error')}`")
                for a in r.get("assertions", []) or []:
                    if not a.get("passed"):
                        md_lines.append(f"- Failed assertion: {a.get('name')}")
                resp = r.get("response_body", {})
                if isinstance(resp, dict) and resp.get("error"):
                    md_lines.append(f"- Server error: `{resp['error']}`")
                md_lines.append("")

    md_lines.extend([
        "",
        f"## Artifacts",
        f"",
        f"- Full log (JSONL): `{LOG_FILE.name}`",
        f"- Summary (JSON): `{SUMMARY_JSON.name}`",
        f"- Summary (MD): `{SUMMARY_MD.name}`",
    ])

    SUMMARY_MD.write_text("\n".join(md_lines), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    args = ap.parse_args()

    runner = run_all_tests(args.base_url, args.project_id)
    summary = generate_reports(runner)

    print(f"\n{'=' * 70}")
    print(f"Test Run Complete")
    print(f"{'=' * 70}")
    print(f"Total:    {summary['total_tests']}")
    print(f"Passed:   {summary['passed']}")
    print(f"Failed:   {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']}%")
    print(f"Elapsed:   {summary['total_elapsed_ms'] / 1000:.1f}s")
    print(f"\nArtifacts saved to: {RESULTS_DIR}")
    print(f"  - Log:     {LOG_FILE.name}")
    print(f"  - Summary: {SUMMARY_MD.name}")
    print(f"  - JSON:    {SUMMARY_JSON.name}")

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
