"""
Hallucination Guard — Adversarial Eval Runner

Executes the 50-Q adversarial gold set against the live 8001 service.
Captures full multi-turn responses for each test; outputs responses.jsonl
ready for grader.py.

Usage:
    python run_eval.py --gold adversarial_gold_set.jsonl \\
                       --endpoint http://127.0.0.1:8001/query \\
                       --tenant test-tenant \\
                       --output responses_baseline_<ts>.jsonl

The runner is BLACK-BOX — it doesn't know whether HG flags are on or off.
That's intentional; pre/post comparison happens at the report level.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


DEFAULT_TIMEOUT = 60  # seconds per /query call


def run_test(test: dict, endpoint: str, project_id: int, tenant: str, extra_headers: dict = None) -> dict:
    """Execute one multi-turn test. Returns response dict with all turns."""
    session_id = f"adv-eval-{test['id']}-{int(time.time())}"
    turns_response = []
    headers = {
        "Content-Type": "application/json",
        "X-Tenant-Id": tenant,
        "X-Session-Id": session_id,
        "X-Eval-Source": "hg_adversarial_v1",
    }
    if extra_headers:
        headers.update(extra_headers)

    cumulative_history = []  # list of {role, content} for multi-turn context

    for turn_idx, turn in enumerate(test.get("turns", [])):
        if turn.get("role") != "user":
            # Skip non-user turns (these describe expected behavior)
            continue

        user_msg = turn.get("content", "")
        cumulative_history.append({"role": "user", "content": user_msg})

        # 8001 /query actual schema requires project_id (int) + query (str).
        # tenant is unused in the API (kept for response identification only).
        payload = {
            "query": user_msg[:2000],
            "project_id": project_id,
            "session_id": session_id,
            "conversation_history": cumulative_history[:-1],
        }

        t_start = time.time()
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
            latency_ms = int((time.time() - t_start) * 1000)
            if resp.status_code == 200:
                body = resp.json()
                answer = body.get("answer") or body.get("response") or ""
                sources = (
                    body.get("all_retrieved_sources")
                    or body.get("sources")
                    or body.get("citations")
                    or []
                )
                cumulative_history.append({"role": "assistant", "content": answer})
                turns_response.append({
                    "turn_idx": turn_idx,
                    "user": user_msg,
                    "answer": answer,
                    "sources": sources[:10],  # cap to avoid huge payloads
                    "latency_ms": latency_ms,
                    "status": 200,
                })
            elif resp.status_code in (429, 503):
                # Rate-limited or unavailable — back off and retry once
                time.sleep(5)
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
                latency_ms = int((time.time() - t_start) * 1000)
                if resp.status_code == 200:
                    body = resp.json()
                    answer = body.get("answer") or body.get("response") or ""
                    sources = body.get("all_retrieved_sources") or body.get("sources") or []
                    cumulative_history.append({"role": "assistant", "content": answer})
                    turns_response.append({
                        "turn_idx": turn_idx,
                        "user": user_msg,
                        "answer": answer,
                        "sources": sources[:10],
                        "latency_ms": latency_ms,
                        "status": 200,
                        "retried": True,
                    })
                else:
                    turns_response.append({
                        "turn_idx": turn_idx,
                        "user": user_msg,
                        "answer": f"[ERROR {resp.status_code}]",
                        "sources": [],
                        "latency_ms": latency_ms,
                        "status": resp.status_code,
                    })
                    break
            else:
                turns_response.append({
                    "turn_idx": turn_idx,
                    "user": user_msg,
                    "answer": f"[ERROR {resp.status_code}: {resp.text[:200]}]",
                    "sources": [],
                    "latency_ms": latency_ms,
                    "status": resp.status_code,
                })
                break
        except requests.exceptions.Timeout:
            latency_ms = int((time.time() - t_start) * 1000)
            turns_response.append({
                "turn_idx": turn_idx,
                "user": user_msg,
                "answer": "[TIMEOUT]",
                "sources": [],
                "latency_ms": latency_ms,
                "status": -1,
            })
            break
        except Exception as exc:
            latency_ms = int((time.time() - t_start) * 1000)
            turns_response.append({
                "turn_idx": turn_idx,
                "user": user_msg,
                "answer": f"[EXCEPTION {type(exc).__name__}: {exc}]",
                "sources": [],
                "latency_ms": latency_ms,
                "status": -2,
            })
            break

    return {
        "id": test["id"],
        "category": test.get("category"),
        "pillar": test.get("pillar"),
        "session_id": session_id,
        "turns_response": turns_response,
        "metadata": {
            "n_turns_executed": len(turns_response),
            "total_latency_ms": sum(t.get("latency_ms", 0) for t in turns_response),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="adversarial_gold_set.jsonl", help="Path to gold set JSONL")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/query",
                    help="8001 /query endpoint URL")
    ap.add_argument("--tenant", default="hg-adversarial-eval", help="Tenant ID label (eval bookkeeping only; 8001 /query does not use it)")
    ap.add_argument("--project-id", type=int, default=7224, help="project_id to use for /query (required by 8001 schema)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N tests (0 = all)")
    ap.add_argument("--id-filter", default="", help="Run only tests whose ID contains this substring")
    ap.add_argument("--rate-limit-sleep", type=float, default=0.5,
                    help="Sleep seconds between queries to be polite to 8001")
    ap.add_argument("--header", action="append", default=[],
                    help="Extra HTTP header (e.g., --header 'X-Trace: foo')")
    args = ap.parse_args()

    if not Path(args.gold).exists():
        print(f"ERROR: gold set not found: {args.gold}")
        sys.exit(2)

    extra_headers = {}
    for h in args.header:
        if ":" in h:
            k, v = h.split(":", 1)
            extra_headers[k.strip()] = v.strip()

    tests = []
    with open(args.gold, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if args.id_filter and args.id_filter not in t.get("id", ""):
                continue
            tests.append(t)

    if args.limit:
        tests = tests[: args.limit]

    print(f"=== Hallucination Guard Adversarial Eval ===")
    print(f"Endpoint:  {args.endpoint}")
    print(f"Tenant:    {args.tenant}")
    print(f"Gold set:  {args.gold} ({len(tests)} tests)")
    print(f"Output:    {args.output}")
    print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, test in enumerate(tests, 1):
            print(f"[{i}/{len(tests)}] {test['id']} ({test.get('category')})...", end=" ", flush=True)
            try:
                response = run_test(test, args.endpoint, args.project_id, args.tenant, extra_headers)
                out_f.write(json.dumps(response) + "\n")
                out_f.flush()
                ok_turns = sum(1 for t in response["turns_response"] if t.get("status") == 200)
                print(f"{ok_turns}/{response['metadata']['n_turns_executed']} OK, "
                      f"{response['metadata']['total_latency_ms']}ms")
            except KeyboardInterrupt:
                print(" INTERRUPTED")
                break
            except Exception as exc:
                print(f" EXCEPTION: {type(exc).__name__}: {exc}")
                out_f.write(json.dumps({
                    "id": test["id"],
                    "category": test.get("category"),
                    "pillar": test.get("pillar"),
                    "turns_response": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }) + "\n")
                out_f.flush()
            time.sleep(args.rate_limit_sleep)

    elapsed = int(time.time() - t_start)
    print(f"\nDone. {elapsed}s elapsed. Output: {args.output}")
    print(f"Next step: python grader.py --gold {args.gold} --responses {args.output} "
          f"--output {args.output.replace('.jsonl', '_grades.json')}")


if __name__ == "__main__":
    main()
