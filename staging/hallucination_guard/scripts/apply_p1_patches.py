"""
Apply Pillar 1 (L1 anaphora resolver) patches to 8001 PROD code.

Patches gateway/router.py:
  - Add import for resolve_anaphora at top
  - In /query handler: after Pillar 11 pre-filter, compute resolved_query
    and pass it to orchestrator (replaces body.query)
  - In /query/stream handler: same

Atomic + AST-verified + py_compile-verified. Idempotent: if already
patched (detect by import marker), exits 0 without changes.
"""

import argparse
import datetime
import os
import py_compile
import shutil
import sys
import tempfile


def must_replace(text, anchor, repl, label: str):
    count = text.count(anchor)
    if count == 0:
        raise RuntimeError(f"anchor missing: {label}")
    if count > 1:
        raise RuntimeError(f"anchor ambiguous ({count} matches): {label}")
    return text.replace(anchor, repl, 1)


def patch_router(text: str) -> str:
    # Idempotency check
    if "_hg_p1_resolve" in text:
        raise RuntimeError("ALREADY_PATCHED")

    # 1. Add import + stub at top-of-file (after existing _hg_p11 imports)
    anchor_imp = """    def _hg_active_pillars():
        return set()


def _hg_build_refusal_response(decision, body, resolved_session_id=None) -> dict:"""
    repl_imp = """    def _hg_active_pillars():
        return set()


# --- Hallucination Guard v2 -- Pillar 1 (L1 anaphora resolver) -------------
try:
    from agentic.hallucination_guard.pillar_1_anaphora import (
        resolve_anaphora as _hg_p1_resolve,
    )
except Exception:  # pragma: no cover -- package absent -> identity stub
    def _hg_p1_resolve(query, history=None, anthropic_client=None, force_enable=None):
        return query


def _hg_build_refusal_response(decision, body, resolved_session_id=None) -> dict:"""
    text = must_replace(text, anchor_imp, repl_imp, "router-p1-import")

    # 2. /query handler: insert resolved_query computation AFTER P11 pre-filter,
    # BEFORE the existing `from gateway.email_hybrid_search import` line.
    # Then replace `query=body.query,` in the orchestrator.query() call with
    # `query=_hg_resolved_query,`
    anchor_q = """        return _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)

    from gateway.email_hybrid_search import search_emails"""
    repl_q = """        return _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)

    # --- Pillar 1 anaphora resolver (Hallucination Guard v2) ---
    # Rewrites follow-up queries with pronouns/references into self-contained queries.
    # No-op when HG_PILLAR_1=false or history empty.
    _hg_resolved_query = _hg_p1_resolve(body.query, history=body.conversation_history or [])

    from gateway.email_hybrid_search import search_emails"""
    text = must_replace(text, anchor_q, repl_q, "router-p1-resolve-query")

    # 3. Replace body.query -> _hg_resolved_query in the orchestrator.query() call
    # inside the /query handler. This is a narrow string match -- the FIRST
    # `query=body.query,` after the resolver definition is the one we want.
    anchor_orch = """    coros: dict[str, Any] = {}
    if "rag" in modes or "web" in modes:
        coros["rag"] = orchestrator.query(
            query=body.query,
            project_id=body.project_id,"""
    repl_orch = """    coros: dict[str, Any] = {}
    if "rag" in modes or "web" in modes:
        coros["rag"] = orchestrator.query(
            query=_hg_resolved_query,
            project_id=body.project_id,"""
    text = must_replace(text, anchor_orch, repl_orch, "router-p1-orchestrator-call")

    # 4. /query/stream handler: same pattern. Insert resolved_query right
    # after P11 pre-filter (which is at start of query_stream function),
    # then replace `query=body.query,` in _run_query_offloop() call.
    anchor_qs = """        return StreamingResponse(_hg_refusal_stream(), media_type="text/event-stream")
    \"\"\"SSE streaming query"""
    repl_qs = """        return StreamingResponse(_hg_refusal_stream(), media_type="text/event-stream")

    # --- Pillar 1 anaphora resolver (Hallucination Guard v2) ---
    _hg_resolved_query_stream = _hg_p1_resolve(body.query, history=body.conversation_history or [])
    \"\"\"SSE streaming query"""
    text = must_replace(text, anchor_qs, repl_qs, "router-p1-stream-resolve")

    # 5. Replace body.query -> _hg_resolved_query_stream in _run_query_offloop call
    anchor_offloop = """                final_result = await _run_query_offloop(
                    orchestrator,
                    query=body.query,
                    project_id=body.project_id,"""
    repl_offloop = """                final_result = await _run_query_offloop(
                    orchestrator,
                    query=_hg_resolved_query_stream,
                    project_id=body.project_id,"""
    text = must_replace(text, anchor_offloop, repl_offloop, "router-p1-offloop-call")

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--ts", default=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()

    bak_suffix = f".bak_hg_p1_{args.ts}"
    print(f"Backup suffix: {bak_suffix}")

    router_path = os.path.join(args.app_dir, "gateway", "router.py")
    if not os.path.exists(router_path):
        print(f"FAIL: missing {router_path}")
        sys.exit(2)

    src_bytes = open(router_path, "rb").read()
    src_str = src_bytes.decode("utf-8", errors="surrogateescape")

    try:
        new_str = patch_router(src_str)
    except RuntimeError as e:
        if "ALREADY_PATCHED" in str(e):
            print("Already patched (idempotent). Exiting 0.")
            sys.exit(0)
        print(f"FAIL patcher: {e}")
        sys.exit(3)

    new_bytes = new_str.encode("utf-8", errors="surrogateescape")

    # Compile-verify
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
        tf.write(new_bytes)
        tmp_path = tf.name
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"FAIL: py_compile error: {e}")
        os.unlink(tmp_path)
        sys.exit(4)
    os.unlink(tmp_path)

    print(f"[plan] {router_path}  src={len(src_bytes)}B  patched={len(new_bytes)}B  +{len(new_bytes) - len(src_bytes)}B  py_compile=OK")

    # Commit
    shutil.copy2(router_path, router_path + bak_suffix)
    tmp = router_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new_bytes)
    os.replace(tmp, router_path)
    print(f"[wrote] {router_path} (backup: {router_path + bak_suffix})")

    # Paranoia
    try:
        py_compile.compile(router_path, doraise=True)
        print("=== ROUTER PATCHED + COMPILED ===")
    except py_compile.PyCompileError as e:
        print(f"FAIL post-write: {e}")
        shutil.copy2(router_path + bak_suffix, router_path)
        sys.exit(5)


if __name__ == "__main__":
    main()
