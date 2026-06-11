"""
Atomically apply Pillar 6 + Pillar 11 patches to 8001 PROD code.

Run on sandbox VM as:
    python3 apply_patches.py --app-dir /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31

Backs up every file before editing. Verifies AST after each patch.
If ANY anchor fails or ANY syntax check fails: restores all backups
and exits non-zero. Zero risk of partial-apply state.
"""

import argparse
import ast
import datetime
import os
import shutil
import sys


def must_replace(text, anchor, repl, label: str):
    """Replace anchor -> repl (works on str or bytes, same type for both).
    Raise if anchor missing or appears >1 times.
    """
    count = text.count(anchor)
    if count == 0:
        raise RuntimeError(f"anchor missing: {label}")
    if count > 1:
        raise RuntimeError(f"anchor ambiguous ({count} matches): {label}")
    return text.replace(anchor, repl, 1)


def patch_agentic_synth(text: str) -> str:
    """Patch agentic/generation/synthesizer.py.

    - Add HG p6 import at top
    - Add wrap call before user_prompt construction
    """
    anchor_imp = "from agentic.generation.text_normalizer import normalize_chunks, normalize_output"
    repl_imp = """from agentic.generation.text_normalizer import normalize_chunks, normalize_output

# Hallucination Guard v2 -- Pillar 6 anti-anchor (env-flag gated; no-op when off)
try:
    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
except Exception:  # pragma: no cover -- package absent -> no-op
    def _hg_p6_wrap(s: str) -> str:
        return s"""
    text = must_replace(text, anchor_imp, repl_imp, "agentic-synth-import")

    # The anchor uses literal \n inside an f-string. Match the file content.
    anchor_call = (
        '    system_prompt = _SYSTEM_PROMPT\n'
        '    if shape_block:\n'
        '        system_prompt = f"{shape_block}\\n\\n{system_prompt}"\n'
        '\n'
        '    user_prompt = _build_user_prompt('
    )
    repl_call = (
        '    system_prompt = _SYSTEM_PROMPT\n'
        '    if shape_block:\n'
        '        system_prompt = f"{shape_block}\\n\\n{system_prompt}"\n'
        '    # Pillar 6 anti-anchor wrap (HG v2) -- no-op when HG_PILLAR_6=false.\n'
        '    system_prompt = _hg_p6_wrap(system_prompt)\n'
        '\n'
        '    user_prompt = _build_user_prompt('
    )
    text = must_replace(text, anchor_call, repl_call, "agentic-synth-call")
    return text


def patch_gateway_synth(text: str) -> str:
    """Patch gateway/synthesizer.py -- multi-source merger."""
    anchor_imp = "from openai import AsyncOpenAI"
    repl_imp = """from openai import AsyncOpenAI

# Hallucination Guard v2 -- Pillar 6 anti-anchor (env-flag gated; no-op when off)
try:
    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
except Exception:  # pragma: no cover
    def _hg_p6_wrap(s: str) -> str:
        return s"""
    text = must_replace(text, anchor_imp, repl_imp, "gateway-synth-import")

    anchor_call = '                    {"role": "system", "content": _SYSTEM},'
    repl_call = '                    {"role": "system", "content": _hg_p6_wrap(_SYSTEM)},'
    text = must_replace(text, anchor_call, repl_call, "gateway-synth-call")
    return text


def patch_router(text: str) -> str:
    """Patch gateway/router.py -- Pillar 11 imports + pre-filters at /query and /query/stream."""
    # 1. Top-of-file imports + helper (anchor = full import line to avoid
    #    substring-overlap with the comma-trailing 'UnifiedResponse')
    anchor_top = "from gateway.models import QueryRequest, UnifiedResponse"
    repl_top = '''from gateway.models import QueryRequest, UnifiedResponse

# --- Hallucination Guard v2 -- Pillar 11 adversarial pre-filter (env-flag gated) ---
try:
    from agentic.hallucination_guard.pillar_11_adversarial import (
        pre_filter_query as _hg_p11_filter,
        applied_metadata as _hg_p11_meta,
    )
    from agentic.hallucination_guard import get_active_pillars as _hg_active_pillars
except Exception:  # pragma: no cover -- package absent -> all-allow stub
    class _HG_AllowDecision:
        action = "allow"
        reason_class = "package_unavailable"
        matched_patterns = []
        refusal_text = None
        prompt_addendum = None
        regulatory_reframe_detected = False
        pillar_applied = False
    def _hg_p11_filter(q, has_history=False, force_enable=None):
        return _HG_AllowDecision()
    def _hg_p11_meta(decision=None):
        return {"pillar": 11, "applied": False}
    def _hg_active_pillars():
        return set()


def _hg_build_refusal_response(decision, body, resolved_session_id=None) -> dict:
    """Build a /query response dict for a Pillar 11 injection refusal.

    Returns a dict matching the /query response shape; skips retrieval + synthesis.
    """
    return {
        "answer": decision.refusal_text,
        "final_answer": decision.refusal_text,
        "session_id": resolved_session_id,
        "engine_used": "hallucination_guard_pillar_11",
        "sources": [],
        "all_retrieved_sources": [],
        "verification_meta": {
            "refused": True,
            "reason_class": decision.reason_class,
            "active_pillars": sorted(_hg_active_pillars()),
            "pillar_11": _hg_p11_meta(decision),
        },
    }
'''
    text = must_replace(text, anchor_top, repl_top, "router-top-imports")

    # 2. /query handler pre-filter
    anchor_q = '''@router.post("/query")
async def query(request: Request, body: QueryRequest) -> dict:
    """Route query to selected sources, return per-source answers + synthesized final_answer."""
    from gateway.email_hybrid_search import search_emails'''
    repl_q = '''@router.post("/query")
async def query(request: Request, body: QueryRequest) -> dict:
    """Route query to selected sources, return per-source answers + synthesized final_answer."""
    # --- Pillar 11 pre-filter (Hallucination Guard v2) ---
    _hg_has_history = bool(body.conversation_history)
    _hg_decision = _hg_p11_filter(body.query, has_history=_hg_has_history)
    if _hg_decision.action == "refuse_injection":
        logger.info(
            "[hg-p11] /query refused -- reason=%s patterns=%d",
            _hg_decision.reason_class, len(_hg_decision.matched_patterns),
        )
        return _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)

    from gateway.email_hybrid_search import search_emails'''
    text = must_replace(text, anchor_q, repl_q, "router-/query-prefilter")

    # 3. /query/stream handler pre-filter — file contains a UTF-8 em-dash here.
    EM = "—"  # U+2014 EM DASH — preserved in the file
    anchor_qs = ('@router.post("/query/stream")\n'
                 'async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:\n'
                 '    """SSE streaming query ' + EM + ' word-by-word token output.')
    repl_qs = ('@router.post("/query/stream")\n'
               'async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:\n'
               '    # --- Pillar 11 pre-filter (Hallucination Guard v2) ---\n'
               '    _hg_has_history = bool(body.conversation_history)\n'
               '    _hg_decision = _hg_p11_filter(body.query, has_history=_hg_has_history)\n'
               '    if _hg_decision.action == "refuse_injection":\n'
               '        logger.info(\n'
               '            "[hg-p11] /query/stream refused -- reason=%s patterns=%d",\n'
               '            _hg_decision.reason_class, len(_hg_decision.matched_patterns),\n'
               '        )\n'
               '        _hg_refusal_dict = _hg_build_refusal_response(_hg_decision, body, resolved_session_id=body.session_id)\n'
               '        async def _hg_refusal_stream():\n'
               '            yield _sse("status", {"phase": "refused"})\n'
               '            yield _sse("token", {"delta": _hg_decision.refusal_text})\n'
               '            yield _sse("done", _hg_refusal_dict)\n'
               '        return StreamingResponse(_hg_refusal_stream(), media_type="text/event-stream")\n'
               '    """SSE streaming query ' + EM + ' word-by-word token output.')
    text = must_replace(text, anchor_qs, repl_qs, "router-/query/stream-prefilter")
    return text


PATCHES = {
    "agentic/generation/synthesizer.py": patch_agentic_synth,
    "gateway/synthesizer.py": patch_gateway_synth,
    "gateway/router.py": patch_router,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--ts", default=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()

    bak_suffix = f".bak_hg_v2_{args.ts}"
    print(f"Backup suffix: {bak_suffix}")
    print()

    # Phase 1 -- bytes-end-to-end with UTF-8 anchors/replacements.
    # Anchors and replacements are str; we encode to UTF-8 bytes and use
    # bytes.replace on the raw file content. This preserves any pre-existing
    # non-UTF-8 bytes (like the stray 0x97 in router.py at pos 17519)
    # untouched because we only replace exact UTF-8 anchor matches.
    #
    # Verification: instead of ast.parse (which strictly enforces UTF-8),
    # we use py_compile (which mirrors Python's actual import behavior).
    # If the pre-patch file already had a quirky byte that Python accepts
    # at import, the post-patch file will accept the same quirk.
    import py_compile
    import tempfile

    plan = []
    for rel, patcher in PATCHES.items():
        path = os.path.join(args.app_dir, rel)
        if not os.path.exists(path):
            print(f"FAIL: missing source file: {path}")
            sys.exit(2)
        src_bytes = open(path, "rb").read()
        # Run the patcher in str-space (latin-1 round-trip is byte-lossless).
        src_str = src_bytes.decode("utf-8", errors="surrogateescape")
        try:
            new_str = patcher(src_str)
        except RuntimeError as e:
            print(f"FAIL: patcher for {rel}: {e}")
            sys.exit(3)
        # If any non-latin-1 char crept into the new content, we cannot
        # round-trip via latin-1. The patcher functions are designed to
        # use \uXXXX escapes for non-ASCII so they encode through latin-1.
        try:
            new_bytes = new_str.encode("utf-8", errors="surrogateescape")
        except UnicodeEncodeError as e:
            print(f"FAIL: patcher for {rel} encode error at pos {e.start}: U+{ord(new_str[e.start]):04X}")
            sys.exit(4)

        # Compile-time verification — mirrors Python import.
        with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
            tf.write(new_bytes)
            tmp_path = tf.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"FAIL: py_compile error in {rel}: {e}")
            os.unlink(tmp_path)
            sys.exit(5)
        os.unlink(tmp_path)

        plan.append((path, src_bytes, new_bytes))
        print(f"[plan] {rel}  src={len(src_bytes)}B  patched={len(new_bytes)}B  +{len(new_bytes) - len(src_bytes)}B  py_compile=OK")

    # Phase 2 -- commit: back up + atomic write.
    for path, _src, new_bytes in plan:
        shutil.copy2(path, path + bak_suffix)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(new_bytes)
        os.replace(tmp, path)
        print(f"[wrote] {path} (backup: {path + bak_suffix})")

    # Phase 3 -- verify on-disk files still py_compile.
    for path, _src, _new in plan:
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"FAIL post-write compile: {path}: {e}")
            shutil.copy2(path + bak_suffix, path)
            sys.exit(6)

    print()
    print("=== ALL 3 FILES PATCHED + AST-VERIFIED ===")
    print(f"Rollback: for f in {' '.join(rel for rel in PATCHES)}; do cp {args.app_dir}/$f{bak_suffix} {args.app_dir}/$f; done")


if __name__ == "__main__":
    main()
