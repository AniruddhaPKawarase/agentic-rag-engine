"""
Apply Pillar 7 (citation validator) patches to 8001 PROD code.

Patches agentic/generation/synthesizer.py:
  - Add import for post_validate at top
  - After generate() returns (non-stream branch), pass the result through
    pillar_7_citation.post_validate() which is a no-op when HG_PILLAR_7=false

Atomic + py_compile-verified. Idempotent: detects already-patched and exits 0.
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


def patch_synthesizer(text: str) -> str:
    if "_hg_p7_validate" in text:
        raise RuntimeError("ALREADY_PATCHED")

    # 1. Add import at top after the existing Pillar 6 import block
    anchor_imp = """# Hallucination Guard v2 -- Pillar 6 anti-anchor (env-flag gated; no-op when off)
try:
    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
except Exception:  # pragma: no cover -- package absent -> no-op
    def _hg_p6_wrap(s: str) -> str:
        return s"""
    repl_imp = """# Hallucination Guard v2 -- Pillar 6 anti-anchor (env-flag gated; no-op when off)
try:
    from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt as _hg_p6_wrap
except Exception:  # pragma: no cover -- package absent -> no-op
    def _hg_p6_wrap(s: str) -> str:
        return s

# Hallucination Guard v2 -- Pillar 7 citation validator (env-flag gated; no-op when off)
try:
    from agentic.hallucination_guard.pillar_7_citation import post_validate as _hg_p7_validate
except Exception:  # pragma: no cover -- package absent -> identity stub
    def _hg_p7_validate(answer, force_enable=None):
        return answer, {"pillar": 7, "applied": False}"""
    text = must_replace(text, anchor_imp, repl_imp, "synth-p7-import")

    # 2. Wrap the return statement at the end of synthesize().
    # The synthesizer has this pattern at the end:
    #   if stream:
    #       return normalize_chunks(result)
    #   return normalize_output(result) if isinstance(result, str) else result
    #
    # We patch the non-stream path to run post_validate before returning.
    anchor_ret = """    if stream:
        return normalize_chunks(result)
    return normalize_output(result) if isinstance(result, str) else result"""
    repl_ret = """    if stream:
        return normalize_chunks(result)
    final = normalize_output(result) if isinstance(result, str) else result
    # Pillar 7 post-hoc citation validation (HG v2). When HG_PILLAR_7=false,
    # _hg_p7_validate is a no-op and returns (final, {applied: False}).
    if isinstance(final, str):
        final, _ = _hg_p7_validate(final)
    return final"""
    text = must_replace(text, anchor_ret, repl_ret, "synth-p7-return")

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--ts", default=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()

    bak_suffix = f".bak_hg_p7_{args.ts}"
    print(f"Backup suffix: {bak_suffix}")

    path = os.path.join(args.app_dir, "agentic", "generation", "synthesizer.py")
    if not os.path.exists(path):
        print(f"FAIL: missing {path}")
        sys.exit(2)

    src_bytes = open(path, "rb").read()
    src_str = src_bytes.decode("utf-8", errors="surrogateescape")

    try:
        new_str = patch_synthesizer(src_str)
    except RuntimeError as e:
        if "ALREADY_PATCHED" in str(e):
            print("Already patched (idempotent). Exiting 0.")
            sys.exit(0)
        print(f"FAIL patcher: {e}")
        sys.exit(3)

    new_bytes = new_str.encode("utf-8", errors="surrogateescape")

    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as tf:
        tf.write(new_bytes)
        tmp_path = tf.name
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"FAIL py_compile: {e}")
        os.unlink(tmp_path)
        sys.exit(4)
    os.unlink(tmp_path)

    print(f"[plan] {path}  src={len(src_bytes)}B  patched={len(new_bytes)}B  +{len(new_bytes) - len(src_bytes)}B  py_compile=OK")

    shutil.copy2(path, path + bak_suffix)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new_bytes)
    os.replace(tmp, path)
    print(f"[wrote] {path} (backup: {path + bak_suffix})")

    try:
        py_compile.compile(path, doraise=True)
        print("=== SYNTHESIZER PATCHED + COMPILED ===")
    except py_compile.PyCompileError as e:
        print(f"FAIL post-write: {e}")
        shutil.copy2(path + bak_suffix, path)
        sys.exit(5)


if __name__ == "__main__":
    main()
