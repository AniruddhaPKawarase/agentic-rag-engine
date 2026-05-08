"""Interactive helper to add ANTHROPIC_API_KEY to the v3.1 worktree .env.

Why this exists: an automated agent should never write your credentials into
a file. This script lets you paste the key once, validates and stores it
locally, and verifies it loads — all without echoing the secret to the
terminal, the shell history, or any log.

Usage:
    cd <v3.1 worktree>
    python eval/set_anthropic_key.py

The script:
  1. Prompts you to paste the key (input hidden via getpass).
  2. Sanity-checks the prefix and length.
  3. Writes it to ./.env (creating or updating ANTHROPIC_API_KEY).
  4. Prints the key length only — never the key itself.
  5. Optionally fires a 1-token test call to confirm Anthropic accepts it.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
ENV_PATH = WORKTREE / ".env"


def read_env_lines() -> list[str]:
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def write_env_lines(lines: list[str]) -> None:
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upsert_env_var(key: str, value: str) -> str:
    """Add or replace a key=value line in the .env. Returns 'added' or 'updated'."""
    lines = read_env_lines()
    found = False
    out = []
    for line in lines:
        # Preserve comments and blank lines
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            out.append(line)
            continue
        if "=" in line:
            k, _ = line.split("=", 1)
            if k.strip() == key:
                out.append(f"{key}={value}")
                found = True
                continue
        out.append(line)
    if not found:
        out.append(f"{key}={value}")
    write_env_lines(out)
    return "updated" if found else "added"


def validate_anthropic_key_format(key: str) -> tuple[bool, str]:
    """Cheap structural check. Real validation happens via the API call."""
    if not key:
        return False, "empty input"
    if not key.startswith("sk-ant-api03-"):
        return False, f"unexpected prefix (Anthropic keys start with 'sk-ant-api03-'); got first 12 chars only"
    if len(key) < 80:
        return False, f"key length {len(key)} chars; expected ~108 — looks truncated"
    if " " in key or "\t" in key or "\n" in key:
        return False, "key contains whitespace; likely a paste error"
    return True, "format ok"


def verify_loads() -> bool:
    """Reload .env and check the key is now visible to os.environ."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        print("dotenv not installed; cannot verify loading. (Eval still works if key is in .env.)")
        return True
    # override=True to overwrite any stale value already in this shell
    load_dotenv(ENV_PATH, override=True)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def test_live_call() -> tuple[bool, str]:
    """Make a 1-token Anthropic call to confirm the key is actually valid."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        return False, "anthropic SDK not installed (pip install anthropic)"

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "1"}],
        )
        # We don't care about the content — we care that the API accepted the key.
        _ = resp.id
        return True, "Anthropic API accepted the key"
    except anthropic.AuthenticationError as e:  # type: ignore[attr-defined]
        return False, f"Anthropic rejected the key: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"Live call failed: {type(e).__name__}: {e}"


def main() -> int:
    print(f"Target .env: {ENV_PATH}")
    if not ENV_PATH.exists():
        print("  (file does not exist yet — will be created)")
    else:
        # Show whether the key is already there, without revealing the value
        existing_lines = read_env_lines()
        has_existing = any(
            line.split("=", 1)[0].strip() == "ANTHROPIC_API_KEY"
            for line in existing_lines
            if "=" in line and not line.strip().startswith("#")
        )
        if has_existing:
            print("  (ANTHROPIC_API_KEY is already present — will be replaced)")

    print()
    print("Paste your Anthropic API key (input is hidden, nothing echoed):")
    print("  - Format: sk-ant-api03-...")
    print("  - You will not see the characters as you paste")
    print()

    try:
        key = getpass.getpass(prompt="ANTHROPIC_API_KEY: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        return 130

    ok, msg = validate_anthropic_key_format(key)
    if not ok:
        print(f"\nFormat check failed: {msg}")
        print("Nothing was written. Run again with the correct key.")
        return 2

    print(f"\nFormat: ok  (length: {len(key)} chars, prefix: sk-ant-api03-...)")

    action = upsert_env_var("ANTHROPIC_API_KEY", key)
    print(f"Wrote .env  ({action})")

    # Best-effort: scrub the key from local memory now that it's persisted.
    key = "x" * len(key)
    del key

    if not verify_loads():
        print("\nWarning: key written to .env but did not appear in os.environ.")
        print("Check that the file is at the path above and dotenv is installed.")
        return 3

    print("Verified: ANTHROPIC_API_KEY is now visible to os.environ")

    # Optional live test
    print()
    answer = input("Run a 1-token Anthropic API call to confirm the key actually works? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        ok, msg = test_live_call()
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {msg}")
        if not ok:
            print("\nThe key was written but Anthropic rejected it. You may need to rotate again.")
            return 4

    print()
    print("Done. You can now run the eval:")
    print('  python eval\\run_v31_eval.py --mode single_turn \\')
    print('    --questions eval_results\\questions_50_focused.xlsx \\')
    print('    --project-id 7325 --set-id 4987 --sample 2 \\')
    print('    --output-dir eval_results\\smoke_v3 --model-set haiku')
    return 0


if __name__ == "__main__":
    sys.exit(main())
