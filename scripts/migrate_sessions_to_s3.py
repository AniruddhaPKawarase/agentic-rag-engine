"""
Standalone migration: upload existing conversation sessions to S3.
Run: python scripts/migrate_sessions_to_s3.py
"""
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
PROD_ROOT = AGENT_ROOT.parent.parent
sys.path.insert(0, str(PROD_ROOT))
sys.path.insert(0, str(AGENT_ROOT))

from dotenv import load_dotenv
load_dotenv(AGENT_ROOT / ".env")

from s3_utils.operations import upload_file, object_exists
from s3_utils.helpers import session_key
from s3_utils.config import get_s3_config


def migrate():
    config = get_s3_config()
    if not config.is_s3_enabled:
        print("ERROR: STORAGE_BACKEND is not 's3'. Set it in .env first.")
        return

    sessions_dir = AGENT_ROOT / os.getenv("SESSION_STORAGE_PATH", "./conversation_sessions")
    if not sessions_dir.exists():
        print(f"No sessions directory found at {sessions_dir}")
        return

    files = list(sessions_dir.glob("session_*.json"))
    print(f"Found {len(files)} sessions to migrate.")

    migrated = skipped = failed = 0
    for f in files:
        s3_key = session_key(config.agent_prefix, f.name)
        if object_exists(s3_key):
            print(f"  EXISTS: {s3_key}")
            skipped += 1
            continue
        if upload_file(str(f), s3_key):
            print(f"  UPLOADED: {s3_key}")
            migrated += 1
        else:
            print(f"  FAILED: {f.name}")
            failed += 1

    print(f"\nMigration complete: {migrated} uploaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    migrate()
