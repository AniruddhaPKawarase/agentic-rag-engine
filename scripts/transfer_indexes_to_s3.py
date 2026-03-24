#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  FAISS Index Transfer: Local/VM → S3
═══════════════════════════════════════════════════════════════════════════════

Transfers ALL FAISS index (.bin) and metadata (.jsonl) files from the local
INDEX_ROOT directory to the S3 bucket under rag-agent/faiss_indexes/.

Features:
  - Reads INDEX_ROOT and S3 credentials from .env (auto-loaded)
  - Real-time progress bar per file (tqdm) with speed + ETA
  - Overall progress bar across all files
  - Overwrites existing S3 files (no skip — always fresh copy)
  - Multipart upload for large files (>50 MB)
  - Also transfers conversation sessions if found
  - Prints summary with per-file status table

Usage (run from the RAG agent directory):
    cd PROD_SETUP/RAG_agent_VCS/RAG
    python scripts/transfer_indexes_to_s3.py

Or from anywhere:
    python PROD_SETUP/RAG_agent_VCS/RAG/scripts/transfer_indexes_to_s3.py

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
PROD_ROOT = AGENT_ROOT.parent.parent
sys.path.insert(0, str(PROD_ROOT))
sys.path.insert(0, str(AGENT_ROOT))

# ── Load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(AGENT_ROOT / ".env")

# ── Dependencies ──────────────────────────────────────────────────────────────
import boto3
from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (all from .env)
# ═══════════════════════════════════════════════════════════════════════════════
BUCKET = os.getenv("S3_BUCKET_NAME", "")
REGION = os.getenv("S3_REGION", "us-east-1")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AGENT_PREFIX = os.getenv("S3_AGENT_PREFIX", "rag-agent")
INDEX_ROOT = os.getenv("INDEX_ROOT", "")
SESSION_PATH = os.getenv("SESSION_STORAGE_PATH", "./conversation_sessions")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")

# S3 target prefixes
S3_FAISS_PREFIX = f"{AGENT_PREFIX}/faiss_indexes"
S3_SESSION_PREFIX = f"{AGENT_PREFIX}/conversation_sessions"


# ═══════════════════════════════════════════════════════════════════════════════
# Progress callback for tqdm
# ═══════════════════════════════════════════════════════════════════════════════
class ProgressCallback:
    """tqdm callback for boto3 upload_file."""

    def __init__(self, file_path: Path, bar: tqdm):
        self._file_path = file_path
        self._bar = bar
        self._seen = 0

    def __call__(self, bytes_amount):
        self._seen += bytes_amount
        self._bar.update(bytes_amount)


# ═══════════════════════════════════════════════════════════════════════════════
# S3 Client setup
# ═══════════════════════════════════════════════════════════════════════════════
def create_s3_client():
    """Create boto3 S3 client with multipart + retry config."""
    boto_config = BotoConfig(
        max_pool_connections=10,
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=15,
        read_timeout=60,
    )
    return boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=boto_config,
    )


def get_transfer_config():
    """Multipart upload config for large FAISS files."""
    return TransferConfig(
        multipart_threshold=50 * 1024 * 1024,   # 50 MB → trigger multipart
        multipart_chunksize=25 * 1024 * 1024,   # 25 MB chunks
        max_concurrency=5,
        use_threads=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Upload functions
# ═══════════════════════════════════════════════════════════════════════════════
def upload_file_with_progress(client, local_path: Path, s3_key: str, transfer_config) -> dict:
    """Upload a single file to S3 with tqdm progress bar. Always overwrites."""
    size = local_path.stat().st_size
    size_mb = size / (1024 * 1024)

    bar = tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  {local_path.name}",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    callback = ProgressCallback(local_path, bar)
    start = time.time()

    try:
        client.upload_file(
            str(local_path),
            BUCKET,
            s3_key,
            Config=transfer_config,
            Callback=callback,
        )
        elapsed = time.time() - start
        speed = size_mb / elapsed if elapsed > 0 else 0
        bar.close()
        return {
            "file": local_path.name,
            "s3_key": s3_key,
            "size_mb": round(size_mb, 1),
            "elapsed_sec": round(elapsed, 1),
            "speed_mbps": round(speed, 1),
            "status": "SUCCESS",
        }
    except Exception as e:
        bar.close()
        return {
            "file": local_path.name,
            "s3_key": s3_key,
            "size_mb": round(size_mb, 1),
            "elapsed_sec": 0,
            "speed_mbps": 0,
            "status": f"FAILED: {e}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main transfer
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("═" * 80)
    print("  FAISS Index & Session Transfer → S3")
    print("═" * 80)
    print()

    # ── Validate config ───────────────────────────────────────────────────────
    errors = []
    if STORAGE_BACKEND != "s3":
        errors.append(f"STORAGE_BACKEND={STORAGE_BACKEND} (must be 's3')")
    if not BUCKET:
        errors.append("S3_BUCKET_NAME is empty")
    if not ACCESS_KEY or not SECRET_KEY:
        errors.append("AWS credentials missing")
    if not INDEX_ROOT:
        errors.append("INDEX_ROOT is empty in .env")

    if errors:
        print("  CONFIGURATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        print("\n  Fix your .env file and try again.")
        return

    index_dir = Path(INDEX_ROOT)
    if not index_dir.exists():
        print(f"  INDEX_ROOT path does not exist: {INDEX_ROOT}")
        print("  Make sure you're running this from the correct machine (VM or local).")
        return

    # ── Discover files ────────────────────────────────────────────────────────
    faiss_files = sorted(index_dir.glob("faiss_index_*.bin"))
    meta_files = sorted(index_dir.glob("metadata_*.jsonl"))
    all_index_files = faiss_files + meta_files

    session_dir = AGENT_ROOT / SESSION_PATH
    session_files = sorted(session_dir.glob("session_*.json")) if session_dir.exists() else []

    total_files = len(all_index_files) + len(session_files)
    total_size = sum(f.stat().st_size for f in all_index_files + session_files)
    total_size_mb = total_size / (1024 * 1024)

    print(f"  Bucket:        s3://{BUCKET}")
    print(f"  Region:        {REGION}")
    print(f"  INDEX_ROOT:    {INDEX_ROOT}")
    print(f"  S3 prefix:     {S3_FAISS_PREFIX}/")
    print()
    print(f"  FAISS indexes: {len(faiss_files)} files")
    print(f"  Metadata:      {len(meta_files)} files")
    print(f"  Sessions:      {len(session_files)} files")
    print(f"  Total:         {total_files} files ({total_size_mb:.0f} MB)")
    print()

    if total_files == 0:
        print("  No files found to transfer.")
        return

    # ── Test S3 connection ────────────────────────────────────────────────────
    print("  Testing S3 connection...", end=" ", flush=True)
    client = create_s3_client()
    try:
        client.head_bucket(Bucket=BUCKET)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    transfer_config = get_transfer_config()
    results = []
    overall_start = time.time()

    # ── Upload FAISS indexes ──────────────────────────────────────────────────
    if all_index_files:
        print()
        print(f"  ── Uploading FAISS Indexes ({len(all_index_files)} files) ──")
        print()

        overall_bar = tqdm(
            total=sum(f.stat().st_size for f in all_index_files),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  OVERALL",
            ncols=100,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for f in all_index_files:
            s3_key = f"{S3_FAISS_PREFIX}/{f.name}"
            result = upload_file_with_progress(client, f, s3_key, transfer_config)
            results.append(result)
            overall_bar.update(f.stat().st_size)

        overall_bar.close()

    # ── Upload conversation sessions ──────────────────────────────────────────
    if session_files:
        print()
        print(f"  ── Uploading Conversation Sessions ({len(session_files)} files) ──")
        print()

        for f in tqdm(session_files, desc="  Sessions", ncols=80):
            s3_key = f"{S3_SESSION_PREFIX}/{f.name}"
            size_mb = f.stat().st_size / (1024 * 1024)
            start = time.time()
            try:
                client.upload_file(str(f), BUCKET, s3_key)
                elapsed = time.time() - start
                results.append({
                    "file": f.name,
                    "s3_key": s3_key,
                    "size_mb": round(size_mb, 3),
                    "elapsed_sec": round(elapsed, 1),
                    "speed_mbps": round(size_mb / elapsed, 1) if elapsed > 0 else 0,
                    "status": "SUCCESS",
                })
            except Exception as e:
                results.append({
                    "file": f.name,
                    "s3_key": s3_key,
                    "size_mb": round(size_mb, 3),
                    "elapsed_sec": 0,
                    "speed_mbps": 0,
                    "status": f"FAILED: {e}",
                })

    # ── Summary ───────────────────────────────────────────────────────────────
    overall_elapsed = time.time() - overall_start
    succeeded = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] != "SUCCESS")

    print()
    print("═" * 80)
    print("  TRANSFER SUMMARY")
    print("═" * 80)
    print()
    print(f"  {'File':<45} {'Size':>8} {'Time':>8} {'Speed':>10} {'Status':<10}")
    print(f"  {'─' * 44} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10}")

    for r in results:
        size_str = f"{r['size_mb']:.1f} MB"
        time_str = f"{r['elapsed_sec']:.1f}s"
        speed_str = f"{r['speed_mbps']:.1f} MB/s"
        status_str = "OK" if r["status"] == "SUCCESS" else "FAIL"
        print(f"  {r['file']:<45} {size_str:>8} {time_str:>8} {speed_str:>10} {status_str:<10}")

    print()
    print(f"  Total files:    {len(results)}")
    print(f"  Succeeded:      {succeeded}")
    print(f"  Failed:         {failed}")
    print(f"  Total time:     {overall_elapsed:.0f}s ({overall_elapsed / 60:.1f} min)")
    print(f"  Total size:     {total_size_mb:.0f} MB")
    if overall_elapsed > 0:
        print(f"  Avg speed:      {total_size_mb / overall_elapsed:.1f} MB/s")
    print()

    # ── Save results log ──────────────────────────────────────────────────────
    log_dir = AGENT_ROOT / "scripts"
    log_file = log_dir / "transfer_results.json"
    log_data = {
        "transfer_date": datetime.utcnow().isoformat() + "Z",
        "bucket": BUCKET,
        "region": REGION,
        "index_root": INDEX_ROOT,
        "s3_faiss_prefix": S3_FAISS_PREFIX,
        "s3_session_prefix": S3_SESSION_PREFIX,
        "total_files": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "total_size_mb": round(total_size_mb, 1),
        "total_elapsed_sec": round(overall_elapsed, 1),
        "files": results,
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  Results saved: {log_file}")

    # ── Verify uploads ────────────────────────────────────────────────────────
    print()
    print("  ── Verifying S3 uploads ──")
    print()
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{AGENT_PREFIX}/")
    s3_objects = []
    for page in pages:
        for obj in page.get("Contents", []):
            s3_objects.append(obj)

    print(f"  Objects in s3://{BUCKET}/{AGENT_PREFIX}/: {len(s3_objects)}")
    for obj in sorted(s3_objects, key=lambda x: x["Key"]):
        size_mb = obj["Size"] / (1024 * 1024)
        print(f"    {obj['Key']:<65} {size_mb:>8.1f} MB")

    print()
    if failed == 0:
        print("  ALL TRANSFERS COMPLETED SUCCESSFULLY")
    else:
        print(f"  WARNING: {failed} file(s) failed. Check logs above.")
    print("═" * 80)
    print()


if __name__ == "__main__":
    main()
