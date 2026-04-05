"""
Core S3 operations: upload, download, list, delete.
All functions are synchronous (suitable for asyncio.to_thread if needed).
All functions fail gracefully — return False/None on error, never raise.
"""

import logging
from pathlib import Path
from typing import Optional

from .client import get_s3_client
from .config import get_s3_config

logger = logging.getLogger(__name__)


# ── Upload Operations ────────────────────────────────────────────────────────


def upload_file(local_path: str, s3_key: str) -> bool:
    """
    Upload a local file to S3.

    Args:
        local_path: Absolute path to local file.
        s3_key: S3 object key (e.g. "construction-intelligence-agent/generated_documents/...").

    Returns:
        True on success, False on failure.
    """
    client = get_s3_client()
    if client is None:
        return False

    config = get_s3_config()
    local = Path(local_path)

    if not local.exists():
        logger.error("upload_file: local file not found: %s", local_path)
        return False

    try:
        client.upload_file(str(local), config.bucket_name, s3_key)
        logger.info("Uploaded %s → s3://%s/%s", local.name, config.bucket_name, s3_key)
        return True
    except Exception as e:
        logger.error("upload_file failed for %s: %s", s3_key, e)
        return False


def upload_bytes(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> bool:
    """
    Upload raw bytes to S3 (for JSON, small data).

    Args:
        data: Bytes to upload.
        s3_key: S3 object key.
        content_type: MIME type (default: application/octet-stream).

    Returns:
        True on success, False on failure.
    """
    client = get_s3_client()
    if client is None:
        return False

    config = get_s3_config()

    try:
        client.put_object(
            Bucket=config.bucket_name,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("Uploaded bytes → s3://%s/%s (%d bytes)", config.bucket_name, s3_key, len(data))
        return True
    except Exception as e:
        logger.error("upload_bytes failed for %s: %s", s3_key, e)
        return False


# ── Download Operations ──────────────────────────────────────────────────────


def download_file(s3_key: str, local_path: str) -> bool:
    """
    Download an S3 object to a local file.

    Args:
        s3_key: S3 object key.
        local_path: Local destination path (parent directory must exist).

    Returns:
        True on success, False on failure.
    """
    client = get_s3_client()
    if client is None:
        return False

    config = get_s3_config()
    local = Path(local_path)

    # Ensure parent directory exists
    local.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.download_file(config.bucket_name, s3_key, str(local))
        logger.info("Downloaded s3://%s/%s → %s", config.bucket_name, s3_key, local.name)
        return True
    except Exception as e:
        logger.error("download_file failed for %s: %s", s3_key, e)
        return False


def download_bytes(s3_key: str) -> Optional[bytes]:
    """
    Download an S3 object as raw bytes.

    Args:
        s3_key: S3 object key.

    Returns:
        Bytes on success, None on failure.
    """
    client = get_s3_client()
    if client is None:
        return None

    config = get_s3_config()

    try:
        response = client.get_object(Bucket=config.bucket_name, Key=s3_key)
        data = response["Body"].read()
        logger.info("Downloaded s3://%s/%s (%d bytes)", config.bucket_name, s3_key, len(data))
        return data
    except Exception as e:
        logger.error("download_bytes failed for %s: %s", s3_key, e)
        return None


# ── List Operations ──────────────────────────────────────────────────────────


def list_objects(prefix: str, max_keys: int = 1000) -> list[dict]:
    """
    List objects under an S3 prefix.

    Args:
        prefix: S3 key prefix (e.g. "rag-agent/conversation_sessions/").
        max_keys: Maximum number of keys to return (default 1000).

    Returns:
        List of dicts with 'Key', 'Size', 'LastModified' fields.
        Empty list on failure.
    """
    client = get_s3_client()
    if client is None:
        return []

    config = get_s3_config()
    results = []

    try:
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=config.bucket_name,
            Prefix=prefix,
            PaginationConfig={"MaxItems": max_keys},
        )

        for page in page_iterator:
            for obj in page.get("Contents", []):
                results.append(
                    {
                        "Key": obj["Key"],
                        "Size": obj["Size"],
                        "LastModified": obj["LastModified"],
                    }
                )

        logger.info("Listed %d objects under s3://%s/%s", len(results), config.bucket_name, prefix)
        return results
    except Exception as e:
        logger.error("list_objects failed for prefix %s: %s", prefix, e)
        return []


def object_exists(s3_key: str) -> bool:
    """
    Check if an S3 object exists.

    Args:
        s3_key: S3 object key.

    Returns:
        True if exists, False otherwise.
    """
    client = get_s3_client()
    if client is None:
        return False

    config = get_s3_config()

    try:
        client.head_object(Bucket=config.bucket_name, Key=s3_key)
        return True
    except Exception:
        return False


# ── Delete Operations ────────────────────────────────────────────────────────


def delete_object(s3_key: str) -> bool:
    """
    Delete a single S3 object.

    Args:
        s3_key: S3 object key.

    Returns:
        True on success, False on failure.
    """
    client = get_s3_client()
    if client is None:
        return False

    config = get_s3_config()

    try:
        client.delete_object(Bucket=config.bucket_name, Key=s3_key)
        logger.info("Deleted s3://%s/%s", config.bucket_name, s3_key)
        return True
    except Exception as e:
        logger.error("delete_object failed for %s: %s", s3_key, e)
        return False


def delete_prefix(prefix: str) -> int:
    """
    Delete ALL objects under an S3 prefix (e.g. deleting a session folder).

    Args:
        prefix: S3 key prefix.

    Returns:
        Number of objects deleted, 0 on failure.
    """
    client = get_s3_client()
    if client is None:
        return 0

    config = get_s3_config()
    objects = list_objects(prefix)

    if not objects:
        return 0

    try:
        delete_keys = [{"Key": obj["Key"]} for obj in objects]
        # S3 delete_objects supports up to 1000 keys per call
        for i in range(0, len(delete_keys), 1000):
            batch = delete_keys[i : i + 1000]
            client.delete_objects(
                Bucket=config.bucket_name,
                Delete={"Objects": batch, "Quiet": True},
            )

        count = len(delete_keys)
        logger.info("Deleted %d objects under s3://%s/%s", count, config.bucket_name, prefix)
        return count
    except Exception as e:
        logger.error("delete_prefix failed for %s: %s", prefix, e)
        return 0


# ── Presigned URLs ───────────────────────────────────────────────────────────


def generate_presigned_url(s3_key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for downloading an S3 object.

    Args:
        s3_key: S3 object key.
        expiration: URL validity in seconds (default 1 hour).

    Returns:
        Presigned URL string, or None on failure.
    """
    client = get_s3_client()
    if client is None:
        return None

    config = get_s3_config()

    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config.bucket_name, "Key": s3_key},
            ExpiresIn=expiration,
        )
        return url
    except Exception as e:
        logger.error("generate_presigned_url failed for %s: %s", s3_key, e)
        return None
