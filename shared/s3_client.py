"""
S3 download helper for the RAG-to-DocQA bridge.

Downloads PDFs from S3 using the s3_path from source_documents.
Falls back to download_url (HTTPS) if boto3 is not configured.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def _get_s3_client():
    """Lazy-init boto3 S3 client. Returns None if not configured."""
    try:
        import boto3
        return boto3.client(
            "s3",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
    except ImportError:
        logger.warning("boto3 not installed — S3 downloads unavailable")
        return None
    except Exception as exc:
        logger.warning("Failed to create S3 client: %s", exc)
        return None


def download_from_s3(
    s3_path: str,
    bucket: Optional[str] = None,
) -> Optional[str]:
    """Download a file from S3 to a temp path.

    Parameters
    ----------
    s3_path : str
        The S3 object key (e.g., "0104202614084657M401MECHANICALROOFPLAN1-1.pdf")
        or a full "bucket/key" path.
    bucket : str, optional
        S3 bucket name. Defaults to S3_BUCKET_NAME env var or "ifieldsmart-drawings".

    Returns
    -------
    str or None
        Path to the downloaded temp file, or None if download failed.
        Caller is responsible for cleaning up the temp file.
    """
    client = _get_s3_client()
    if client is None:
        return None

    bucket = bucket or os.environ.get("S3_BUCKET_NAME", "ifieldsmart-drawings")

    # If s3_path contains a slash, it might be "bucket/key" format
    if "/" in s3_path and not s3_path.startswith("s3://"):
        parts = s3_path.split("/", 1)
        if len(parts) == 2 and not parts[0].endswith(".pdf"):
            bucket = parts[0]
            s3_path = parts[1]

    # Strip s3:// prefix if present
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
        if "/" in s3_path:
            bucket, s3_path = s3_path.split("/", 1)

    # Ensure .pdf extension for temp file
    suffix = ".pdf" if s3_path.lower().endswith(".pdf") else ""

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()

        logger.info("Downloading s3://%s/%s → %s", bucket, s3_path, tmp_path)
        client.download_file(bucket, s3_path, tmp_path)

        file_size = os.path.getsize(tmp_path)
        logger.info("Downloaded %d bytes from S3", file_size)

        if file_size == 0:
            os.unlink(tmp_path)
            logger.warning("Downloaded file is empty")
            return None

        return tmp_path
    except Exception as exc:
        logger.error("S3 download failed for %s/%s: %s", bucket, s3_path, exc)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def download_from_url(url: str) -> Optional[str]:
    """Download a file from an HTTPS URL to a temp path.

    Fallback when S3 boto3 access is not configured.

    Returns
    -------
    str or None
        Path to the downloaded temp file, or None if download failed.
    """
    if not url:
        return None

    try:
        import httpx

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()

        logger.info("Downloading from URL: %s", url[:100])
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

            with open(tmp_path, "wb") as f:
                f.write(resp.content)

        file_size = os.path.getsize(tmp_path)
        logger.info("Downloaded %d bytes from URL", file_size)

        if file_size == 0:
            os.unlink(tmp_path)
            return None

        return tmp_path
    except Exception as exc:
        logger.error("URL download failed: %s", exc)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None
