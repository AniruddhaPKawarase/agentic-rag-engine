"""Local unit tests for Fix #8 — ``_ensure_signed_source_urls``.

Runs without the full agent stack. Exercises the in-place re-signing logic on
hand-crafted source_documents shaped like traditional and agentic outputs.

Run:
    python -m pytest PROD_SETUP/unified-rag-agent/tests/test_url_resigning.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make gateway package importable when running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _load_env() -> None:
    """Ensure AWS creds + bucket env vars are loaded from .env for SigV4."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    os.environ.setdefault("S3_BUCKET_NAME", "ifieldsmart")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def _fake_signed(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{key}?X-Amz-Signature=fakesig&X-Amz-Expires=3600"


def test_ensure_signed_noop_when_empty() -> None:
    from gateway.orchestrator import _ensure_signed_source_urls
    _ensure_signed_source_urls(None)
    _ensure_signed_source_urls([])
    _ensure_signed_source_urls("not a list")  # type: ignore[arg-type]
    # No exception == pass


def test_keeps_already_signed_url_untouched() -> None:
    from gateway.orchestrator import _ensure_signed_source_urls
    signed = "https://ifieldsmart.s3.amazonaws.com/x.pdf?X-Amz-Signature=deadbeef"
    docs = [{"s3_path": "ifieldsmart/a/b", "pdf_name": "x", "download_url": signed}]
    _ensure_signed_source_urls(docs)
    assert docs[0]["download_url"] == signed  # unchanged


def test_resigns_unsigned_url_from_traditional_shape() -> None:
    """Traditional engine returns download_url without X-Amz-Signature."""
    from gateway.orchestrator import _ensure_signed_source_urls
    unsigned = "https://ifieldsmart.s3.amazonaws.com/proj/Drawings/pdf1/file.pdf"
    docs = [{
        "s3_path": "ifieldsmart/proj/Drawings/pdf1",
        "pdf_name": "file",
        "download_url": unsigned,
    }]
    with patch("gateway.orchestrator._build_download_url", return_value=_fake_signed("ifieldsmart", "proj/Drawings/pdf1/file.pdf")):
        _ensure_signed_source_urls(docs)
    assert "X-Amz-Signature" in docs[0]["download_url"]
    assert docs[0]["download_url"] != unsigned


def test_resigns_when_download_url_missing() -> None:
    from gateway.orchestrator import _ensure_signed_source_urls
    docs = [{
        "s3_path": "ifieldsmart/proj/Specification/pdf2",
        "pdf_name": "spec1",
    }]
    with patch("gateway.orchestrator._build_download_url", return_value=_fake_signed("ifieldsmart", "proj/Specification/pdf2/spec1.pdf")):
        _ensure_signed_source_urls(docs)
    assert docs[0]["download_url"].startswith("https://ifieldsmart.s3.amazonaws.com/")
    assert "X-Amz-Signature" in docs[0]["download_url"]


def test_skips_when_no_path_and_no_name() -> None:
    from gateway.orchestrator import _ensure_signed_source_urls
    docs = [{"s3_path": "", "pdf_name": "", "download_url": ""}]
    _ensure_signed_source_urls(docs)
    assert docs[0]["download_url"] == ""  # untouched, no regeneration attempted


def test_build_failure_does_not_raise() -> None:
    from gateway.orchestrator import _ensure_signed_source_urls
    docs = [{"s3_path": "ifieldsmart/a/b", "pdf_name": "x", "download_url": "http://unsigned"}]
    with patch("gateway.orchestrator._build_download_url", side_effect=RuntimeError("boom")):
        _ensure_signed_source_urls(docs)  # should not raise
    # URL stays as-is since we couldn't regenerate
    assert docs[0]["download_url"] == "http://unsigned"


def test_handles_mixed_list_agentic_and_traditional() -> None:
    """Agentic sources are already signed; traditional sources aren't. Re-sign only the unsigned ones."""
    from gateway.orchestrator import _ensure_signed_source_urls
    already_signed = "https://ifieldsmart.s3.amazonaws.com/a.pdf?X-Amz-Signature=abc"
    unsigned = "https://ifieldsmart.s3.amazonaws.com/b.pdf"
    docs = [
        {"s3_path": "ifieldsmart/a", "pdf_name": "a", "download_url": already_signed},  # agentic
        {"s3_path": "ifieldsmart/b", "pdf_name": "b", "download_url": unsigned},         # traditional
    ]
    with patch("gateway.orchestrator._build_download_url", return_value=_fake_signed("ifieldsmart", "b/b.pdf")):
        _ensure_signed_source_urls(docs)
    assert docs[0]["download_url"] == already_signed  # agentic untouched
    assert "X-Amz-Signature" in docs[1]["download_url"]  # traditional re-signed


def test_live_resign_produces_working_url() -> None:
    """Integration-style test: with real AWS creds, re-signing produces a URL that
    actually fetches via ranged GET. Skipped when creds are missing."""
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        pytest.skip("AWS creds not available in env")
    import requests
    from gateway.orchestrator import _ensure_signed_source_urls, _build_download_url
    # Use a known-good key from project 7222
    s3_path = "ifieldsmart/jrparkwayhotel2511202509120993/Drawings/pdf2511202514114083"
    pdf_name = "2511202514133993JRpkwyHotelMechPlans1-1"
    # Sanity: the helper produces a URL
    url = _build_download_url(s3_path, pdf_name)
    assert url and "X-Amz-Signature" in url
    # Re-signing a doc that came in unsigned should give same shape
    docs = [{"s3_path": s3_path, "pdf_name": pdf_name, "download_url": f"https://unsigned/foo/bar.pdf"}]
    _ensure_signed_source_urls(docs)
    resigned = docs[0]["download_url"]
    assert "X-Amz-Signature" in resigned
    # Verify it actually fetches
    r = requests.get(resigned, headers={"Range": "bytes=0-1023"}, timeout=15)
    assert r.status_code in (200, 206), f"re-signed URL failed with {r.status_code}: {r.text[:200]}"
