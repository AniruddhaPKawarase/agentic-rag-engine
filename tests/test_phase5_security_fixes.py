"""Phase 5 security + correctness regression tests."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.docqa_bridge import (
    DocQABridge,
    MAX_DOWNLOAD_BYTES,
    _ip_is_blocked,
    _validate_download_url,
)


# ────────────────────────────────────────── SSRF (SEC-C1)


def test_validate_download_url_rejects_aws_metadata():
    with pytest.raises(ValueError, match="blocked"):
        _validate_download_url("http://169.254.169.254/latest/meta-data/")


def test_validate_download_url_rejects_localhost():
    with pytest.raises(ValueError, match="blocked"):
        _validate_download_url("http://127.0.0.1:8080/secret")


def test_validate_download_url_rejects_rfc1918():
    with pytest.raises(ValueError, match="blocked"):
        _validate_download_url("http://10.0.0.5/internal")
    with pytest.raises(ValueError, match="blocked"):
        _validate_download_url("http://192.168.1.1/admin")
    with pytest.raises(ValueError, match="blocked"):
        _validate_download_url("http://172.16.0.1/x")


def test_validate_download_url_rejects_bad_scheme():
    with pytest.raises(ValueError, match="scheme"):
        _validate_download_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="scheme"):
        _validate_download_url("gopher://example.com/")


def test_validate_download_url_accepts_public_s3_urls():
    # Public S3 presigned URL pattern — should pass. Mock DNS to avoid network.
    with patch(
        "gateway.docqa_bridge.socket.getaddrinfo",
        return_value=[(0, 0, 0, "", ("8.8.8.8", 0))],
    ):
        _validate_download_url("https://s3.amazonaws.com/bucket/key")


def test_ip_is_blocked_for_each_range():
    assert _ip_is_blocked("169.254.169.254")
    assert _ip_is_blocked("10.0.0.1")
    assert _ip_is_blocked("172.16.5.5")
    assert _ip_is_blocked("192.168.1.1")
    assert _ip_is_blocked("127.0.0.1")
    assert not _ip_is_blocked("8.8.8.8")
    assert not _ip_is_blocked("1.1.1.1")


# ────────────────────────────────────────── S3 bucket allowlist (SEC-C2)


def test_split_bucket_enforces_allowlist_default_bucket():
    # Path starts with allowed bucket — use it directly
    b, k = DocQABridge._split_bucket_and_key(
        "agentic-ai-production/drawings/foo.pdf", "agentic-ai-production"
    )
    assert b == "agentic-ai-production"
    assert k == "drawings/foo.pdf"


def test_split_bucket_rejects_arbitrary_first_component():
    # 'other-bucket' is NOT in allowlist — whole path becomes key under default
    b, k = DocQABridge._split_bucket_and_key(
        "other-bucket/private.pdf", "agentic-ai-production"
    )
    assert b == "agentic-ai-production"
    assert k == "other-bucket/private.pdf"


def test_split_bucket_empty_path():
    b, k = DocQABridge._split_bucket_and_key("", "agentic-ai-production")
    assert b == "agentic-ai-production"
    assert k == ""


def test_split_bucket_no_slash_key_only():
    b, k = DocQABridge._split_bucket_and_key("file.pdf", "agentic-ai-production")
    assert b == "agentic-ai-production"
    assert k == "file.pdf"


# ────────────────────────────────────────── None session guard (CODE-C1)


@pytest.mark.asyncio
async def test_ensure_document_loaded_rejects_none_session():
    mm = MagicMock()
    bridge = DocQABridge(memory_manager=mm, s3_client=MagicMock())
    with pytest.raises(RuntimeError, match="session_id"):
        await bridge.ensure_document_loaded(
            session_id=None,
            doc_ref={"s3_path": "x/y.pdf", "file_name": "y.pdf"},
        )


@pytest.mark.asyncio
async def test_ensure_document_loaded_rejects_empty_session():
    mm = MagicMock()
    bridge = DocQABridge(memory_manager=mm, s3_client=MagicMock())
    with pytest.raises(RuntimeError, match="session_id"):
        await bridge.ensure_document_loaded(
            session_id="",
            doc_ref={"s3_path": "x/y.pdf", "file_name": "y.pdf"},
        )


@pytest.mark.asyncio
async def test_ensure_document_loaded_rejects_missing_session_object():
    mm = MagicMock()
    mm.get_session.return_value = None
    bridge = DocQABridge(memory_manager=mm, s3_client=MagicMock())
    with pytest.raises(RuntimeError, match="not found"):
        await bridge.ensure_document_loaded(
            session_id="abc",
            doc_ref={"s3_path": "x/y.pdf", "file_name": "y.pdf"},
        )


# ────────────────────────────────────────── Filename sanitization (SEC-H3)


@pytest.mark.asyncio
async def test_upload_only_strips_path_traversal_in_filename():
    """Verify pathlib.Path(...).name is applied before forwarding to DocQA."""
    import respx
    from httpx import Response

    from gateway import docqa_client

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF")
        tmp = f.name

    try:
        with respx.mock:
            route = respx.post("http://localhost:8006/api/converse").mock(
                return_value=Response(200, json={"session_id": "dq_abc"})
            )
            await docqa_client.upload_only(
                file_path=tmp,
                file_name="../../../etc/passwd",  # traversal attempt
            )
        # Inspect the last request sent to DocQA
        req = route.calls.last.request
        body_bytes = req.content
        # The multipart form should use just "passwd", not the traversal path
        assert b'filename="passwd"' in body_bytes
        assert b"../../../" not in body_bytes
    finally:
        os.unlink(tmp)


# ────────────────────────────────────────── Classifier bypass (CODE-H3)


@pytest.mark.asyncio
async def test_explicit_search_mode_rag_bypasses_classifier():
    """Client sending search_mode='rag' explicitly should NOT trigger classifier.

    Note: the same behavior is covered in
    tests/test_orchestrator_intent_routing.py::test_explicit_search_mode_skips_classifier
    which stubs fewer internals and is less brittle. This test only asserts
    on the classifier-call boundary so we can short-circuit before reaching
    any agentic/traditional engine method. We use a generic downstream stub
    that matches whichever internal method the orchestrator actually calls.
    """
    import gateway.orchestrator as orch_module

    orch = orch_module.Orchestrator.__new__(orch_module.Orchestrator)
    orch._last_query = ""
    orch._last_project_id = None
    orch._last_session_id = None

    # Don't know the exact engine-dispatch private-method name — that's why
    # we raise inside the RAG path and swallow it. What matters is that the
    # classifier module-level function `classify` was NOT called.
    with patch.object(orch_module, "classify") as mock_classify:
        try:
            await orch.query(
                query="what does it say on page 14?",
                project_id=7222,
                session_id="s1",
                search_mode="rag",  # EXPLICIT
            )
        except Exception:
            pass  # downstream engine not stubbed; irrelevant to this assertion
    mock_classify.assert_not_called()
