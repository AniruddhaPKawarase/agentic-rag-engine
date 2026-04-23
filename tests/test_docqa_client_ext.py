"""Phase 2.2: gateway.docqa_client.upload_only helper tests."""
from __future__ import annotations

import pytest
import respx
from httpx import Response

from gateway import docqa_client


@pytest.mark.asyncio
@respx.mock
async def test_upload_only_calls_converse_with_placeholder():
    route = respx.post("http://localhost:8006/api/converse").mock(
        return_value=Response(200, json={
            "session_id": "dq_abc",
            "answer": "Document uploaded.",
            "total_session_chunks": 42,
        })
    )
    # Write a small bytes file to act as the upload source.
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4 tiny")
        tmp_path = f.name
    try:
        result = await docqa_client.upload_only(
            file_path=tmp_path,
            file_name="x.pdf",
            session_id=None,
        )
    finally:
        os.unlink(tmp_path)
    assert result.get("session_id") == "dq_abc"
    assert result.get("total_session_chunks") == 42
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_upload_only_reuses_existing_session_id():
    respx.post("http://localhost:8006/api/converse").mock(
        return_value=Response(200, json={
            "session_id": "dq_abc", "total_session_chunks": 84,
        })
    )
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4")
        tmp_path = f.name
    try:
        result = await docqa_client.upload_only(
            file_path=tmp_path,
            file_name="y.pdf",
            session_id="dq_abc",
        )
    finally:
        os.unlink(tmp_path)
    assert result.get("session_id") == "dq_abc"


@pytest.mark.asyncio
@respx.mock
async def test_upload_only_handles_timeout_gracefully():
    import httpx
    respx.post("http://localhost:8006/api/converse").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4")
        tmp_path = f.name
    try:
        result = await docqa_client.upload_only(
            file_path=tmp_path, file_name="x.pdf",
        )
    finally:
        os.unlink(tmp_path)
    assert "error" in result


@pytest.mark.asyncio
@respx.mock
async def test_upload_only_handles_http_error_gracefully():
    respx.post("http://localhost:8006/api/converse").mock(
        return_value=Response(500, json={"detail": "server boom"})
    )
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4")
        tmp_path = f.name
    try:
        result = await docqa_client.upload_only(
            file_path=tmp_path, file_name="x.pdf",
        )
    finally:
        os.unlink(tmp_path)
    assert "error" in result
    assert "500" in result["error"]
