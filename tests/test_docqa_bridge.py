"""Phase 2.3: DocQABridge — single source of truth for RAG→DocQA handoff."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from httpx import Response

from gateway.docqa_bridge import DocQABridge, DOCQA_FALLBACK_MESSAGE


class _FakeSession:
    def __init__(self):
        self.active_agent = "rag"
        self.docqa_session_id = None
        self.selected_documents = []
        self.last_intent_decision = None


@pytest.fixture
def fake_session():
    return _FakeSession()


@pytest.fixture
def mock_mm(fake_session):
    mm = MagicMock()
    mm.get_session.return_value = fake_session
    return mm


# ---------------------------------------------------------------- S3 download

@pytest.mark.asyncio
async def test_download_to_temp_uses_boto3_when_available(mock_mm, tmp_path):
    fake_s3 = MagicMock()
    def _download(Bucket, Key, Fileobj):
        Fileobj.write(b"%PDF-1.4 faked via boto3")
    fake_s3.download_fileobj.side_effect = _download

    bridge = DocQABridge(memory_manager=mock_mm, s3_client=fake_s3)
    path = await bridge._download_to_temp(
        {"s3_path": "agentic-ai-production/drawings/foo.pdf", "file_name": "foo.pdf"}
    )
    try:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        fake_s3.download_fileobj.assert_called_once()
        args, _ = fake_s3.download_fileobj.call_args
        # The key should have bucket prefix stripped
        assert args[0] == "agentic-ai-production"
        assert args[1] == "drawings/foo.pdf"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
@respx.mock
async def test_download_to_temp_falls_back_to_presigned_url(mock_mm, tmp_path):
    fake_s3 = MagicMock()
    fake_s3.download_fileobj.side_effect = RuntimeError("boto3 boom")

    respx.get("https://example.com/presigned").mock(
        return_value=Response(200, content=b"%PDF-1.4 via http")
    )
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=fake_s3)
    path = await bridge._download_to_temp(
        {
            "s3_path": "bucket/foo.pdf",
            "file_name": "foo.pdf",
            "download_url": "https://example.com/presigned",
        }
    )
    try:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_download_to_temp_raises_when_no_fallback(mock_mm):
    fake_s3 = MagicMock()
    fake_s3.download_fileobj.side_effect = RuntimeError("s3 boom")
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=fake_s3)
    with pytest.raises(RuntimeError):
        await bridge._download_to_temp(
            {"s3_path": "bucket/foo.pdf", "file_name": "foo.pdf"}
        )


# ---------------------------------------------------- ensure_document_loaded

@pytest.mark.asyncio
async def test_ensure_document_loaded_uploads_and_persists(mock_mm, fake_session, tmp_path):
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF")

    with patch.object(bridge, "_download_to_temp",
                      new=AsyncMock(return_value=str(fake_pdf))), \
         patch("gateway.docqa_bridge.docqa_client.upload_only",
               new=AsyncMock(return_value={
                   "session_id": "dq_new", "total_session_chunks": 12,
               })):
        dq_sid = await bridge.ensure_document_loaded(
            session_id="rag_s1",
            doc_ref={"s3_path": "bucket/x.pdf", "file_name": "x.pdf",
                     "download_url": "https://presigned"},
        )
    assert dq_sid == "dq_new"
    assert fake_session.active_agent == "docqa"
    assert fake_session.docqa_session_id == "dq_new"
    assert len(fake_session.selected_documents) == 1
    assert fake_session.selected_documents[0]["s3_path"] == "bucket/x.pdf"
    assert fake_session.selected_documents[0]["docqa_session_id"] == "dq_new"
    assert "loaded_at" in fake_session.selected_documents[0]


@pytest.mark.asyncio
async def test_ensure_document_loaded_reuses_existing_if_same_doc(mock_mm, fake_session):
    fake_session.docqa_session_id = "dq_existing"
    fake_session.selected_documents = [{
        "s3_path": "bucket/x.pdf", "file_name": "x.pdf",
        "docqa_session_id": "dq_existing",
    }]
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    # Should not call upload or download
    with patch.object(bridge, "_download_to_temp",
                      new=AsyncMock(side_effect=AssertionError("should not download"))), \
         patch("gateway.docqa_bridge.docqa_client.upload_only",
               new=AsyncMock(side_effect=AssertionError("should not upload"))):
        dq_sid = await bridge.ensure_document_loaded(
            session_id="rag_s1",
            doc_ref={"s3_path": "bucket/x.pdf", "file_name": "x.pdf"},
        )
    assert dq_sid == "dq_existing"


@pytest.mark.asyncio
async def test_ensure_document_loaded_cleans_tempfile_on_success(mock_mm, fake_session, tmp_path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF")
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    with patch.object(bridge, "_download_to_temp",
                      new=AsyncMock(return_value=str(fake_pdf))), \
         patch("gateway.docqa_bridge.docqa_client.upload_only",
               new=AsyncMock(return_value={"session_id": "dq_new"})):
        await bridge.ensure_document_loaded(
            session_id="rag_s1",
            doc_ref={"s3_path": "b/x.pdf", "file_name": "x.pdf"},
        )
    assert not os.path.exists(str(fake_pdf)), "temp file must be deleted"


@pytest.mark.asyncio
async def test_ensure_document_loaded_cleans_tempfile_on_upload_error(mock_mm, fake_session, tmp_path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF")
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    with patch.object(bridge, "_download_to_temp",
                      new=AsyncMock(return_value=str(fake_pdf))), \
         patch("gateway.docqa_bridge.docqa_client.upload_only",
               new=AsyncMock(return_value={"error": "timeout"})):
        with pytest.raises(RuntimeError):
            await bridge.ensure_document_loaded(
                session_id="rag_s1",
                doc_ref={"s3_path": "b/x.pdf", "file_name": "x.pdf"},
            )
    assert not os.path.exists(str(fake_pdf)), "temp file must be deleted even on upload error"


# -------------------------------------------------------------- ask + normalize

@pytest.mark.asyncio
async def test_ask_forwards_to_query_document(mock_mm):
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    with patch("gateway.docqa_bridge.docqa_client.query_document",
               new=AsyncMock(return_value={
                   "answer": "on page 14", "session_id": "dq_x",
                   "source_documents": [{"file_name": "x.pdf", "page": 14}],
                   "groundedness_score": 0.92,
               })) as m:
        result = await bridge.ask(
            docqa_session_id="dq_x", query="where is fire damper?"
        )
    m.assert_awaited_once_with(session_id="dq_x", query="where is fire damper?")
    assert result["answer"] == "on page 14"


def test_normalize_success_response(mock_mm):
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    dq_resp = {
        "answer": "page 14",
        "session_id": "dq_x",
        "source_documents": [{"file_name": "HVAC.pdf", "page": 14, "snippet": "…"}],
        "groundedness_score": 0.92,
    }
    unified = bridge.normalize(
        docqa_response=dq_resp,
        rag_session_id="rag_s1",
        selected_document={"file_name": "HVAC.pdf"},
    )
    assert unified["success"] is True
    assert unified["answer"] == "page 14"
    assert unified["active_agent"] == "docqa"
    assert unified["selected_document"]["file_name"] == "HVAC.pdf"
    assert unified["docqa_session_id"] == "dq_x"
    assert unified["session_id"] == "rag_s1"
    assert unified["engine_used"] == "docqa"
    assert unified["source_documents"][0]["page"] == 14
    assert unified["needs_clarification"] is False


def test_normalize_error_response(mock_mm):
    bridge = DocQABridge(memory_manager=mock_mm, s3_client=MagicMock())
    unified = bridge.normalize(
        docqa_response={"error": "timeout"},
        rag_session_id="rag_s1",
        selected_document={"file_name": "x.pdf"},
    )
    assert unified["success"] is False
    assert unified["engine_used"] == "docqa_fallback"
    assert unified["active_agent"] == "rag"  # degrade so user can continue
    assert unified["fallback_used"] is True
    assert DOCQA_FALLBACK_MESSAGE in unified["answer"] or "timeout" in unified["answer"].lower()
