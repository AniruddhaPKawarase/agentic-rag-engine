"""Phase 7.2: RAG Agent S3 migration tests."""
import json
import os
import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

AGENT_ROOT = Path(__file__).resolve().parent.parent
PROD_ROOT = AGENT_ROOT.parent.parent
sys.path.insert(0, str(PROD_ROOT))
sys.path.insert(0, str(AGENT_ROOT))

TEST_BUCKET = "test-vcs-agents"


@pytest.fixture(autouse=True)
def s3_env(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET_NAME", TEST_BUCKET)
    monkeypatch.setenv("S3_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("S3_AGENT_PREFIX", "rag-agent")
    monkeypatch.setenv("S3_ENDPOINT_URL", "")
    monkeypatch.setenv("S3_FAISS_PREFIX", "rag-agent/faiss_indexes")
    from s3_utils.config import get_s3_config
    from s3_utils.client import get_s3_client
    get_s3_config.cache_clear()
    get_s3_client.cache_clear()


@pytest.fixture
def s3_bucket():
    with mock_aws():
        from s3_utils.config import get_s3_config
        from s3_utils.client import get_s3_client
        get_s3_config.cache_clear()
        get_s3_client.cache_clear()
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=TEST_BUCKET)
        yield conn
        get_s3_config.cache_clear()
        get_s3_client.cache_clear()


class TestSessionS3Upload:
    """Test session write-through to S3."""

    def test_upload_session_json(self, s3_bucket):
        from s3_utils.operations import upload_bytes, download_bytes
        from s3_utils.helpers import session_key
        session_data = {
            "session_id": "session_abc123",
            "created_at": 1700000000.0,
            "last_accessed": 1700000100.0,
            "messages": [
                {"role": "user", "content": "What about HVAC?", "timestamp": 1700000000.0, "tokens": 10}
            ],
            "context": {"project_id": 7212},
            "summaries": [],
            "total_tokens": 10,
            "metadata": {},
        }
        s3_key = session_key("rag-agent", "session_abc123.json")
        data = json.dumps(session_data).encode("utf-8")
        assert upload_bytes(data, s3_key) is True
        restored = json.loads(download_bytes(s3_key))
        assert restored["session_id"] == "session_abc123"
        assert len(restored["messages"]) == 1

    def test_multiple_sessions(self, s3_bucket):
        from s3_utils.operations import upload_bytes, list_objects
        from s3_utils.helpers import session_key
        for i in range(5):
            s3_key = session_key("rag-agent", f"session_{i}.json")
            upload_bytes(json.dumps({"session_id": f"session_{i}"}).encode(), s3_key)
        objects = list_objects("rag-agent/conversation_sessions/")
        assert len(objects) == 5


class TestSessionS3Delete:
    def test_delete_session_from_s3(self, s3_bucket):
        from s3_utils.operations import upload_bytes, delete_object, object_exists
        from s3_utils.helpers import session_key
        s3_key = session_key("rag-agent", "session_to_delete.json")
        upload_bytes(b'{"session_id": "del"}', s3_key)
        assert object_exists(s3_key) is True
        delete_object(s3_key)
        assert object_exists(s3_key) is False


class TestFAISSIndexS3:
    """Test FAISS index backup/restore via S3."""

    def test_upload_faiss_index(self, s3_bucket, tmp_path):
        from s3_utils.operations import upload_file, object_exists
        from s3_utils.helpers import faiss_index_key
        fake_index = tmp_path / "faiss_index_7166.bin"
        fake_index.write_bytes(b"\x00" * 1024)  # Fake binary
        s3_key = faiss_index_key("faiss_index_7166.bin")
        assert upload_file(str(fake_index), s3_key) is True
        assert object_exists(s3_key) is True

    def test_download_faiss_index(self, s3_bucket, tmp_path):
        from s3_utils.operations import upload_file, download_file
        from s3_utils.helpers import faiss_index_key
        source = tmp_path / "faiss_index_7201.bin"
        source.write_bytes(b"\x01" * 2048)
        s3_key = faiss_index_key("faiss_index_7201.bin")
        upload_file(str(source), s3_key)
        dest = tmp_path / "downloaded_index.bin"
        assert download_file(s3_key, str(dest)) is True
        assert dest.read_bytes() == source.read_bytes()

    def test_upload_metadata_jsonl(self, s3_bucket, tmp_path):
        from s3_utils.operations import upload_file, download_bytes
        from s3_utils.helpers import faiss_index_key
        meta = tmp_path / "metadata_7166.jsonl"
        lines = [json.dumps({"text": f"chunk {i}", "project_id": 7166}) for i in range(3)]
        meta.write_text("\n".join(lines))
        s3_key = faiss_index_key("metadata_7166.jsonl")
        assert upload_file(str(meta), s3_key) is True
        content = download_bytes(s3_key).decode()
        assert len(content.strip().split("\n")) == 3


class TestRollback:
    def test_local_mode_no_s3(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        from s3_utils.config import get_s3_config
        get_s3_config.cache_clear()
        assert get_s3_config().is_s3_enabled is False

    def test_operations_noop_in_local_mode(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        from s3_utils.config import get_s3_config
        from s3_utils.client import get_s3_client
        get_s3_config.cache_clear()
        get_s3_client.cache_clear()
        from s3_utils.operations import upload_bytes, download_bytes
        assert upload_bytes(b"test", "key") is False
        assert download_bytes("key") is None
