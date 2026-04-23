"""
DocQA Bridge — adapter layer between the unified RAG gateway and the DocQA agent.

Single source of truth for:
  1. Fetching source documents from S3 (or via presigned URL fallback)
  2. Uploading them to the DocQA agent on port 8006 (via docqa_client.upload_only)
  3. Persisting DocQA session state in ConversationSession (active_agent,
     docqa_session_id, selected_documents)
  4. Forwarding follow-up queries to DocQA /api/chat (via docqa_client.query_document)
  5. Normalizing DocQA responses into UnifiedResponse-shaped dicts

This module must stay thin. Future RAG changes or DocQA changes should not
ripple into other gateway modules — everything DocQA-related lives here.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import tempfile
import urllib.parse
from datetime import datetime
from typing import Any, Optional

import httpx

from gateway import docqa_client

logger = logging.getLogger(__name__)

DOCQA_FALLBACK_MESSAGE = (
    "Could not load document for deep-dive. Try selecting a different "
    "source or ask a general question to continue."
)

S3_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("DOCQA_BRIDGE_S3_TIMEOUT", "60"))

# [SEC-H1] Maximum bytes allowed per download (boto3 path + presigned path).
MAX_DOWNLOAD_BYTES = int(
    os.environ.get("DOCQA_BRIDGE_MAX_BYTES", str(100 * 1024 * 1024))  # 100 MB default
)

# [SEC-C2] Allowlist of S3 bucket names that may appear as the first path component.
_ALLOWED_BUCKETS: frozenset[str] = frozenset(
    b.strip()
    for b in os.environ.get(
        "S3_ALLOWED_BUCKETS",
        os.environ.get("S3_BUCKET", "agentic-ai-production"),
    ).split(",")
    if b.strip()
)

# [SEC-C1] SSRF-blocked IP ranges.
_SSRF_BLOCKED_RANGES = tuple(
    ipaddress.ip_network(n)
    for n in (
        "169.254.0.0/16",  # AWS metadata + link-local
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "0.0.0.0/8",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)


def _ip_is_blocked(ip: str) -> bool:
    """Return True if the IP address falls within a blocked (internal) range."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in _SSRF_BLOCKED_RANGES)


def _validate_download_url(url: str) -> None:
    """Raise ValueError if url scheme is unsupported or resolves to a blocked range.

    Guards against SSRF attacks via the presigned-URL fallback path.
    """
    if not url:
        raise ValueError("empty download_url")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("missing hostname")
    # Literal IP — check directly.
    try:
        ipaddress.ip_address(host)
        if _ip_is_blocked(host):
            raise ValueError(f"blocked literal IP: {host}")
        return
    except ValueError as exc:
        # Re-raise our own ValueError; swallow ipaddress parse errors (not a literal IP).
        if "blocked literal IP" in str(exc):
            raise
    # DNS resolve and check all returned addresses.
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for {host}: {exc}") from exc
    for res in results:
        sockaddr = res[4]
        ip = sockaddr[0]
        if _ip_is_blocked(ip):
            raise ValueError(f"blocked resolved IP: {host} -> {ip}")


class DocQABridge:
    """Thin adapter. One instance per gateway process.

    The memory_manager is injected so DocQA session state is persisted
    keyed by the RAG session_id. The s3_client is optional — if not
    provided, boto3 is lazy-initialized on first download.
    """

    def __init__(self, memory_manager: Any, s3_client: Any = None) -> None:
        self.mm = memory_manager
        self.s3_client = s3_client

    # ------------------------------------------------------------------ S3

    def _init_s3(self) -> Any:
        if self.s3_client is not None:
            return self.s3_client
        import boto3
        from botocore.config import Config
        self.s3_client = boto3.client(
            "s3",
            config=Config(
                max_pool_connections=20,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        return self.s3_client

    @staticmethod
    def _split_bucket_and_key(s3_path: str, default_bucket: str) -> tuple[str, str]:
        """Split 's3_path' into (bucket, key). Enforces the S3 bucket allowlist.

        Rules:
          - Empty/None  → (default_bucket, "")
          - No slash    → (default_bucket, p)  [legacy single-key paths]
          - First component IS in _ALLOWED_BUCKETS → use it as bucket, rest as key
          - First component NOT in _ALLOWED_BUCKETS → treat whole path as key under
            default_bucket (covers paths like "drawings/foo.pdf" where the first
            dir is not a bucket name, and also prevents cross-bucket injection).

        Examples (with default='agentic-ai-production' in allowlist):
          'agentic-ai-production/drawings/foo.pdf' -> ('agentic-ai-production', 'drawings/foo.pdf')
          'drawings/foo.pdf'                       -> ('agentic-ai-production', 'drawings/foo.pdf')
          'other-bucket/key.pdf'                   -> ('agentic-ai-production', 'other-bucket/key.pdf')
          'foo.pdf'                                -> ('agentic-ai-production', 'foo.pdf')
        """
        p = (s3_path or "").lstrip("/")
        if not p:
            return default_bucket, ""
        if "/" not in p:
            return default_bucket, p
        first, rest = p.split("/", 1)
        if first in _ALLOWED_BUCKETS:
            return first, rest
        # First component is NOT a known bucket — treat whole path as key.
        return default_bucket, p

    async def _download_to_temp(self, doc_ref: dict) -> str:
        """Download S3 object to a NamedTemporaryFile, return the path.

        Order of attempts:
          1. boto3 S3 client (fastest, AWS SDK)
          2. httpx GET against doc_ref['download_url'] (presigned fallback)

        Caller is responsible for deleting the returned path.
        Raises RuntimeError if both paths fail.
        """
        bucket_env = os.environ.get("S3_BUCKET", "agentic-ai-production")
        s3_path = doc_ref.get("s3_path") or ""
        bucket, key = self._split_bucket_and_key(s3_path, default_bucket=bucket_env)

        file_name = doc_ref.get("file_name") or (key.rsplit("/", 1)[-1] or "document.pdf")
        suffix = os.path.splitext(file_name)[1] or ".pdf"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()

        # --- Attempt 1: boto3 ---
        try:
            client = self._init_s3()
            with open(tmp_path, "wb") as fp:
                client.download_fileobj(bucket, key, fp)
            # [SEC-H1] Post-download size guard for the boto3 path.
            size = os.path.getsize(tmp_path)
            if size > MAX_DOWNLOAD_BYTES:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"DocQABridge: S3 object exceeded {MAX_DOWNLOAD_BYTES} bytes "
                    f"(got {size}); rejecting to avoid disk exhaustion"
                )
            logger.info(
                "DocQABridge S3 download ok: bucket=%s key=%s size=%s",
                bucket, key, size,
            )
            return tmp_path
        except Exception as s3_exc:
            logger.warning(
                "DocQABridge S3 download failed (%s); falling back to presigned URL",
                type(s3_exc).__name__,
            )

            # --- Attempt 2: presigned URL ---
            url = doc_ref.get("download_url")
            if not url:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise RuntimeError(
                    "DocQABridge: S3 download failed and no download_url to fall back to"
                ) from s3_exc

            try:
                # [SEC-C1] Validate URL against SSRF-blocked ranges before fetching.
                _validate_download_url(url)
                # [SEC-C1] follow_redirects=False prevents redirect-based SSRF.
                async with httpx.AsyncClient(
                    timeout=S3_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=False
                ) as hc:
                    async with hc.stream("GET", url) as resp:
                        resp.raise_for_status()
                        with open(tmp_path, "wb") as fp:
                            # [SEC-H1] Stream with size cap to prevent disk exhaustion.
                            total = 0
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                total += len(chunk)
                                if total > MAX_DOWNLOAD_BYTES:
                                    raise RuntimeError(
                                        f"DocQABridge: download exceeded "
                                        f"{MAX_DOWNLOAD_BYTES} bytes (got {total}); "
                                        f"aborting to avoid disk exhaustion"
                                    )
                                fp.write(chunk)
                logger.info(
                    "DocQABridge presigned download ok: size=%s",
                    os.path.getsize(tmp_path),
                )
                return tmp_path
            except Exception as url_exc:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"DocQABridge: both S3 and presigned URL download failed: "
                    f"s3={type(s3_exc).__name__} url={type(url_exc).__name__}"
                ) from url_exc

    # -------------------------------------------------------------- Public

    async def ensure_document_loaded(
        self,
        session_id: str,
        doc_ref: dict,
    ) -> str:
        """Ensure doc is loaded in DocQA; return docqa_session_id.

        If the document (matched by s3_path) is already loaded in this
        RAG session, reuses the existing docqa_session_id. Otherwise
        downloads from S3 and uploads to DocQA, persisting state.
        """
        # [CODE-C1] Guard against None/empty session_id before any mm call.
        if not session_id:
            raise RuntimeError(
                "DocQABridge.ensure_document_loaded requires a session_id; "
                "caller did not supply one"
            )
        sess = self.mm.get_session(session_id)
        if sess is None:
            raise RuntimeError(
                f"DocQABridge: session {session_id!r} not found in memory manager"
            )
        existing_sid = getattr(sess, "docqa_session_id", None)
        s3_path = doc_ref.get("s3_path") or ""
        already_loaded = any(
            (d.get("s3_path") or "") == s3_path
            for d in (getattr(sess, "selected_documents", None) or [])
        )
        if existing_sid and already_loaded:
            logger.info(
                "DocQABridge reuse: rag_session=%s docqa=%s",
                session_id, existing_sid,
            )
            return existing_sid

        tmp_path: Optional[str] = None
        try:
            tmp_path = await self._download_to_temp(doc_ref)
            # DocQA validates by file extension. RAG-sourced names sometimes
            # arrive without ".pdf" — append it when absent so the downstream
            # multipart filename passes DocQA's extension allowlist.
            upload_name = doc_ref.get("file_name") or "document.pdf"
            if not os.path.splitext(upload_name)[1]:
                upload_name = f"{upload_name}.pdf"
            result = await docqa_client.upload_only(
                file_path=tmp_path,
                file_name=upload_name,
                session_id=existing_sid,
            )
            if result.get("error"):
                raise RuntimeError(
                    f"DocQABridge upload failed: {result['error']}"
                )
            new_sid = result.get("session_id")
            if not new_sid:
                raise RuntimeError(
                    "DocQABridge upload returned no session_id"
                )

            # Persist in session
            sess.active_agent = "docqa"
            sess.docqa_session_id = new_sid
            sess.selected_documents.append({
                "s3_path": s3_path,
                "file_name": doc_ref.get("file_name"),
                "docqa_session_id": new_sid,
                "loaded_at": datetime.utcnow().isoformat(),
            })
            # [CODE-H4] Persist mutation so docqa_session_id survives restart.
            # _save_session does blocking I/O (S3 upload or disk write) — run in
            # thread to avoid blocking the event loop.
            try:
                if hasattr(self.mm, "save_session"):
                    await asyncio.to_thread(self.mm.save_session, session_id)
                elif hasattr(self.mm, "_save_session"):
                    await asyncio.to_thread(self.mm._save_session, sess)
                # If neither exists, state is in-memory-only (sandbox OK; Phase 8 adds persistence)
            except Exception as save_exc:
                logger.warning(
                    "DocQABridge: failed to persist session state: %s", save_exc
                )
            logger.info(
                "DocQABridge loaded: rag_session=%s docqa=%s chunks=%s",
                session_id, new_sid, result.get("total_session_chunks"),
            )
            return new_sid
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def ask(self, docqa_session_id: str, query: str) -> dict:
        """Forward follow-up query to DocQA /api/chat. Returns raw DocQA dict."""
        return await docqa_client.query_document(
            session_id=docqa_session_id, query=query,
        )

    def normalize(
        self,
        docqa_response: dict,
        rag_session_id: str,
        selected_document: Optional[dict] = None,
    ) -> dict:
        """Map DocQA response → UnifiedResponse-shaped dict (schema-frozen)."""
        if docqa_response.get("error"):
            return {
                "success": False,
                "answer": f"{DOCQA_FALLBACK_MESSAGE} ({docqa_response['error']})",
                "sources": [],
                "source_documents": [],
                "active_agent": "rag",  # degrade to RAG so user can continue
                "selected_document": None,
                "docqa_session_id": None,
                "session_id": rag_session_id,
                "engine_used": "docqa_fallback",
                "confidence": "low",
                "needs_clarification": False,
                "fallback_used": True,
            }
        return {
            "success": True,
            "answer": docqa_response.get("answer", ""),
            "sources": [],
            "source_documents": docqa_response.get("source_documents", []),
            "active_agent": "docqa",
            "selected_document": selected_document,
            "docqa_session_id": docqa_response.get("session_id"),
            "session_id": rag_session_id,
            "engine_used": "docqa",
            "confidence": "high",
            "groundedness_score": docqa_response.get("groundedness_score"),
            "needs_clarification": False,
            "fallback_used": False,
        }
