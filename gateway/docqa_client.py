"""
HTTP client for the Document QA Agent (port 8006).

Handles document upload + query and follow-up queries.
Used by the orchestrator when search_mode="docqa".
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DOCQA_BASE_URL = os.environ.get("DOCQA_BASE_URL", "http://localhost:8006")
DOCQA_TIMEOUT = int(os.environ.get("DOCQA_TIMEOUT_SECONDS", "120"))


async def upload_and_query(
    file_path: str,
    file_name: str,
    query: str,
    session_id: Optional[str] = None,
) -> dict:
    """Upload a document to DocQA and ask an initial question.

    Calls POST /api/converse with multipart form data.
    Returns the full DocQA response dict.
    """
    url = f"{DOCQA_BASE_URL}/api/converse"
    logger.info(
        "DocQA upload_and_query: file=%s, query=%s, session=%s",
        file_name, query[:50], session_id,
    )

    try:
        async with httpx.AsyncClient(timeout=DOCQA_TIMEOUT) as client:
            with open(file_path, "rb") as f:
                files = {"files": (file_name, f, "application/pdf")}
                data = {"query": query}
                if session_id:
                    data["session_id"] = session_id

                response = await client.post(url, files=files, data=data)
                response.raise_for_status()
                result = response.json()

                logger.info(
                    "DocQA upload_and_query success: session=%s, chunks=%s",
                    result.get("session_id"),
                    result.get("total_session_chunks"),
                )
                return result
    except httpx.TimeoutException:
        logger.error("DocQA upload_and_query timed out after %ds", DOCQA_TIMEOUT)
        return {"error": "Document processing timed out. The document may be too large."}
    except httpx.HTTPStatusError as exc:
        logger.error("DocQA upload_and_query HTTP error: %s", exc.response.status_code)
        return {"error": f"Document QA service returned {exc.response.status_code}"}
    except Exception as exc:
        logger.error("DocQA upload_and_query failed: %s", exc)
        return {"error": f"Failed to connect to Document QA service: {type(exc).__name__}"}


async def query_document(
    session_id: str,
    query: str,
) -> dict:
    """Ask a follow-up question on an existing DocQA session.

    Calls POST /api/chat with JSON body.
    The session must already have uploaded documents.
    """
    url = f"{DOCQA_BASE_URL}/api/chat"
    logger.info("DocQA query_document: session=%s, query=%s", session_id, query[:50])

    try:
        async with httpx.AsyncClient(timeout=DOCQA_TIMEOUT) as client:
            response = await client.post(
                url,
                json={"session_id": session_id, "query": query},
            )
            response.raise_for_status()
            result = response.json()

            logger.info(
                "DocQA query_document success: groundedness=%s",
                result.get("groundedness_score"),
            )
            return result
    except httpx.TimeoutException:
        logger.error("DocQA query_document timed out after %ds", DOCQA_TIMEOUT)
        return {"error": "Document QA query timed out."}
    except httpx.HTTPStatusError as exc:
        logger.error("DocQA query_document HTTP error: %s", exc.response.status_code)
        return {"error": f"Document QA service returned {exc.response.status_code}"}
    except Exception as exc:
        logger.error("DocQA query_document failed: %s", exc)
        return {"error": f"Failed to connect to Document QA service: {type(exc).__name__}"}


async def check_health() -> bool:
    """Check if the DocQA agent is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{DOCQA_BASE_URL}/health")
            return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False
