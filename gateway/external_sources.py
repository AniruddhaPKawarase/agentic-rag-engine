"""
HTTP clients for external agent services (Meeting, RFI).

Normalises remote responses to:
  {"answer": str, "sources": list[dict], "error": str | None}

Errors are swallowed - a failed external source never crashes the main query.

Environment variables
---------------------
MEETING_AGENT_URL          base URL  (default: http://localhost:8007)
RFI_AGENT_URL              base URL  (default: https://ai5.ifieldsmart.com/sql)
EXTERNAL_AGENT_TIMEOUT     seconds   (default: 30)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MEETING_AGENT_URL: str = os.environ.get("MEETING_AGENT_URL", "https://ai5.ifieldsmart.com/meeting")
RFI_AGENT_URL: str = os.environ.get("RFI_AGENT_URL", "https://ai5.ifieldsmart.com/sql")
EXTERNAL_TIMEOUT: int = int(os.environ.get("EXTERNAL_AGENT_TIMEOUT", "30"))


def _empty(error: str) -> dict:
    return {"answer": "", "sources": [], "error": error}


async def search_meetings(question: str, project_id: str, top_k: int = 5) -> dict:
    """POST /ask to the Meeting Agent.

    Request:  {"question": str, "project_id": str, "top_k": int}
    Response: {"answer": str, "sources": [...], "cached": bool}
    """
    url = f"{MEETING_AGENT_URL}/ask"
    payload: dict[str, Any] = {
        "question": question,
        "project_id": project_id,
        "top_k": top_k,
    }
    logger.info("Meeting agent -> project=%s q=%.80s", project_id, question)
    try:
        async with httpx.AsyncClient(timeout=EXTERNAL_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "error": None,
            }
    except httpx.TimeoutException:
        logger.warning("Meeting agent timed out after %ds", EXTERNAL_TIMEOUT)
        return _empty("Meeting agent timed out")
    except httpx.HTTPStatusError as exc:
        logger.warning("Meeting agent HTTP %s", exc.response.status_code)
        return _empty(f"Meeting agent returned {exc.response.status_code}")
    except Exception as exc:
        logger.warning("Meeting agent error: %s", exc)
        return _empty(f"Meeting agent unavailable: {type(exc).__name__}")


async def search_rfis(question: str, project_id: str, top_k: int = 5) -> dict:
    """POST /api/rfi/query to the RFI Agent.

    Request:  {"query": str, "project_id": str, "top_k": int}
    Response: {"success": bool, "data": {"summary": str, "list": [...], "count": int}, ...}
    """
    url = f"{RFI_AGENT_URL}/api/rfi/query"
    payload: dict[str, Any] = {
        "query": question,
        "project_id": project_id,
        "top_k": top_k,
    }
    logger.info("RFI agent -> project=%s q=%.80s", project_id, question)
    try:
        async with httpx.AsyncClient(timeout=EXTERNAL_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            rfi_data = data.get("data") or {}
            return {
                "answer": rfi_data.get("summary", ""),
                "sources": rfi_data.get("list", []),
                "error": data.get("error"),
            }
    except httpx.TimeoutException:
        logger.warning("RFI agent timed out after %ds", EXTERNAL_TIMEOUT)
        return _empty("RFI agent timed out")
    except httpx.HTTPStatusError as exc:
        logger.warning("RFI agent HTTP %s", exc.response.status_code)
        return _empty(f"RFI agent returned {exc.response.status_code}")
    except Exception as exc:
        logger.warning("RFI agent error: %s", exc)
        return _empty(f"RFI agent unavailable: {type(exc).__name__}")
