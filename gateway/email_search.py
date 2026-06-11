"""
Email search via MongoDB Atlas Vector Search over the email_agent3 collection.

Flow
----
1. Embed the user question (text-embedding-3-small, async).
2. $vectorSearch on Atlas with project_id pre-filter - returns top-K matching emails.
3. Expand matched emails to full threads: fetch all emails sharing the same thread_id.
4. LLM synthesis over complete thread context (not just matched fragments).
5. Return {"answer": str, "sources": list[dict], "error": str | None}.

Thread-level retrieval ensures the LLM sees the full conversation arc -
final decisions and latest replies - not just the most semantically similar fragment.

Errors are caught and returned in "error" - never raised to the caller.

Environment variables
---------------------
EMAIL_VECTOR_INDEX      Atlas Search index name   (default: email_agent3_embeddings)
EMAIL_PROJECT_FIELD     field used to filter docs (default: project.id)
EMAIL_SEARCH_TOP_K      threads passed to LLM     (default: 8)
EMAIL_SEARCH_TIMEOUT    per-step timeout seconds  (default: 30)
AGENTIC_MODEL           synthesis model           (default: gpt-4.1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI
from pymongo import MongoClient, ASCENDING

logger = logging.getLogger(__name__)

EMAIL_COLLECTION = "email_agent3"
VECTOR_INDEX: str = os.environ.get("EMAIL_VECTOR_INDEX", "email_agent3_embeddings")
PROJECT_FIELD: str = os.environ.get("EMAIL_PROJECT_FIELD", "project.id")
EMBEDDING_MODEL = "text-embedding-3-small"
SYNTHESIS_MODEL: str = os.environ.get("AGENT_MODEL", "gpt-4.1")
TOP_K: int = int(os.environ.get("EMAIL_SEARCH_TOP_K", "8"))          # sent to LLM
CANDIDATE_K: int = int(os.environ.get("EMAIL_CANDIDATE_K", "20"))    # retrieved before re-ranking
TIMEOUT: int = int(os.environ.get("EMAIL_SEARCH_TIMEOUT", "30"))
RECENCY_WEIGHT: float = float(os.environ.get("EMAIL_RECENCY_WEIGHT", "0.3"))
RECENCY_HALF_LIFE_DAYS: float = float(os.environ.get("EMAIL_RECENCY_HALF_LIFE_DAYS", "30"))

_openai_client: Optional[AsyncOpenAI] = None
_mongo_client: Optional[MongoClient] = None
_mongo_lock = threading.Lock()


def _recency_score(timestamp) -> float:
    """Return 0-1 recency score: 1.0 = today, decays to 0.5 at RECENCY_HALF_LIFE_DAYS."""
    if not timestamp:
        return 0.5
    try:
        now = datetime.now(timezone.utc)
        ts = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days_old = max(0.0, (now - ts).total_seconds() / 86400)
        return 0.5 ** (days_old / RECENCY_HALF_LIFE_DAYS)
    except Exception:
        return 0.5


def _openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _openai_client


def _collection():
    global _mongo_client
    if _mongo_client is None:
        with _mongo_lock:
            if _mongo_client is None:
                from shared.config import get_config
                cfg = get_config()
                _mongo_client = MongoClient(
                    cfg.mongodb_uri,
                    serverSelectionTimeoutMS=10_000,
                    connectTimeoutMS=10_000,
                    socketTimeoutMS=30_000,
                    maxPoolSize=5,
                )
    from shared.config import get_config
    cfg = get_config()
    return _mongo_client[cfg.mongo_db][EMAIL_COLLECTION]


def _vector_search_sync(query_vector: list[float], project_id: int) -> list[dict]:
    """Return top-K matching emails with thread_id for thread expansion."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": CANDIDATE_K * 10,
                "limit": CANDIDATE_K,
                "filter": {PROJECT_FIELD: project_id},
            }
        },
        {
            "$project": {
                "_id": 0,
                "thread_id": 1,
                "subject": 1,
                "timestamp": 1,
                "from": 1,
                "summary": "$ai_analysis.summary",
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(_collection().aggregate(pipeline))


def _fetch_threads_sync(thread_ids: list[str], project_id: int) -> dict[str, list[dict]]:
    """Fetch all emails for the given thread_ids, grouped by thread_id, sorted oldest-first."""
    if not thread_ids:
        return {}
    col = _collection()
    emails = list(col.find(
        {"thread_id": {"$in": thread_ids}, PROJECT_FIELD: project_id},
        {
            "_id": 0,
            "thread_id": 1,
            "subject": 1,
            "from": 1,
            "timestamp": 1,
            "thread_root": 1,
            "thread_depth": 1,
            "ai_analysis": 1,
        },
        sort=[("timestamp", ASCENDING)],
    ))
    threads: dict[str, list[dict]] = {}
    for email in emails:
        tid = email.get("thread_id") or "__no_thread__"
        threads.setdefault(tid, []).append(email)
    return threads


def _format_thread(emails: list[dict]) -> str:
    """Format a thread as a chronological conversation for LLM context."""
    if not emails:
        return ""

    subject = emails[0].get("subject", "")
    senders = list(dict.fromkeys(e.get("from", "") for e in emails if e.get("from")))
    first_date = emails[0].get("timestamp", "")
    last_date = emails[-1].get("timestamp", "")

    lines = [
        f"Subject: {subject}",
        f"Participants: {', '.join(senders[:4])}{'...' if len(senders) > 4 else ''}",
        f"Date range: {first_date} ? {last_date}  ({len(emails)} messages)",
        "",
    ]

    for i, email in enumerate(emails):
        label = "ROOT" if email.get("thread_root") else f"Reply {i}"

        ai = email.get("ai_analysis") or {}   
        summary = (ai.get("summary") or "").strip()
        bullets = ai.get("bullets") or []
        priority = ai.get("priority")  # optional

        bullet_str = "; ".join(str(b).strip() for b in bullets[:3] if b)

        lines.append(
            f"  [{label} | {email.get('from', '')} | {email.get('timestamp', '')}]"
        )

        if summary:
            lines.append(f"  Summary: {summary}")

        if bullet_str:
            lines.append(f"  Points: {bullet_str}")

        if priority is not None:
            lines.append(f"  Priority: {priority}")

        lines.append("")

    return "\n".join(lines)


_SYNTHESIS_PROMPT = """\
You are a project assistant. Answer the user's question using the email threads below.
Each thread is shown in chronological order - pay attention to the latest replies for \
final decisions, resolved issues, and current status.
Be concise and factual. If the threads do not contain enough information, say so briefly.

Email threads:
{threads}

Question: {question}

Answer:"""


async def search_emails(question: str, project_id: str) -> dict:
    """Search project emails via Atlas Vector Search, expand to full threads, synthesise answer."""
    try:
        pid = int(project_id)
    except ValueError:
        return {"answer": "", "sources": [], "error": f"Invalid project_id: {project_id!r}"}

    try:
        # Step 1 - embed question
        emb_response = await asyncio.wait_for(
            _openai().embeddings.create(model=EMBEDDING_MODEL, input=question),
            timeout=TIMEOUT,
        )
        query_vector = emb_response.data[0].embedding

        # Step 2 - vector search: find top-K matching emails
        matches: list[dict] = await asyncio.wait_for(
            asyncio.to_thread(_vector_search_sync, query_vector, pid),
            timeout=TIMEOUT,
        )

        if not matches:
            return {"answer": "No relevant emails found for this project.", "sources": [], "error": None}

        # Re-rank by combined semantic + recency score
        for m in matches:
            sem = m.get("score", 0.0)
            rec = _recency_score(m.get("timestamp"))
            m["combined_score"] = (1 - RECENCY_WEIGHT) * sem + RECENCY_WEIGHT * rec
        matches.sort(key=lambda m: m["combined_score"], reverse=True)
        matches = matches[:TOP_K]

        # Step 3 - expand to full threads (deduplicated by thread_id, preserving match order)
        seen: dict[str, float] = {}
        for m in matches:
            tid = m.get("thread_id") or "__no_thread__"
            if tid not in seen:
                seen[tid] = m.get("combined_score", m.get("score", 0.0))

        thread_ids = list(seen.keys())
        threads: dict[str, list[dict]] = await asyncio.wait_for(
            asyncio.to_thread(_fetch_threads_sync, thread_ids, pid),
            timeout=TIMEOUT,
        )

        # Step 4 - format threads for LLM (highest-scoring thread first)
        sorted_thread_ids = sorted(thread_ids, key=lambda t: seen.get(t, 0.0), reverse=True)
        thread_blocks = "\n\n---\n\n".join(
            f"Thread {i + 1}:\n{_format_thread(threads.get(tid, []))}"
            for i, tid in enumerate(sorted_thread_ids)
            if threads.get(tid)
        )

        if not thread_blocks:
            return {"answer": "No relevant email threads found for this project.", "sources": [], "error": None}

        prompt = _SYNTHESIS_PROMPT.format(threads=thread_blocks, question=question)

        # Step 5 - LLM synthesis
        completion = await asyncio.wait_for(
            _openai().chat.completions.create(
                model=SYNTHESIS_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        answer = completion.choices[0].message.content.strip()

        # Sources: one entry per matched email (with thread context)
        sources = [
            {
                "source_type": "email",
                "subject": m.get("subject", ""),
                "sender": m.get("from", ""),
                "timestamp": str(m.get("timestamp", "")),
                "thread_id": m.get("thread_id", ""),
                "thread_size": len(threads.get(m.get("thread_id", ""), [])),
                "score": round(m.get("combined_score", m.get("score", 0.0)), 4),
                "vector_score": round(m.get("score", 0.0), 4),
            }
            for m in matches
        ]

        logger.info(
            "Email search: project=%s matched=%d threads=%d answer_len=%d",
            project_id, len(matches), len(thread_ids), len(answer),
        )
        return {"answer": answer, "sources": sources, "error": None}

    except asyncio.TimeoutError:
        logger.warning("Email search timed out (project=%s)", project_id)
        return {"answer": "", "sources": [], "error": "Email search timed out"}
    except Exception as exc:
        logger.warning("Email search failed (project=%s): %s", project_id, exc)
        return {"answer": "", "sources": [], "error": f"Email search error: {type(exc).__name__}"}
