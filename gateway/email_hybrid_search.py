"""
Hybrid email search: Atlas Vector Search (semantic) + Atlas full-text Search (BM25)
combined via Reciprocal Rank Fusion (RRF), with date-aware boost and filtering.

Pipeline
--------
1.  Date extraction    - regex, zero latency, no LLM
2.  Embed query        - text-embedding-3-small
3.  Parallel retrieval - vector search + BM25 text search (subject/summary/bullets/actions/body)
4.  RRF merge          - Reciprocal Rank Fusion (k=60)
5.  Date boost         - subject match x1.5, summary/body match x1.2 (soft signal)
6.  Date filter        - hard cut: keep only date-matched emails (if any found)
7.  Recency blend      - combined = 0.7 x boosted_rrf_norm + 0.3 x recency
8.  Cut to TOP_K       - top 8 by combined score
9.  Thread dedup+expand- best email per thread -> fetch full thread from MongoDB
10. LLM synthesis      - gpt-4.1

Why hybrid?
-----------
Pure vector search misses queries where the keyword exists in the email but the
embedding space is far from the query vector (e.g. "resubmit submittals by 5/27").
BM25 catches exact/fuzzy keyword hits that vector misses, including inside the
email body which may contain specifics not surfaced in subject/summary/bullets.
RRF merges both ranked lists without needing normalised scores.

Why RRF normalisation before recency blend?
-------------------------------------------
RRF scores are in the 0.01-0.02 range. Without normalisation, blending with
recency (0-1) would let recency dominate. We normalise RRF to 0-1 (divide by max)
before computing combined = 0.7*rrf_norm + 0.3*recency.

Atlas Search indexes required
-----------------------------
Vector: email_agent3_embeddings  (existing)

Text:   email_agent3_text - create or UPDATE in Atlas UI with this definition:
  {
    "mappings": {
      "dynamic": false,
      "fields": {
        "subject":               {"type": "string"},
        "body":                  {"type": "string"},
        "ai_analysis.summary":   {"type": "string"},
        "ai_analysis.bullets":   {"type": "string"},
        "ai_analysis.actions":   {"type": "string"},
        "project.id":            {"type": "number"}
      }
    }
  }

Environment variables
---------------------
EMAIL_VECTOR_INDEX           Atlas vector index name   (default: email_agent3_embeddings)
EMAIL_TEXT_INDEX             Atlas text index name     (default: email_agent3_text)
EMAIL_PROJECT_FIELD          project filter field      (default: project.id)
EMAIL_SEARCH_TOP_K           threads passed to LLM     (default: 8)
EMAIL_CANDIDATE_K            candidates per pipeline   (default: 20)
EMAIL_SEARCH_TIMEOUT         per-step timeout seconds  (default: 30)
EMAIL_RECENCY_WEIGHT         0-1 recency blend weight  (default: 0.3)
EMAIL_RECENCY_HALF_LIFE_DAYS recency half-life in days (default: 30)
RRF_K                        RRF constant k            (default: 60)
AGENT_MODEL                  synthesis model           (default: gpt-4.1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from datetime import date as date_type, datetime, timezone
from typing import Optional
from datetime import timedelta

from openai import AsyncOpenAI
from pymongo import MongoClient, ASCENDING

logger = logging.getLogger(__name__)

EMAIL_COLLECTION = "email_agent3"
VECTOR_INDEX: str = os.environ.get("EMAIL_VECTOR_INDEX", "email_agent3_embeddings")
TEXT_INDEX: str = os.environ.get("EMAIL_TEXT_INDEX", "email_agent3_text")
PROJECT_FIELD: str = os.environ.get("EMAIL_PROJECT_FIELD", "project.id")
EMBEDDING_MODEL = "text-embedding-3-small"
SYNTHESIS_MODEL: str = os.environ.get("AGENT_MODEL", "gpt-4.1")
TOP_K: int = int(os.environ.get("EMAIL_SEARCH_TOP_K", "8"))
CANDIDATE_K: int = int(os.environ.get("EMAIL_CANDIDATE_K", "20"))
TIMEOUT: int = int(os.environ.get("EMAIL_SEARCH_TIMEOUT", "30"))
RECENCY_WEIGHT: float = float(os.environ.get("EMAIL_RECENCY_WEIGHT", "0.3"))
RECENCY_HALF_LIFE_DAYS: float = float(os.environ.get("EMAIL_RECENCY_HALF_LIFE_DAYS", "30"))
RRF_K: int = int(os.environ.get("RRF_K", "60"))
DATE_BOOST_SUBJECT: float = 1.5
DATE_BOOST_CONTENT: float = 1.2   # summary, bullets, or body

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_openai_client: Optional[AsyncOpenAI] = None
_mongo_client: Optional[MongoClient] = None
_mongo_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def _extract_time_window(query: str) -> Optional[tuple[date_type, date_type]]:
    """
    Extract a relative date range from query. Returns (start, end) inclusive.
    Supports: "last 7 days", "last 14 days", "last 30 days", "last week",
              "last month", "yesterday", "today", "past N days"
    """
    today = date_type.today()
    q = query.lower()

    # "last N days" / "past N days" / "last N day"
    m = re.search(r"(?:last|past)\s+(\d+)\s+days?", q)
    if m:
        return (today - timedelta(days=int(m.group(1))), today)

    # "last week" ? 7 days
    if re.search(r"\blast\s+week\b", q):
        return (today - timedelta(days=7), today)

    # "last 2 weeks" / "last two weeks"
    m = re.search(r"\blast\s+(\d+)\s+weeks?\b", q)
    if m:
        return (today - timedelta(weeks=int(m.group(1))), today)

    # "last month" ? 30 days
    if re.search(r"\blast\s+month\b", q):
        return (today - timedelta(days=30), today)

    # "yesterday"
    if "yesterday" in q:
        yesterday = today - timedelta(days=1)
        return (yesterday, yesterday)

    # "today"
    if re.search(r"\btoday\b", q):
        return (today, today)

    return None


def _extract_query_date(query: str) -> Optional[date_type]:
    """
    Extract a date from query string. Zero latency, no LLM.
    Supports: 5/26  05/26  5/26/2026  5-26  05-26  5-26-2026
              2026-05-26  May 26  26 May  May 26th
    """
    q = query.strip()

    # M/D, M/D/YYYY, MM/DD/YYYY
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", q)
    if m:
        month, day, year_str = m.groups()
        year = int(year_str) if year_str else datetime.now().year
        if year < 100:
            year += 2000
        try:
            return date_type(year, int(month), int(day))
        except ValueError:
            pass

    # ISO: 2026-05-26 (must check before M-D to avoid partial match)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
    if m:
        year, month, day = m.groups()
        try:
            return date_type(int(year), int(month), int(day))
        except ValueError:
            pass

    # M-D, M-D-YYYY, MM-DD-YYYY (hyphen separator e.g. "5-26", "05-26")
    m = re.search(r"\b(\d{1,2})-(\d{1,2})(?:-(\d{2,4}))?\b", q)
    if m:
        month, day, year_str = m.groups()
        year = int(year_str) if year_str else datetime.now().year
        if year < 100:
            year += 2000
        try:
            return date_type(year, int(month), int(day))
        except ValueError:
            pass

    # "May 26", "May 26th"
    month_pat = "|".join(_MONTH_NAMES.keys())
    m = re.search(rf"\b({month_pat})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", q.lower())
    if m:
        month_str, day = m.groups()
        try:
            return date_type(datetime.now().year, _MONTH_NAMES[month_str], int(day))
        except ValueError:
            pass

    # "26 May", "26th May"
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pat})\b", q.lower())
    if m:
        day, month_str = m.groups()
        try:
            return date_type(datetime.now().year, _MONTH_NAMES[month_str], int(day))
        except ValueError:
            pass

    return None


def _date_variants(d: date_type) -> list[str]:
    """All string representations of a date used for matching."""
    return [
        f"{d.month}/{d.day}",
        f"{d.month:02d}/{d.day:02d}",
        f"{d.strftime('%B')} {d.day}",  # May 26
        str(d),                          # 2026-05-26
    ]


def _date_match_boost(doc: dict, query_date: date_type) -> float:
    """
    Score multiplier for date relevance:
      1.5 - date in subject (authoritative: 'BIM Follow-up 5/26')
      1.2 - date in summary, bullets, or body (contextual: '...due by 5/26...')
      1.0 - no match
    """
    variants = [v.lower() for v in _date_variants(query_date)]

    if any(v in (doc.get("subject") or "").lower() for v in variants):
        return DATE_BOOST_SUBJECT

    content = " ".join([
        (doc.get("summary") or ""),
        (doc.get("body_snippet") or ""),
    ]).lower()
    if any(v in content for v in variants):
        return DATE_BOOST_CONTENT

    return 1.0


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------

def _recency_score(timestamp) -> float:
    """0-1 score: 1.0 = today, decays to 0.5 at RECENCY_HALF_LIFE_DAYS."""
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


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _vector_search_sync(query_vector: list[float], project_id: int) -> list[dict]:
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
                "_id": 1,
                "thread_id": 1,
                "subject": 1,
                "timestamp": 1,
                "from": 1,
                "summary": "$ai_analysis.summary",
                "body_snippet": {"$substr": [{"$ifNull": ["$body", ""]}, 0, 500]},
                "vector_score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(_collection().aggregate(pipeline))


def _text_search_sync(query: str, project_id: int) -> list[dict]:
    """BM25 full-text search across subject, body, summary, bullets, actions."""
    try:
        pipeline = [
            {
                "$search": {
                    "index": TEXT_INDEX,
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "query": query,
                                    "path": [
                                        "subject",
                                        "body",
                                        "ai_analysis.summary",
                                        "ai_analysis.bullets",
                                        "ai_analysis.actions",
                                    ],
                                    "fuzzy": {"maxEdits": 1},
                                }
                            }
                        ],
                        "filter": [
                            {"equals": {"path": PROJECT_FIELD, "value": project_id}}
                        ],
                    },
                }
            },
            {"$limit": CANDIDATE_K},
            {
                "$project": {
                    "_id": 1,
                    "thread_id": 1,
                    "subject": 1,
                    "timestamp": 1,
                    "from": 1,
                    "summary": "$ai_analysis.summary",
                    "body_snippet": {"$substr": [{"$ifNull": ["$body", ""]}, 0, 500]},
                    "text_score": {"$meta": "searchScore"},
                }
            },
        ]
        return list(_collection().aggregate(pipeline))
    except Exception as exc:
        logger.warning("BM25 text search failed (index missing?): %s", exc)
        return []


def _sender_search_sync(sender_name: str, project_id: int, limit: int = 15) -> list[dict]:
    """Search emails by sender name, sorted newest first.

    Used when query contains 'from [Name]' - guarantees that person's emails
    are in the candidate pool even if vector/BM25 miss them semantically.
    """
    try:
        results = list(_collection().find(
            {
                PROJECT_FIELD: project_id,
                "from": {"$regex": sender_name, "$options": "i"},
            },
            {
                "_id": 1,
                "thread_id": 1,
                "subject": 1,
                "timestamp": 1,
                "from": 1,
                "ai_analysis": 1,
                "body": 1,
            },
            sort=[("timestamp", -1)],
            limit=limit,
        ))
        normalized = []
        for doc in results:
            ai = doc.get("ai_analysis") or {}
            normalized.append({
                "_id": doc["_id"],
                "thread_id": doc.get("thread_id"),
                "subject": doc.get("subject", ""),
                "timestamp": doc.get("timestamp"),
                "from": doc.get("from", ""),
                "summary": ai.get("summary", ""),
                "body_snippet": (doc.get("body") or "")[:500],
                "rrf_score": 0.0,
                "vector_score": 0.0,
            })
        return normalized
    except Exception as exc:
        logger.warning("Sender search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(vector_results: list[dict], text_results: list[dict]) -> list[dict]:
    """Merge two ranked lists via Reciprocal Rank Fusion: score = sum(1/(k+rank+1))."""
    scores: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        if doc_id not in scores:
            scores[doc_id] = {"doc": doc, "rrf": 0.0}
        scores[doc_id]["rrf"] += 1.0 / (RRF_K + rank + 1)

    for rank, doc in enumerate(text_results):
        doc_id = str(doc["_id"])
        if doc_id not in scores:
            scores[doc_id] = {"doc": doc, "rrf": 0.0}
        scores[doc_id]["rrf"] += 1.0 / (RRF_K + rank + 1)

    merged = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    result = []
    for item in merged[:CANDIDATE_K]:
        doc = item["doc"]
        doc["rrf_score"] = item["rrf"]
        result.append(doc)
    return result


# ---------------------------------------------------------------------------
# Thread fetch + format
# ---------------------------------------------------------------------------

def _fetch_threads_sync(thread_ids: list[str], project_id: int) -> dict[str, list[dict]]:
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
            "body": 1,
        },
        sort=[("timestamp", ASCENDING)],
    ))
    threads: dict[str, list[dict]] = {}
    for email in emails:
        tid = email.get("thread_id") or "__no_thread__"
        threads.setdefault(tid, []).append(email)
    return threads


def _format_thread(emails: list[dict]) -> str:
    if not emails:
        return ""
    subject = emails[0].get("subject", "")
    senders = list(dict.fromkeys(e.get("from", "") for e in emails if e.get("from")))
    first_date = emails[0].get("timestamp", "")
    last_date = emails[-1].get("timestamp", "")

    lines = [
        f"Subject: {subject}",
        f"Participants: {', '.join(senders[:4])}{'...' if len(senders) > 4 else ''}",
        f"Date range: {first_date} -> {last_date}  ({len(emails)} messages)",
        "",
    ]
    for i, email in enumerate(emails):
        label = "ROOT" if email.get("thread_root") else f"Reply {i}"
        ai = email.get("ai_analysis") or {}
        summary = (ai.get("summary") or "").strip()
        bullets = ai.get("bullets") or []
        actions = ai.get("actions") or []
        priority = ai.get("priority")
        body = (email.get("body") or "").strip()[:1500]
        bullet_str = "\n    ".join(f"- {str(b).strip()}" for b in bullets if b)
        action_str = "\n    ".join(f"- {str(a).strip()}" for a in actions if a)
        lines.append(f"  [{label} | {email.get('from', '')} | {email.get('timestamp', '')}]")
        if summary:
            lines.append(f"  Summary: {summary}")
        if bullet_str:
            lines.append(f"  Points:\n    {bullet_str}")
        if action_str:
            lines.append(f"  Actions:\n    {action_str}")
        if priority is not None:
            lines.append(f"  Priority: {priority}")
        if body:
            lines.append(f"  Body:\n    {body}")
        lines.append("")
    return "\n".join(lines)


_SYNTHESIS_PROMPT = """\
You are a project assistant. Answer the user's question using the email threads below.
Each thread is shown in chronological order - pay attention to the latest replies for \
final decisions, resolved issues, and current status.
Be concise and factual. If the threads do not contain enough information, say so briefly.

{history_context}Email threads:
{threads}

Question: {question}

Answer:"""

_DETAIL_SYNTHESIS_PROMPT = """\
You are a project assistant. The user wants detailed information from the email threads below.
Extract and present ALL relevant information - do NOT summarize or omit details.

Rules:
- List every point, request, decision, and specification mentioned.
- Include names, dates, measurements, system numbers, and technical specs exactly as written.
- Preserve the sender and timestamp context for each point.
- Use bullet points per email/message. Be exhaustive, not brief.

{history_context}Email threads:
{threads}

Question: {question}

Answer:"""

_DATE_SYNTHESIS_PROMPT = """\
You are a project assistant. The user is asking about a specific date. \
Extract precise details from the email threads below - do NOT summarize or paraphrase.

Rules:
- List every deliverable, action item, and deadline mentioned for the requested date.
- Include specific deadlines per item/level.
- Include assignees or responsible parties where mentioned.
- Include RFI numbers, coordination issue numbers, and level-by-level breakdown.
- If an item is already completed, note it as done.
- Use bullet points. Be exhaustive, not brief.

{history_context}Email threads:
{threads}

Question: {question}

Answer:"""

_PERIOD_SYNTHESIS_PROMPT = """\
You are a project assistant. Summarize all email activity from the requested time period.

Group by topic or thread. For each group include:
- What was discussed or decided
- Action items and deadlines
- Responsible parties
- Outstanding issues

Be comprehensive -- cover every thread. Use headers per topic.

{history_context}Email threads:
{threads}

Question: {question}

Summary:"""

_QUERY_EXPAND_PROMPT = """\
You are a search query optimizer for a construction project email assistant.

Rewrite the user's question into a keyword-rich search query that will retrieve \
the most relevant emails from a construction project database.

Rules:
- Expand vague terms into specific construction keywords:
  "meeting minutes" -> "follow-up action items coordination decisions"
  "deliverables" -> "action items submittals RFIs deadlines assignments"
  "issues" -> "conflicts clashes RFIs coordination problems"
  "status" -> "progress update completion pending outstanding"
  "responsibility" -> "assigned responsible owner trade contractor"
- Keep all dates (5/26, May 26), names, RFI numbers, level names exactly as-is.
- If conversation history is provided, resolve pronouns and implicit references \
(he/she/they/it/this/that) using context from history.
- Return ONLY the optimized query -- no explanation, no quotes.

{history_section}Question: {question}

Optimized search query:"""


async def _expand_query(question: str, history: list | None = None) -> str:
    """Expand query intent and optionally resolve follow-up references using history."""
    try:
        history_section = ""
        if history:
            history_text = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {(m.get('content') or '')[:400]}"
                for m in history[-6:]
            )
            history_section = f"Conversation history:\n{history_text}\n\n"
        prompt = _QUERY_EXPAND_PROMPT.format(
            history_section=history_section,
            question=question,
        )
        response = await asyncio.wait_for(
            _openai().chat.completions.create(
                model=SYNTHESIS_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            ),
            timeout=10,
        )
        expanded = (response.choices[0].message.content or "").strip()
        return expanded if expanded else question
    except Exception as exc:
        logger.warning("Query expansion failed: %s - using original question", exc)
        return question


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def search_emails(question: str, project_id: str, session_id: str | None = None) -> dict:
    """
    Hybrid search: vector + BM25 -> RRF -> date boost -> date filter
    -> recency blend -> thread expand -> LLM synthesis.

    If session_id is provided, conversation history is loaded from MemoryManager
    for query rewriting. The caller (router) is responsible for saving Q&A to session.
    """
    try:
        pid = int(project_id)
    except ValueError:
        return {"answer": "", "sources": [], "error": f"Invalid project_id: {project_id!r}"}

    try:
        # Step 0 - load conversation history from session for query rewriting
        conversation_history: list = []
        if session_id:
            try:
                from traditional.memory_manager import get_memory_manager
                _mm = get_memory_manager()
                _session = _mm.get_session(session_id)
                if _session:
                    conversation_history = [
                        m for m in _session.get_formatted_messages(include_system=False)
                        if m.get("role") in ("user", "assistant")
                    ]
                    logger.debug("Loaded %d history messages from session %s", len(conversation_history), session_id)
            except Exception as _exc:
                logger.debug("Session history load skipped: %s", _exc)

        # Step 0b - expand query intent + resolve follow-up references (always runs)
        retrieval_query = await _expand_query(question, history=conversation_history or None)
        if retrieval_query != question:
            logger.info("Query expanded: %r -> %r", question, retrieval_query)

        # Step 0c - detect sender name for "from [Name]" queries (e.g. "email from person")
        _sender_match = re.search(r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", question)
        sender_name: str | None = _sender_match.group(1) if _sender_match else None

        # Step 1 - time window (relative dates) takes priority over single-date extraction
        time_window: Optional[tuple[date_type, date_type]] = _extract_time_window(question)
        if time_window:
            logger.info("Time window detected: %s -> %s", time_window[0], time_window[1])

        # Single-date extraction (only if no time window)
        query_date = None if time_window else (
            _extract_query_date(question) or _extract_query_date(retrieval_query)
        )
        if query_date:
            logger.info("Date detected in query: %s", query_date)

        # Step 2 - embed rewritten query
        emb_response = await asyncio.wait_for(
            _openai().embeddings.create(model=EMBEDDING_MODEL, input=retrieval_query),
            timeout=TIMEOUT,
        )
        query_vector = emb_response.data[0].embedding

        # Step 3 - vector + BM25 + optional sender search in parallel
        gather_tasks = [
            asyncio.to_thread(_vector_search_sync, query_vector, pid),
            asyncio.to_thread(_text_search_sync, retrieval_query, pid),
        ]
        if sender_name:
            gather_tasks.append(asyncio.to_thread(_sender_search_sync, sender_name, pid))

        gather_results = await asyncio.wait_for(
            asyncio.gather(*gather_tasks),
            timeout=TIMEOUT,
        )
        vector_results = gather_results[0]
        text_results = gather_results[1]
        sender_results = gather_results[2] if sender_name else []
        logger.info(
            "Hybrid retrieval: project=%s vector=%d text=%d sender=%d",
            project_id, len(vector_results), len(text_results), len(sender_results),
        )

        # Step 4 - RRF merge
        matches = _rrf_merge(vector_results, text_results)

        # Inject sender results not already in RRF pool (guarantees sender's emails are present)
        if sender_results:
            existing_ids = {str(m["_id"]) for m in matches}
            injected = 0
            for doc in sender_results:
                if str(doc["_id"]) not in existing_ids:
                    doc["rrf_score"] = 0.005  # baseline - recency will do the ranking
                    matches.append(doc)
                    injected += 1
            if injected:
                logger.info("Injected %d sender emails into candidate pool", injected)

        if not matches:
            return {"answer": "No relevant emails found for this project.", "sources": [], "error": None}

        # Step 4b - time window filter (timestamp-based, for "last N days" queries)
        if time_window:
            tw_start, tw_end = time_window
            start_dt = datetime(tw_start.year, tw_start.month, tw_start.day, tzinfo=timezone.utc)
            end_dt = datetime(tw_end.year, tw_end.month, tw_end.day, 23, 59, 59, tzinfo=timezone.utc)
            tw_matched = []
            for m in matches:
                ts = m.get("timestamp")
                if not ts:
                    continue
                try:
                    ts_dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    if start_dt <= ts_dt <= end_dt:
                        tw_matched.append(m)
                except Exception:
                    pass
            if tw_matched:
                logger.info("Time window filter: kept %d/%d emails", len(tw_matched), len(matches))
                matches = tw_matched
            else:
                logger.info("Time window filter: no matches, using full results")
            for m in matches:
                m["date_match"] = False

        # Step 5 - date boost (soft signal applied to rrf_score before normalisation)
        if query_date:
            for m in matches:
                boost = _date_match_boost(m, query_date)
                m["rrf_score"] *= boost
                m["date_match"] = boost >= DATE_BOOST_SUBJECT  # subject match only
        else:
            for m in matches:
                m["date_match"] = False

        # Step 6 - date filter (hard constraint: keep only date-matched if any exist)
        date_filter_applied = False
        if query_date:
            date_matched = [m for m in matches if m.get("date_match")]
            if date_matched:
                logger.info(
                    "Date filter applied: %s - kept %d/%d",
                    query_date, len(date_matched), len(matches),
                )
                matches = date_matched
                date_filter_applied = True
            else:
                logger.info(
                    "Date %s detected - no subject/summary/body matches, using full results",
                    query_date,
                )

        # Step 7 - recency blend: normalise boosted RRF to 0-1 then blend with recency
        # Recency-intent queries ("latest", "last", "most recent") get 0.9 recency weight
        _recency_keywords = {"latest", "last", "most recent", "recent", "newest", "new"}
        recency_intent = (
            any(kw in question.lower() for kw in _recency_keywords)
            or (bool(re.search(r"\bfrom\s+[A-Z][a-z]+", question)) and not query_date)
        )
        effective_recency_weight = 0.9 if recency_intent else RECENCY_WEIGHT
        if recency_intent:
            logger.info("Recency intent detected (explicit kw or 'from Person') - weight=0.9")

        max_rrf = max((m["rrf_score"] for m in matches), default=1.0) or 1.0
        for m in matches:
            rrf_norm = m["rrf_score"] / max_rrf
            rec = _recency_score(m.get("timestamp"))
            m["combined_score"] = (1 - effective_recency_weight) * rrf_norm + effective_recency_weight * rec

        matches.sort(key=lambda m: m["combined_score"], reverse=True)

        # Step 8 - cut to TOP_K
        matches = matches[:TOP_K]

        # Step 9 - thread dedup (best combined_score per thread_id)
        seen: dict[str, float] = {}
        for m in matches:
            tid = m.get("thread_id") or "__no_thread__"
            if tid not in seen:
                seen[tid] = m["combined_score"]

        thread_ids = list(seen.keys())

        # Thread expansion - fetch ALL emails per thread
        threads = await asyncio.wait_for(
            asyncio.to_thread(_fetch_threads_sync, thread_ids, pid),
            timeout=TIMEOUT,
        )

        # Step 10 - format threads for LLM (highest-scoring first)
        sorted_thread_ids = sorted(thread_ids, key=lambda t: seen.get(t, 0.0), reverse=True)
        thread_blocks = "\n\n---\n\n".join(
            f"Thread {i + 1}:\n{_format_thread(threads.get(tid, []))}"
            for i, tid in enumerate(sorted_thread_ids)
            if threads.get(tid)
        )

        if not thread_blocks:
            return {"answer": "No relevant email threads found for this project.", "sources": [], "error": None}

        # Build history context prefix for synthesis (last 4 turns, truncated)
        history_context = ""
        if conversation_history:
            recent = conversation_history[-8:]
            lines = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {(m.get('content') or '')[:300]}"
                for m in recent
            )
            history_context = f"Prior conversation:\n{lines}\n\n"

        _detail_keywords = {"detail", "details", "detailed", "full", "all", "everything", "elaborate", "explain"}
        detail_intent = any(kw in question.lower().split() for kw in _detail_keywords)
        if time_window:
            template = _PERIOD_SYNTHESIS_PROMPT
        elif date_filter_applied:
            template = _DATE_SYNTHESIS_PROMPT
        elif detail_intent or sender_name:
            template = _DETAIL_SYNTHESIS_PROMPT
        else:
            template = _SYNTHESIS_PROMPT
        prompt = template.format(
            threads=thread_blocks,
            question=question,
            history_context=history_context,
        )

        # Period/detail/date queries get more tokens
        max_tokens = 2000 if (time_window or date_filter_applied or detail_intent or sender_name) else 1000
        completion = await asyncio.wait_for(
            _openai().chat.completions.create(
                model=SYNTHESIS_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        answer = completion.choices[0].message.content.strip()

        sources = [
            {
                "source_type": "email",
                "subject": m.get("subject", ""),
                "sender": m.get("from", ""),
                "timestamp": str(m.get("timestamp", "")),
                "thread_id": m.get("thread_id", ""),
                "thread_size": len(threads.get(m.get("thread_id", ""), [])),
                "score": round(m["combined_score"], 4),
                "rrf_score": round(m.get("rrf_score", 0.0), 6),
                "vector_score": round(m.get("vector_score", 0.0), 4),
                "date_match": m.get("date_match", False),
            }
            for m in matches
        ]

        logger.info(
            "Hybrid search done: project=%s matched=%d threads=%d date_filter=%s answer_len=%d",
            project_id, len(matches), len(thread_ids), date_filter_applied, len(answer),
        )
        return {
            "answer": answer,
            "sources": sources,
            "error": None,
            "date_filter_applied": date_filter_applied,
            "query_date": str(query_date) if query_date else None,
            "time_window": f"{time_window[0]} to {time_window[1]}" if time_window else None,
        }

    except asyncio.TimeoutError:
        logger.warning("Hybrid email search timed out (project=%s)", project_id)
        return {"answer": "", "sources": [], "error": "Email search timed out"}
    except Exception as exc:
        logger.warning("Hybrid email search failed (project=%s): %s", project_id, exc)
        return {"answer": "", "sources": [], "error": f"Email search error: {type(exc).__name__}"}
