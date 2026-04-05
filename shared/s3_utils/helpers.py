"""
Path builder helpers for structured S3 key generation.
All functions return S3 key strings (no leading slash).
"""

import re
from datetime import date, datetime
from typing import Optional


def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use in S3 key paths.
    Removes special characters, replaces spaces with underscores.

    Examples:
        "Granville Hotel" → "Granville_Hotel"
        "HVAC / Mechanical" → "HVAC_Mechanical"
    """
    # Replace spaces and slashes with underscores
    name = re.sub(r"[\s/\\]+", "_", name.strip())
    # Remove anything that isn't alphanumeric, underscore, or hyphen
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# ── Construction Agent Paths ─────────────────────────────────────────────────


def generated_document_key(
    agent_prefix: str,
    project_name: Optional[str],
    project_id: int,
    trade: str,
    filename: str,
) -> str:
    """
    Build S3 key for a generated document.

    Returns:
        e.g. "construction-intelligence-agent/generated_documents/GranvilleHotel_7298/Electrical/scope_electrical_7298_a1b2c3d4.docx"
    """
    if project_name:
        project_folder = f"{sanitize_name(project_name)}_{project_id}"
    else:
        project_folder = f"Project_{project_id}"

    trade_folder = sanitize_name(trade) if trade else "General"

    return f"{agent_prefix}/generated_documents/{project_folder}/{trade_folder}/{filename}"


def conversation_memory_key(agent_prefix: str, session_id: str) -> str:
    """
    Build S3 key for a conversation session export.

    Returns:
        e.g. "construction-intelligence-agent/conversation_memory/session_abc123.json"
    """
    return f"{agent_prefix}/conversation_memory/session_{session_id}.json"


# ── RAG Agent Paths ──────────────────────────────────────────────────────────


def session_key(agent_prefix: str, session_filename: str) -> str:
    """
    Build S3 key for a RAG conversation session file.

    Returns:
        e.g. "rag-agent/conversation_sessions/session_43d5ff4a46e3.json"
    """
    return f"{agent_prefix}/conversation_sessions/{session_filename}"


def faiss_index_key(index_filename: str, faiss_prefix: str = "rag-agent/faiss_indexes") -> str:
    """
    Build S3 key for a FAISS index or metadata file.
    FAISS indexes are stored ONLY under rag-agent/faiss_indexes/ —
    both RAG Agent (reader) and Ingestion Agent (writer) use the same location.

    Args:
        index_filename: e.g. "faiss_index_7166.bin" or "metadata_7201.jsonl"
        faiss_prefix: S3 prefix for FAISS storage (default: "rag-agent/faiss_indexes").
                      Override via S3_FAISS_PREFIX env var for ingestion agent.

    Returns:
        e.g. "rag-agent/faiss_indexes/faiss_index_7166.bin"
    """
    import os
    prefix = os.getenv("S3_FAISS_PREFIX", faiss_prefix)
    return f"{prefix}/{index_filename}"


# ── Ingestion Agent Paths ────────────────────────────────────────────────────


def resume_state_key(agent_prefix: str, project_id: int) -> str:
    """
    Build S3 key for ingestion resume state.

    Returns:
        e.g. "ingestion-agent/resume_state/ingest_state_7166.json"
    """
    return f"{agent_prefix}/resume_state/ingest_state_{project_id}.json"


def ingestion_log_key(agent_prefix: str, project_id: int) -> str:
    """
    Build S3 key for ingestion log file.

    Returns:
        e.g. "ingestion-agent/logs/faiss_indexing_7166.log"
    """
    return f"{agent_prefix}/logs/faiss_indexing_{project_id}.log"


def dedup_db_key(agent_prefix: str) -> str:
    """
    Build S3 key for dedup SQLite database.

    Returns:
        e.g. "ingestion-agent/dedup_db/duplicate_tracker.db"
    """
    return f"{agent_prefix}/dedup_db/duplicate_tracker.db"


# ── Document QA Agent Paths ──────────────────────────────────────────────────


def docqa_session_meta_key(agent_prefix: str, session_id: str) -> str:
    """
    Build S3 key for DocQA session metadata.

    Returns:
        e.g. "document-qa-agent/session_data/abc123/session_meta.json"
    """
    return f"{agent_prefix}/session_data/{session_id}/session_meta.json"


def docqa_session_index_key(agent_prefix: str, session_id: str) -> str:
    """
    Build S3 key for DocQA per-session FAISS index.

    Returns:
        e.g. "document-qa-agent/session_data/abc123/faiss_index.bin"
    """
    return f"{agent_prefix}/session_data/{session_id}/faiss_index.bin"


def docqa_session_chunks_key(agent_prefix: str, session_id: str) -> str:
    """
    Build S3 key for DocQA per-session chunk metadata.

    Returns:
        e.g. "document-qa-agent/session_data/abc123/chunks.jsonl"
    """
    return f"{agent_prefix}/session_data/{session_id}/chunks.jsonl"


# ── Log Paths (all agents) ──────────────────────────────────────────────────


def dated_log_key(agent_prefix: str, log_filename: str, log_date: Optional[date] = None) -> str:
    """
    Build S3 key for a dated log file.

    Args:
        agent_prefix: Agent S3 prefix.
        log_filename: Log file name (e.g. "app.log").
        log_date: Date for folder (defaults to today).

    Returns:
        e.g. "sql-intelligence-agent/api_logs/2026-03-20/app.log"
    """
    if log_date is None:
        log_date = date.today()
    date_str = log_date.isoformat()
    return f"{agent_prefix}/api_logs/{date_str}/{log_filename}"
