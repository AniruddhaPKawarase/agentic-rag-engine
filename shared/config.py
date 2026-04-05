"""
Unified configuration — single frozen dataclass loaded from environment.

All agents (agentic, traditional, gateway) read from the same config.
Defaults match the task specification; override via environment variables.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


def _bool_env(key: str, default: bool) -> bool:
    """Parse a boolean environment variable (true/false/1/0)."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


def _int_env(key: str, default: int) -> int:
    """Parse an integer environment variable."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return int(raw)


def _float_env(key: str, default: float) -> float:
    """Parse a float environment variable."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class UnifiedConfig:
    """Immutable configuration for the Unified RAG Agent system."""

    # --- Server ---
    host: str
    port: int
    log_level: str

    # --- OpenAI ---
    openai_api_key: str

    # --- AgenticRAG ---
    agentic_model: str
    agentic_model_fallback: str
    agentic_max_steps: int
    agentic_max_context_tokens: int
    agentic_max_request_cost: float
    agentic_daily_budget: float
    agentic_rate_limit: int

    # --- Traditional RAG ---
    traditional_model: str
    traditional_embedding_model: str
    web_search_model: str
    index_root: str
    max_sessions: int
    max_tokens_per_session: int
    confidence_threshold: float

    # --- MongoDB ---
    mongodb_uri: str
    mongo_db: str

    # --- S3 ---
    storage_backend: str
    s3_bucket_name: str
    s3_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_agent_prefix: str

    # --- Orchestrator ---
    fallback_enabled: bool
    fallback_timeout_seconds: int
    faiss_lazy_load: bool

    # --- Auth ---
    api_key: str


@lru_cache(maxsize=1)
def get_config() -> UnifiedConfig:
    """
    Load unified configuration from environment variables.

    Cached via lru_cache — returns the same instance for the process lifetime.
    Call ``get_config.cache_clear()`` in tests to force reload.
    """
    return UnifiedConfig(
        # Server
        host=os.getenv("HOST", "0.0.0.0"),
        port=_int_env("PORT", 8001),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        # OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        # AgenticRAG
        agentic_model=os.getenv("AGENTIC_MODEL", "gpt-4.1"),
        agentic_model_fallback=os.getenv("AGENTIC_MODEL_FALLBACK", "gpt-4.1-mini"),
        agentic_max_steps=_int_env("AGENTIC_MAX_STEPS", 8),
        agentic_max_context_tokens=_int_env("AGENTIC_MAX_CONTEXT_TOKENS", 100_000),
        agentic_max_request_cost=_float_env("AGENTIC_MAX_REQUEST_COST", 0.50),
        agentic_daily_budget=_float_env("AGENTIC_DAILY_BUDGET", 50.0),
        agentic_rate_limit=_int_env("AGENTIC_RATE_LIMIT", 20),
        # Traditional RAG
        traditional_model=os.getenv("TRADITIONAL_MODEL", "gpt-4o"),
        traditional_embedding_model=os.getenv(
            "TRADITIONAL_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        web_search_model=os.getenv("WEB_SEARCH_MODEL", "gpt-4.1"),
        index_root=os.getenv("INDEX_ROOT", "./index"),
        max_sessions=_int_env("MAX_SESSIONS", 200),
        max_tokens_per_session=_int_env("MAX_TOKENS_PER_SESSION", 10_000),
        confidence_threshold=_float_env("CONFIDENCE_THRESHOLD", 0.30),
        # MongoDB
        mongodb_uri=os.getenv("MONGODB_URI", ""),
        mongo_db=os.getenv("MONGO_DB", "iField"),
        # S3
        storage_backend=os.getenv("STORAGE_BACKEND", "s3"),
        s3_bucket_name=os.getenv("S3_BUCKET_NAME", "agentic-ai-production"),
        s3_region=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        s3_agent_prefix=os.getenv("S3_AGENT_PREFIX", "unified-rag-agent"),
        # Orchestrator
        fallback_enabled=_bool_env("FALLBACK_ENABLED", True),
        fallback_timeout_seconds=_int_env("FALLBACK_TIMEOUT_SECONDS", 30),
        faiss_lazy_load=_bool_env("FAISS_LAZY_LOAD", True),
        # Auth
        api_key=os.getenv("API_KEY", ""),
    )
