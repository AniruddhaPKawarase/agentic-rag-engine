"""AgenticRAG Configuration — fail-fast on missing required vars."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────

def _require_env(key: str) -> str:
    """Get a required env var or raise at startup."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable {key} is not set. See .env.example")
    return value


# ── OpenAI ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1")
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))

# ── MongoDB ────────────────────────────────────────────────────────────
MONGODB_URI = _require_env("MONGODB_URI")
MONGO_DB = os.getenv("MONGO_DB", "iField")
VISION_COLLECTION = os.getenv("VISION_COLLECTION", "drawingVision")

# ── API ────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8010"))
API_KEY = os.getenv("API_KEY", "")

# ── Agent ──────────────────────────────────────────────────────────────
AGENT_MODEL_FALLBACK = os.getenv("AGENT_MODEL_FALLBACK", "gpt-4.1-mini")
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "100000"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))

# ── Cost Controls ──────────────────────────────────────────────────────
MAX_REQUEST_COST_USD = float(os.getenv("MAX_REQUEST_COST_USD", "0.50"))
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "50.0"))

# ── Rate Limiting ──────────────────────────────────────────────────────
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

# ── Logging ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
