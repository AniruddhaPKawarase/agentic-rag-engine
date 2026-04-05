"""Tests for shared.config — UnifiedConfig dataclass."""

import os
from unittest.mock import patch

import pytest


class TestConfigLoadsDefaults:
    """test_config_loads_defaults — minimal env produces correct defaults."""

    def test_config_loads_defaults(self) -> None:
        from shared.config import UnifiedConfig, get_config

        # Clear lru_cache so fresh config is loaded
        get_config.cache_clear()

        env = {"OPENAI_API_KEY": "test-key-123"}
        with patch.dict(os.environ, env, clear=False):
            cfg = get_config()

        assert isinstance(cfg, UnifiedConfig)
        # Server defaults
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8001
        assert cfg.log_level == "INFO"
        # OpenAI
        assert cfg.openai_api_key == "test-key-123"
        # AgenticRAG defaults
        assert cfg.agentic_model == "gpt-4.1"
        assert cfg.agentic_model_fallback == "gpt-4.1-mini"
        assert cfg.agentic_max_steps == 8
        assert cfg.agentic_max_context_tokens == 100_000
        assert cfg.agentic_max_request_cost == 0.50
        assert cfg.agentic_daily_budget == 50.0
        assert cfg.agentic_rate_limit == 20
        # Traditional defaults
        assert cfg.traditional_model == "gpt-4o"
        assert cfg.traditional_embedding_model == "text-embedding-3-small"
        assert cfg.web_search_model == "gpt-4.1"
        assert cfg.index_root == "./index"
        assert cfg.max_sessions == 200
        assert cfg.max_tokens_per_session == 10_000
        assert cfg.confidence_threshold == 0.30
        # MongoDB defaults
        assert cfg.mongo_db == "iField"
        # S3 defaults
        assert cfg.storage_backend == "s3"
        assert cfg.s3_bucket_name == "agentic-ai-production"
        assert cfg.s3_region == "us-east-1"
        assert cfg.s3_agent_prefix == "unified-rag-agent"
        # Orchestrator defaults
        assert cfg.fallback_enabled is True
        assert cfg.fallback_timeout_seconds == 30
        assert cfg.faiss_lazy_load is True
        # Auth default
        assert cfg.api_key == ""

        get_config.cache_clear()


class TestConfigReadsEnvOverrides:
    """test_config_reads_env_overrides — env vars override defaults."""

    def test_config_reads_env_overrides(self) -> None:
        from shared.config import get_config

        get_config.cache_clear()

        env = {
            "OPENAI_API_KEY": "override-key",
            "PORT": "9999",
            "HOST": "127.0.0.1",
            "LOG_LEVEL": "DEBUG",
            "AGENTIC_MODEL": "gpt-5",
            "FALLBACK_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = get_config()

        assert cfg.port == 9999
        assert cfg.host == "127.0.0.1"
        assert cfg.log_level == "DEBUG"
        assert cfg.openai_api_key == "override-key"
        assert cfg.agentic_model == "gpt-5"
        assert cfg.fallback_enabled is False

        get_config.cache_clear()


class TestConfigIsImmutable:
    """test_config_is_immutable — setting attribute raises AttributeError."""

    def test_config_is_immutable(self) -> None:
        from shared.config import get_config

        get_config.cache_clear()

        env = {"OPENAI_API_KEY": "immutable-test"}
        with patch.dict(os.environ, env, clear=False):
            cfg = get_config()

        with pytest.raises(AttributeError):
            cfg.port = 1234  # type: ignore[misc]

        with pytest.raises(AttributeError):
            cfg.openai_api_key = "hacked"  # type: ignore[misc]

        get_config.cache_clear()
