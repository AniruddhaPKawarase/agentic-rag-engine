"""Tests for shared.session — EngineUsage and UnifiedSessionMeta."""

import pytest


class TestEngineUsageTracking:
    """test_engine_usage_tracking — record multiple engines, verify counts."""

    def test_engine_usage_tracking(self) -> None:
        from shared.session.manager import (
            clear_meta,
            get_meta,
            record_engine_use,
        )

        sid = "test-session-tracking"
        clear_meta(sid)

        record_engine_use(sid, "agentic", cost_usd=0.10)
        record_engine_use(sid, "agentic", cost_usd=0.15)
        record_engine_use(sid, "fallback", cost_usd=0.05)

        meta = get_meta(sid)
        assert meta.engine_usage.agentic == 2
        assert meta.engine_usage.fallback == 1
        assert meta.engine_usage.traditional == 0
        assert meta.last_engine == "fallback"
        assert abs(meta.total_cost_usd - 0.30) < 1e-9

        clear_meta(sid)


class TestGetSessionStatsExtended:
    """test_get_session_stats_extended — record 1 agentic, verify stats dict."""

    def test_get_session_stats_extended(self) -> None:
        from shared.session.manager import (
            clear_meta,
            get_session_stats_extended,
            record_engine_use,
        )

        sid = "test-session-stats"
        clear_meta(sid)

        record_engine_use(sid, "agentic", cost_usd=0.25)

        stats = get_session_stats_extended(sid)
        assert stats["session_id"] == sid
        assert stats["last_engine"] == "agentic"
        assert abs(stats["total_cost_usd"] - 0.25) < 1e-9
        assert stats["engine_usage"]["agentic"] == 1
        assert stats["engine_usage"]["traditional"] == 0
        assert stats["engine_usage"]["fallback"] == 0

        clear_meta(sid)


class TestEngineUsageToDict:
    """test_engine_usage_to_dict — verify dict output from EngineUsage."""

    def test_engine_usage_to_dict(self) -> None:
        from shared.session.models import EngineUsage

        usage = EngineUsage()
        usage.record("agentic")
        usage.record("traditional")
        usage.record("agentic")

        d = usage.to_dict()
        assert d == {"agentic": 2, "traditional": 1, "fallback": 0}
