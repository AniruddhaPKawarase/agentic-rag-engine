"""Tests for gateway.router and gateway.app — route registration and app startup."""

import pytest


class TestRouterHasQueryEndpoint:
    """test_router_has_query_endpoint — /query registered on router."""

    def test_router_has_query_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/query" in paths


class TestRouterHasHealthEndpoint:
    """test_router_has_health_endpoint — /health registered on router."""

    def test_router_has_health_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/health" in paths


class TestRouterHasSessionEndpoints:
    """test_router_has_session_endpoints — session CRUD routes registered."""

    def test_router_has_session_endpoints(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/sessions/create" in paths
        assert "/sessions" in paths
        assert "/sessions/{session_id}/stats" in paths
        assert "/sessions/{session_id}/conversation" in paths
        assert "/sessions/{session_id}/update" in paths
        assert "/sessions/{session_id}" in paths
        assert "/sessions/{session_id}/pin-document" in paths


class TestRouterHasStreamEndpoint:
    """test_router_has_stream_endpoint — /query/stream registered."""

    def test_router_has_stream_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/query/stream" in paths


class TestRouterHasWebSearchEndpoint:
    """test_router_has_web_search_endpoint — /web-search registered."""

    def test_router_has_web_search_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/web-search" in paths


class TestRouterHasQuickQueryEndpoint:
    """test_router_has_quick_query_endpoint — /quick-query registered."""

    def test_router_has_quick_query_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/quick-query" in paths


class TestRouterHasConfigEndpoint:
    """test_router_has_config_endpoint — /config registered."""

    def test_router_has_config_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/config" in paths


class TestRouterHasRootEndpoint:
    """test_router_has_root_endpoint — / registered."""

    def test_router_has_root_endpoint(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/" in paths


class TestRouterHasDebugEndpoints:
    """test_router_has_debug_endpoints — /test-retrieve and /debug-pipeline."""

    def test_router_has_debug_endpoints(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert "/test-retrieve" in paths
        assert "/debug-pipeline" in paths


class TestAppLoads:
    """test_app_loads — FastAPI app imports and has correct title."""

    def test_app_loads(self) -> None:
        from gateway.app import app

        assert app.title == "Unified RAG Agent"
        routes = [r.path for r in app.routes]
        assert "/query" in routes


class TestAppHasCorsMiddleware:
    """test_app_has_cors_middleware — CORS middleware is attached."""

    def test_app_has_cors_middleware(self) -> None:
        from gateway.app import app

        # FastAPI stores middleware stack in user_middleware as Middleware objects
        middleware_classes = [
            m.cls.__name__
            for m in getattr(app, "user_middleware", [])
            if hasattr(m, "cls")
        ]
        assert "CORSMiddleware" in middleware_classes


class TestRouterEndpointCount:
    """test_router_endpoint_count — at least 16 routes registered."""

    def test_router_endpoint_count(self) -> None:
        from gateway.router import router

        paths = [r.path for r in router.routes]
        assert len(paths) >= 16
