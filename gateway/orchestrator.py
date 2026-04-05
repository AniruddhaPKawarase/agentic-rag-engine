"""
Gateway Orchestrator — Agentic-first with Traditional RAG fallback.

Runs AgenticRAG first, evaluates the result quality, and falls back
to Traditional RAG (FAISS) if the agentic answer is insufficient.

All imports from agentic/ and traditional/ are lazy (inside methods)
to prevent import errors if one engine is misconfigured.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure fallback decision function
# ---------------------------------------------------------------------------

def _should_fallback(result: Any) -> bool:
    """Determine whether the orchestrator should fall back to traditional RAG.

    Returns True when:
    - result is None
    - answer is empty or shorter than 20 characters
    - confidence is "low"
    - sources list is empty
    - needs_escalation flag is set
    """
    if result is None:
        return True

    answer: str = getattr(result, "answer", "") or ""
    if len(answer.strip()) < 20:
        return True

    confidence: str = getattr(result, "confidence", "low") or "low"
    if confidence == "low":
        return True

    sources: list = getattr(result, "sources", []) or []
    if len(sources) == 0:
        return True

    needs_escalation: bool = getattr(result, "needs_escalation", False)
    if needs_escalation:
        return True

    return False


# ---------------------------------------------------------------------------
# Agentic Engine — wraps agentic.core.agent
# ---------------------------------------------------------------------------

class AgenticEngine:
    """Wrapper around the agentic RAG engine with lazy initialization."""

    def __init__(self) -> None:
        self._initialized: bool = False

    def ensure_initialized(self) -> None:
        """Import and initialize the agentic engine (MongoDB indexes, etc.).

        Called once on first query.  All imports are lazy to avoid startup
        failures when agentic dependencies are not installed.
        """
        if self._initialized:
            return
        try:
            from agentic.core.db import get_client, ensure_indexes  # type: ignore[import-untyped]
            get_client()
            ensure_indexes()
            self._initialized = True
            logger.info("AgenticEngine initialized successfully")
        except Exception as exc:
            logger.error("AgenticEngine initialization failed: %s", exc)
            raise

    async def query(
        self,
        query: str,
        project_id: int,
        set_id: Optional[int] = None,
        conversation_history: Optional[list] = None,
    ) -> Any:
        """Run the agentic RAG pipeline in a thread (blocking call)."""
        self.ensure_initialized()
        from agentic.core.agent import run_agent  # type: ignore[import-untyped]

        result = await asyncio.to_thread(
            run_agent,
            query=query,
            project_id=project_id,
            set_id=set_id,
            conversation_history=conversation_history,
        )
        return result


# ---------------------------------------------------------------------------
# Traditional Engine — wraps traditional.rag
# ---------------------------------------------------------------------------

class TraditionalEngine:
    """Wrapper around the traditional FAISS-based RAG engine."""

    def __init__(self) -> None:
        self._faiss_loaded: bool = False
        self._lock: threading.Lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the FAISS index has been loaded."""
        return self._faiss_loaded

    def ensure_loaded(self, project_id: int) -> None:
        """Double-checked locking to load a project's FAISS index once.

        Imports are lazy to avoid startup failures when traditional
        dependencies (faiss-cpu, numpy) are not installed.
        """
        if self._faiss_loaded:
            return
        with self._lock:
            if self._faiss_loaded:
                return
            try:
                start = time.monotonic()
                from traditional.rag.retrieval.loaders import _load_project  # type: ignore[import-untyped]
                from traditional.rag.retrieval.state import PROJECTS  # type: ignore[import-untyped]

                _load_project(PROJECTS[project_id])
                elapsed = (time.monotonic() - start) * 1000
                logger.info(
                    "TraditionalEngine loaded project %d in %.0f ms",
                    project_id,
                    elapsed,
                )
                self._faiss_loaded = True
            except Exception as exc:
                logger.error(
                    "TraditionalEngine failed to load project %d: %s",
                    project_id,
                    exc,
                )
                raise

    async def query(
        self,
        query: str,
        project_id: int,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Run the traditional RAG pipeline in a thread."""
        self.ensure_loaded(project_id)
        from traditional.rag.api.generation_unified import generate_unified_answer  # type: ignore[import-untyped]

        result = await asyncio.to_thread(
            generate_unified_answer,
            user_query=query,
            project_id=project_id,
            session_id=session_id,
        )
        return result


# ---------------------------------------------------------------------------
# Orchestrator — ties both engines together
# ---------------------------------------------------------------------------

class Orchestrator:
    """Agentic-first orchestrator with optional traditional RAG fallback.

    Parameters
    ----------
    fallback_enabled : bool
        When True, automatically falls back to traditional RAG if the
        agentic result is deemed insufficient.
    fallback_timeout : int
        Maximum seconds to wait for the traditional engine fallback.
    """

    def __init__(
        self,
        fallback_enabled: bool = True,
        fallback_timeout: int = 30,
    ) -> None:
        self.fallback_enabled = fallback_enabled
        self.fallback_timeout = fallback_timeout
        self.agentic = AgenticEngine()
        self.traditional = TraditionalEngine()

    async def query(
        self,
        query: str,
        project_id: int,
        engine: Optional[str] = None,
        session_id: Optional[str] = None,
        set_id: Optional[int] = None,
        conversation_history: Optional[list] = None,
        **kwargs: Any,
    ) -> dict:
        """Route a query through the appropriate engine(s).

        Flow
        ----
        1. If ``engine="traditional"`` -- skip agentic, go straight to FAISS.
        2. Otherwise try agentic first.
        3. If ``engine="agentic"`` or fallback is disabled -- return as-is.
        4. Evaluate agentic result via ``_should_fallback``.
        5. If fallback needed -- run traditional with timeout.
        6. If traditional also fails -- return whatever agentic produced.
        """
        start = time.monotonic()

        # --- Direct traditional request ---
        if engine == "traditional":
            return await self._run_traditional(
                query, project_id, session_id, start, **kwargs,
            )

        # --- Try agentic ---
        agentic_result = None
        agentic_error: Optional[str] = None
        try:
            agentic_result = await self.agentic.query(
                query=query,
                project_id=project_id,
                set_id=set_id,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            agentic_error = str(exc)
            logger.warning("Agentic engine failed: %s", exc)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # --- If forced agentic or fallback disabled, return as-is ---
        if engine == "agentic" or not self.fallback_enabled:
            return self._build_response(
                result=agentic_result,
                engine="agentic",
                elapsed_ms=elapsed_ms,
                fallback_used=False,
                error=agentic_error,
            )

        # --- Evaluate and possibly fallback ---
        if not _should_fallback(agentic_result):
            return self._build_response(
                result=agentic_result,
                engine="agentic",
                elapsed_ms=elapsed_ms,
                fallback_used=False,
                error=None,
            )

        # --- Fallback to traditional ---
        agentic_confidence = getattr(agentic_result, "confidence", None)
        logger.info(
            "Falling back to traditional RAG (agentic confidence=%s)",
            agentic_confidence,
        )
        try:
            trad_response = await asyncio.wait_for(
                self._run_traditional(
                    query, project_id, session_id, start, **kwargs,
                ),
                timeout=self.fallback_timeout,
            )
            trad_response["fallback_used"] = True
            trad_response["agentic_confidence"] = agentic_confidence
            return trad_response
        except Exception as exc:
            logger.warning("Traditional fallback also failed: %s", exc)
            # Return whatever agentic had
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return self._build_response(
                result=agentic_result,
                engine="agentic",
                elapsed_ms=elapsed_ms,
                fallback_used=False,
                error=f"Fallback failed: {exc}",
            )

    async def _run_traditional(
        self,
        query: str,
        project_id: int,
        session_id: Optional[str],
        start: float,
        **kwargs: Any,
    ) -> dict:
        """Execute traditional RAG and format the response."""
        result = await self.traditional.query(
            query=query,
            project_id=project_id,
            session_id=session_id,
            **kwargs,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._build_response(
            result=result,
            engine="traditional",
            elapsed_ms=elapsed_ms,
            fallback_used=False,
            error=None,
        )

    def _build_response(
        self,
        result: Any,
        engine: str,
        elapsed_ms: int,
        fallback_used: bool,
        error: Optional[str],
    ) -> dict:
        """Convert an engine result (any type) to a unified response dict.

        Uses ``getattr`` throughout so both agentic dataclass results and
        traditional dict results are handled safely.
        """
        if result is None:
            return {
                "success": error is None,
                "answer": "",
                "sources": [],
                "confidence": "low",
                "engine_used": engine,
                "fallback_used": fallback_used,
                "agentic_confidence": None,
                "elapsed_ms": elapsed_ms,
                "error": error,
            }

        # Handle dict results (traditional engine)
        if isinstance(result, dict):
            return {
                "success": result.get("success", True),
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", "medium"),
                "engine_used": engine,
                "fallback_used": fallback_used,
                "agentic_confidence": None,
                "elapsed_ms": elapsed_ms,
                "error": error,
            }

        # Handle dataclass / object results (agentic engine)
        return {
            "success": True,
            "answer": getattr(result, "answer", ""),
            "sources": getattr(result, "sources", []),
            "confidence": getattr(result, "confidence", "medium"),
            "engine_used": engine,
            "fallback_used": fallback_used,
            "agentic_confidence": getattr(result, "confidence", None),
            "cost_usd": getattr(result, "cost_usd", 0.0),
            "total_steps": getattr(result, "total_steps", 0),
            "model": getattr(result, "model", ""),
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
