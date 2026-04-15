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

# Confidence string → numeric score mapping
CONFIDENCE_SCORE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.2}


def _base_response(**overrides: Any) -> dict:
    """Return a complete response dict with ALL fields at safe defaults.

    Every response in the system starts from this template so that the
    frontend ALWAYS receives a consistent shape — regardless of which
    engine answered.  New fields for future phases are included here
    with backward-compatible defaults.

    Callers pass only the fields they want to override.
    """
    base = {
        # --- Core answer ---
        "query": "",
        "answer": "",
        "rag_answer": None,
        "web_answer": None,
        # --- Retrieval metrics ---
        "retrieval_count": 0,
        "average_score": 0.0,
        "confidence": "low",
        "confidence_score": 0.0,
        "is_clarification": False,
        # --- Follow-up & query enhancement ---
        "follow_up_questions": [],
        "improved_queries": [],
        "query_tips": [],
        # --- Model info ---
        "model_used": "",
        "token_usage": None,
        "token_tracking": None,
        # --- Source documents (frontend display) ---
        "s3_paths": [],
        "s3_path_count": 0,
        "source_documents": [],
        "retrieved_chunks": [],
        # --- Debug ---
        "debug_info": None,
        # --- Timing & context ---
        "processing_time_ms": 0,
        "project_id": None,
        "session_id": None,
        "session_stats": None,
        # --- Search mode ---
        "search_mode": "rag",
        # --- Web sources ---
        "web_sources": [],
        "web_source_count": 0,
        # --- Document pinning / scoping ---
        "pin_status": None,
        "needs_document_selection": False,
        "available_documents": [],
        "scoped_to": None,
        # --- Engine metadata ---
        "success": True,
        "engine_used": "agentic",
        "fallback_used": False,
        "agentic_confidence": None,
        "error": None,
    }
    base.update(overrides)
    return base


def _build_download_url(s3_path: str, pdf_name: str) -> str | None:
    """Construct a public HTTPS download URL for a source document.

    Uses the same logic as traditional engine metadata — the s3_path
    contains ``bucket/prefix`` and the pdf_name is appended.

    Format: https://{bucket}.s3.amazonaws.com/{prefix}/{pdf_name}.pdf
    """
    if not s3_path:
        return None
    # Use the traditional engine's builder if available
    try:
        from traditional.rag.retrieval.metadata import build_pdf_download_url
        return build_pdf_download_url(s3_path, pdf_name or s3_path.rsplit("/", 1)[-1])
    except ImportError:
        pass
    # Inline fallback: same logic
    if not pdf_name:
        pdf_name = s3_path.rsplit("/", 1)[-1] if "/" in s3_path else s3_path
    bucket, _, key_prefix = s3_path.partition("/")
    if not bucket:
        return None
    filename = pdf_name if pdf_name.lower().endswith(".pdf") else f"{pdf_name}.pdf"
    return f"https://{bucket}.s3.amazonaws.com/{key_prefix}/{filename}"


def _extract_source_documents(
    sources: list,
) -> tuple[list[dict], list[str]]:
    """Convert agentic sources (str or dict) to frontend-compatible format.

    Returns (source_documents, s3_paths) where each source_document has:
    s3_path, file_name, display_title, download_url, pdf_name, drawing_name,
    drawing_title, page — all fields the Angular frontend needs.

    download_url is constructed from s3_path + pdf_name using the same
    pattern as the traditional engine's metadata module.
    """
    source_documents: list[dict] = []
    s3_paths: list[str] = []

    for src in sources or []:
        if isinstance(src, dict):
            s3_path = src.get("s3_path", src.get("s3BucketPath", src.get("sourceFile", "")))
            pdf_name = src.get("pdfName", src.get("pdf_name", ""))
            download_url = src.get("download_url") or _build_download_url(s3_path, pdf_name)
            source_documents.append({
                "s3_path": s3_path,
                "file_name": src.get("name", src.get("sourceFile", pdf_name or "")),
                "display_title": src.get("sheet_number", src.get("name", src.get("drawingName", ""))),
                "download_url": download_url,
                "pdf_name": pdf_name,
                "drawing_name": src.get("drawingName", src.get("drawing_name", "")),
                "drawing_title": src.get("drawingTitle", src.get("drawing_title", "")),
                "page": src.get("page"),
            })
            if s3_path:
                s3_paths.append(s3_path)
        elif isinstance(src, str):
            download_url = _build_download_url(src, src)
            source_documents.append({
                "s3_path": src,
                "file_name": src,
                "display_title": src,
                "download_url": download_url,
                "pdf_name": src,
                "drawing_name": "",
                "drawing_title": "",
                "page": None,
            })
            s3_paths.append(src)

    return source_documents, s3_paths


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
        scope: Optional[dict] = None,
    ) -> Any:
        """Run the agentic RAG pipeline in a thread (blocking call).

        Parameters
        ----------
        scope : dict or None
            Document scope filter. When set, injected into every tool call
            to restrict results to a specific drawingTitle/drawingName.
        """
        self.ensure_initialized()
        from agentic.core.agent import run_agent  # type: ignore[import-untyped]

        result = await asyncio.to_thread(
            run_agent,
            query=query,
            project_id=project_id,
            set_id=set_id,
            conversation_history=conversation_history,
            scope=scope,
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
        search_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Route a query through the appropriate engine(s).

        Flow
        ----
        1. If ``search_mode="web"`` — run web search only.
        2. If ``search_mode="hybrid"`` — run agentic + web in parallel.
        3. Check session scope — if active, inject scope filters into agentic.
        4. Try agentic first.
        5. If agentic fails → document discovery (list available drawing titles)
           instead of blind FAISS fallback.
        """
        start = time.monotonic()
        search_mode = search_mode or "rag"

        # --- Web-only mode ---
        if search_mode == "web":
            return await self._run_web_search(query, start)

        # --- Hybrid mode: agentic + web in parallel ---
        if search_mode == "hybrid":
            return await self._run_hybrid(
                query, project_id, set_id, conversation_history, start,
            )

        # --- Direct traditional request (backward compat, not used in scoped mode) ---
        if engine == "traditional":
            return await self._run_traditional(
                query, project_id, session_id, start, **kwargs,
            )

        # --- Resolve session scope ---
        scope = None
        if session_id:
            try:
                from shared.session.manager import get_document_scope
                scope_state = get_document_scope(session_id)
                if scope_state.get("is_active"):
                    scope = scope_state
                    logger.info(
                        "Query scoped to: %s (%s)",
                        scope.get("drawing_title") or scope.get("section_title"),
                        scope.get("document_type"),
                    )
            except ImportError:
                pass

        # --- Try agentic (with scope if active) ---
        agentic_result = None
        agentic_error: Optional[str] = None
        try:
            agentic_result = await self.agentic.query(
                query=query,
                project_id=project_id,
                set_id=set_id,
                conversation_history=conversation_history,
                scope=scope,
            )
        except Exception as exc:
            agentic_error = str(exc)
            logger.warning("Agentic engine failed: %s", exc)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # --- Touch scope timer if active ---
        if scope and session_id:
            try:
                from shared.session.manager import get_meta
                get_meta(session_id).scope.touch()
            except ImportError:
                pass

        # --- Success: agentic answered well ---
        if not _should_fallback(agentic_result):
            resp = self._build_response(
                result=agentic_result,
                engine="agentic",
                elapsed_ms=elapsed_ms,
                fallback_used=False,
                error=None,
            )
            if scope:
                resp["scoped_to"] = scope.get("drawing_title") or scope.get("section_title")
            return resp

        # --- Agentic failed: document discovery instead of FAISS fallback ---
        agentic_confidence = getattr(agentic_result, "confidence", None)
        logger.info(
            "Agentic insufficient (confidence=%s) — running document discovery",
            agentic_confidence,
        )

        # If already scoped and failed → report "doc doesn't have this info"
        if scope:
            scoped_name = scope.get("drawing_title") or scope.get("section_title")
            resp = self._build_response(
                result=agentic_result,
                engine="agentic",
                elapsed_ms=elapsed_ms,
                fallback_used=False,
                error=None,
            )
            resp["answer"] = (
                f"The document '{scoped_name}' does not contain information "
                f"about this topic. You can try a different document or "
                f"return to full project scope to search all documents."
            )
            resp["scoped_to"] = scoped_name
            resp["needs_document_selection"] = True
            # Discover other documents for suggestions
            available = await self._discover_documents(project_id, set_id)
            resp["available_documents"] = available
            return resp

        # Not scoped: run document discovery for the full project
        resp = self._build_response(
            result=agentic_result,
            engine="agentic",
            elapsed_ms=elapsed_ms,
            fallback_used=False,
            error=agentic_error,
        )
        if not resp.get("answer") or len(resp.get("answer", "")) < 20:
            resp["answer"] = (
                "I couldn't find specific information in a broad search. "
                "Your project has the following document groups — try "
                "selecting one for a focused search."
            )
        resp["needs_document_selection"] = True
        resp["agentic_confidence"] = agentic_confidence
        available = await self._discover_documents(project_id, set_id)
        resp["available_documents"] = available

        # Generate improved queries and tips (query enhancement)
        try:
            from gateway.query_enhancer import generate_query_enhancement
            enhancement = generate_query_enhancement(
                original_query=query,
                available_documents=available,
                project_id=project_id,
            )
            resp["improved_queries"] = enhancement.get("improved_queries", [])
            resp["query_tips"] = enhancement.get("query_tips", [])
        except Exception as exc:
            logger.warning("Query enhancement failed: %s", exc)

        return resp

    # ------------------------------------------------------------------
    # Document discovery
    # ------------------------------------------------------------------

    async def _discover_documents(
        self,
        project_id: int,
        set_id: Optional[int] = None,
    ) -> list[dict]:
        """Fetch unique drawing titles + spec titles for document discovery.

        Uses the title cache (1hr TTL) to avoid re-aggregating on every query.
        Returns a merged list of drawings and specifications.
        """
        try:
            from gateway.title_cache import get_cached_titles, set_cached_titles

            # Check cache first
            cached = get_cached_titles(project_id)
            if cached:
                return self._merge_available_documents(
                    cached["drawings"], cached["specifications"],
                )

            # Cache miss — fetch from MongoDB
            drawings = await asyncio.to_thread(
                self._fetch_drawing_titles, project_id, set_id,
            )
            specifications = await asyncio.to_thread(
                self._fetch_spec_titles, project_id,
            )

            # Store in cache
            set_cached_titles(project_id, drawings, specifications)

            return self._merge_available_documents(drawings, specifications)

        except Exception as exc:
            logger.error("Document discovery failed: %s", exc)
            return []

    @staticmethod
    def _fetch_drawing_titles(project_id: int, set_id: Optional[int] = None) -> list[dict]:
        """Fetch unique drawing titles from MongoDB."""
        try:
            from agentic.tools.drawing_tools import list_unique_drawing_titles  # type: ignore
            return list_unique_drawing_titles(project_id, set_id)
        except Exception as exc:
            logger.error("Failed to fetch drawing titles: %s", exc)
            return []

    @staticmethod
    def _fetch_spec_titles(project_id: int) -> list[dict]:
        """Fetch unique spec titles from MongoDB."""
        try:
            from agentic.tools.specification_tools import list_unique_spec_titles  # type: ignore
            return list_unique_spec_titles(project_id)
        except Exception as exc:
            logger.error("Failed to fetch spec titles: %s", exc)
            return []

    @staticmethod
    def _merge_available_documents(
        drawings: list[dict],
        specifications: list[dict],
    ) -> list[dict]:
        """Merge drawing and spec title lists into a unified available_documents list.

        Groups by document type, includes trade info for drawings.
        """
        result: list[dict] = []

        for d in drawings:
            result.append({
                "type": "drawing",
                "drawing_title": d.get("drawingTitle", ""),
                "drawing_name": d.get("drawingName", ""),
                "trade": d.get("trade", ""),
                "pdf_name": d.get("pdfName", ""),
                "fragment_count": d.get("fragment_count", 0),
            })

        for s in specifications:
            result.append({
                "type": "specification",
                "section_title": s.get("sectionTitle", ""),
                "pdf_name": s.get("pdfName", ""),
                "specification_number": s.get("specificationNumber", ""),
                "fragment_count": s.get("fragment_count", 0),
            })

        return result

    # ------------------------------------------------------------------
    # Web search (shared by web-only and hybrid modes)
    # ------------------------------------------------------------------

    async def _run_web_search(self, query: str, start: float) -> dict:
        """Run web search only using traditional engine's web_search service."""
        try:
            from traditional.services.web_search import web_search
            result = await asyncio.to_thread(web_search, query)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            web_sources = result.get("sources", [])
            return _base_response(
                query=query,
                answer=result.get("answer", ""),
                web_answer=result.get("answer", ""),
                confidence_score=0.8,
                model_used="gpt-4.1",
                processing_time_ms=elapsed_ms,
                search_mode="web",
                web_sources=web_sources,
                web_source_count=len(web_sources),
            )
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error("Web search failed: %s", exc)
            return _base_response(
                processing_time_ms=elapsed_ms,
                success=False,
                error="Web search failed. Please try again.",
            )

    async def _run_hybrid(
        self,
        query: str,
        project_id: int,
        set_id: Optional[int],
        conversation_history: Optional[list],
        start: float,
    ) -> dict:
        """Run agentic (project data) + web search in PARALLEL.

        Returns both ``rag_answer`` and ``web_answer`` separately.
        """
        # Launch both in parallel
        agentic_task = asyncio.create_task(
            self.agentic.query(
                query=query, project_id=project_id,
                set_id=set_id, conversation_history=conversation_history,
            )
        )
        web_task = asyncio.create_task(
            asyncio.to_thread(self._sync_web_search, query)
        )

        # Wait for both (don't fail if one errors)
        agentic_result = None
        web_result = None
        try:
            agentic_result = await agentic_task
        except Exception as exc:
            logger.warning("Hybrid: agentic failed: %s", exc)
        try:
            web_result = await web_task
        except Exception as exc:
            logger.warning("Hybrid: web search failed: %s", exc)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Extract answers
        rag_answer = getattr(agentic_result, "answer", None) if agentic_result else None
        web_answer = web_result.get("answer", None) if web_result else None
        sources = getattr(agentic_result, "sources", []) if agentic_result else []
        web_sources = web_result.get("sources", []) if web_result else []
        confidence = getattr(agentic_result, "confidence", "low") if agentic_result else "low"

        # Build combined answer
        combined = ""
        if rag_answer:
            combined += f"**From Project Data:**\n{rag_answer}"
        if web_answer:
            if combined:
                combined += "\n\n---\n\n"
            combined += f"**From Web Search:**\n{web_answer}"
        if not combined:
            combined = "No results from either project data or web search."

        source_documents, s3_paths = _extract_source_documents(sources)

        return _base_response(
            query=query,
            answer=combined,
            rag_answer=rag_answer,
            web_answer=web_answer,
            retrieval_count=len(sources),
            confidence=confidence,
            average_score=CONFIDENCE_SCORE_MAP.get(confidence, 0.5),
            confidence_score=CONFIDENCE_SCORE_MAP.get(confidence, 0.5),
            model_used=getattr(agentic_result, "model", "gpt-4.1") if agentic_result else "gpt-4.1",
            s3_paths=s3_paths,
            s3_path_count=len(s3_paths),
            source_documents=source_documents,
            debug_info={
                "agentic_steps": getattr(agentic_result, "total_steps", 0) if agentic_result else 0,
                "agentic_cost_usd": getattr(agentic_result, "cost_usd", 0.0) if agentic_result else 0.0,
                "web_sources_count": len(web_sources),
            },
            processing_time_ms=elapsed_ms,
            project_id=project_id,
            search_mode="hybrid",
            web_sources=web_sources,
            web_source_count=len(web_sources),
            agentic_confidence=confidence,
        )

    @staticmethod
    def _sync_web_search(query: str) -> dict:
        """Synchronous web search wrapper for asyncio.to_thread."""
        try:
            from traditional.services.web_search import web_search
            return web_search(query)
        except Exception as exc:
            logger.error("Sync web search error: %s", exc)
            return {"answer": None, "sources": []}

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
        """Convert an engine result to a response dict matching the QueryResponse schema.

        Handles three result types:
        1. None → empty response with error
        2. dict / Pydantic model (traditional engine) → pass through ALL fields, inject engine metadata
        3. dataclass (agentic engine) → map to the full response schema

        Every path returns a dict starting from _base_response() so ALL fields
        (including source_documents, s3_paths, pdf_name, drawing_name, etc.)
        are always present for the Angular frontend.
        """
        # --- None result (engine error or timeout) ---
        if result is None:
            return _base_response(
                processing_time_ms=elapsed_ms,
                success=error is None,
                engine_used=engine,
                fallback_used=fallback_used,
                error=error,
            )

        # --- Dict results (traditional engine) — pass through ALL original fields ---
        if isinstance(result, dict):
            resp = _base_response()  # start from full template
            resp.update(result)      # overlay all traditional fields (preserves everything)
            resp["engine_used"] = engine
            resp["fallback_used"] = fallback_used
            resp["agentic_confidence"] = resp.get("agentic_confidence")
            if "processing_time_ms" not in result:
                resp["processing_time_ms"] = elapsed_ms
            if error:
                resp["error"] = error
            return resp

        # --- Pydantic model results (traditional engine returning model) ---
        if hasattr(result, "model_dump"):
            resp = _base_response()
            resp.update(result.model_dump())
            resp["engine_used"] = engine
            resp["fallback_used"] = fallback_used
            resp["agentic_confidence"] = None
            if "processing_time_ms" not in result.model_dump():
                resp["processing_time_ms"] = elapsed_ms
            return resp

        # --- Dataclass / object results (agentic engine) → map to full schema ---
        answer = getattr(result, "answer", str(result))
        sources = getattr(result, "sources", []) or []
        confidence = getattr(result, "confidence", "medium")
        cost = getattr(result, "cost_usd", 0.0)
        steps = getattr(result, "total_steps", 0)
        model = getattr(result, "model", "")
        follow_ups = getattr(result, "follow_up_questions", []) or []

        # Use structured source_docs (with s3BucketPath/pdfName) when available
        # for proper download URL construction; fall back to plain string sources
        source_docs = getattr(result, "source_docs", []) or []
        raw_sources = source_docs if source_docs else sources

        confidence_score = CONFIDENCE_SCORE_MAP.get(confidence, 0.5)
        source_documents, s3_paths = _extract_source_documents(raw_sources)

        return _base_response(
            answer=answer,
            rag_answer=answer,
            retrieval_count=len(sources),
            confidence=confidence,
            average_score=confidence_score,
            confidence_score=confidence_score,
            is_clarification=confidence == "low",
            follow_up_questions=follow_ups,
            model_used=model,
            token_usage={
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            s3_paths=s3_paths,
            s3_path_count=len(s3_paths),
            source_documents=source_documents,
            debug_info={
                "agentic_steps": steps,
                "agentic_cost_usd": cost,
            },
            processing_time_ms=elapsed_ms,
            search_mode="agentic",
            engine_used=engine,
            fallback_used=fallback_used,
            agentic_confidence=confidence,
            error=error,
        )
