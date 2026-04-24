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
import os
import re
import threading
import time
from typing import Any, Optional

from gateway.docqa_bridge import DocQABridge
from gateway.intent_classifier import classify, IntentDecision

logger = logging.getLogger(__name__)

# Confidence string → numeric score mapping
CONFIDENCE_SCORE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.2}


# ---------------------------------------------------------------------------
# Feature flags (additive, default-off). Flip via env vars without redeploying.
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


MULTI_QUERY_RRF_ENABLED = _env_flag("MULTI_QUERY_RRF_ENABLED", False)
RERANKER_ENABLED = _env_flag("RERANKER_ENABLED", False)
RERANKER_KEEP_TOP_K = int(os.environ.get("RERANKER_KEEP_TOP_K", "0")) or None
# Drop sources below this rerank score (0 = off). Default 5.0 kills the
# "90% of references are irrelevant drawings" UX complaint while leaving the
# reranker's top-scored sources intact. RERANKER_MIN_KEEP guarantees a floor
# so a hyper-strict query never ends up with an empty source list.
RERANKER_SCORE_THRESHOLD = float(os.environ.get("RERANKER_SCORE_THRESHOLD", "5.0"))
RERANKER_MIN_KEEP = int(os.environ.get("RERANKER_MIN_KEEP", "3"))
SELF_RAG_ENABLED = _env_flag("SELF_RAG_ENABLED", False)

# --- Fix B: evasive-answer heuristic ---------------------------------------
# When agentic returns a high-confidence answer whose body reads like
# "I was unable to find / there is no / documents do not contain ...", we
# still want to try the traditional FAISS engine. Traditional RAG frequently
# finds the information in a different chunking / index that the agentic
# MongoDB tools missed.
_EVASIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(i\s+)?(was\s+)?unable\s+to\s+(find|locate|identify|determine|retrieve)\b", re.I),
    re.compile(r"\bcould\s+not\s+(find|locate|determine|identify|retrieve)\b", re.I),
    re.compile(r"\bno\s+(direct|explicit|specific)?\s*(evidence|mention|record|reference|references)\b", re.I),
    re.compile(r"\b(do|does|did)\s+not\s+(contain|provide|include|specify|mention|list|yield)\b", re.I),
    re.compile(r"\bnot\s+(available|found|present|explicitly|specifically|listed)\s+in\b", re.I),
    re.compile(r"\bthere\s+(are|is)\s+no\s+(sections?|direct|explicit|specific|mention|record|records|information|unit|listings?|references?)\b", re.I),
    re.compile(r"\bnot\s+(explicitly|directly)\s+(mentioned|provided|specified|listed)\b", re.I),
    re.compile(r"\breached\s+the\s+maximum\s+number\s+of\s+(search\s+)?steps\b", re.I),
    re.compile(r"\b(only|just)\s+returned\s+results?\s+in\b", re.I),
    re.compile(r"\bbased\s+on\s+what\s+i\s+found\s+so\s+far\b", re.I),
)

# Phrases that almost always mean "agent failed to answer" when they appear
# near the start of the response — trigger fallback even on a single hit.
_EVASIVE_OPENERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*I\s+(was\s+)?unable\s+to\b", re.I),
    re.compile(r"^\s*I\s+could\s+not\b", re.I),
    re.compile(r"^\s*I\s+reached\s+the\s+maximum\b", re.I),
    re.compile(r"^\s*(After|Based\s+on)\s+.{0,60}\s+(there|no)\s+(are|is)\s+no\b", re.I),
    re.compile(r"^\s*There\s+(are|is)\s+no\b", re.I),
    re.compile(r"^\s*The\s+available\s+documents\s+do\s+not\b", re.I),
    re.compile(r"^\s*A\s+review\s+of\s+the\s+available\b.{0,120}\s+did\s+not\b", re.I),
)


def _answer_is_evasive(answer: str) -> bool:
    """Return True when an agentic answer reads like a dodge.

    Heuristic (ordered cheapest-first):
    - Empty answer → evasive.
    - Evasive opener in the first ~200 chars → evasive (agent starts by bailing).
    - 2+ evasive phrase matches anywhere → evasive.
    - 1 evasive phrase + short answer (< 400 chars) → evasive.

    A dodge that slips through this filter still costs nothing — Fix A's
    traditional fallback will take over and, if traditional also can't
    answer, the original agentic response is preserved via the document-
    discovery path.
    """
    if not answer:
        return True
    opener = answer[:200]
    if any(p.search(opener) for p in _EVASIVE_OPENERS):
        return True
    hits = sum(1 for pat in _EVASIVE_PATTERNS if pat.search(answer))
    if hits >= 2:
        return True
    if hits >= 1 and len(answer) < 400:
        return True
    return False


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
        # --- Agent switching (RAG <-> DocQA) ---
        "active_agent": "rag",
        "suggest_switch": None,
        "selected_document": None,
        # --- Phase 3.2: intent classifier ---
        "needs_clarification": False,
        "clarification_prompt": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# DocQA bridge — lazy singleton (Phase 2.4)
# ---------------------------------------------------------------------------

_docqa_bridge: Optional[DocQABridge] = None


def _get_docqa_bridge() -> DocQABridge:
    """Lazy-initialize the module-level DocQABridge singleton.

    Uses the same memory_manager instance the orchestrator already uses for
    RAG sessions so DocQA state is keyed by the same session_id.
    """
    global _docqa_bridge
    if _docqa_bridge is None:
        from traditional.memory_manager import get_memory_manager  # type: ignore
        _docqa_bridge = DocQABridge(memory_manager=get_memory_manager())
    return _docqa_bridge


# ---------------------------------------------------------------------------
# Phase 3.2: session access + clarify response helpers
# ---------------------------------------------------------------------------

def _get_session_for_classifier(session_id: Optional[str]):
    """Return the ConversationSession for session_id, or a minimal shim
    with empty selected_documents when session_id is missing or unknown.

    The classifier only reads `.selected_documents`; a shim is safe.
    """
    class _Shim:
        selected_documents: list = []
        active_agent = "rag"
        docqa_session_id = None
        last_intent_decision = None
    if not session_id:
        return _Shim()
    try:
        from traditional.memory_manager import get_memory_manager
        mm = get_memory_manager()
        sess = mm.get_session(session_id)
        if sess is None:
            return _Shim()
        return sess
    except Exception:
        return _Shim()


def _build_clarify_response(
    decision: IntentDecision,
    rag_session_id: Optional[str],
    selected_document: Optional[dict],
    query: str,
) -> dict:
    """Phase 3.2: build a UnifiedResponse-shaped dict for a clarify decision."""
    return _base_response(
        query=query,
        answer="",
        success=True,
        needs_clarification=True,
        clarification_prompt=decision.clarification_prompt,
        active_agent="rag",
        selected_document=selected_document,
        session_id=rag_session_id or "",
        engine_used="classifier",
        confidence="low",
    )


def _build_download_url(s3_path: str, pdf_name: str) -> str | None:
    """Generate a pre-signed HTTPS download URL for a source document.

    Uses AWS SigV4 to produce a time-limited (default 1 hour) URL that
    works with private S3 buckets. The URL includes signed headers that
    authorize the request without exposing AWS credentials.

    The S3 key is used as-is (not parsed for bucket). The bucket comes
    from the ``S3_BUCKET_NAME`` environment variable.

    If S3 credentials are unavailable, falls back to a plain public-style
    URL (which will 403 on private buckets but preserves backward compat).

    Returns
    -------
    str or None
        Pre-signed URL valid for ``DOWNLOAD_URL_EXPIRATION_SECONDS`` (default 3600).
    """
    if not s3_path and not pdf_name:
        return None

    import os as _os
    # s3_path from MongoDB is typically "bucket/prefix/to/folder" (s3BucketPath field).
    # pdf_name is the filename without extension.
    # Full key = stripped(s3BucketPath) + "/" + pdfName + ".pdf"
    bucket_env = _os.environ.get("S3_BUCKET_NAME", "ifieldsmart")

    s3_key = (s3_path or "").lstrip("/")
    # If s3_path starts with the bucket name, strip it
    if s3_key.startswith(f"{bucket_env}/"):
        s3_key = s3_key[len(bucket_env) + 1:]
    elif s3_key == bucket_env:
        s3_key = ""

    # Compose with pdf_name
    if pdf_name:
        filename = pdf_name if pdf_name.lower().endswith(".pdf") else f"{pdf_name}.pdf"
        if s3_key and not s3_key.endswith("/"):
            s3_key = f"{s3_key}/{filename}"
        elif s3_key:
            s3_key = f"{s3_key}{filename}"
        else:
            s3_key = filename

    if not s3_key:
        return None

    # Try to generate a pre-signed URL using SigV4
    try:
        import boto3
        from botocore.config import Config

        region = _os.environ.get("AWS_DEFAULT_REGION") or _os.environ.get("S3_REGION", "us-east-1")
        expires = int(_os.environ.get("DOWNLOAD_URL_EXPIRATION_SECONDS", "3600"))

        client = boto3.client(
            "s3",
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_env, "Key": s3_key},
            ExpiresIn=expires,
        )
        return url
    except Exception as exc:
        logger.debug("Presigned URL generation failed for %s: %s", s3_key, exc)

    # Fallback: plain public-style URL (works only if bucket allows public reads)
    try:
        from traditional.rag.retrieval.metadata import build_pdf_download_url
        return build_pdf_download_url(s3_path, pdf_name or s3_path.rsplit("/", 1)[-1])
    except ImportError:
        pass

    # Final inline fallback
    if not pdf_name:
        pdf_name = s3_path.rsplit("/", 1)[-1] if "/" in s3_path else s3_path
    bucket, _, key_prefix = s3_path.partition("/")
    if not bucket:
        return None
    filename = pdf_name if pdf_name.lower().endswith(".pdf") else f"{pdf_name}.pdf"
    return f"https://{bucket}.s3.amazonaws.com/{key_prefix}/{filename}"


# Boilerplate title patterns: drawings that appear on every project set as
# procedural front-matter or general safety info. They are almost never the
# correct source for a specific user question (material quantities, equipment
# locations, schedules, sizing, etc.). We hard-drop them before any LLM
# reranker can accidentally keep them because the title shares a keyword like
# "1ST FLOOR" with the query.
_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"cover\s*sheets?|area\s*plans?|site\s*photographs?|"
    r"list\s*of\s*drawings?|general\s*notes?|abbreviations?\s*(?:and|&)\s*general\s*notes?|"
    r"ADA\s*requirements?|building\s*code\s*data|life\s*safety\s*plans?|"
    r"slab\s*edge\s*plans?|link\s*beam\s*schedules?|"
    r"title\s*sheets?|drawing\s*index(?:es)?|sheet\s*index(?:es)?|"
    r"symbol\s*legends?|north\s*arrow"
    r")",
    re.IGNORECASE,
)


def _strip_boilerplate_sources(source_documents: Any) -> list[dict] | None:
    """Return a copy of the list with boilerplate entries removed.

    If every source is boilerplate (rare — means the agent only retrieved
    front-matter), we keep the originals so the caller never sees an empty
    list for a question that retrieved something.
    """
    if not isinstance(source_documents, list) or not source_documents:
        return source_documents
    kept: list[dict] = []
    dropped = 0
    for sd in source_documents:
        if not isinstance(sd, dict):
            kept.append(sd)
            continue
        blob = " ".join(
            str(sd.get(k) or "")
            for k in (
                "display_title", "drawing_title", "drawingTitle",
                "pdf_name", "pdfName", "file_name",
                "drawing_name", "drawingName",
            )
        )
        if _BOILERPLATE_RE.search(blob):
            dropped += 1
            continue
        kept.append(sd)
    if not kept:
        return source_documents  # keep originals rather than return empty
    if dropped:
        logger.info("Stripped %d boilerplate sources from response", dropped)
    return kept


def _maybe_self_rag(query: str, answer: str, source_documents: Any) -> dict | None:
    """Fix #4: Self-RAG groundedness check on the final answer.

    Feature-flagged. Returns a small dict with ``groundedness_score``,
    ``claims_total``, ``claims_supported``, ``flagged_claims`` — or ``None``
    when the flag is off, the answer is too short, or the verifier errored.
    Strictly additive: never mutates the answer or source_documents.
    """
    if not SELF_RAG_ENABLED:
        return None
    if not answer or len(answer.strip()) < 40:
        return None
    try:
        from gateway.self_rag import evaluate_groundedness
        result = evaluate_groundedness(answer=answer, sources=source_documents or [])
        if result is None:
            return None
        return result.to_public()
    except Exception as exc:
        logger.warning("Self-RAG evaluation failed: %s", exc)
        return None


def _maybe_rerank(query: str, source_documents: Any) -> Any:
    """Fix #2: LLM-as-judge reorder of source_documents.

    No-op when the flag is off, the list is empty, or ≤1 item. Returns a new
    list (same dicts) ordered best-first. Safe to call multiple times.
    """
    if not RERANKER_ENABLED:
        return source_documents
    if not isinstance(source_documents, list) or len(source_documents) <= 1:
        return source_documents
    try:
        from gateway.reranker import rerank_source_documents
        return rerank_source_documents(
            query=query,
            source_documents=source_documents,
            keep_top_k=RERANKER_KEEP_TOP_K,
            score_threshold=RERANKER_SCORE_THRESHOLD,
            min_keep=RERANKER_MIN_KEEP,
        )
    except Exception as exc:
        logger.warning("Reranker failed, keeping original order: %s", exc)
        return source_documents


def _ensure_signed_source_urls(source_documents: Any) -> None:
    """Fix #8: In-place re-sign any unsigned download URLs in source_documents.

    The traditional engine builds download URLs from pdf_name + s3_path but does
    not sign them. Our S3 bucket is private (AccessDenied on unsigned GET), so
    those URLs 403 in the UI. We detect "no X-Amz-Signature" and regenerate via
    ``_build_download_url`` which uses SigV4.

    Mutates the list in place. No-op when the list is missing, empty, or all
    entries already have signed URLs. Safe to call on output from any engine.
    """
    if not isinstance(source_documents, list):
        return
    for sd in source_documents:
        if not isinstance(sd, dict):
            continue
        existing = sd.get("download_url") or ""
        if existing and "X-Amz-Signature" in existing:
            continue  # already signed, leave alone
        s3_path = sd.get("s3_path") or sd.get("s3BucketPath") or ""
        pdf_name = sd.get("pdf_name") or sd.get("pdfName") or sd.get("file_name") or ""
        if not s3_path and not pdf_name:
            continue
        try:
            new_url = _build_download_url(s3_path, pdf_name)
        except Exception as exc:  # defensive: never break the response over URL re-signing
            logger.debug("URL re-sign failed for %s/%s: %s", s3_path, pdf_name, exc)
            continue
        if new_url:
            sd["download_url"] = new_url


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
            existing_url = src.get("download_url")
            # Fix #8: Traditional engine returns UNSIGNED URLs which 403 against
            # our private bucket. Only keep the existing URL if it already has a
            # SigV4 signature; otherwise regenerate from s3_path + pdf_name so
            # the client gets a working presigned link.
            if existing_url and "X-Amz-Signature" in existing_url:
                download_url = existing_url
            else:
                download_url = _build_download_url(s3_path, pdf_name)
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
    - Fix B: answer reads like an evasive "unable to find / no evidence" dodge
      even at high confidence — traditional FAISS frequently finds info the
      agentic MongoDB tools missed.
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

    if _answer_is_evasive(answer):
        logger.info(
            "Fix B: agentic high-confidence answer detected as evasive (len=%d) — routing to traditional fallback",
            len(answer),
        )
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
        # Per-request context (set at start of query(), used by _build_response)
        self._last_query: str = ""
        self._last_project_id: Optional[int] = None
        self._last_session_id: Optional[str] = None

    async def query(
        self,
        query: str,
        project_id: int,
        engine: Optional[str] = None,
        session_id: Optional[str] = None,
        set_id: Optional[int] = None,
        conversation_history: Optional[list] = None,
        search_mode: Optional[str] = None,
        docqa_document: Optional[dict] = None,
        mode_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Route a query through the appropriate engine(s).

        Flow
        ----
        1. If ``search_mode="docqa"`` — route to Document QA agent.
        2. If ``search_mode="web"`` — run web search only.
        3. If ``search_mode="hybrid"`` — run agentic + web in parallel.
        4. Phase 3.2: auto-route via intent classifier (only when caller did
           not specify search_mode explicitly).
        5. Check session scope — if active, inject scope filters into agentic.
        6. Try agentic first.
        7. If agentic fails → document discovery (list available drawing titles)
           instead of blind FAISS fallback.
        """
        start = time.monotonic()
        # [CODE-H3] Preserve original caller intent before defaulting.
        # explicit_search_mode is None when the caller omitted the field entirely.
        explicit_search_mode = search_mode
        search_mode = search_mode or "rag"

        # Store per-request context for _build_response
        self._last_query = query
        self._last_project_id = project_id
        self._last_session_id = session_id

        # --- Phase 3.2: auto-route via intent classifier ---
        # Only classify when caller did NOT explicitly pick a search_mode.
        # An explicit search_mode (even "rag") must be honored without classifier injection.
        if explicit_search_mode is None and not docqa_document:  # auto-route path only
            try:
                sess = _get_session_for_classifier(session_id)
                decision = classify(query=query, session=sess, mode_hint=mode_hint)
                logger.info(
                    "intent_decision: target=%s conf=%.2f reason=%s",
                    decision.target, decision.confidence, decision.reason,
                )
                # Record for downstream debugging (non-persistent, best-effort)
                try:
                    sess.last_intent_decision = {
                        "target": decision.target,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                    }
                except Exception:
                    pass

                if decision.target == "clarify":
                    return _build_clarify_response(
                        decision,
                        rag_session_id=session_id,
                        selected_document=(
                            sess.selected_documents[-1]
                            if getattr(sess, "selected_documents", None)
                            else None
                        ),
                        query=query,
                    )

                if decision.target == "docqa":
                    # Promote last selected doc into docqa_document so _run_docqa works
                    effective_docqa_doc = docqa_document
                    if not effective_docqa_doc and getattr(sess, "selected_documents", None):
                        effective_docqa_doc = sess.selected_documents[-1]
                    if effective_docqa_doc:
                        return await self._run_docqa(
                            query=query,
                            project_id=project_id,
                            session_id=session_id,
                            docqa_document=effective_docqa_doc,
                            start=start,
                        )
                    # No doc available to hand off — fall through to RAG
            except Exception as exc:
                logger.exception("intent classifier error — falling back to RAG: %s", exc)

        # --- DocQA mode: route to Document QA agent ---
        if search_mode == "docqa":
            return await self._run_docqa(
                query=query,
                project_id=project_id,
                session_id=session_id,
                docqa_document=docqa_document,
                start=start,
            )

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

        # --- Fix #1: Multi-Query + RAG-Fusion pre-retrieval hint (feature-flagged) ---
        # When MULTI_QUERY_RRF_ENABLED=true, we decompose the user's question,
        # fan out to the agent's own search tools, fuse results with RRF, and
        # inject the top candidates as an rrf_hint via scope so the agent can
        # append it to the LAST user message (cache-preserving, Phase 1.3).
        # The hint is no longer prepended into conversation_history as a system
        # message — that approach busted the OpenAI auto-cache on every turn.
        rrf_scope: dict | None = None
        multi_query_diagnostic: dict | None = None
        if MULTI_QUERY_RRF_ENABLED and not scope:
            try:
                from gateway.retrieval_enrichment import build_context_hint, format_hint_for_agent
                hint = await asyncio.to_thread(
                    build_context_hint, query=query, project_id=project_id,
                )
                hint_text = format_hint_for_agent(hint)
                if hint_text:
                    # Pass via scope so agent appends to last user message only
                    rrf_scope = {"rrf_hint": hint_text}
                    multi_query_diagnostic = {
                        "sub_queries": hint.get("sub_queries"),
                        "fused_top": len(hint.get("fused") or []),
                        "total_candidates_considered": hint.get("total_candidates_considered"),
                        "num_source_lists": hint.get("num_source_lists"),
                    }
                    logger.info(
                        "Multi-query RRF hint injected via scope (cache-safe): %d sub-queries, %d fused, %d candidates",
                        len(hint.get("sub_queries") or []),
                        len(hint.get("fused") or []),
                        hint.get("total_candidates_considered") or 0,
                    )
            except Exception as exc:
                logger.warning("Multi-query RRF enrichment failed: %s — proceeding without hint", exc)

        # Merge rrf_scope into scope (rrf_scope is only set when scope was None above)
        effective_scope = scope or rrf_scope

        # --- Try agentic (with scope if active) ---
        agentic_result = None
        agentic_error: Optional[str] = None
        try:
            agentic_result = await self.agentic.query(
                query=query,
                project_id=project_id,
                set_id=set_id,
                conversation_history=conversation_history,
                scope=effective_scope,
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

        # --- Session management for agentic queries ---
        # The traditional engine handles sessions internally, but the agentic
        # engine does not. We create/update sessions here for agentic results.
        active_session_id = session_id
        session_stats = None
        try:
            from traditional.memory_manager import get_memory_manager, estimate_tokens  # type: ignore
            mm = get_memory_manager()
            if not active_session_id:
                active_session_id = mm.create_session(
                    user_query=query,
                    project_id=project_id,
                )
            # Add user message
            mm.add_to_session(
                session_id=active_session_id,
                role="user",
                content=query,
                tokens=estimate_tokens(query),
                metadata={"search_mode": "agentic", "project_id": project_id},
            )
            # Add assistant message if we have an answer
            answer_text = getattr(agentic_result, "answer", "") if agentic_result else ""
            if answer_text:
                mm.add_to_session(
                    session_id=active_session_id,
                    role="assistant",
                    content=answer_text,
                    tokens=estimate_tokens(answer_text),
                    metadata={
                        "engine": "agentic",
                        "confidence": getattr(agentic_result, "confidence", ""),
                        "cost_usd": getattr(agentic_result, "total_cost_usd", 0),
                    },
                )
            session_stats = mm.get_session_stats(active_session_id)
        except (ImportError, Exception) as exc:
            logger.debug("Session management for agentic: %s", exc)

        # Update per-request context with resolved session_id
        self._last_session_id = active_session_id

        # --- Record engine usage ---
        if active_session_id:
            try:
                from shared.session.manager import record_engine_use
                cost = getattr(agentic_result, "total_cost_usd", 0.0) if agentic_result else 0.0
                record_engine_use(active_session_id, "agentic", cost)
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
            resp["session_id"] = active_session_id
            resp["session_stats"] = session_stats
            if scope:
                resp["scoped_to"] = scope.get("drawing_title") or scope.get("section_title")
            return resp

        # --- Agentic insufficient: try Traditional RAG fallback, then document discovery ---
        agentic_confidence = getattr(agentic_result, "confidence", None)
        logger.info(
            "Agentic insufficient (confidence=%s) — attempting traditional RAG fallback",
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
            resp["session_id"] = active_session_id
            resp["session_stats"] = session_stats
            return resp

        # Not scoped: Try Traditional RAG fallback BEFORE document discovery.
        # This restores the pre-regression behavior where FAISS/gpt-4o answered
        # when agentic returned low confidence / empty / no sources.
        if self.fallback_enabled:
            try:
                trad_result = await asyncio.wait_for(
                    self.traditional.query(
                        query=query,
                        project_id=project_id,
                        session_id=active_session_id,
                    ),
                    timeout=self.fallback_timeout,
                )
                trad_answer = (
                    trad_result.get("answer")
                    if isinstance(trad_result, dict)
                    else getattr(trad_result, "answer", "")
                ) or ""

                if len(trad_answer.strip()) >= 20:
                    # Traditional produced a real answer — return it.
                    if active_session_id:
                        try:
                            from shared.session.manager import record_engine_use
                            record_engine_use(active_session_id, "traditional", 0.0)
                        except ImportError:
                            pass
                    trad_elapsed_ms = int((time.monotonic() - start) * 1000)
                    resp = self._build_response(
                        result=trad_result,
                        engine="traditional",
                        elapsed_ms=trad_elapsed_ms,
                        fallback_used=True,
                        error=None,
                    )
                    resp["agentic_confidence"] = agentic_confidence
                    resp["session_id"] = active_session_id
                    resp["session_stats"] = session_stats
                    return resp
            except asyncio.TimeoutError:
                logger.warning(
                    "Traditional RAG fallback timed out after %ss — falling through to document discovery",
                    self.fallback_timeout,
                )
            except Exception as exc:
                logger.warning(
                    "Traditional RAG fallback failed: %s — falling through to document discovery",
                    exc,
                )

        # Both engines insufficient: run document discovery for the full project
        elapsed_ms = int((time.monotonic() - start) * 1000)
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

        resp["session_id"] = active_session_id
        resp["session_stats"] = session_stats
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
    # Document QA Agent bridge
    # ------------------------------------------------------------------

    async def _run_docqa(
        self,
        query: str,
        project_id: int,
        session_id: Optional[str],
        docqa_document: Optional[dict],
        start: float,
    ) -> dict:
        """Phase 2.4: bridge-driven DocQA handoff.

        Delegates S3 fetch + DocQA upload + session state persistence +
        /api/chat forward + response normalization to DocQABridge.
        Returns a UnifiedResponse-shaped dict.

        Exceptions inside the bridge flow are caught and converted into a
        graceful docqa_fallback response (active_agent back to 'rag') via
        bridge.normalize() so the user never sees a 5xx.
        """
        bridge = _get_docqa_bridge()
        try:
            dq_sid = await bridge.ensure_document_loaded(
                session_id=session_id,
                doc_ref=docqa_document or {},
            )
            dq_resp = await bridge.ask(docqa_session_id=dq_sid, query=query)
            return bridge.normalize(
                docqa_response=dq_resp,
                rag_session_id=session_id,
                selected_document=docqa_document,
            )
        except Exception as exc:
            logger.exception(
                "DocQA bridge error for session=%s: %s", session_id, exc
            )
            return bridge.normalize(
                docqa_response={"error": str(exc)},
                rag_session_id=session_id,
                selected_document=docqa_document,
            )

    def _format_docqa_response(
        self,
        docqa_result: dict,
        query: str,
        elapsed_ms: int,
        file_name: str = "",
    ) -> dict:
        """Convert DocQA agent response to the unified RAG response format."""
        sources = docqa_result.get("sources", [])
        source_documents = [
            {
                "s3_path": s.get("file_name", ""),
                "file_name": s.get("file_name", ""),
                "display_title": s.get("file_name", ""),
                "download_url": None,
                "page": s.get("page_number"),
                "drawing_name": "",
                "drawing_title": "",
                "text_preview": s.get("text_preview", "")[:200],
                "relevance_score": s.get("score", 0),
            }
            for s in sources
        ]

        return _base_response(
            query=query,
            answer=docqa_result.get("answer", ""),
            confidence="high" if docqa_result.get("groundedness_score", 0) > 0.5 else "medium",
            confidence_score=docqa_result.get("groundedness_score", 0.0),
            follow_up_questions=docqa_result.get("follow_up_questions", []),
            source_documents=source_documents,
            s3_paths=[s.get("file_name", "") for s in sources],
            s3_path_count=len(sources),
            model_used=docqa_result.get("model_used", "gpt-4o"),
            token_usage=docqa_result.get("token_usage"),
            processing_time_ms=elapsed_ms,
            search_mode="docqa",
            engine_used="docqa",
            session_id=docqa_result.get("session_id", ""),
            is_clarification=docqa_result.get("needs_clarification", False),
            active_agent="docqa",
            selected_document=file_name,
        )

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
                query=self._last_query or "",
                processing_time_ms=elapsed_ms,
                project_id=self._last_project_id,
                session_id=self._last_session_id,
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
            # Fix #8: Force-sign any download URLs the traditional engine returned
            # unsigned (private bucket returns 403 without a SigV4 signature).
            _ensure_signed_source_urls(resp.get("source_documents"))
            # Hard drop of boilerplate sheets (cover, life-safety, general notes, etc.)
            # These are never the correct source for a specific question.
            resp["source_documents"] = _strip_boilerplate_sources(resp.get("source_documents"))
            # Fix #2: optional LLM-as-judge reorder (no-op when flag off)
            resp["source_documents"] = _maybe_rerank(
                self._last_query or "", resp.get("source_documents")
            )
            # Fix #4: optional self-RAG groundedness (nested under debug_info so
            # no new top-level fields are introduced)
            grounded = _maybe_self_rag(
                self._last_query or "", resp.get("answer", ""), resp.get("source_documents")
            )
            if grounded is not None:
                di = resp.get("debug_info") or {}
                if not isinstance(di, dict):
                    di = {}
                di["groundedness"] = grounded
                resp["debug_info"] = di
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
            _ensure_signed_source_urls(resp.get("source_documents"))
            resp["source_documents"] = _strip_boilerplate_sources(resp.get("source_documents"))
            resp["source_documents"] = _maybe_rerank(
                self._last_query or "", resp.get("source_documents")
            )
            grounded = _maybe_self_rag(
                self._last_query or "", resp.get("answer", ""), resp.get("source_documents")
            )
            if grounded is not None:
                di = resp.get("debug_info") or {}
                if not isinstance(di, dict):
                    di = {}
                di["groundedness"] = grounded
                resp["debug_info"] = di
            return resp

        # --- Dataclass / object results (agentic engine) → map to full schema ---
        answer = getattr(result, "answer", str(result))
        sources = getattr(result, "sources", []) or []
        confidence = getattr(result, "confidence", "medium")
        cost = getattr(result, "total_cost_usd", 0.0)
        steps = getattr(result, "total_steps", 0)
        model = getattr(result, "model", "")
        follow_ups = getattr(result, "follow_up_questions", []) or []
        input_tokens = getattr(result, "total_input_tokens", 0)
        output_tokens = getattr(result, "total_output_tokens", 0)

        # Use structured source_docs (with s3BucketPath/pdfName) when available
        # for proper download URL construction; fall back to plain string sources
        source_docs = getattr(result, "source_docs", []) or []
        raw_sources = source_docs if source_docs else sources

        confidence_score = CONFIDENCE_SCORE_MAP.get(confidence, 0.5)
        source_documents, s3_paths = _extract_source_documents(raw_sources)
        # Hard drop of boilerplate sheets before rerank so they can't leak
        # through on a title-keyword coincidence.
        source_documents = _strip_boilerplate_sources(source_documents)
        # Fix #2: optional reorder of agentic sources by direct relevance
        source_documents = _maybe_rerank(self._last_query or "", source_documents)

        # Fix #4: optional groundedness verification (nested under debug_info)
        debug_info_dict: dict[str, Any] = {
            "agentic_steps": steps,
            "agentic_cost_usd": cost,
        }
        grounded = _maybe_self_rag(self._last_query or "", answer, source_documents)
        if grounded is not None:
            debug_info_dict["groundedness"] = grounded

        return _base_response(
            query=self._last_query or "",
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
                "total_tokens": input_tokens + output_tokens,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
            s3_paths=s3_paths,
            s3_path_count=len(s3_paths),
            source_documents=source_documents,
            debug_info=debug_info_dict,
            processing_time_ms=elapsed_ms,
            project_id=self._last_project_id,
            session_id=self._last_session_id,
            search_mode="agentic",
            engine_used=engine,
            fallback_used=fallback_used,
            agentic_confidence=confidence,
            error=error,
        )
