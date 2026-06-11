"""
agentic.memory.writer
=====================

Memory Writer (Agent 6) — fire-and-forget side channel that persists
conversation turns to the existing ``MemoryManager`` and (when the
``MEMORY_WRITER_VECTOR_ENABLED`` flag is true) the new dual-layer
``SessionVectorStore``.

Why a thread pool, not asyncio
------------------------------
Some FastAPI handlers in this codebase are sync (``def`` not
``async def``) and the existing ``MemoryManager`` is sync. Spinning a
new event loop per request is awkward; a tiny thread pool is the
straightforward fit.

Contract
--------
* ``write_turn_async`` returns immediately. The actual work runs in a
  background thread.
* No exception escapes the worker — the writer is best-effort by
  design. A failure here must NEVER affect the user-visible response.
* When the flag is off, only ``MemoryManager.add_to_session`` runs
  (today's behaviour). Embedding, vector storage, and rolling summary
  are all gated behind the flag.

Rolling summary
---------------
Every ``ROLLING_SUMMARY_INTERVAL`` *user* turns we regenerate a <=200
word summary of the last 30 turns and stash it on
``ConversationContext``. We add a new ``rolling_summary`` field if it
exists on the dataclass (newer build); otherwise we fall back to
``custom_instructions`` (older build) for backward compatibility.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_rag.memory.writer")


# ---------------------------------------------------------------------------
# Module-level executors (shared across writer instances).
#
# We split dispatch from embedding work to avoid a self-deadlock:
# the dispatch worker submits two embedding sub-tasks and blocks on
# their results. If both pools were the same one with N workers, N
# concurrent dispatches would saturate the pool — the embed sub-tasks
# would never get scheduled.
# ---------------------------------------------------------------------------

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memwriter")
_EMBED_EXECUTOR = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="memory-embed"
)


def _shutdown_executor() -> None:
    """Drain in-flight memory writes on interpreter / uvicorn shutdown.

    ``cancel_futures=False`` ensures already-queued writes complete so
    we don't silently lose data on graceful shutdown.
    """
    try:
        _EXECUTOR.shutdown(wait=True, cancel_futures=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        _EMBED_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    except Exception:  # noqa: BLE001
        pass


atexit.register(_shutdown_executor)


def _flag_enabled(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


ROLLING_SUMMARY_INTERVAL = int(os.environ.get("ROLLING_SUMMARY_INTERVAL", "10"))
ROLLING_SUMMARY_MODEL = os.environ.get("ROLLING_SUMMARY_MODEL", "gpt-4o-mini")
ROLLING_SUMMARY_TURN_WINDOW = int(os.environ.get("ROLLING_SUMMARY_TURN_WINDOW", "30"))


# ---------------------------------------------------------------------------
# MemoryWriter
# ---------------------------------------------------------------------------


class MemoryWriter:
    """Background writer that persists turns to memory + vector store.

    Parameters
    ----------
    memory_manager:
        Existing ``MemoryManager`` instance (sync). If ``None``, the
        global instance from ``traditional.memory_manager`` is fetched
        lazily on first use.
    vector_store:
        Optional ``SessionVectorStore`` instance. If ``None`` and the
        flag is enabled, a default instance is constructed lazily.
    embed_fn:
        Callable ``str -> list[float]`` used to embed turn text. Defaults
        to ``embed_text`` from ``agentic.memory.embeddings``. Injectable
        for tests.
    summary_fn:
        Callable that takes a list of message dicts and returns a
        summary string. Defaults to an OpenAI gpt-4o-mini call;
        injectable for tests.
    executor:
        ``concurrent.futures.Executor`` to dispatch work onto. Defaults
        to the module-level pool.
    """

    def __init__(
        self,
        memory_manager: Any = None,
        vector_store: Any = None,
        embed_fn: Optional[Any] = None,
        summary_fn: Optional[Any] = None,
        executor: Optional[Any] = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._vector_store = vector_store
        self._embed_fn = embed_fn
        self._summary_fn = summary_fn
        self._executor = executor or _EXECUTOR
        self._summary_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    def _get_memory_manager(self):
        if self._memory_manager is not None:
            return self._memory_manager
        # Imported lazily so tests that mock the manager don't pull the
        # real module.
        from traditional.memory_manager import get_memory_manager  # type: ignore

        self._memory_manager = get_memory_manager()
        return self._memory_manager

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store
        from agentic.memory.vector_store import SessionVectorStore  # local import

        self._vector_store = SessionVectorStore()
        return self._vector_store

    def _get_embed_fn(self):
        if self._embed_fn is not None:
            return self._embed_fn
        from agentic.memory.embeddings import embed_text  # local import

        self._embed_fn = embed_text
        return self._embed_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_turn_async(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        project_id: Optional[int],
        set_id: Optional[str],
        turn_id: Optional[str] = None,
        turn_type: str = "rag",
        parent_turn_id: Optional[str] = None,
        source_documents: Optional[list] = None,
    ) -> None:
        """Submit a turn-write to the background pool and return.

        This must NEVER raise. Any submission failure is logged and the
        caller continues unaffected.

        Parameters
        ----------
        turn_id : str | None
            Stable UUID identifying the (user, assistant) turn pair. The
            caller mints this synchronously *before* dispatch so the value
            can flow back to the HTTP client without waiting on the
            background write. When None, no turn_id is attached.
        turn_type : str
            "rag" (default — normal Q&A turn) or "deep_dive" (vision-based
            secondary analysis). Used by Deep Dive recall + audit pipelines.
        parent_turn_id : str | None
            For ``turn_type == "deep_dive"``, the turn_id of the original
            RAG turn that this Deep Dive enhances. Enables traversal of
            the (rag -> deep_dive) chain in conversation history.
        """
        try:
            self._executor.submit(
                self._write_turn,
                session_id,
                user_text,
                assistant_text,
                project_id,
                set_id,
                turn_id,
                turn_type,
                parent_turn_id,
                source_documents,
            )
        except RuntimeError as exc:
            # Executor shut down or rejected — don't crash the request.
            logger.warning("MemoryWriter submit failed: %s", exc)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _write_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        project_id: Optional[int],
        set_id: Optional[str],
        turn_id: Optional[str] = None,
        turn_type: str = "rag",
        parent_turn_id: Optional[str] = None,
        source_documents: Optional[list] = None,
    ) -> None:
        """Top-level worker — wraps everything in a global try/except.

        We never let an exception bubble out of this function.
        """
        try:
            self._do_write_turn(
                session_id, user_text, assistant_text, project_id, set_id,
                turn_id, turn_type, parent_turn_id, source_documents,
            )
        except Exception as exc:  # noqa: BLE001 - intentional safety net
            # Truly last-resort: any uncaught error is swallowed and
            # logged. The Memory Writer is fire-and-forget by contract.
            logger.error(
                "MemoryWriter unexpected failure for session=%s: %s",
                session_id, exc, exc_info=True,
            )

    def _do_write_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        project_id: Optional[int],
        set_id: Optional[str],
        turn_id: Optional[str] = None,
        turn_type: str = "rag",
        parent_turn_id: Optional[str] = None,
        source_documents: Optional[list] = None,
    ) -> None:
        vector_enabled = _flag_enabled("MEMORY_WRITER_VECTOR_ENABLED")

        # --- Step 1: embed (parallel) — only when flag is on ---
        user_vec: Optional[List[float]] = None
        asst_vec: Optional[List[float]] = None
        if vector_enabled:
            user_vec, asst_vec = self._embed_pair(user_text, assistant_text)

        # --- Step 2: append to MemoryManager (always) ---
        mm = self._get_memory_manager()
        # turn_id/turn_type/parent_turn_id/source_documents are optional
        # ADDITIVE fields. Older readers ignore them; newer readers (Deep
        # Dive recall + audit) use them to traverse (rag -> deep_dive)
        # chains and reuse the cited source_documents without a fresh
        # MongoDB hit during Deep Dive.
        meta: Dict[str, Any] = {"project_id": project_id, "set_id": set_id}
        if turn_id is not None:
            meta["turn_id"] = turn_id
        if turn_type and turn_type != "rag":
            meta["turn_type"] = turn_type
        if parent_turn_id is not None:
            meta["parent_turn_id"] = parent_turn_id
        if source_documents:
            # Trim source_documents to the essential fields Deep Dive needs.
            # Accept BOTH key shapes so this works regardless of whether
            # the caller passes normalized docs (post _extract_source_documents:
            # `s3_path`, `pdf_name`, `display_title`) or raw agent output
            # (`s3BucketPath`, `pdfName`, `drawingName`, `drawingTitle`,
            # `sheet_number`).
            def _pick(sd: Dict[str, Any], *keys: str) -> Any:
                for k in keys:
                    v = sd.get(k)
                    if v not in (None, ""):
                        return v
                return ""

            trimmed: list = []
            for sd in source_documents:
                if not isinstance(sd, dict):
                    continue
                # Normalize keys at write time so the reader (Deep Dive)
                # sees a single consistent shape.
                s3_path = _pick(sd, "s3_path", "s3BucketPath", "sourceFile")
                pdf_name = _pick(sd, "pdf_name", "pdfName")
                drawing_name = _pick(sd, "drawing_name", "drawingName")
                drawing_title = _pick(sd, "drawing_title", "drawingTitle", "sectionTitle")
                display_title = _pick(
                    sd, "display_title", "sheet_number", "drawingTitle",
                    "sectionTitle", "drawingName",
                ) or pdf_name
                trimmed.append({
                    "s3_path": s3_path,
                    "pdf_name": pdf_name,
                    "file_name": _pick(sd, "file_name", "name", "sourceFile") or pdf_name,
                    "display_title": display_title,
                    "drawing_name": drawing_name,
                    "drawing_title": drawing_title,
                    "page": sd.get("page") or 1,
                    # Carry png_url + download_url forward if upstream
                    # already built them (prod orchestrator does;
                    # v31 chain does not — Deep Dive derives in agent).
                    "png_url": sd.get("png_url"),
                    "download_url": sd.get("download_url"),
                    # v3.3 (2026-05-18) — spec chunk identity for Deep Dive
                    # follow-ups. parent_id lets the next Deep Dive call
                    # group chunks of the same spec section even when the
                    # UI passes a single representative pdf_name. Null on
                    # drawings — they're already 1-doc-per-sheet.
                    "parent_id": (
                        sd.get("parent_id")
                        or sd.get("parentId")
                    ),
                    "fragment_count": sd.get("fragment_count"),
                })
            meta["source_documents"] = trimmed
        user_turn_index, asst_turn_index = self._append_to_memory_manager(
            mm, session_id, user_text, assistant_text, meta
        )

        # --- Step 3: vector store (flag-gated) ---
        if vector_enabled:
            self._write_vectors(
                session_id,
                user_text, user_vec, user_turn_index,
                assistant_text, asst_vec, asst_turn_index,
                meta,
            )

        # --- Step 4: rolling summary (flag-gated) ---
        if vector_enabled:
            self._maybe_regenerate_summary(mm, session_id)

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _embed_pair(
        self, user_text: str, assistant_text: str
    ) -> tuple[Optional[List[float]], Optional[List[float]]]:
        """Embed both turns in parallel via the dedicated embed pool.

        We deliberately do NOT use ``self._executor`` here: that pool
        runs the worker submitting these sub-tasks, so reusing it
        would deadlock under concurrent ``write_turn_async`` load.
        """
        embed = self._get_embed_fn()
        try:
            f_user = _EMBED_EXECUTOR.submit(embed, user_text)
            f_asst = _EMBED_EXECUTOR.submit(embed, assistant_text)
            return f_user.result(timeout=10), f_asst.result(timeout=10)
        except Exception as exc:  # noqa: BLE001 - logged + continue
            logger.warning("Embedding failed; vectors will be skipped: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    @staticmethod
    def _append_to_memory_manager(
        mm: Any,
        session_id: str,
        user_text: str,
        assistant_text: str,
        meta: Dict[str, Any],
    ) -> tuple[int, int]:
        """Add the user + assistant turns and return their turn indices.

        The ``MemoryManager.add_to_session`` API doesn't return the turn
        index, so we compute it from ``len(messages)`` after each call.
        """
        # User turn
        try:
            mm.add_to_session(session_id, "user", user_text, metadata=meta)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "add_to_session(user) failed for %s: %s", session_id, exc
            )

        user_idx = MemoryWriter._safe_message_count(mm, session_id) - 1

        # Assistant turn
        try:
            mm.add_to_session(session_id, "assistant", assistant_text, metadata=meta)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "add_to_session(assistant) failed for %s: %s", session_id, exc
            )

        asst_idx = MemoryWriter._safe_message_count(mm, session_id) - 1
        return max(user_idx, 0), max(asst_idx, 0)

    @staticmethod
    def _safe_message_count(mm: Any, session_id: str) -> int:
        try:
            session = mm.get_session(session_id)
            if session is None:
                return 0
            return len(getattr(session, "messages", []) or [])
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    def _write_vectors(
        self,
        session_id: str,
        user_text: str,
        user_vec: Optional[List[float]],
        user_idx: int,
        assistant_text: str,
        asst_vec: Optional[List[float]],
        asst_idx: int,
        meta: Dict[str, Any],
    ) -> None:
        if user_vec is None and asst_vec is None:
            return
        try:
            store = self._get_vector_store()
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector_store unavailable: %s", exc)
            return

        if user_vec is not None:
            try:
                store.add(session_id, user_idx, "user", user_text, user_vec, meta)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vector_store.add(user) failed: %s", exc)

        if asst_vec is not None:
            try:
                store.add(
                    session_id, asst_idx, "assistant", assistant_text, asst_vec, meta
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vector_store.add(assistant) failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4 — rolling summary
    # ------------------------------------------------------------------

    def _maybe_regenerate_summary(self, mm: Any, session_id: str) -> None:
        try:
            session = mm.get_session(session_id)
        except Exception:  # noqa: BLE001
            return
        if session is None:
            return

        user_turns = sum(
            1 for m in getattr(session, "messages", []) or []
            if getattr(m, "role", None) == "user"
        )
        if user_turns == 0 or user_turns % ROLLING_SUMMARY_INTERVAL != 0:
            return

        # Avoid two threads racing to regenerate the same summary.
        if not self._summary_lock.acquire(blocking=False):
            return
        try:
            self._regenerate_summary(mm, session_id)
        finally:
            self._summary_lock.release()

    def _regenerate_summary(self, mm: Any, session_id: str) -> None:
        try:
            session = mm.get_session(session_id)
            if session is None:
                return
            messages = getattr(session, "messages", []) or []
            recent = messages[-ROLLING_SUMMARY_TURN_WINDOW:]
            history: List[Dict[str, str]] = [
                {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                for m in recent
            ]
            if not history:
                return

            summary = self._call_summary_fn(history)
            if not summary:
                return
            self._persist_summary(mm, session_id, summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rolling summary regeneration failed for %s: %s", session_id, exc
            )

    def _call_summary_fn(self, history: List[Dict[str, str]]) -> str:
        if self._summary_fn is not None:
            return self._summary_fn(history) or ""
        return _default_summarize(history)

    @staticmethod
    def _persist_summary(mm: Any, session_id: str, summary: str) -> None:
        """Write the summary back onto ``ConversationContext``.

        Prefer a dedicated ``rolling_summary`` field if the dataclass
        supports it (newer build); otherwise fall back to
        ``custom_instructions`` (older build).
        """
        try:
            session = mm.get_session(session_id)
            if session is None:
                return
            ctx = getattr(session, "context", None)
            if ctx is None:
                return

            field_names = {f.name for f in dataclass_fields(ctx)} \
                if hasattr(ctx, "__dataclass_fields__") else set()

            if "rolling_summary" in field_names or hasattr(ctx, "rolling_summary"):
                try:
                    setattr(ctx, "rolling_summary", summary)
                    return
                except (AttributeError, TypeError):
                    pass

            # Fallback path — write to custom_instructions via the
            # public update_context helper so downstream persistence
            # (S3 / disk) picks it up.
            mm.update_context(session_id, custom_instructions=summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("persist_summary failed for %s: %s", session_id, exc)


# ---------------------------------------------------------------------------
# Default summariser (OpenAI gpt-4o-mini)
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You compress construction-domain RAG conversations. Summarise the "
    "transcript in <=200 words. Preserve every concrete entity: drawing "
    "names, project IDs, trade names, equipment names, document titles. "
    "Output prose only — no bullet lists."
)


def _default_summarize(history: List[Dict[str, str]]) -> str:
    """Call gpt-4o-mini with the rolling-summary prompt.

    Errors are swallowed by the caller; we just propagate them here so
    the caller can log a single coherent message.
    """
    try:
        from openai import OpenAI  # local import keeps test mocking simple
    except ImportError:  # pragma: no cover
        return ""

    transcript_lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = (msg.get("content", "") or "").strip()
        if content:
            transcript_lines.append(f"{role.upper()}: {content}")
    transcript = "\n".join(transcript_lines)
    if not transcript:
        return ""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=ROLLING_SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    choice = resp.choices[0] if resp.choices else None
    if choice is None or choice.message is None:
        return ""
    return (choice.message.content or "").strip()
