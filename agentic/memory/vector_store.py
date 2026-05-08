"""
agentic.memory.vector_store
===========================

Dual-layer vector store for conversation turns.

Layout
------
* **Durable layer** — Mongo Atlas collection
  ``<MONGO_DB>.session_turn_embeddings`` with one document per turn.
  Atlas Vector Search is queried via ``$vectorSearch`` with a hard
  ``maxTimeMS`` budget so a slow Atlas region can never block the user.

* **Hot layer** — local FAISS ``IndexFlatIP`` per session, kept beside the
  existing JSON session files at
  ``./conversation_sessions/embeddings/<session_id>.faiss`` plus a
  ``<session_id>.meta.jsonl`` sidecar that maps FAISS row index → turn
  metadata. FAISS handles ~5ms searches; we only fall back to Atlas when
  the local file is missing or too small.

Recovery
--------
On a FAISS miss we still answer the caller from Atlas, then kick off a
background thread to rehydrate the FAISS file from the Atlas results so
the next call is fast again. Failures in the background rebuild are
logged, never raised.

Public API
----------
``SessionVectorStore.add(...)``
    Persist a single turn to both layers.
``SessionVectorStore.search(...)``
    Top-k cosine similarity search (FAISS-first, Atlas fallback).
``SessionVectorStore.delete_session(...)``
    Remove all traces of a session from both layers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("agentic_rag.memory.vector_store")

# Lazy imports for faiss / pymongo so the module loads cleanly in tests
# even when those backends are mocked.
try:  # pragma: no cover - import-time guard
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

try:  # pragma: no cover
    from pymongo import ASCENDING, MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover
    MongoClient = None  # type: ignore
    PyMongoError = Exception  # type: ignore
    ASCENDING = 1

# ---------------------------------------------------------------------------
# Tunables / constants
# ---------------------------------------------------------------------------

EMBEDDING_DIMS = 1536
DEFAULT_DB = "iField"
COLLECTION_NAME = "session_turn_embeddings"
DEFAULT_TIMEOUT_MS = 150
DEFAULT_EMBEDDING_DIR = Path("./conversation_sessions/embeddings")
TEXT_EXCERPT_CHARS = 500
VECTOR_SEARCH_INDEX_NAME = os.getenv(
    "MEMORY_ATLAS_VECTOR_INDEX", "session_turn_embeddings_vec_idx"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _l2_normalize(vec: List[float]) -> np.ndarray:
    """Convert to a 2-D float32 row vector L2-normalized in place.

    FAISS ``IndexFlatIP`` returns inner-product scores; we normalise so
    that the inner product equals cosine similarity.
    """
    arr = np.asarray(vec, dtype="float32").reshape(1, -1)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


# ---------------------------------------------------------------------------
# SessionVectorStore
# ---------------------------------------------------------------------------


class SessionVectorStore:
    """Dual-layer (FAISS + Atlas) per-session embedding store.

    Construction is cheap; the Mongo client is created lazily on first
    use so unit tests that never touch Atlas don't pay the connection
    cost.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        atlas_timeout_ms: Optional[int] = None,
        embedding_dir: Optional[Path] = None,
        mongo_client: Optional[Any] = None,
    ) -> None:
        self.mongo_uri = mongo_uri or os.environ.get("MONGODB_URI", "")
        self.db_name = db_name or os.environ.get("MONGO_DB", DEFAULT_DB)
        self.atlas_timeout_ms = int(
            atlas_timeout_ms
            or os.environ.get("MEMORY_ATLAS_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)
        )
        self.embedding_dir = Path(embedding_dir or DEFAULT_EMBEDDING_DIR)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[Any] = mongo_client
        self._client_lock = threading.Lock()
        self._fs_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mongo client (lazy)
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Any]:
        if self._client is not None:
            return self._client
        if MongoClient is None or not self.mongo_uri:
            return None
        with self._client_lock:
            if self._client is None:
                try:
                    self._client = MongoClient(
                        self.mongo_uri,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=10000,
                        retryWrites=True,
                        retryReads=True,
                    )
                except PyMongoError as exc:
                    logger.warning("Mongo client init failed: %s", exc)
                    self._client = None
        return self._client

    def _collection(self) -> Optional[Any]:
        client = self._get_client()
        if client is None:
            return None
        try:
            return client[self.db_name][COLLECTION_NAME]
        except PyMongoError as exc:  # pragma: no cover - extremely unlikely
            logger.warning("Mongo collection lookup failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _faiss_path(self, session_id: str) -> Path:
        return self.embedding_dir / f"{session_id}.faiss"

    def _meta_path(self, session_id: str) -> Path:
        return self.embedding_dir / f"{session_id}.meta.jsonl"

    def _load_faiss_index(self, session_id: str):
        if faiss is None:
            return None
        path = self._faiss_path(session_id)
        if not path.exists():
            return None
        try:
            return faiss.read_index(str(path))
        except (RuntimeError, OSError) as exc:
            logger.warning("FAISS read failed for %s: %s", session_id, exc)
            return None

    def _write_faiss_index(self, session_id: str, index) -> None:
        if faiss is None or index is None:
            return
        path = self._faiss_path(session_id)
        try:
            faiss.write_index(index, str(path))
        except (RuntimeError, OSError) as exc:
            logger.warning("FAISS write failed for %s: %s", session_id, exc)

    def _read_meta(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._meta_path(session_id)
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("FAISS meta read failed for %s: %s", session_id, exc)
            return []
        return out

    def _append_meta(self, session_id: str, record: Dict[str, Any]) -> None:
        path = self._meta_path(session_id)
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("FAISS meta append failed for %s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Public: add
    # ------------------------------------------------------------------

    def add(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Persist a turn to Atlas and FAISS.

        Either layer can fail independently — we log and continue so a
        Mongo outage doesn't take down the local cache and vice versa.
        """
        if len(embedding) != EMBEDDING_DIMS:
            raise ValueError(
                f"Embedding dim {len(embedding)} != expected {EMBEDDING_DIMS}"
            )

        text_excerpt = (text or "")[:TEXT_EXCERPT_CHARS]
        doc = {
            "session_id": session_id,
            "turn_index": int(turn_index),
            "role": role,
            "text_excerpt": text_excerpt,
            "embedding": list(embedding),
            "created_at": _utc_now(),
            "metadata": metadata or {},
        }

        # --- Atlas write (durable) ---
        coll = self._collection()
        if coll is not None:
            try:
                coll.insert_one(doc)
            except PyMongoError as exc:
                logger.warning(
                    "Atlas insert failed for session=%s turn=%s: %s",
                    session_id, turn_index, exc,
                )

        # --- FAISS write (hot path) ---
        if faiss is None:
            logger.debug("faiss unavailable; skipping local write")
            return

        with self._fs_lock:
            try:
                index = self._load_faiss_index(session_id)
                if index is None:
                    index = faiss.IndexFlatIP(EMBEDDING_DIMS)
                vec = _l2_normalize(embedding)
                index.add(vec)
                self._write_faiss_index(session_id, index)
                self._append_meta(
                    session_id,
                    {
                        "turn_index": int(turn_index),
                        "role": role,
                        "text_excerpt": text_excerpt,
                        "metadata": metadata or {},
                    },
                )
            except (RuntimeError, OSError) as exc:
                logger.warning(
                    "FAISS local write failed for session=%s turn=%s: %s",
                    session_id, turn_index, exc,
                )

    # ------------------------------------------------------------------
    # Public: search
    # ------------------------------------------------------------------

    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 3,
        atlas_timeout_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Top-k semantic recall for a session.

        Try FAISS first (~5ms). If the local index is missing or has
        fewer than ``top_k`` rows, fall through to Atlas
        ``$vectorSearch`` with a hard ``maxTimeMS`` budget. On Atlas
        success after a FAISS miss, schedule a background rebuild of the
        local index so the next call is fast again.

        Returns an empty list (never raises) if both layers fail.
        """
        if len(query_embedding) != EMBEDDING_DIMS:
            logger.warning(
                "search rejected: query dim %d != %d",
                len(query_embedding), EMBEDDING_DIMS,
            )
            return []

        timeout_ms = int(atlas_timeout_ms or self.atlas_timeout_ms)

        # --- FAISS first ---
        faiss_results = self._search_faiss(session_id, query_embedding, top_k)
        if faiss_results is not None and len(faiss_results) >= top_k:
            return faiss_results[:top_k]

        # --- Atlas fallback ---
        atlas_results = self._search_atlas(
            session_id, query_embedding, top_k, timeout_ms
        )
        if atlas_results is None:
            # Both layers down — return whatever (possibly partial) FAISS
            # gave us, or empty list. Never raise.
            return faiss_results or []

        # Schedule async FAISS rehydrate so the next call hits hot path.
        if faiss is not None and atlas_results:
            try:
                threading.Thread(
                    target=self._rebuild_faiss_from_atlas,
                    args=(session_id, atlas_results),
                    daemon=True,
                ).start()
            except RuntimeError as exc:  # pragma: no cover - thread start
                logger.debug("FAISS rebuild thread failed to start: %s", exc)

        return atlas_results

    def _search_faiss(
        self, session_id: str, query_embedding: List[float], top_k: int
    ) -> Optional[List[Dict[str, Any]]]:
        if faiss is None:
            return None
        index = self._load_faiss_index(session_id)
        if index is None or index.ntotal == 0:
            return None
        meta = self._read_meta(session_id)
        try:
            qvec = _l2_normalize(query_embedding)
            k = min(top_k, index.ntotal)
            scores, ids = index.search(qvec, k)
        except (RuntimeError, ValueError) as exc:
            logger.warning("FAISS search failed for %s: %s", session_id, exc)
            return None

        results: List[Dict[str, Any]] = []
        for score, row in zip(scores[0].tolist(), ids[0].tolist()):
            if row < 0 or row >= len(meta):
                continue
            entry = meta[row]
            results.append(
                {
                    "turn_index": entry.get("turn_index"),
                    "role": entry.get("role"),
                    "text_excerpt": entry.get("text_excerpt", ""),
                    "score": float(score),
                    "metadata": entry.get("metadata", {}),
                }
            )
        return results

    def _search_atlas(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int,
        timeout_ms: int,
    ) -> Optional[List[Dict[str, Any]]]:
        coll = self._collection()
        if coll is None:
            return None

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_SEARCH_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": list(query_embedding),
                    "numCandidates": max(50, top_k * 10),
                    "limit": top_k,
                    "filter": {"session_id": session_id},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "turn_index": 1,
                    "role": 1,
                    "text_excerpt": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            cursor = coll.aggregate(pipeline, maxTimeMS=timeout_ms)
            results = list(cursor)
        except PyMongoError as exc:
            logger.warning(
                "Atlas $vectorSearch failed for %s: %s", session_id, exc
            )
            return None
        return results

    def _rebuild_faiss_from_atlas(
        self, session_id: str, atlas_results: List[Dict[str, Any]]
    ) -> None:
        """Best-effort rebuild of the local FAISS index from Atlas hits.

        Runs in a daemon thread; all errors are swallowed.
        """
        if faiss is None:
            return
        coll = self._collection()
        if coll is None:
            return
        try:
            # Fetch the full per-turn docs (with vectors) for this session.
            cursor = coll.find(
                {"session_id": session_id},
                {"_id": 0, "turn_index": 1, "role": 1, "text_excerpt": 1,
                 "embedding": 1, "metadata": 1},
                max_time_ms=2000,
            ).sort("turn_index", ASCENDING)
            docs = list(cursor)
        except PyMongoError as exc:
            logger.debug("FAISS rebuild fetch failed: %s", exc)
            return

        if not docs:
            return

        with self._fs_lock:
            try:
                index = faiss.IndexFlatIP(EMBEDDING_DIMS)
                meta_path = self._meta_path(session_id)
                # Truncate the meta sidecar before rewriting.
                if meta_path.exists():
                    meta_path.unlink()
                for doc in docs:
                    emb = doc.get("embedding")
                    if not emb or len(emb) != EMBEDDING_DIMS:
                        continue
                    index.add(_l2_normalize(emb))
                    self._append_meta(
                        session_id,
                        {
                            "turn_index": doc.get("turn_index"),
                            "role": doc.get("role"),
                            "text_excerpt": doc.get("text_excerpt", ""),
                            "metadata": doc.get("metadata", {}),
                        },
                    )
                self._write_faiss_index(session_id, index)
                logger.info(
                    "FAISS rebuilt for session=%s (%d turns)",
                    session_id, len(docs),
                )
            except (RuntimeError, OSError) as exc:
                logger.debug("FAISS rebuild write failed: %s", exc)

    # ------------------------------------------------------------------
    # Public: delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> None:
        """Remove a session's embeddings from both layers (best-effort)."""
        coll = self._collection()
        if coll is not None:
            try:
                coll.delete_many({"session_id": session_id})
            except PyMongoError as exc:
                logger.warning("Atlas delete failed for %s: %s", session_id, exc)

        for path in (self._faiss_path(session_id), self._meta_path(session_id)):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("FAISS file delete failed for %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Index bootstrap helpers
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        """Create the basic ``session_id`` btree index (idempotent).

        The Atlas Vector Search index has to be created via the Atlas UI
        or admin API — see ``IMPLEMENTATION_NOTES.md`` for the JSON spec.
        """
        coll = self._collection()
        if coll is None:
            return
        try:
            coll.create_index([("session_id", ASCENDING), ("turn_index", ASCENDING)])
        except PyMongoError as exc:
            logger.warning("ensure_indexes failed: %s", exc)
