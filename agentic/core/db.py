"""
MongoDB connection management with thread-safe connection pooling.
Single shared client for all tools — no duplicate clients.
"""

import logging
import threading
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config import MONGODB_URI, MONGO_DB

logger = logging.getLogger("agentic_rag.db")

_client: Optional[MongoClient] = None
_lock = threading.Lock()
_indexes_initialized = False


def get_client() -> MongoClient:
    """Get or create the shared MongoDB client (thread-safe, double-checked locking)."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            # NOTE: mongodb+srv:// already enforces TLS — no explicit tls=True needed
            _client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=20000,
                socketTimeoutMS=60000,
                maxPoolSize=20,
                minPoolSize=2,
                retryWrites=True,
                retryReads=True,
            )
            _client.admin.command("ping")
            logger.info("MongoDB connected (TLS enforced, pool=20)")
    return _client


def get_db() -> Database:
    """Get the application database."""
    return get_client()[MONGO_DB]


def get_collection(name: str) -> Collection:
    """Get a named collection from the application database."""
    return get_db()[name]


def ensure_indexes() -> None:
    """Create required indexes on all collections (call once at startup)."""
    global _indexes_initialized
    if _indexes_initialized:
        return

    with _lock:
        if _indexes_initialized:
            return

        _ensure_drawing_indexes()
        _ensure_spec_indexes()
        _indexes_initialized = True
        logger.info("All collection indexes verified")


def _ensure_drawing_indexes() -> None:
    """Create indexes on the legacy drawing collection (2.8M docs)."""
    coll = get_collection("drawing")
    existing = coll.index_information()

    indexes = [
        {"keys": [("projectId", 1), ("drawingId", 1)], "name": "project_drawing_idx"},
        {"keys": [("projectId", 1), ("setTrade", 1)], "name": "project_trade_idx"},
        {"keys": [("projectId", 1), ("trade", 1)], "name": "project_trade2_idx"},
        {"keys": [("projectId", 1), ("text", 1)], "name": "project_text_idx"},
        {"keys": [("projectId", 1), ("drawingTitle", 1), ("drawingName", 1)], "name": "project_title_name_idx"},
    ]

    for idx in indexes:
        if idx["name"] not in existing:
            try:
                coll.create_index(idx["keys"], name=idx["name"], background=True)
                logger.info(f"Created index {idx['name']} on drawing collection")
            except Exception as e:
                logger.warning(f"Index creation failed for {idx['name']}: {e}")


def _ensure_spec_indexes() -> None:
    """Create indexes on the specification collection."""
    coll = get_collection("specification")
    existing = coll.index_information()

    if "project_spec_idx" not in existing:
        try:
            coll.create_index(
                [("projectId", 1), ("specificationNumber", 1)],
                name="project_spec_idx",
            )
            logger.info("Created project_spec_idx on specification")
        except Exception:
            pass


def close() -> None:
    """Close the shared MongoDB client."""
    global _client, _indexes_initialized
    with _lock:
        if _client:
            _client.close()
            _client = None
            _indexes_initialized = False
            logger.info("MongoDB connection closed")
