#!/usr/bin/env python3
"""
Chunk and embed email attachments: email_attachment2 → email_attachment_chunks.

Flow
----
1. Read email_attachment2 docs that have extracted_text.
2. Chunk each attachment into ~900-token segments with ~125-token overlap (cl100k_base).
3. Embed each chunk with text-embedding-3-small (batched, 100 chunks/call).
4. Write chunks to email_attachment_chunks collection.
5. Mark source doc with chunked_at so re-runs are idempotent.

Schema written to email_attachment_chunks
------------------------------------------
  attachment_id   : ObjectId  — source doc _id from email_attachment2
  email_id        : any       — email_attachment2.email_id (if present)
  project_id      : int       — Atlas pre-filter field
  thread_id       : str|None  — for thread-scoped queries
  filename        : str
  chunk_index     : int       — 0-based index within this attachment
  chunk_text      : str
  token_count     : int
  embedding       : [float]   — 1536 dims, text-embedding-3-small
  embedding_model : str
  embedded_at     : datetime

Atlas Vector Index (create in Atlas UI → Search → Create Index → JSON editor)
-------------------------------------------------------------------------------
  Collection : email_attachment_chunks
  Index name : email_attachment_chunks_embeddings
  {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "project_id"
      }
    ]
  }

Usage
-----
    # from unified-rag-agent/ directory
    python scripts/ingest_attachment_chunks.py
    python scripts/ingest_attachment_chunks.py --force
    python scripts/ingest_attachment_chunks.py --dry-run

Environment variables
---------------------
    MONGODB_URI      MongoDB Atlas connection string
    OPENAI_API_KEY   OpenAI API key
    MONGO_DB         database name  (default: iField)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterator

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# SOURCE_COLLECTION = "email_attachments2"
# EMAIL_COLLECTION = "email_agent2"   #(Sandbox)    # joined to resolve project_id + thread_id
EMAIL_COLLECTION = "email_agent3"   ##(Production)    # joined to resolve project_id + thread_id
# CHUNK_COLLECTION = "email_attachment2_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 900
OVERLAP_TOKENS = 125
EMBED_BATCH = 100

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _build_email_lookup(email_col) -> dict:
    """Build map: str(email_id) → {project_id, thread_id} from email_agent2. (Sandbox)"""
    lookup = {}
    for doc in email_col.find({}, {"_id": 1, "project": 1, "thread_id": 1}):
        pid = None
        project = doc.get("project")
        if isinstance(project, dict):
            try:
                pid = int(project.get("id", 0)) or None
            except (TypeError, ValueError):
                pass
        lookup[str(doc["_id"])] = {
            "project_id": pid,
            "thread_id": doc.get("thread_id"),
        }
    return lookup


def _extract_project_id(doc: dict) -> int | None:
    if "project_id" in doc:
        try:
            return int(doc["project_id"])
        except (TypeError, ValueError):
            pass
    project = doc.get("project")
    if isinstance(project, dict):
        try:
            return int(project.get("id", 0))
        except (TypeError, ValueError):
            pass
    return None


def _chunk_text(text: str) -> list[tuple[str, int]]:
    """Split text into (chunk_text, token_count) pairs with overlap."""
    tokens = _tokenizer.encode(text)
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append((_tokenizer.decode(chunk_tokens), len(chunk_tokens)))
        if end == len(tokens):
            break
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


def _iter_attachments(collection, force: bool) -> Iterator[dict]:
    query = {"extracted_text": {"$exists": True, "$ne": ""}}
    if not force:
        query["chunked_at"] = {"$exists": False}
    yield from collection.find(query, {
        "_id": 1,
        "email_id": 1,
        "thread_id": 1,
        "filename": 1,
        "extracted_text": 1,
        "project_id": 1,
        "project": 1,
    })


def ingest(mongo_uri: str, db_name: str, force: bool = False, dry_run: bool = False) -> None:
    logger.info("Connecting to MongoDB Atlas…")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=15_000)
    client.admin.command("ping")
    db = client[db_name]
    source_col = db[SOURCE_COLLECTION]
    chunk_col = db[CHUNK_COLLECTION]

    count_query = {"extracted_text": {"$exists": True, "$ne": ""}}
    total_all = source_col.count_documents(count_query)
    need_query = count_query if force else {**count_query, "chunked_at": {"$exists": False}}
    total_need = source_col.count_documents(need_query)
    logger.info(
        "Source: %s | has_text=%d | need_chunking=%d",
        SOURCE_COLLECTION, total_all, total_need,
    )

    if dry_run:
        sample = list(source_col.find(count_query, {"_id": 1, "filename": 1, "extracted_text": 1}, limit=3))
        for i, doc in enumerate(sample, 1):
            text = doc.get("extracted_text") or ""
            chunks = _chunk_text(text)
            logger.info(
                "--- sample %d: %s | len=%d chars | %d chunks ---\n%s…",
                i, doc.get("filename", "?"), len(text), len(chunks), text[:200],
            )
        logger.info("Dry run complete.")
        return

    if total_need == 0:
        logger.info("All attachments already chunked. Use --force to re-chunk.")
        return

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    email_lookup = _build_email_lookup(db[EMAIL_COLLECTION])
    logger.info("Email lookup built: %d emails indexed", len(email_lookup))

    pending_chunks: list[dict] = []
    pending_texts: list[str] = []
    pending_source_ids: list = []
    inserted_chunks = 0
    processed = 0
    t0 = time.time()

    def _flush() -> None:
        nonlocal inserted_chunks
        if not pending_chunks:
            return
        try:
            response = oai.embeddings.create(model=EMBEDDING_MODEL, input=pending_texts)
        except Exception as exc:
            logger.error("OpenAI embeddings call failed: %s — skipping batch", exc)
            pending_chunks.clear()
            pending_texts.clear()
            pending_source_ids.clear()
            return

        now = datetime.now(timezone.utc)
        for chunk_doc, emb_obj in zip(pending_chunks, response.data):
            chunk_doc["embedding"] = emb_obj.embedding
            chunk_doc["embedding_model"] = EMBEDDING_MODEL
            chunk_doc["embedded_at"] = now

        chunk_col.insert_many(pending_chunks, ordered=False)
        inserted_chunks += len(pending_chunks)

        unique_ids = list(set(pending_source_ids))
        source_col.update_many(
            {"_id": {"$in": unique_ids}},
            {"$set": {"chunked_at": now}},
        )

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(
            "Progress: %d/%d attachments, %d chunks — %.1f att/s",
            processed, total_need, inserted_chunks, rate,
        )

        pending_chunks.clear()
        pending_texts.clear()
        pending_source_ids.clear()

    for doc in _iter_attachments(source_col, force):
        text = (doc.get("extracted_text") or "").strip()
        if not text:
            continue

        attachment_id = doc["_id"]

        if force:
            chunk_col.delete_many({"attachment_id": attachment_id})

        # Resolve project_id + thread_id via email join (attachment has no project field)
        email_meta = email_lookup.get(str(doc.get("email_id", "")), {})
        project_id = email_meta.get("project_id") or _extract_project_id(doc)
        thread_id = doc.get("thread_id") or email_meta.get("thread_id")
        chunks = _chunk_text(text)

        for idx, (chunk_text, token_count) in enumerate(chunks):
            pending_chunks.append({
                "attachment_id": attachment_id,
                "email_id": doc.get("email_id"),
                "project_id": project_id,
                "thread_id": thread_id,
                "filename": doc.get("filename", ""),
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "token_count": token_count,
            })
            pending_texts.append(chunk_text)
            pending_source_ids.append(attachment_id)

            if len(pending_chunks) >= EMBED_BATCH:
                _flush()

        processed += 1

    _flush()

    elapsed = time.time() - t0
    logger.info(
        "Done. attachments=%d chunks=%d elapsed=%.1fs",
        processed, inserted_chunks, elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk and embed email attachments into MongoDB")
    parser.add_argument("--force", action="store_true", help="Re-chunk attachments already processed")
    parser.add_argument("--dry-run", action="store_true", help="Print sample info and exit without writing")
    args = parser.parse_args()

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        logger.error("MONGODB_URI not set")
        sys.exit(1)

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    db_name = os.environ.get("MONGO_DB", "iField")
    ingest(mongo_uri, db_name, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
