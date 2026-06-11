#!/usr/bin/env python3
"""
Ingest embeddings for the email_agent3 MongoDB collection.

Generates an "embedding" field on each document using the text:
    "{subject}\\n{summary}\\n{bullet_1}\\n{bullet_2}\\n{bullet_3}"

og_body is intentionally excluded - it contains noise (quoted threads,
signatures, legal disclaimers) that degrades retrieval quality.

The script is idempotent: documents that already have an "embedding"
field are skipped unless --force is passed.

Usage
-----
    # from the unified-rag-agent/ directory
    python scripts/ingest_email_embeddings.py

    # re-embed everything (e.g. after changing the embedding model)
    python scripts/ingest_email_embeddings.py --force

    # dry run - count docs, print first 3 texts, exit
    python scripts/ingest_email_embeddings.py --dry-run

Environment variables required
-------------------------------
    MONGODB_URI         MongoDB Atlas connection string
    OPENAI_API_KEY      OpenAI API key
    MONGO_DB            database name          (default: iField)
    EMAIL_PROJECT_FIELD field name for project filter (default: project.id)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne

# Load .env from the parent directory (unified-rag-agent/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# COLLECTION: str = os.environ.get("EMAIL_COLLECTION", "email_agent2")        ## Sandbox
COLLECTION: str = os.environ.get("EMAIL_COLLECTION", "email_agent3")      ## Production
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
BATCH_SIZE = 100        # docs per OpenAI embeddings call (max 2048, 100 is safe)
WRITE_BATCH = 500       # bulk_write batch size


def _extract_body_text(doc: dict, max_chars: int = 2000) -> str:
    body = (doc.get("body") or "").strip()
    if not body:
        return ""
    for sep in ["________________________________", "-----Original Message-----"]:
        idx = body.find(sep)
        if idx < 0:
            continue
        if idx > 100:
            # Separator mid-body: strip quoted reply tail
            body = body[:idx].strip()
        else:
            # Separator at start: whole body is forwarded/quoted.
            # Skip past headers (From/To/Cc/Subject) to actual content.
            subj_idx = body.find("Subject:")
            if subj_idx != -1:
                newline = body.find("\n", subj_idx)
                body = body[newline:].strip() if newline != -1 else body[subj_idx + 80:].strip()
            else:
                body = body[idx + len(sep):].strip()
        break
    return body[:max_chars]


def _build_text(doc: dict) -> str:
    """Build the text to embed for a single email document."""
    subject = (doc.get("subject") or "").strip()
    ai = doc.get("ai_analysis") or {}
    summary = (ai.get("summary") or "").strip()
    bullets = ai.get("bullets") or []
    actions = ai.get("actions") or []
    bullet_text = "\n".join(str(b).strip() for b in bullets if b)
    action_text = "\n".join(str(a).strip() for a in actions if a)
    body_text = _extract_body_text(doc)
    parts = [p for p in [subject, summary, bullet_text, action_text, body_text] if p]
    return "\n".join(parts)


def _iter_docs(collection, force: bool) -> Iterator[dict]:
    """Yield docs that need embedding."""
    query = {} if force else {"embedding": {"$exists": False}}
    yield from collection.find(query, {
        "_id": 1,
        "subject": 1,
        "ai_analysis": 1,
        "body": 1,
    })


def ingest(mongo_uri: str, db_name: str, force: bool = False, dry_run: bool = False) -> None:
    logger.info("Connecting to MongoDB Atlas...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=15_000)
    client.admin.command("ping")
    collection = client[db_name][COLLECTION]

    total_need = collection.count_documents({} if force else {"embedding": {"$exists": False}})
    total_all = collection.count_documents({})
    logger.info(
        "Collection: %s | total=%d | need embedding=%d",
        COLLECTION, total_all, total_need,
    )

    if dry_run:
        sample = list(collection.find(
            {} if force else {"embedding": {"$exists": False}},
            {"_id": 1, "subject": 1, "ai_analysis": 1},
            limit=3,
        ))
        for i, doc in enumerate(sample, 1):
            logger.info("--- sample %d ---\n%s", i, _build_text(doc))
        logger.info("Dry run complete. Exiting.")
        return

    if total_need == 0:
        logger.info("All documents already have embeddings. Use --force to re-embed.")
        return

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_docs: list[dict] = []
    batch_texts: list[str] = []
    writes: list[UpdateOne] = []
    processed = 0
    skipped = 0
    t0 = time.time()

    def _flush_batch() -> None:
        nonlocal processed
        if not batch_docs:
            return
        try:
            response = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        except Exception as exc:
            logger.error("OpenAI embeddings call failed: %s - skipping batch", exc)
            batch_docs.clear()
            batch_texts.clear()
            return

        now = datetime.now(timezone.utc)
        for doc, emb_obj in zip(batch_docs, response.data):
            writes.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {
                    "embedding": emb_obj.embedding,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedded_at": now,
                }},
            ))
        processed += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

        if len(writes) >= WRITE_BATCH:
            _flush_writes()

    def _flush_writes() -> None:
        if not writes:
            return
        collection.bulk_write(writes, ordered=False)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(
            "Progress: %d/%d (%.0f%%) - %.1f docs/s",
            processed, total_need, processed / total_need * 100, rate,
        )
        writes.clear()

    for doc in _iter_docs(collection, force):
        text = _build_text(doc)
        if not text:
            skipped += 1
            continue

        batch_docs.append(doc)
        batch_texts.append(text)

        if len(batch_docs) >= BATCH_SIZE:
            _flush_batch()

    _flush_batch()
    _flush_writes()

    elapsed = time.time() - t0
    logger.info(
        "Done. embedded=%d skipped(empty)=%d elapsed=%.1fs",
        processed, skipped, elapsed,
    )


def watch(mongo_uri: str, db_name: str) -> None:
    """Watch email_agent3 for new inserts and embed them in real-time."""
    logger.info("Connecting to MongoDB Atlas (watch mode)...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=15_000)
    client.admin.command("ping")
    collection = client[db_name][COLLECTION]
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Backfill any docs that arrived while this watcher was down
    backfill = collection.count_documents({"embedding": {"$exists": False}})
    if backfill:
        logger.info("Backfilling %d un-embedded docs before watching...", backfill)
        ingest(mongo_uri, db_name)

    logger.info("Watching %s for new inserts...", COLLECTION)
    pipeline = [{"$match": {"operationType": "insert"}}]

    with collection.watch(pipeline, full_document="updateLookup") as stream:
        for change in stream:
            doc = change.get("fullDocument") or {}
            if "embedding" in doc:
                continue

            text = _build_text(doc)
            if not text:
                logger.debug("Skipping doc %s - empty text", doc.get("_id"))
                continue

            try:
                response = oai.embeddings.create(model=EMBEDDING_MODEL, input=text)
                emb = response.data[0].embedding
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "embedding": emb,
                        "embedding_model": EMBEDDING_MODEL,
                        "embedded_at": datetime.now(timezone.utc),
                    }},
                )
                logger.info("Embedded new doc: %s", (doc.get("subject") or "")[:80])
            except Exception as exc:
                logger.error("Failed to embed doc %s: %s", doc.get("_id"), exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest email embeddings into MongoDB Atlas")
    parser.add_argument("--force", action="store_true", help="Re-embed docs that already have embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Print sample texts and exit without writing")
    parser.add_argument("--watch", action="store_true", help="Watch for new inserts and embed in real-time")
    args = parser.parse_args()

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        logger.error("MONGODB_URI environment variable is not set")
        sys.exit(1)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key and not args.dry_run:
        logger.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    db_name = os.environ.get("MONGO_DB", "iField")

    if args.watch:
        watch(mongo_uri, db_name)
    else:
        ingest(mongo_uri, db_name, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
