"""Feedback attachment upload helpers (Phase 2 of the feedback feature).

Single responsibility: take a FastAPI UploadFile, validate it (size, MIME,
magic bytes), upload to S3 under the canonical per-turn path, and return
the metadata dict that gets embedded in the agent_feedback Mongo doc.

NO Mongo writes here — that's the caller's job (feedback_store.submit_feedback).
NO router logic — that's gateway/feedback_router.py.

Strictly additive: importing this module does NOT change any existing
behavior. It's only invoked from the new multipart handler.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from fastapi import HTTPException, UploadFile

logger = logging.getLogger("agentic_rag.feedback_attachments")


# -------- Configuration (env-tunable, sensible defaults) --------------------

MAX_FILE_BYTES         = 25 * 1024 * 1024       # 25 MB per file
MAX_FILES_PER_SUBMIT   = 5                       # 5 files max per POST
MAX_TOTAL_BYTES        = 125 * 1024 * 1024      # 125 MB total payload
PRESIGN_EXPIRES_SEC    = 7 * 24 * 3600           # 7 days
FILENAME_MAX_CHARS     = 120

# Canonical MIME allowlist. Magic bytes are checked separately against the
# declared content type to defeat "rename .exe to .pdf" tricks.
ALLOWED_MIME_TYPES = {
    # Documents
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.ms-excel",                                                  # XLS
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",         # XLSX
    "text/plain",
    "text/csv",
    # Images
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

# Magic byte signatures we explicitly accept. Empty tuple = skip magic check
# (used for text/* where there's no reliable magic).
MAGIC_SIGNATURES: Dict[str, Tuple[bytes, ...]] = {
    "application/pdf":   (b"%PDF-",),
    "image/png":         (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg":        (b"\xff\xd8\xff",),
    "image/gif":         (b"GIF87a", b"GIF89a"),
    "image/webp":        (b"RIFF",),                  # RIFF....WEBP — first 4 are RIFF
    "application/msword":                                              (b"\xd0\xcf\x11\xe0",),  # CFB header
    "application/vnd.ms-excel":                                        (b"\xd0\xcf\x11\xe0",),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  (b"PK\x03\x04",),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":        (b"PK\x03\x04",),
    "text/plain":        (),
    "text/csv":          (),
}

# Maps MIME to a coarse "kind" the UI uses for icon rendering.
MIME_TO_KIND = {
    "image/png":   "screenshot", "image/jpeg": "screenshot",
    "image/webp":  "screenshot", "image/gif":  "screenshot",
    "application/pdf":  "document",
    "application/msword": "document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "application/vnd.ms-excel": "document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "document",
    "text/plain": "document", "text/csv": "document",
}


# -------- S3 client (lazy) --------------------------------------------------

_s3 = None


def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _bucket() -> str:
    return os.environ.get("S3_BUCKET_NAME", "agentic-ai-production")


# -------- Filename sanitization --------------------------------------------

_FILENAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    """Strip path components, lowercase, replace runs of unsafe chars with '-'.
    Preserves the extension (last dot segment).
    """
    if not name:
        return "unnamed"
    base = os.path.basename(name).strip()
    # Replace unsafe runs
    safe = _FILENAME_SAFE_RE.sub("-", base).strip("-_.")
    safe = safe.lower()[:FILENAME_MAX_CHARS] or "unnamed"
    return safe


# -------- Validation -------------------------------------------------------

def _check_magic(data: bytes, content_type: str) -> bool:
    """Best-effort first-bytes signature check.

    Returns False ONLY when we have a known signature and it doesn't match.
    Unknown signatures (empty tuple) pass through — caller already validated
    the MIME is allowed.
    """
    sigs = MAGIC_SIGNATURES.get(content_type, ())
    if not sigs:
        return True
    # webp special case: first 4 bytes RIFF, bytes 8-12 'WEBP'
    if content_type == "image/webp":
        return data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP"
    return any(data.startswith(s) for s in sigs)


async def validate_uploads(files: List[UploadFile]) -> List[bytes]:
    """Validate each file. Returns the in-memory content bytes for each
    (so we can also compute sha256 + upload to S3 in one pass without
    re-reading the file).

    Raises HTTPException(422) on any failure — the whole submission is
    rejected, no S3 uploads happen.
    """
    if files is None:
        files = []
    if len(files) > MAX_FILES_PER_SUBMIT:
        raise HTTPException(
            status_code=422,
            detail=f"too many files: {len(files)} (max {MAX_FILES_PER_SUBMIT})",
        )

    contents: List[bytes] = []
    total_bytes = 0
    for idx, up in enumerate(files):
        if up is None:
            continue
        data = await up.read()
        # rewind so a later consumer could re-read (we won't, but be defensive)
        await up.seek(0)
        size = len(data)
        if size == 0:
            raise HTTPException(status_code=422, detail=f"file {idx}: empty upload")
        if size > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"file {idx}: size {size // (1024*1024)} MB exceeds {MAX_FILE_BYTES // (1024*1024)} MB limit",
            )
        total_bytes += size
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"total payload exceeds {MAX_TOTAL_BYTES // (1024*1024)} MB",
            )
        ctype = (up.content_type or "").lower().strip()
        if ctype not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"file {idx}: type {ctype!r} not allowed (allowed: {sorted(ALLOWED_MIME_TYPES)})",
            )
        if not _check_magic(data[:64], ctype):
            raise HTTPException(
                status_code=422,
                detail=f"file {idx}: magic bytes don't match declared type {ctype!r}",
            )
        contents.append(data)
    return contents


# -------- Path builder -----------------------------------------------------

def build_attachment_key(
    *,
    user_id: int,
    project_id: int,
    agent_id: str,
    session_id: str,
    turn_id: str,
    feedback_id: str,
    original_filename: str,
) -> str:
    """Path layout (locked):
        sessions/{user_id}/{project_id}/{agent_id}/{session_id}
            /feedback/{turn_id}/{feedback_id}/{<salt>__<safe-name>}
    """
    safe = _sanitize_filename(original_filename)
    salt = uuid.uuid4().hex[:8]
    return (
        f"sessions/{int(user_id)}/{int(project_id)}/{agent_id}/{session_id}"
        f"/feedback/{turn_id}/{feedback_id}/{salt}__{safe}"
    )


# -------- Upload + metadata -----------------------------------------------

def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def upload_attachment(
    *,
    data: bytes,
    content_type: str,
    original_filename: str,
    user_id: int,
    project_id: int,
    agent_id: str,
    session_id: str,
    turn_id: str,
    feedback_id: str,
    user_id_for_record: Optional[int] = None,
) -> Dict[str, Any]:
    """Upload one file's bytes to S3 and return the metadata dict that
    should be appended to the agent_feedback doc's attachments array.

    The caller has already validated `data` (size + magic). This function
    only does the S3 put + sign + metadata composition.
    """
    bucket = _bucket()
    key = build_attachment_key(
        user_id=user_id, project_id=project_id, agent_id=agent_id,
        session_id=session_id, turn_id=turn_id, feedback_id=feedback_id,
        original_filename=original_filename,
    )
    s3 = _get_s3()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        ServerSideEncryption="AES256",
        Metadata={
            "user-id":    str(user_id),
            "project-id": str(project_id),
            "agent-id":   agent_id,
            "session-id": session_id[:128],
            "turn-id":    turn_id[:128],
            "feedback-id": feedback_id,
        },
    )

    presigned = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGN_EXPIRES_SEC,
    )

    now = datetime.now(timezone.utc)
    return {
        "attachment_id":        f"att_{uuid.uuid4().hex[:8]}",
        "s3_bucket":            bucket,
        "s3_key":               key,
        "presigned_url":        presigned,
        "presigned_expires_at": int(now.timestamp()) + PRESIGN_EXPIRES_SEC,
        "original_filename":    original_filename,
        "content_type":         content_type,
        "size_bytes":           len(data),
        "sha256":               _sha256(data),
        "kind":                 MIME_TO_KIND.get(content_type, "other"),
        "uploaded_at":          now,
        "uploaded_by_user_id":  user_id_for_record if user_id_for_record is not None else user_id,
        "is_pii":               True,  # tag for Phase-3 encryption migration
    }


def refresh_presigned_url(attachment: Dict[str, Any]) -> Dict[str, Any]:
    """Re-sign the URL if it's within 1 day of expiry. Returns the
    (possibly mutated) attachment dict.
    """
    try:
        exp = int(attachment.get("presigned_expires_at") or 0)
        now = int(datetime.now(timezone.utc).timestamp())
        if exp - now > 86400:                      # >1 day left, keep as-is
            return attachment
        s3 = _get_s3()
        new_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": attachment["s3_bucket"], "Key": attachment["s3_key"]},
            ExpiresIn=PRESIGN_EXPIRES_SEC,
        )
        attachment["presigned_url"] = new_url
        attachment["presigned_expires_at"] = now + PRESIGN_EXPIRES_SEC
        return attachment
    except Exception as exc:                       # noqa: BLE001
        logger.warning("refresh_presigned_url failed for %s: %s",
                       attachment.get("s3_key"), exc)
        return attachment
