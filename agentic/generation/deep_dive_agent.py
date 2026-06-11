"""
agentic.generation.deep_dive_agent
==================================

Deep Dive orchestrator (v3.3).

Given a previously-answered (Q, A) turn and a session history, this module
performs a vision-grounded re-analysis by:

1. Loading the parent turn (question, answer, source_documents) from
   session memory — NO fresh DB hits per the Deep Dive contract.
2. Picking the cited PNG URLs from source_documents (already presigned
   by the v3.2 orchestrator).
3. Loading optional conversation history (rolling summary + last N turns).
4. Building a Deep Dive prompt that frames the vision call as a
   secondary-pass review of the prior text-RAG answer.
5. Streaming the vision response back to the caller through the
   ``vision_client`` multi-provider façade.
6. Persisting the enhanced turn back into session memory with
   ``turn_type="deep_dive"`` + ``parent_turn_id=<rag_turn_id>``.
7. Writing an audit row to the ``deep_dive_audits_projectqna_rag`` Mongo
   collection (separate write-path — does not interrupt the response).

Schema-preserving design:
* Response payload is the SAME shape as the normal /query response.
  Only difference is ``turn_type="deep_dive"`` and a populated
  ``parent_turn_id``.
* No existing field changes type or semantics.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

logger = logging.getLogger("agentic_rag.deep_dive_agent")


# ---------------------------------------------------------------------------
# Config / feature flag
# ---------------------------------------------------------------------------


def _flag_enabled(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Master kill-switch for Deep Dive — must be true for the endpoint to serve."""
    return _flag_enabled("DEEP_DIVE_ENABLED", "false")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class DeepDiveResult:
    """Outcome of a Deep Dive call.

    Mirrors the same response shape as /query so the gateway can return
    it through the existing response envelope.
    """

    deep_dive_id: str
    turn_id: str
    parent_turn_id: str
    session_id: Optional[str]
    project_id: Optional[int]
    original_question: str
    initial_rag_answer: str
    deep_dive_answer: str
    source_documents: List[Dict[str, Any]] = field(default_factory=list)
    image_urls_used: List[str] = field(default_factory=list)
    model_used: str = ""
    provider: str = ""
    fallback_used: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    processing_ms: int = 0
    status: str = "completed"  # "completed" | "failed" | "streaming_aborted"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


DEEP_DIVE_SYSTEM_PROMPT = """You are a senior construction document analyst performing a Deep Dive review of a prior AI answer.

You have been given:
  (a) The original user question
  (b) The AI's first answer (text-only RAG — may have OCR gaps or imprecision)
  (c) The actual page images of every cited document (attached above the text)
  (d) Optional prior conversation history

================================================================
ABSOLUTE RULES — NON-NEGOTIABLE
================================================================
A. **VERBATIM ONLY.** When transcribing notes, labels, dimensions, elevations, sizes, or any specific text from the drawing, you MUST quote the exact wording, numbers, and units as visible on the page. Do not paraphrase. Do not infer. Do not "fill in" plausible content.

B. **NO FABRICATION.** If you cannot clearly read a specific note, value, dimension, or label, you MUST say so explicitly using the exact phrase: "I cannot clearly read [what] in the attached image at this resolution." Then state what would help — a higher-resolution image, a different sheet, etc.

C. **NO TRAINING-DATA FALLBACK.** Do NOT supplement with "typical construction practice" or "standard plumbing/electrical/structural conventions." Only state what is visibly on THIS specific page. If you find yourself writing a sentence that could apply to any drawing of this type, DELETE IT — it's hallucination.

D. **COUNT IS A CLAIM.** "There are 5 general notes" is a claim that requires you to actually count what is visible. If you cannot count them with certainty, say "I see at least N notes but cannot determine the total" — don't invent a number.

E. **CITATION = EVIDENCE.** Every fact must include [Source: <pdf_name> p<page>] AND a short description of where on the page it is (e.g. "in the upper-right general notes block", "in detail 7 LEGEND SANITARY RISER bottom-right"). If you cannot point to a location, the fact is not grounded — omit it.

F. **PRIOR ANSWER ≠ GROUND TRUTH.** The earlier AI answer may itself contain hallucinations. Do NOT treat its specific values as confirmed. Re-verify EVERY value against the image. If the prior answer says "X = 6 inches" and you cannot see that on the page, do not echo it — correct the prior answer or say you cannot verify it.

================================================================
YOUR JOB
================================================================
1. Examine the actual page images carefully — read visible text in tables, annotations, schedules, dimensions, callouts, and legend blocks.
2. Compare what you SEE on the page against what the prior answer CLAIMED. Flag disagreements explicitly.
3. Produce an enhanced answer that is:
   - VERBATIM where text appears (quote it exactly)
   - SPECIFIC about page locations (which region of the sheet)
   - HONEST about what you cannot read clearly
   - CORRECTIVE when the prior answer was wrong (state the correction)
4. Start your response with "**Deep Dive Analysis:**" so the UI can label it.
"""


DEEP_DIVE_FOLLOWUP_SYSTEM_PROMPT = """You are a senior construction document analyst answering a follow-up question in an ongoing Deep Dive conversation.

You have been given:
  (a) Context from earlier in this session — an earlier question and AI answer (background only, NOT the thing you are reviewing)
  (b) The actual page images of the documents the user picked
  (c) A NEW question from the user — this is what you must answer
  (d) Optional prior conversation history

================================================================
ABSOLUTE RULES — NON-NEGOTIABLE
================================================================
A. **VERBATIM ONLY.** When stating specific notes, labels, dimensions, sizes, elevations, or any text-based value from the drawing, you MUST quote it exactly as it appears on the page. Do not paraphrase. Do not infer. Do not substitute synonyms.

B. **NO FABRICATION.** If you cannot clearly read what the new question is asking about, say so explicitly: "I cannot clearly read [what] in the attached image at this resolution." Suggest what would help (higher-resolution sheet, a different sheet, the unit isometric, etc.).

C. **NO TRAINING-DATA FALLBACK.** Do NOT supplement with generic construction knowledge ("typical practice", "industry standard", "usually X"). Only state what you can verify on THIS specific page. Drop any sentence that could apply to any drawing of this type.

D. **COUNT IS A CLAIM.** Don't invent quantities (DFU values, fixture counts, note counts). If you cannot count with certainty, say "at least N visible" or "exact count not legible at this resolution".

E. **CITATION = LOCATION.** Every fact must include [Source: <pdf_name>] AND a short description of WHERE on the page it appears (e.g. "in the title-block keynotes", "in detail 7 lower-right legend", "in the riser diagram column"). Vague citations are not citations.

F. **EARLIER CONTEXT ≠ GROUND TRUTH.** The earlier Q/A may contain errors. Do NOT treat earlier-stated values as confirmed. Re-verify everything against the image.

================================================================
YOUR JOB
================================================================
1. Answer the user's NEW question directly. Do NOT re-summarize the earlier answer.
2. Read the attached page images for content that bears on the NEW question.
3. State VERBATIM what is visible; state HONESTLY what is not legible.
4. If the new question cannot be answered from the attached pages, say so in 1-2 sentences and suggest what would help.
5. Reference earlier conversation ONLY for pronouns/references ("that detail").
6. Start your response with "**Deep Dive Analysis:**" so the UI can label it.
"""


def _build_user_prompt(
    *,
    original_question: str,
    initial_answer: str,
    source_documents: List[Dict[str, Any]],
    rolling_summary: Optional[str],
    followup_question: Optional[str] = None,
) -> str:
    """Assemble the per-call user prompt body.

    Two modes:
      * **Re-analyze mode** (followup_question is None) — caller is doing
        the original Deep Dive workflow: verify the prior text-RAG answer
        against the actual page images.
      * **Follow-up mode** (followup_question is set) — caller is asking
        a NEW question on the same picked documents within a multi-turn
        Deep Dive conversation. The original question + prior answer are
        included only as context; the new question is what the model
        should answer.
    """
    lines: List[str] = []
    if followup_question:
        lines.append("CONVERSATION CONTEXT — earlier in this same session:")
        lines.append(f"  Earlier question:   {original_question.strip()}")
        lines.append(f"  Earlier AI answer:  {initial_answer.strip()[:600]}")
        if rolling_summary:
            lines.append(f"  Rolling summary:    {rolling_summary.strip()}")
        lines.append("")
        lines.append("NEW QUESTION TO ANSWER (on the same picked documents):")
        lines.append(followup_question.strip())
    else:
        lines.append("ORIGINAL QUESTION:")
        lines.append(original_question.strip())
        lines.append("")
        lines.append("PRIOR AI ANSWER (text-RAG only — may have OCR gaps):")
        lines.append(initial_answer.strip())
        if rolling_summary:
            lines.append("")
            lines.append("CONVERSATION CONTEXT (summary of prior thread):")
            lines.append(rolling_summary.strip())

    lines.append("")
    lines.append("PICKED DOCUMENTS (images attached above, in this order):")
    for i, sd in enumerate(source_documents, start=1):
        label = (
            sd.get("display_title")
            or sd.get("drawing_name")
            or sd.get("drawing_title")
            or sd.get("pdf_name")
            or sd.get("file_name")
            or f"source_{i}"
        )
        page = sd.get("page")
        pdf = sd.get("pdf_name") or sd.get("file_name") or ""
        lines.append(f"  {i}. {label} — {pdf} (page {page})")

    lines.append("")
    if followup_question:
        lines.append(
            "Answer the NEW QUESTION based ONLY on what you can see in the "
            "attached page images. Reference earlier conversation context when "
            "the user uses pronouns or refers back to prior discussion. Cite "
            "every fact in [Source: <pdf_name>] format."
        )
    else:
        lines.append(
            "Now, with the actual page images in hand, produce the Deep Dive "
            "Analysis. Be specific about what you see on each page and cite "
            "every fact."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parent-turn loader — memory cache only, NO fresh DB hits
# ---------------------------------------------------------------------------


def _load_parent_turn(
    session_id: str,
    turn_id: str,
) -> Optional[Dict[str, Any]]:
    """Look up the (question, answer, source_documents) by turn_id.

    Reads the session from MemoryManager (in-process cache) — does NOT
    hit MongoDB. This is the "no fresh DB hits during Deep Dive" rule.

    Returns ``None`` when the turn cannot be located (caller surfaces 404).
    """
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
    except ImportError:
        try:
            from shared.session import get_memory_manager  # type: ignore
        except ImportError as exc:
            logger.error("deep_dive: no memory_manager importable: %s", exc)
            return None

    try:
        mm = get_memory_manager()
        session = mm.get_session(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("deep_dive: get_session(%s) failed: %s", session_id, exc)
        return None
    if session is None:
        return None

    # Walk messages from the back to find the assistant message whose
    # metadata.turn_id matches. The previous message in the list is the
    # user question that paired with it.
    messages = getattr(session, "messages", []) or []
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        meta = getattr(msg, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        if meta.get("turn_id") == turn_id and getattr(msg, "role", None) == "assistant":
            # Find the most recent user message before this one.
            user_text = ""
            for back in range(idx - 1, -1, -1):
                pmsg = messages[back]
                if getattr(pmsg, "role", None) == "user":
                    user_text = getattr(pmsg, "content", "") or ""
                    break
            return {
                "user_text": user_text,
                "assistant_text": getattr(msg, "content", "") or "",
                "source_documents": meta.get("source_documents", []) or [],
                "project_id": meta.get("project_id"),
                "set_id": meta.get("set_id"),
                "metadata": meta,
            }
    return None


# ---------------------------------------------------------------------------
# Rolling-summary loader — also memory cache only
# ---------------------------------------------------------------------------


def _resolve_most_recent_turn(session_id: str) -> Optional[str]:
    """Return the turn_id of the most recent assistant turn in a session.

    Supports multi-turn Deep Dive conversations where the UI doesn't
    track turn_id between calls. We walk messages from the back and
    return the first turn_id we find in metadata.
    """
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
    except ImportError:
        try:
            from shared.session import get_memory_manager  # type: ignore
        except ImportError:
            return None
    try:
        mm = get_memory_manager()
        sess = mm.get_session(session_id)
        if sess is None:
            return None
        messages = getattr(sess, "messages", []) or []
        for msg in reversed(messages):
            meta = getattr(msg, "metadata", None) or {}
            if isinstance(meta, dict) and meta.get("turn_id"):
                if getattr(msg, "role", None) == "assistant":
                    return meta["turn_id"]
        return None
    except Exception:  # noqa: BLE001
        return None


def _find_last_deep_dive_selection(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """Find the source_documents from the most recent Deep Dive turn.

    Walks back through session messages looking for an assistant turn
    whose metadata says ``turn_type == "deep_dive"``. Returns its
    persisted source_documents (the docs the user picked last time)
    so a follow-up Deep Dive call can reuse the same selection
    without the UI re-sending it.
    """
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
    except ImportError:
        try:
            from shared.session import get_memory_manager  # type: ignore
        except ImportError:
            return None
    try:
        mm = get_memory_manager()
        sess = mm.get_session(session_id)
        if sess is None:
            return None
        messages = getattr(sess, "messages", []) or []
        for msg in reversed(messages):
            meta = getattr(msg, "metadata", None) or {}
            if not isinstance(meta, dict):
                continue
            if meta.get("turn_type") == "deep_dive" and getattr(msg, "role", None) == "assistant":
                docs = meta.get("source_documents")
                if docs:
                    return docs
        return None
    except Exception:  # noqa: BLE001
        return None


def _load_rolling_summary(session_id: str) -> Optional[str]:
    """Best-effort fetch of the session's rolling summary from memory."""
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
    except ImportError:
        try:
            from shared.session import get_memory_manager  # type: ignore
        except ImportError:
            return None
    try:
        mm = get_memory_manager()
        sess = mm.get_session(session_id)
        if sess is None:
            return None
        ctx = getattr(sess, "context", None)
        if ctx is None:
            return None
        summary = getattr(ctx, "rolling_summary", None) or getattr(
            ctx, "custom_instructions", None
        )
        return (summary or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Recent turns loader (optional, for history context)
# ---------------------------------------------------------------------------


def _load_recent_turns(
    session_id: str,
    n: int,
    *,
    exclude_turn_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return last N (user, assistant) pairs as conversation history.

    Excludes the turn that's being deep-dived (we already inject it
    explicitly as the prompt's PRIOR QUESTION / PRIOR ANSWER sections).
    """
    if n <= 0:
        return []
    try:
        from traditional.memory_manager import get_memory_manager  # type: ignore
    except ImportError:
        try:
            from shared.session import get_memory_manager  # type: ignore
        except ImportError:
            return []
    try:
        mm = get_memory_manager()
        sess = mm.get_session(session_id)
        if sess is None:
            return []
        messages = getattr(sess, "messages", []) or []
    except Exception:  # noqa: BLE001
        return []

    out: List[Dict[str, str]] = []
    for m in messages:
        meta = getattr(m, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("turn_id") == exclude_turn_id:
            continue
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "")
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})

    # Each conversational turn is 2 messages — cap N pairs == 2*N items.
    return out[-(2 * n):]


# ---------------------------------------------------------------------------
# Audit writer — Mongo (write-only, never blocks)
# ---------------------------------------------------------------------------


_AUDIT_COLLECTION = "deep_dive_audits_projectqna_rag"


# ---------------------------------------------------------------------------
# PNG URL derivation (v3.3 — same-folder convention)
#
# Per Q1 clarification (2026-05-12): "every drawings pdf has only one page so
# we are saving pdf and its single page png in a same folder". So the PNG
# sits *sibling* to the PDF — NOT in a subfolder named after the stem.
#
# Convention:    s3://{bucket}/{prefix}/{pdf_stem}.png
# NOT:           s3://{bucket}/{prefix}/{pdf_stem}/page_1.png
#
# This helper takes the source_doc dict (whatever shape the upstream
# orchestrator produced) and returns the best vision-usable URL it can
# build. Order of preference:
#   1. doc["png_url"]   — when the orchestrator already builds it
#   2. derived via s3_path + pdf_name with .png sibling convention
#   3. doc["download_url"] (presigned PDF) — last resort; works on
#      models with native PDF support (GPT-4.1, Gemini), fails on
#      Claude / GPT-4o
# ---------------------------------------------------------------------------


def _derive_png_url(doc: Dict[str, Any]) -> Optional[str]:
    """Best-effort PNG URL for a source_document.

    Returns ``None`` only when neither s3_path+pdf_name nor a fallback URL
    is recoverable.
    """
    if not isinstance(doc, dict):
        return None
    # 1. trust upstream png_url when present (prod orchestrator path)
    upstream = doc.get("png_url")
    if upstream:
        return str(upstream)

    s3_path = doc.get("s3_path") or ""
    pdf_name = doc.get("pdf_name") or doc.get("file_name") or ""
    if s3_path and pdf_name:
        try:
            url = _presign_sibling_png(s3_path, pdf_name)
            if url:
                return url
        except Exception as exc:  # noqa: BLE001
            logger.debug("deep_dive: png derivation failed for %s: %s", pdf_name, exc)

    # 3. last resort — return the PDF presigned URL. Vision models with
    # native PDF support (GPT-4.1, Gemini) will accept it; Claude / 4o
    # won't. The fallback ladder handles that gracefully.
    return doc.get("download_url")


def _derive_pdf_url(doc: Dict[str, Any]) -> Optional[str]:
    """Best-effort PDF presigned URL for a source_document.

    Mirrors ``_derive_png_url`` but resolves to the .pdf sibling. Returns
    ``None`` when neither upstream nor derivable.
    """
    if not isinstance(doc, dict):
        return None
    # 1. trust upstream download_url (v3.2 orchestrator builds this)
    upstream = doc.get("download_url")
    if upstream:
        return str(upstream)
    # 2. derive from s3_path + pdf_name
    s3_path = doc.get("s3_path") or ""
    pdf_name = doc.get("pdf_name") or doc.get("file_name") or ""
    if s3_path and pdf_name:
        try:
            return _presign_sibling_pdf(s3_path, pdf_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("deep_dive: pdf derivation failed for %s: %s", pdf_name, exc)
    return None


def _presign_sibling_pdf(s3_path: str, pdf_name: str, expires: int = 3600) -> Optional[str]:
    """Presigned URL for the PDF at {prefix}/{pdf_stem}.pdf."""
    if not s3_path or not pdf_name:
        return None
    client = _get_s3_client()
    if client is None:
        return None
    bucket, _, key_prefix = s3_path.partition("/")
    if not bucket:
        return None
    stem = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
    key = f"{key_prefix}/{stem}.pdf" if key_prefix else f"{stem}.pdf"
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("deep_dive: pdf presign failed for %s/%s: %s", bucket, key, exc)
        return None


def _resolve_doc_urls(
    doc: Dict[str, Any],
    source_format: str,
) -> List[str]:
    """Return the list of URLs to send to the vision model for ONE source doc.

    ``source_format`` controls which assets are picked. Per the
    2026-05-15 product direction "use both pdf and png depend on user for
    upload and further process, whichever gives best and accurate
    extraction":

    * ``"png"``   — PNG only (fastest, default)
    * ``"pdf"``   — PDF only (preserves vector text, best for dense specs;
                   only providers with native PDF support — GPT-4.1, Gemini —
                   accept it. Claude / GPT-4o will skip the doc.)
    * ``"both"``  — PNG AND PDF for the same doc (max extraction quality
                   at 2× image budget — vision model cross-references)
    * ``"auto"``  — PNG primary; if PNG derivation fails fall back to PDF
                   (current default behavior, fully backward-compatible)

    Returns 0, 1, or 2 URLs. Empty list means no usable asset.
    """
    fmt = (source_format or "auto").strip().lower()
    out: List[str] = []
    if fmt == "png":
        u = _derive_png_url(doc)
        if u:
            out.append(u)
    elif fmt == "pdf":
        u = _derive_pdf_url(doc)
        if u:
            out.append(u)
    elif fmt == "both":
        png = _derive_png_url(doc)
        pdf = _derive_pdf_url(doc)
        if png:
            out.append(png)
        if pdf and pdf != png:
            out.append(pdf)
    else:  # "auto"
        png = _derive_png_url(doc)
        if png:
            out.append(png)
        else:
            pdf = _derive_pdf_url(doc)
            if pdf:
                out.append(pdf)
    return out


import threading as _threading
_S3_CLIENT: Any = None
# C3 fix (2026-06-02): init lock at module load — the prior lazy-init had a
# TOCTOU race where two threads could both create a lock and one would be
# silently discarded.
_S3_CLIENT_LOCK = _threading.Lock()


def _get_s3_client():
    """Lazy-init a single SigV4 S3 client per process. Mirrors the prod
    orchestrator's helper so we don't import boto3 at module load time.
    """
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    with _S3_CLIENT_LOCK:
        if _S3_CLIENT is not None:
            return _S3_CLIENT
        try:
            import boto3  # type: ignore  # type: ignore
            from botocore.config import Config  # type: ignore
            region = (
                os.getenv("S3_REGION")
                or os.getenv("AWS_REGION")
                or "us-east-1"
            )
            _S3_CLIENT = boto3.client(
                "s3",
                region_name=region,
                config=Config(signature_version="s3v4", retries={"max_attempts": 2}),
            )
            return _S3_CLIENT
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep_dive: S3 client init failed: %s", exc)
            return None


def _presign_sibling_png(s3_path: str, pdf_name: str, expires: int = 3600) -> Optional[str]:
    """Build a presigned URL for the PNG sibling of {prefix}/{pdf_stem}.pdf.

    Per Q1 clarification: PNGs sit in the SAME folder as the PDFs, with
    matching stems. So {prefix}/{stem}.pdf → {prefix}/{stem}.png.
    """
    if not s3_path or not pdf_name:
        return None
    client = _get_s3_client()
    if client is None:
        return None
    bucket, _, key_prefix = s3_path.partition("/")
    if not bucket:
        return None
    # Strip any trailing .pdf — pdf_name already arrives stem-form usually
    stem = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
    key = f"{key_prefix}/{stem}.png" if key_prefix else f"{stem}.png"
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("deep_dive: presign failed for %s/%s: %s", bucket, key, exc)
        return None


def _write_audit_async(result: DeepDiveResult) -> None:
    """Fire-and-forget write to the audit collection.

    Per Q11 clarification: Mongo writes are allowed; only Mongo *reads*
    during deep-dive execution are banned. The audit row is metadata,
    not document content.
    """
    import threading

    def _strip_data_urls(urls):
        # 2026-06-02 fix: base64 data: URLs are 5-10 MB each; two of them
        # blow the 16 MB BSON limit. Replace each with a compact stub so
        # the audit row stays under the cap. https URLs pass through.
        out = []
        for u in urls or []:
            if isinstance(u, str) and u.startswith("data:"):
                header, _, payload = u.partition(",")
                out.append(f"{header},<{len(payload)} b64 chars elided>")
            else:
                out.append(u)
        return out

    def _do_write() -> None:
        try:
            from agentic.core.db import get_collection
            coll = get_collection(_AUDIT_COLLECTION)
            coll.insert_one({
                "deep_dive_id": result.deep_dive_id,
                "session_id": result.session_id,
                "parent_turn_id": result.parent_turn_id,
                "turn_id": result.turn_id,
                "project_id": result.project_id,
                "original_question": result.original_question,
                "initial_rag_answer": result.initial_rag_answer,
                "deep_dive_answer": result.deep_dive_answer,
                "source_references": result.source_documents,
                "image_urls_used": _strip_data_urls(result.image_urls_used),
                "model_used": result.model_used,
                "provider": result.provider,
                "fallback_used": result.fallback_used,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "image_count": len(result.image_urls_used),
                "processing_ms": result.processing_ms,
                "processing_status": result.status,
                "error": result.error,
                "created_at_ms": int(time.time() * 1000),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep_dive: audit insert failed: %s", exc)

    threading.Thread(target=_do_write, daemon=True, name="deep-dive-audit").start()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_deep_dive(
    *,
    session_id: str,
    parent_turn_id: Optional[str] = None,
    project_id: Optional[int] = None,
    history_turns: int = 1,
    max_images: Optional[int] = None,
    primary_model: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    selected_pdf_names: Optional[List[str]] = None,
    selected_indices: Optional[List[int]] = None,
    source_format: str = "auto",
    user_question: Optional[str] = None,
    reuse_last_selection: bool = True,
    stream: bool = True,
) -> "DeepDiveStreamHandle":
    """Run a Deep Dive call — supports multi-turn conversation on picked docs.

    Returns a ``DeepDiveStreamHandle`` that exposes both:
      * ``handle.tokens()`` — iterator of streamed text chunks
      * ``handle.result()`` — populated after streaming completes

    Non-streaming callers can simply drain ``tokens()`` and then read
    ``result()``.

    Multi-turn flow (2026-05-18):
      1. First call: caller passes turn_id (a /rag/query turn) + selection
      2. Follow-up calls: caller may omit turn_id (auto-resolves to the
         most recent session turn) and may omit selection (auto-reuses
         the last Deep Dive turn's picks when reuse_last_selection=True)
      3. Caller may pass `user_question` to override the parent question
         and continue the conversation with a fresh query on same docs
    """
    started = time.monotonic()

    # --- 1. Resolve parent_turn_id ---
    # When the UI doesn't track turn_id (e.g. continuing a Deep Dive
    # conversation), fall back to the most recent turn in the session.
    if not parent_turn_id:
        parent_turn_id = _resolve_most_recent_turn(session_id) or ""
    parent = _load_parent_turn(session_id, parent_turn_id) if parent_turn_id else None
    if parent is None:
        return DeepDiveStreamHandle.error_handle(
            session_id=session_id,
            parent_turn_id=parent_turn_id or "<unset>",
            project_id=project_id,
            error="parent_turn_not_found",
        )

    original_question = parent["user_text"]
    initial_answer = parent["assistant_text"]
    source_docs = parent["source_documents"] or []

    # ----- Reuse last Deep Dive selection for multi-turn conversations -----
    # When the UI omits both selected_pdf_names and selected_indices AND the
    # session already has a prior Deep Dive turn, reuse that turn's picked
    # documents. This is the "continue the conversation on the same docs"
    # workflow — user keeps asking new questions on the same selection.
    selection_reused_from_prior = False
    if (
        reuse_last_selection
        and not selected_pdf_names
        and not selected_indices
    ):
        prior_picks = _find_last_deep_dive_selection(session_id)
        if prior_picks:
            # H3 fix (2026-06-02): strip cached presigned URLs from prior
            # picks before reuse — they expire in 3600s and following calls
            # would 403. _derive_png_url / _derive_pdf_url will re-derive
            # from s3_path + pdf_name on demand.
            cleaned_picks: List[Dict[str, Any]] = []
            for d in prior_picks:
                if not isinstance(d, dict):
                    continue
                d2 = dict(d)
                d2.pop("png_url", None)
                d2.pop("download_url", None)
                cleaned_picks.append(d2)
            source_docs = cleaned_picks
            selection_reused_from_prior = True
            logger.info(
                "deep_dive: reusing %d source_docs from prior Deep Dive in session %s (presigned URLs re-derived)",
                len(cleaned_picks), session_id,
            )

    if not source_docs:
        return DeepDiveStreamHandle.error_handle(
            session_id=session_id,
            parent_turn_id=parent_turn_id,
            project_id=project_id or parent.get("project_id"),
            error="parent_turn_has_no_sources",
            original_question=original_question,
            initial_answer=initial_answer,
        )

    # ----- User selection filter (Mode A from user story) -----
    # If the UI sent a pick-list (selected_pdf_names and/or selected_indices),
    # narrow source_docs to ONLY those. Selection is a UNION: a doc is kept
    # if it matches either selected_pdf_names OR selected_indices.
    # When neither field is provided, the full parent-turn source set is used.
    available_count = len(source_docs)
    selection_filter_used = False
    selected_count_target = 0
    if selected_pdf_names or selected_indices:
        selection_filter_used = True
        wanted_names = {str(n).strip().lower() for n in (selected_pdf_names or []) if n}
        # L2 fix: accept negative indices (Python-style from-end). Coerce
        # string forms safely with try/except so we don't drop "-1".
        def _coerce_idx(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        wanted_indices_raw = {_coerce_idx(i) for i in (selected_indices or [])}
        wanted_indices_raw.discard(None)
        # Normalize negative indices against current source_docs length
        wanted_indices = set()
        for i in wanted_indices_raw:
            if i is None:
                continue
            if i < 0:
                i = len(source_docs) + i
            if 0 <= i < len(source_docs):
                wanted_indices.add(i)
        selected_count_target = len(wanted_names) + len(wanted_indices)

        # H4 fix: two-pass match. Pass 1 = exact (index or stem-equal). Pass 2
        # = fuzzy substring fallback for the names that didn't match exactly.
        # Substring fallback never over-matches against names already exact-hit.
        filtered: List[Dict[str, Any]] = []
        matched_indices: set = set()
        unmatched_names: set = set(wanted_names)
        for idx, sd in enumerate(source_docs):
            if not isinstance(sd, dict):
                continue
            pdf_name_lc = (sd.get("pdf_name") or sd.get("file_name") or "").strip().lower()
            # Strip .pdf extension for stem-equal comparison
            stem_lc = pdf_name_lc[:-4] if pdf_name_lc.endswith(".pdf") else pdf_name_lc
            if idx in wanted_indices:
                filtered.append(sd); matched_indices.add(idx); continue
            if pdf_name_lc in wanted_names:
                filtered.append(sd); matched_indices.add(idx); unmatched_names.discard(pdf_name_lc); continue
            if stem_lc in wanted_names:
                filtered.append(sd); matched_indices.add(idx); unmatched_names.discard(stem_lc); continue
        # Pass 2: substring fallback ONLY for names that didn't exact-match.
        # Require minimum length to prevent 1-2 char names matching everything.
        MIN_FUZZY_LEN = 4
        if unmatched_names:
            for idx, sd in enumerate(source_docs):
                if idx in matched_indices or not isinstance(sd, dict):
                    continue
                pdf_name_lc = (sd.get("pdf_name") or sd.get("file_name") or "").strip().lower()
                if not pdf_name_lc:
                    continue
                for w in list(unmatched_names):
                    if len(w) < MIN_FUZZY_LEN:
                        continue
                    if w in pdf_name_lc:
                        filtered.append(sd); matched_indices.add(idx); break
        if not filtered:
            return DeepDiveStreamHandle.error_handle(
                session_id=session_id,
                parent_turn_id=parent_turn_id,
                project_id=project_id or parent.get("project_id"),
                error="selection_matched_no_documents",
                original_question=original_question,
                initial_answer=initial_answer,
            )
        logger.info(
            "deep_dive: selection filter trimmed %d -> %d source_docs",
            available_count, len(filtered),
        )
        source_docs = filtered

    # Pick the URLs to send to vision per the requested source_format.
    # Per 2026-05-15 direction: "use both pdf and png depend on user for
    # upload and further process, whichever gives best and accurate
    # extraction".
    #   - source_format="auto"  → PNG; PDF fallback (default, current behavior)
    #   - source_format="png"   → PNG only
    #   - source_format="pdf"   → PDF only
    #   - source_format="both"  → PNG + PDF per doc (max accuracy, 2x budget)
    #
    # H1 fix (2026-06-02): build image_docs parallel to image_urls so
    # downstream (hires_render, eager_ocr) can look up the correct doc
    # for each URL even when "both" mode produces 2x entries.
    # H2 fix: dedup on (bucket, key) so presigned-URL signature variation
    # doesn't defeat the dedup.
    def _dedup_key(u: str) -> str:
        if not u:
            return ""
        if u.startswith("data:"):
            return u[:60]  # mime + first 50 b64 chars
        # Strip query (presigned signature) before hashing
        return (u.split("?", 1)[0] or "").lower()
    image_urls: List[str] = []
    image_docs: List[Dict[str, Any]] = []
    seen: set = set()
    for sd in source_docs:
        if not isinstance(sd, dict):
            continue
        for url in _resolve_doc_urls(sd, source_format):
            key = _dedup_key(url)
            if url and key not in seen:
                seen.add(key)
                image_urls.append(url)
                image_docs.append(sd)
    # Apply caller / env cap
    cap = max_images or int(os.getenv("DEEP_DIVE_MAX_IMAGES", "20"))
    cap = max(1, min(cap, 25))
    image_urls = image_urls[:cap]
    image_docs = image_docs[:cap]
    if not image_urls:
        return DeepDiveStreamHandle.error_handle(
            session_id=session_id,
            parent_turn_id=parent_turn_id,
            project_id=project_id or parent.get("project_id"),
            error="no_image_urls_available",
            original_question=original_question,
            initial_answer=initial_answer,
        )

    rolling_summary = _load_rolling_summary(session_id)
    history = _load_recent_turns(session_id, history_turns, exclude_turn_id=parent_turn_id)

    # ----- Determine the EFFECTIVE question to answer in this Deep Dive -----
    # When caller supplies user_question, that's the new question on the same
    # picked docs (multi-turn follow-up). Otherwise re-answer the parent
    # turn's original question (first-turn deep-dive default).
    #
    # In both cases run the v3.1 query rewriter against session memory so
    # anaphora like "the second one" or "that table" resolves correctly —
    # this gives /rag/deep-dive parity with /rag/query's context-awareness.
    raw_effective_question = (user_question or original_question or "").strip()
    effective_question = raw_effective_question
    was_rewritten = False
    if raw_effective_question:
        try:
            from agentic.memory.query_rewriter import rewrite as _rewrite
            from agentic.memory.recall_agent import recall as _recall
            mem_ctx = _recall(session_id=session_id, user_query=raw_effective_question)
            rewrite_out = _rewrite(user_query=raw_effective_question, memory_context=mem_ctx)
            effective_question = (rewrite_out.get("contextualized_query") or raw_effective_question).strip()
            was_rewritten = bool(rewrite_out.get("was_rewritten"))
            if was_rewritten:
                logger.info(
                    "deep_dive: question rewritten %r -> %r",
                    raw_effective_question[:80], effective_question[:80],
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("deep_dive: rewriter skipped (%s); using raw question", exc)

    # Pick the prompt mode based on whether the caller is asking a new
    # question (multi-turn follow-up) or re-analyzing the parent's question.
    is_followup = bool(user_question and user_question.strip())
    user_prompt = _build_user_prompt(
        original_question=original_question,
        initial_answer=initial_answer,
        source_documents=image_docs,
        rolling_summary=rolling_summary,
        followup_question=effective_question if is_followup else None,
    )

    deep_dive_id = str(uuid.uuid4())
    new_turn_id = str(uuid.uuid4())

    return DeepDiveStreamHandle(
        deep_dive_id=deep_dive_id,
        new_turn_id=new_turn_id,
        parent_turn_id=parent_turn_id,
        session_id=session_id,
        project_id=project_id or parent.get("project_id"),
        original_question=original_question,
        initial_answer=initial_answer,
        source_documents=image_docs,
        image_urls=image_urls,
        effective_question=effective_question,
        is_followup=is_followup,
        user_question_raw=user_question,
        rolling_summary=rolling_summary,
        history=history,
        primary_model=primary_model,
        fallback_models=fallback_models,
        started=started,
    )


# ---------------------------------------------------------------------------
# Stream handle
# ---------------------------------------------------------------------------


class DeepDiveStreamHandle:
    """Resumable handle exposing the token stream + final result.

    Lifecycle:
      1. ``run_deep_dive(...)`` returns a handle ready to stream.
      2. Caller iterates ``handle.tokens()`` — vision call runs,
         tokens stream out.
      3. After the iterator drains, ``handle.result()`` carries the
         final DeepDiveResult (with the full assembled answer + provider
         metadata + timing).
      4. Memory + audit writes happen inside ``tokens()`` after the
         vision call completes.
    """

    def __init__(
        self,
        *,
        deep_dive_id: str,
        new_turn_id: str,
        parent_turn_id: str,
        session_id: Optional[str],
        project_id: Optional[int],
        original_question: str,
        initial_answer: str,
        source_documents: List[Dict[str, Any]],
        image_urls: List[str],
        rolling_summary: Optional[str],
        history: List[Dict[str, str]],
        primary_model: Optional[str],
        fallback_models: Optional[List[str]],
        started: float,
        effective_question: Optional[str] = None,
        is_followup: bool = False,
        user_question_raw: Optional[str] = None,
    ) -> None:
        self._dd_id = deep_dive_id
        self._turn_id = new_turn_id
        self._parent_turn_id = parent_turn_id
        self._session_id = session_id
        self._project_id = project_id
        self._original_question = original_question
        self._initial_answer = initial_answer
        self._source_documents = source_documents
        self._image_urls = image_urls
        self._rolling_summary = rolling_summary
        self._history = history
        self._primary_model = primary_model
        self._fallback_models = fallback_models
        self._started = started
        # v3.3 multi-turn: effective_question is what the model actually
        # answers (parent's original OR a user-supplied follow-up, possibly
        # rewritten for anaphora). user_question_raw is the literal text
        # the caller sent (for audit + memory write).
        self._effective_question = effective_question or original_question
        self._is_followup = is_followup
        self._user_question_raw = user_question_raw
        self._result: Optional[DeepDiveResult] = None
        self._error: Optional[str] = None

    # ---------- builders ----------

    @property
    def deep_dive_id(self) -> str:
        return self._dd_id

    @property
    def turn_id(self) -> str:
        return self._turn_id

    @property
    def parent_turn_id(self) -> str:
        return self._parent_turn_id

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def project_id(self) -> Optional[int]:
        return self._project_id

    @property
    def image_urls(self) -> List[str]:
        return list(self._image_urls)

    @property
    def source_documents(self) -> List[Dict[str, Any]]:
        return list(self._source_documents)

    @property
    def original_question(self) -> str:
        return self._original_question

    @property
    def initial_answer(self) -> str:
        return self._initial_answer

    # ---------- pre-error path ----------

    @classmethod
    def error_handle(
        cls,
        *,
        session_id: Optional[str],
        parent_turn_id: str,
        project_id: Optional[int],
        error: str,
        original_question: str = "",
        initial_answer: str = "",
    ) -> "DeepDiveStreamHandle":
        """Build a handle that will yield an error string and produce a
        failed DeepDiveResult on ``result()``.
        """
        h = cls(
            deep_dive_id=str(uuid.uuid4()),
            new_turn_id=str(uuid.uuid4()),
            parent_turn_id=parent_turn_id,
            session_id=session_id,
            project_id=project_id,
            original_question=original_question,
            initial_answer=initial_answer,
            source_documents=[],
            image_urls=[],
            rolling_summary=None,
            history=[],
            primary_model=None,
            fallback_models=None,
            started=time.monotonic(),
        )
        h._error = error
        return h

    # ---------- streaming ----------

    def tokens(self) -> Iterator[str]:
        """Yield text chunks from the vision provider.

        On internal pre-flight failure (no parent turn, no images, etc.)
        yields a single explanation string and stops. After draining,
        ``self.result()`` is populated for the caller.
        """
        if self._error is not None:
            msg = f"[Deep Dive could not run: {self._error}]"
            yield msg
            self._result = self._build_result(
                final_text=msg, status="failed", error=self._error,
                model_used="", provider="",
                fallback_used=False, attempts=[],
            )
            return

        from agentic.generation.vision_client import (
            generate_with_images,
        )

        # Rebuild the user prompt here so the multi-turn fields
        # (effective_question + is_followup) flow through. Without the
        # followup_question kwarg the prompt would silently fall back to
        # "re-analyze parent question" mode even when the caller passed a
        # fresh question — breaking multi-turn Deep Dive conversations.
        chunks: List[str] = []
        # v3.3 multi-turn: re-analyze vs. follow-up mode dictates the framing.
        # Using the re-analyze prompt in follow-up mode causes the model to ignore
        # the user's new question and re-emit a review of the parent answer.
        sys_prompt = (
            DEEP_DIVE_FOLLOWUP_SYSTEM_PROMPT if self._is_followup
            else DEEP_DIVE_SYSTEM_PROMPT
        )
        # Fix C: when ZOOM crops were prepended, augment the system prompt
        # with the spatial-binding rule so the model knows to prefer crops
        # for value-for-tag questions. Initialized below after crop_locator.

        # ---------------------------------------------------------------
        # Phase 1.5+ (2026-05-28): On-demand 300 DPI re-render.
        # When DEEP_DIVE_HIRES_DPI_ENABLED=true, replace the default-DPI PNG
        # URLs with 300 DPI data URLs rendered from the source PDFs.
        # Both OCR (below) and vision call benefit from higher resolution.
        # Additive: when disabled or render fails, original URLs preserved.
        # ---------------------------------------------------------------
        try:
            from agentic.preprocessing import hires_render
            if hires_render.is_enabled() and self._image_urls:
                upgraded_urls = hires_render.upgrade_image_urls(
                    self._image_urls, self._source_documents,
                )
                # Replace in place so downstream OCR + vision see upgraded URLs
                self._image_urls = upgraded_urls
                logger.info(
                    "hires_render: image URL list upgraded (count=%d)",
                    len(upgraded_urls),
                )
        except Exception as _hires_exc:  # noqa: BLE001
            logger.warning(
                "hires_render: upgrade failed (continuing with original URLs): %s",
                _hires_exc,
            )

        # ---------------------------------------------------------------
        # Fix C (2026-06-02): Crop-and-answer for tag-binding questions.
        # When DEEP_DIVE_CROP_ENABLED=true and the query contains a named
        # tag (U-218, P-600, FAN-12, etc.), generate ZOOM crops centered on
        # each tag location and PREPEND them to image_urls. The vision
        # model gets high-signal local context + the full sheet for fallback.
        # Additive: when disabled or no tags found, image_urls unchanged.
        # ---------------------------------------------------------------
        _crop_prompt_addendum = ""
        try:
            from agentic.preprocessing import crop_locator
            if crop_locator.is_enabled() and self._image_urls:
                # Build the same pages list eager_ocr will use, with stable cache_urls
                _crop_pages = []
                for i, url in enumerate(self._image_urls):
                    label = ""
                    cache_url = None
                    if i < len(self._source_documents) and isinstance(self._source_documents[i], dict):
                        sd = self._source_documents[i]
                        label = (sd.get("display_title") or sd.get("pdf_name")
                                 or sd.get("file_name") or f"page_{i+1}")
                        try:
                            cu = _derive_png_url(sd) or sd.get("png_url")
                            if cu and "?" in cu:
                                cu = cu.split("?", 1)[0]
                            cache_url = cu
                        except Exception:
                            cache_url = None
                    _crop_pages.append({
                        "image_url": url,
                        "label": label or f"page_{i+1}",
                        "cache_url": cache_url,
                    })
                _query_for_crop = self._effective_question or self._original_question or ""
                crops = crop_locator.crop_for_pages(_crop_pages, _query_for_crop)
                if crops:
                    # PREPEND crops so the vision model sees them first.
                    # Update source_documents in parallel so audit + prompts
                    # know which page each crop came from.
                    new_urls = [c["image_url"] for c in crops] + list(self._image_urls)
                    new_docs = []
                    for c in crops:
                        si = c.get("source_idx", 0)
                        if 0 <= si < len(self._source_documents):
                            base_doc = dict(self._source_documents[si]) if isinstance(self._source_documents[si], dict) else {}
                        else:
                            base_doc = {}
                        base_doc["display_title"] = c.get("label", "[ZOOM]")
                        base_doc["_crop_source"] = True
                        base_doc["_crop_tag"] = c.get("tag")
                        new_docs.append(base_doc)
                    new_docs.extend(self._source_documents)
                    self._image_urls = new_urls
                    self._source_documents = new_docs
                    _crop_prompt_addendum = crop_locator.CROP_PROMPT_DIRECTIVE
                    logger.info(
                        "crop_locator: prepended %d ZOOM crops, total images=%d",
                        len(crops), len(new_urls),
                    )
        except Exception as _crop_exc:  # noqa: BLE001
            logger.warning("crop_locator: failed (continuing without crops): %s", _crop_exc)

        # ---------------------------------------------------------------
        # Phase 1.5 (2026-05-28): Eager OCR injection.
        # When DEEP_DIVE_EAGER_OCR_ENABLED=true, run gpt-4o-mini OCR on
        # each selected page concurrently and inject the extracted text
        # alongside the image. Combats hallucination on dense engineering
        # drawings (P-600 sanitary riser bug class). Additive: when
        # disabled or OCR fails, falls back to image-only behavior.
        # ---------------------------------------------------------------
        ocr_prompt_block = None
        try:
            from agentic.preprocessing import eager_ocr
            if eager_ocr.is_enabled() and self._image_urls:
                ocr_pages = []
                for i, url in enumerate(self._image_urls):
                    label = ""
                    cache_url = None
                    if i < len(self._source_documents) and isinstance(self._source_documents[i], dict):
                        sd = self._source_documents[i]
                        label = (sd.get("display_title") or sd.get("pdf_name")
                                 or sd.get("file_name") or f"page_{i+1}")
                        # Stable cache key: prefer the original PNG sibling URL
                        # derived from s3_path + pdf_name (not the upgraded data URL).
                        try:
                            cache_url = _derive_png_url(sd) or sd.get("png_url")
                            if cache_url and "?" in cache_url:
                                cache_url = cache_url.split("?", 1)[0]
                        except Exception:
                            cache_url = None
                    ocr_pages.append({
                        "image_url": url,
                        "label": label or f"page_{i+1}",
                        "cache_url": cache_url,
                    })
                ocr_results = eager_ocr.extract_text_for_pages(ocr_pages)
                ocr_prompt_block = eager_ocr.format_for_prompt(ocr_results)
                logger.info(
                    "eager_ocr: built prompt block, ok_pages=%d/%d, chars=%d",
                    sum(1 for r in ocr_results if r.get("ok")),
                    len(ocr_results),
                    len(ocr_prompt_block or ""),
                )
        except Exception as _ocr_exc:  # noqa: BLE001
            logger.warning("eager_ocr: pre-call extraction failed (continuing image-only): %s", _ocr_exc)
            ocr_prompt_block = None

        _built_user_prompt = _build_user_prompt(
            original_question=self._original_question,
            initial_answer=self._initial_answer,
            source_documents=self._source_documents,
            rolling_summary=self._rolling_summary,
            followup_question=(
                self._effective_question if self._is_followup else None
            ),
        )
        if ocr_prompt_block:
            _built_user_prompt = ocr_prompt_block + "\n\n" + _built_user_prompt

        # C1 fix (2026-06-02): mutable attribution dict — populated by
        # _stream_with_fallback when the winning model emits its first chunk.
        # Replaces the previous "always primary" attribution.
        _attribution: Dict[str, Any] = {}
        full_text = ""
        # Fix C: inject CROP DIRECTIVE into sys_prompt if crops are present
        if _crop_prompt_addendum:
            sys_prompt = sys_prompt + "\n" + _crop_prompt_addendum
        try:
            for chunk in generate_with_images(  # type: ignore[misc]
                system_prompt=sys_prompt,
                user_prompt=_built_user_prompt,
                image_urls=self._image_urls,
                conversation_history=self._history,
                primary_model=self._primary_model,
                fallback_models=self._fallback_models,
                stream=True,
                _attribution=_attribution,
            ):
                chunks.append(chunk)
                yield chunk
        finally:
            # C2 fix (2026-06-02): build result + persist inside finally so a
            # client disconnect mid-stream still writes the audit row and
            # memory turn (with status="streaming_aborted" if incomplete).
            full_text = "".join(chunks).strip()
            primary = self._primary_model or os.getenv("DEEP_DIVE_PRIMARY_MODEL", "gpt-4.1")
            model_used = _attribution.get("model_used") or primary
            provider = _attribution.get("provider") or ""
            fallback_used = bool(_attribution.get("fallback_used", False))
            attempts = _attribution.get("attempts") or []
            err = _attribution.get("error")
            if full_text:
                status = "completed"
                error = None
            elif chunks:
                status = "streaming_aborted"
                error = "client_disconnect_or_partial"
            else:
                status = "failed"
                error = err or "empty_response"
            self._result = self._build_result(
                final_text=full_text,
                status=status,
                error=error,
                model_used=model_used,
                provider=provider,
                fallback_used=fallback_used,
                attempts=attempts,
            )
            # Persist to memory + audit (idempotent — wrapped in try/except internally)
            self._persist()

    # ---------- non-streaming convenience ----------

    def collect(self) -> DeepDiveResult:
        """Drain the stream and return the populated result."""
        for _ in self.tokens():
            pass
        assert self._result is not None
        return self._result

    def result(self) -> DeepDiveResult:
        """Return the populated result (only valid after tokens() drains)."""
        if self._result is None:
            return self._build_result(
                final_text="", status="failed",
                error="result_called_before_stream_drained",
                model_used="", provider="",
                fallback_used=False, attempts=[],
            )
        return self._result

    # ---------- internals ----------

    def _build_result(
        self,
        *,
        final_text: str,
        status: str,
        error: Optional[str],
        model_used: str,
        provider: str,
        fallback_used: bool,
        attempts: List[Any],
    ) -> DeepDiveResult:
        elapsed_ms = int((time.monotonic() - self._started) * 1000)
        # The response's "query" field should reflect what the model
        # actually answered — the user's follow-up question if provided,
        # else the parent's original question.
        displayed_question = (
            self._effective_question if self._is_followup
            else self._original_question
        )
        return DeepDiveResult(
            deep_dive_id=self._dd_id,
            turn_id=self._turn_id,
            parent_turn_id=self._parent_turn_id,
            session_id=self._session_id,
            project_id=self._project_id,
            original_question=displayed_question,
            initial_rag_answer=self._initial_answer,
            deep_dive_answer=final_text,
            source_documents=self._source_documents,
            image_urls_used=self._image_urls,
            model_used=model_used,
            provider=provider,
            fallback_used=fallback_used,
            input_tokens=0,
            output_tokens=0,
            processing_ms=elapsed_ms,
            status=status,
            error=error,
        )

    def _persist(self) -> None:
        """Write the Deep Dive turn back to memory + emit audit row.

        Memory user_text is set to the **effective question** so that
        future ``/rag/query`` recall + rewriter calls see a real question
        in the conversation history (not a placeholder). Prefix with
        ``[Deep Dive]`` so transcripts make the call site obvious.

        ``source_documents`` is persisted in metadata so a follow-up
        Deep Dive in the same session can reuse the user's selection.
        """
        if self._result is None or not self._session_id:
            return
        # Compose memory user_text — multi-turn aware
        if self._is_followup and self._user_question_raw:
            mem_user_text = f"[Deep Dive] {self._user_question_raw.strip()}"
        else:
            # First-turn deep-dive — echo the parent's question so memory
            # makes sense in linear reading order.
            mem_user_text = f"[Deep Dive] {self._original_question.strip()}"
        try:
            from agentic.memory.writer import MemoryWriter
            writer = MemoryWriter()
            writer.write_turn_async(
                session_id=self._session_id,
                user_text=mem_user_text,
                assistant_text=self._result.deep_dive_answer or "",
                project_id=self._project_id,
                set_id=None,
                turn_id=self._turn_id,
                turn_type="deep_dive",
                parent_turn_id=self._parent_turn_id,
                # Persist the picked source_documents so follow-up Deep Dive
                # calls with reuse_last_selection=True (default) can find them.
                source_documents=self._source_documents,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep_dive: memory write failed: %s", exc)
        # Audit row — also fire-and-forget
        _write_audit_async(self._result)
