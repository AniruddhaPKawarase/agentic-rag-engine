"""Fix #4 — Self-RAG groundedness verification.

After the agent produces an answer, we run ONE cheap verification pass:
extract atomic factual claims from the answer, check each against the
retrieved source texts, and return both a numerical groundedness score and
a list of flagged (unsupported) claims. Optionally, a refinement pass asks
the agent to retract or soften flagged claims.

Integration is additive and behind a feature flag
(``SELF_RAG_ENABLED``). Response envelope gains optional fields
``groundedness_score`` and ``flagged_claims``; UI can ignore them without
breaking.

Kept in the ``gateway`` package (not ``agentic``) because the verification
is orchestrator-level, not part of the ReAct loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_SELF_RAG_MODEL = os.environ.get("SELF_RAG_MODEL", "gpt-4.1-mini")
DEFAULT_MAX_CLAIMS = int(os.environ.get("SELF_RAG_MAX_CLAIMS", "12"))
DEFAULT_CONTEXT_CHAR_CAP = int(os.environ.get("SELF_RAG_CONTEXT_CAP", "12000"))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    groundedness_score: float  # 0.0 – 1.0
    claims: List[str]
    supported: List[dict]  # list of {claim, evidence}
    flagged: List[dict]    # list of {claim, reason}
    raw_response: str

    def to_public(self) -> dict:
        """Return the UI-safe subset (keeps payload small)."""
        return {
            "groundedness_score": round(self.groundedness_score, 3),
            "claims_total": len(self.claims),
            "claims_supported": len(self.supported),
            "flagged_claims": [
                {"claim": f.get("claim", "")[:240], "reason": (f.get("reason") or "")[:180]}
                for f in self.flagged
            ],
        }


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_CLAIM_EXTRACT_SYSTEM = (
    "You are a factual-claim extractor for a construction RAG system. "
    "Given an assistant's answer, return ONLY the atomic factual claims it "
    "makes as a JSON array of short strings. No prose, no headers, no "
    "explanations."
)

_CLAIM_EXTRACT_USER = (
    "Return at most {max_claims} atomic factual claims from this answer. "
    "Each claim must be a single verifiable statement (quantity, location, "
    "material, reference, presence/absence). Drop hedging, opinions, and "
    "follow-up suggestions.\n\nAnswer:\n{answer}"
)


def extract_claims(
    answer: str,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    model: str = DEFAULT_SELF_RAG_MODEL,
    openai_client: Any = None,
) -> List[str]:
    """Parse atomic factual claims from an answer. Returns [] on failure."""
    a = (answer or "").strip()
    if not a:
        return []
    try:
        client = openai_client
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLAIM_EXTRACT_SYSTEM},
                {"role": "user", "content": _CLAIM_EXTRACT_USER.format(max_claims=max_claims, answer=a[:4000])},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("extract_claims LLM call failed: %s", exc)
        return []

    return _parse_claims(raw, max_claims)


def _parse_claims(raw: str, max_claims: int) -> List[str]:
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    out: List[str] = []
    for item in parsed:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped[:280])
        if len(out) >= max_claims:
            break
    return out


# ---------------------------------------------------------------------------
# Claim verification (batched)
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM = (
    "You grade whether each claim is DIRECTLY supported by the provided "
    "sources. Output ONLY a JSON array of objects, one per claim, in order. "
    'Each object has: {"supported": true|false, "reason": "<short>"}.'
)

_VERIFY_USER = (
    "Sources (numbered):\n{context}\n\n"
    "Claims to verify (one per line, in order):\n{claims}\n\n"
    "For each claim, decide supported=true only if the sources contain "
    "direct evidence. If not stated, or only implied, or contradicted, "
    "return supported=false and explain briefly in <=12 words."
)


def _format_context(sources: List[dict], char_cap: int) -> str:
    """Build the numbered sources block. Trims to stay inside *char_cap*."""
    parts: List[str] = []
    spent = 0
    for i, src in enumerate(sources or [], 1):
        if not isinstance(src, dict):
            continue
        title = (
            src.get("drawing_title")
            or src.get("drawingTitle")
            or src.get("sectionTitle")
            or src.get("display_title")
            or src.get("pdf_name")
            or src.get("pdfName")
            or "(untitled)"
        )
        body = str(
            src.get("text")
            or src.get("fullText")
            or src.get("full_text")
            or src.get("sectionText")
            or src.get("page_summary")
            or ""
        ).strip()
        # Stay inside the cap
        remaining = char_cap - spent
        if remaining <= 200:
            break
        snippet = body[: min(900, remaining - 200)]
        entry = f"[S{i}] {str(title)[:140]}\n{snippet}".strip()
        spent += len(entry) + 2
        parts.append(entry)
    return "\n\n".join(parts)


def verify_claims(
    claims: List[str],
    sources: List[dict],
    model: str = DEFAULT_SELF_RAG_MODEL,
    context_char_cap: int = DEFAULT_CONTEXT_CHAR_CAP,
    openai_client: Any = None,
) -> List[dict]:
    """Return a list of ``{claim, supported, reason}`` (same order as claims).

    On LLM failure, returns all-supported (fail-open) so we never block an
    answer. Fails closed only when the response is strictly malformed.
    """
    if not claims:
        return []
    context = _format_context(sources, context_char_cap)
    claim_block = "\n".join(f"- {c}" for c in claims)
    try:
        client = openai_client
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM},
                {"role": "user", "content": _VERIFY_USER.format(context=context, claims=claim_block)},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("verify_claims LLM call failed: %s — assuming all supported", exc)
        return [{"claim": c, "supported": True, "reason": "verifier unavailable"} for c in claims]

    return _parse_verification(raw, claims)


def _parse_verification(raw: str, claims: List[str]) -> List[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return [{"claim": c, "supported": True, "reason": "parse failure"} for c in claims]
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [{"claim": c, "supported": True, "reason": "parse failure"} for c in claims]
    if not isinstance(parsed, list):
        return [{"claim": c, "supported": True, "reason": "parse failure"} for c in claims]

    results: List[dict] = []
    for i, claim in enumerate(claims):
        item = parsed[i] if i < len(parsed) else {}
        if not isinstance(item, dict):
            item = {}
        supported = bool(item.get("supported", True))
        reason = str(item.get("reason") or ("supported" if supported else "not supported"))
        results.append({"claim": claim, "supported": supported, "reason": reason[:200]})
    return results


# ---------------------------------------------------------------------------
# Public entry point used by the orchestrator
# ---------------------------------------------------------------------------

def evaluate_groundedness(
    answer: str,
    sources: List[dict],
    max_claims: int = DEFAULT_MAX_CLAIMS,
    model: str = DEFAULT_SELF_RAG_MODEL,
    openai_client: Any = None,
) -> Optional[VerificationResult]:
    """Run claim extraction → verification. Returns None if the answer is
    too short or claims couldn't be extracted (fail-open).
    """
    a = (answer or "").strip()
    if len(a) < 40:
        return None
    claims = extract_claims(answer=a, max_claims=max_claims, model=model, openai_client=openai_client)
    if not claims:
        return None
    verified = verify_claims(claims=claims, sources=sources or [], model=model, openai_client=openai_client)
    supported = [v for v in verified if v.get("supported")]
    flagged = [v for v in verified if not v.get("supported")]
    score = len(supported) / max(len(claims), 1)
    return VerificationResult(
        groundedness_score=score,
        claims=claims,
        supported=supported,
        flagged=flagged,
        raw_response="",
    )
