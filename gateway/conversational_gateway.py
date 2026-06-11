"""[CG-V1] Conversational Gateway — intent gate for greetings / small talk.

Answers conversational messages (greetings, thanks, acknowledgements, help,
identity, chit-chat) instantly with a canned, schema-complete /query response —
no retrieval, no LLM call, no sources. Anything ambiguous falls through to the
normal pipeline, so the only possible failure mode is a false positive and the
patterns are therefore conservative:

- Full-string anchored regex AFTER normalization. A mixed query like
  "hi, what is the ceiling height in the lobby" can never match because the
  pattern must consume the entire string.
- Hard guards: length cap, any-digit bailout (protects "A-701", "7201"),
  domain-word bailout, and bare yes/no excluded (may answer an agent
  clarification question).

Flag: CONVERSATIONAL_GATEWAY_ENABLED (default true). Set false + restart to
restore prior behavior exactly.
"""

from __future__ import annotations

import os
import re
import time

GATEWAY_VERSION = "cg-v1"

_MAX_LEN = 64  # normalized; conversational messages are short

# Any of these words ⇒ the user is talking about the project — never intercept.
_DOMAIN_WORDS = re.compile(
    r"\b(drawing|drawings|sheet|sheets|spec|specs|specification|specifications|"
    r"floor|ceiling|wall|walls|slab|roof|plan|plans|detail|details|schedule|"
    r"schedules|rfi|rfis|submittal|door|doors|window|windows|room|rooms|beam|"
    r"column|hvac|plumbing|electrical|mechanical|structural|architectural|"
    r"project|elevation|section|finish|finishes|material|materials|dimension|"
    r"dimensions|height|width|area|zone|level|basement|cellar|lobby|stair|"
    r"document|documents|file|files|pdf|email|emails|meeting|meetings|code|"
    r"insulation|concrete|steel|rebar|duct|pipe|panel|fixture|equipment)\b"
)

# ---------------------------------------------------------------------------
# Intent patterns — applied with re.fullmatch on the normalized query.
# Order matters: first full match wins.
# ---------------------------------------------------------------------------
_PERSON = r"(\s+(there|team|all|everyone|guys|folks|agent|bot|assistant|buddy|sir|madam|friend))?"

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("howru", re.compile(
        r"((hi+|hey+|hello+|yo)[\s,]+)?"
        r"(how\s+are\s+you(\s+doing)?(\s+(today|now))?|how\s+r\s+u|how\s+are\s+u|"
        r"hows\s+it\s+going|how\s+is\s+it\s+going|how\s+goes\s+it|"
        r"hows\s+your\s+day|whats?\s+up|wassup|whats\s+going\s+on|sup|"
        r"how\s+do\s+you\s+do|how\s+have\s+you\s+been)")),
    ("greeting", re.compile(
        r"(hi+|hii+|heyy*|hello+|helo+|heya|hiya|yo|yoo+|howdy|namaste|hola|"
        r"bonjour|greetings|gm|good\s+(morning|afternoon|evening|day)|morning|"
        r"afternoon|evening|welcome|hey\s+hi|hi\s+hello)" + _PERSON)),
    ("thanks", re.compile(
        r"((ok(ay)?|great|perfect|awesome|nice|cool|wow|wonderful|excellent)[\s,]+)?"
        r"((thanks?|thank\s+you|thankyou|thx|tysm|tnx|ty)"
        r"(\s+(so\s+much|very\s+much|a\s+lot|again|man|buddy|for\s+(your|the)\s+help))?|"
        r"appreciate\s+(it|that)|much\s+appreciated|appreciated)")),
    ("farewell", re.compile(
        r"(bye+|goodbye|bbye|good\s+night|gn|see\s+(you|ya|u)(\s+(later|soon|tomorrow|around))?|"
        r"take\s+care|ciao|talk\s+(to\s+you\s+)?later|ttyl|catch\s+you\s+later|"
        r"im\s+done|that'?s?\s+all(\s+for\s+(now|today))?|thats\s+it(\s+for\s+(now|today))?|"
        r"im\s+leaving|gotta\s+go|have\s+a\s+(good|great|nice)\s+(day|evening|night|one)|signing\s+off)")),
    ("ack", re.compile(
        r"(ok+|okay+|kk|cool|nice|great|awesome|perfect|excellent|wonderful|"
        r"got\s+it|gotcha|understood|sounds\s+good|alright|all\s+right|fine|"
        r"hmm+|oh\s+ok(ay)?|oh+|ah+|i\s+see|makes\s+sense|noted|well\s+done|"
        r"good\s+job|good\s+work|nice\s+work|interesting|fair\s+enough)")),
    ("help", re.compile(
        r"(help|help\s+me|i\s+need\s+help|please\s+help(\s+me)?|can\s+you\s+help(\s+me)?(\s+please)?|"
        r"what\s+can\s+you\s+do(\s+for\s+me)?|what\s+do\s+you\s+do|"
        r"what\s+all\s+can\s+you\s+do|how\s+can\s+you\s+(help|assist)(\s+me)?|"
        r"what\s+are\s+your\s+(capabilities|features|functions)|capabilities|"
        r"what\s+is\s+this|whats\s+this|how\s+does\s+this\s+work|how\s+do\s+you\s+work|"
        r"what\s+can\s+i\s+ask(\s+you)?(\s+here)?|what\s+(questions|things|queries)\s+can\s+i\s+ask|"
        r"what\s+should\s+i\s+ask|what\s+kind\s+of\s+(questions|queries)\s+can\s+(i|you)\s+(ask|answer)|"
        r"menu|options|show\s+me\s+(the\s+)?options|guide\s+me|"
        r"how\s+to\s+use\s+(this|you|it)(\s+(chat|bot|chatbot|tool|assistant))?|"
        r"where\s+do\s+i\s+start|getting\s+started|how\s+do\s+i\s+(start|begin)|give\s+me\s+examples?)")),
    ("identity", re.compile(
        r"(who\s+are\s+you|what\s+are\s+you|whats\s+your\s+name|what\s+is\s+your\s+name|"
        r"tell\s+me\s+about\s+yourself|introduce\s+yourself|"
        r"are\s+you\s+(a\s+)?(bot|robot|chatbot|ai|human|real(\s+person)?)|"
        r"who\s+(made|created|built|developed|designed)\s+you|what\s+model\s+are\s+you|"
        r"are\s+you\s+(chatgpt|gpt|claude|gemini))")),
    ("ping", re.compile(
        r"(test|testing|test\s+test|ping|hello\s+test|are\s+you\s+(there|here|online|alive|awake|working)|"
        r"you\s+there|anyone\s+there|anybody\s+there|is\s+this\s+working|"
        r"can\s+you\s+(hear|see)\s+me|do\s+you\s+work)")),
    ("ood_smalltalk", re.compile(
        r"(tell\s+me\s+a\s+joke|joke|make\s+me\s+laugh|"
        r"whats\s+the\s+weather(\s+(like|today|now|outside))?|hows\s+the\s+weather|"
        r"will\s+it\s+rain(\s+today)?|is\s+it\s+(raining|hot|cold)(\s+outside)?|"
        r"sing\s+(me\s+)?a\s+song|tell\s+me\s+a\s+story|write\s+me\s+a\s+poem|"
        r"i\s+love\s+you|do\s+you\s+love\s+me|marry\s+me|will\s+you\s+marry\s+me|"
        r"whats\s+the\s+time|what\s+time\s+is\s+it|what\s+day\s+is\s+(it|today)|"
        r"whats\s+todays\s+date|whats\s+the\s+date(\s+today)?|"
        r"how\s+old\s+are\s+you|where\s+are\s+you(\s+from)?|do\s+you\s+sleep|are\s+you\s+happy)")),
]

_FOLLOW_UPS = [
    "Can you list all the architectural drawings for this project?",
    "What materials are specified for the HVAC system?",
    "Where can I find the general notes or code data for this project?",
]

_CAPABILITY_TEXT = (
    "Here's what I can help you with:\n\n"
    "- **Drawings** — find sheets, dimensions, ceiling heights, schedules, "
    "details, and plan information (e.g. *\"What is the ceiling height in the "
    "first floor lobby?\"*)\n"
    "- **Specifications** — materials, submittal requirements, standards, and "
    "section content (e.g. *\"What are the submittal requirements for "
    "cast-in-place concrete?\"*)\n"
    "- **Search modes** — search drawings only, specifications only, or "
    "everything together; I can also search project emails, RFIs, and meeting "
    "notes\n"
    "- **Cited answers** — every answer references the exact sheet or spec "
    "section it came from, with a link to open the document\n\n"
    "Just ask your question in plain language — I'll find it in your project "
    "documents."
)

_RESPONSES: dict[str, str] = {
    "greeting": (
        "Hello! 👋 I'm your construction documents assistant. I can answer "
        "questions about your project's drawings and specifications — "
        "dimensions, schedules, materials, code data, and more. What would "
        "you like to know?"
    ),
    "howru": (
        "I'm doing great, thank you for asking! 😊 Ready to dig into your "
        "project documents whenever you are. What would you like to know?"
    ),
    "thanks": (
        "You're welcome! Happy to help. Is there anything else you'd like to "
        "check in your project documents?"
    ),
    "ack": (
        "👍 Let me know if there's anything else you'd like to look up in "
        "your project documents."
    ),
    "farewell": (
        "Goodbye! 👋 Feel free to come back anytime you have questions about "
        "your project's drawings or specifications."
    ),
    "help": _CAPABILITY_TEXT,
    "identity": (
        "I'm your AI construction documents assistant. I read your project's "
        "drawings and specifications and answer questions about them with "
        "cited sources — so you can verify every answer against the actual "
        "sheet or spec section. Ask me anything about your project documents!"
    ),
    "ping": (
        "I'm here and working! ✅ Ask me anything about your project's "
        "drawings or specifications."
    ),
    "ood_smalltalk": (
        "That one's a bit outside my toolbox 🙂 — I'm focused on your "
        "project's construction documents. I can help with drawings, "
        "specifications, schedules, dimensions, materials, and more. What "
        "would you like to know about your project?"
    ),
}

_APOSTROPHES = ("'", "’", "‘", "`")


def _normalize(query: str) -> str:
    """Lowercase, drop apostrophes, strip edge punctuation (incl. the
    trailing-backslash case), collapse internal whitespace."""
    q = (query or "").lower()
    for ch in _APOSTROPHES:
        q = q.replace(ch, "")
    # strip non-letter/digit junk from both ends: "hi\\", "hi!!!", "...hello?"
    q = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def _enabled() -> bool:
    return os.getenv("CONVERSATIONAL_GATEWAY_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )


def detect(query: str) -> str | None:
    """Return the conversational intent name, or None to run the pipeline."""
    if not _enabled():
        return None
    if not query or not isinstance(query, str):
        return None
    q = _normalize(query)
    if not q or len(q) > _MAX_LEN:
        return None
    # Digits ⇒ sheet refs / project ids / quantities — never intercept.
    if any(c.isdigit() for c in q):
        return None
    # Domain vocabulary ⇒ never intercept (belt + suspenders; the anchored
    # fullmatch below could not match these anyway).
    if _DOMAIN_WORDS.search(q):
        return None
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.fullmatch(q):
            return intent
    return None


# multi-source modes return per-source answer keys (email_answer, sources_used…)
# in their /query envelope — the CG response must mirror them or mode-specific
# UI panels would render blank on a greeting.
_MULTI_SOURCE_MODES = ("email", "rfi", "meeting", "web")


def build_response(
    intent: str,
    query: str,
    project_id=None,
    session_id=None,
    search_mode=None,
) -> dict:
    """Schema-complete /query response — mirrors generation_chain's contract."""
    answer = _RESPONSES.get(intent) or _RESPONSES["greeting"]
    requested = {
        tok for tok in re.split(r"[,+\s]+", (search_mode or "").lower())
        if tok in _MULTI_SOURCE_MODES
    }
    extra: dict = {}
    if requested or (search_mode or "").lower() == "hybrid":
        extra["sources_used"] = []
        for m in _MULTI_SOURCE_MODES:
            extra[f"{m}_answer"] = answer if m in requested else None
    return {
        **extra,
        "query": query,
        "contextualized_query": query,
        "answer": answer,
        "rag_answer": answer,
        "final_answer": answer,
        "retrieval_count": 0,
        "confidence": "high",
        "confidence_score": 1.0,
        "average_score": 1.0,
        "is_clarification": False,
        "follow_up_questions": list(_FOLLOW_UPS),
        "model_used": "conversational_gateway",
        "token_usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "source_documents": [],
        "all_retrieved_sources": [],
        "sources": [],
        "s3_paths": [],
        "s3_path_count": 0,
        "debug_info": {
            "conversational_intent": intent,
            "gateway": GATEWAY_VERSION,
            "retrieval_skipped": True,
        },
        "processing_time_ms": 0,
        "project_id": project_id,
        "session_id": session_id,
        "search_mode": search_mode or "rag",
        "engine_used": "conversational_gateway",
        "fallback_used": False,
        "verification_meta": {
            "conversational": True,
            "intent": intent,
            "refused": False,
            "guardrails_version": GATEWAY_VERSION,
        },
    }
