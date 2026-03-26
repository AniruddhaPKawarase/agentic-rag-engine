"""Lightweight regex-based intent detector.

Zero-latency classification of user input into:
  - "greeting"              : hello, hi, good morning, etc.
  - "small_talk"            : how are you, what can you do, etc.
  - "thanks"                : thank you, thanks, etc.
  - "farewell"              : bye, goodbye, see you, etc.
  - "meta_conversation"     : what was my first question, how many questions, etc.
  - "follow_up"             : tell me more, continue, elaborate, expand on that
  - "reference_previous"    : what size you mentioned, you said, as you mentioned
  - "drawing_specific"      : queries referencing a drawing name (A0.01, CG-107, M-101)
  - "drawing_title_specific": queries referencing a drawing title (Partition Schedule, Floor Plan)
  - "document_query"        : anything else (default -- proceed to RAG pipeline)

NO LLM call. NO embedding. NO API call. Pure regex. ~0ms.
"""
import re
from typing import Optional, Tuple

# ── PATTERN DEFINITIONS ──────────────────────────────────────────────────────
# Each pattern is compiled once at import time for maximum performance.
# Greeting/small-talk/thanks/farewell patterns use full-match anchors (^ ... $)
# so that queries like "Hi, what are the fire safety specs?" do NOT match.

_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|howdy|hiya|yo|sup)\s*[!.?,]*\s*$",
    r"^\s*(hi|hello|hey)\s+(there|everyone|buddy|friend|team)\s*[!.?,]*\s*$",
    r"^\s*good\s+(morning|afternoon|evening|day|night)\s*[!.?,]*\s*$",
    r"^\s*(greetings|salutations|namaste|hola|bonjour|salam)\s*[!.?,]*\s*$",
    r"^\s*what'?s?\s+up\s*[!.?,]*\s*$",
    r"^\s*(hii+|heyyy*|helloo+)\s*[!.?,]*\s*$",
]

_SMALL_TALK_PATTERNS = [
    r"^\s*how\s+are\s+you(\s+doing)?\s*[!.?]*\s*$",
    r"^\s*how('?s|\s+is)\s+(it\s+going|everything|life)\s*[!.?]*\s*$",
    r"^\s*what\s+can\s+you\s+do\s*[!.?]*\s*$",
    r"^\s*who\s+are\s+you\s*[!.?]*\s*$",
    r"^\s*tell\s+me\s+about\s+yourself\s*[!.?]*\s*$",
    r"^\s*what\s+are\s+you\s*[!.?]*\s*$",
    r"^\s*are\s+you\s+(a\s+)?(bot|ai|robot|human|real)\s*[!.?]*\s*$",
    r"^\s*help\s*[!.?]*\s*$",
    r"^\s*what\s+do\s+you\s+do\s*[!.?]*\s*$",
]

_THANKS_PATTERNS = [
    r"^\s*(thanks?(\s+you)?|thank\s+you(\s+(very|so)\s+much)?|thx|ty|cheers|appreciated)\s*[!.?]*\s*$",
    r"^\s*(great|awesome|perfect|wonderful|excellent|nice|cool|ok|okay)[\s,]*(thanks?|thank\s+you)?\s*[!.?]*\s*$",
    r"^\s*that('?s|\s+is)\s+(helpful|great|perfect|awesome|exactly\s+what\s+i\s+needed)\s*[!.?]*\s*$",
    r"^\s*i\s+appreciate\s+(it|that|your\s+help)\s*[!.?]*\s*$",
    r"^\s*much\s+appreciated\s*[!.?]*\s*$",
]

_FAREWELL_PATTERNS = [
    r"^\s*(bye|goodbye|good\s*bye|see\s+you|take\s+care|ciao|adios|later)\s*[!.?]*\s*$",
    r"^\s*(have\s+a\s+(good|nice|great|wonderful)\s+(day|one|evening|night|weekend))\s*[!.?]*\s*$",
    r"^\s*see\s+you\s+(later|soon|next\s+time|around)\s*[!.?]*\s*$",
    r"^\s*good\s*night\s*[!.?]*\s*$",
]

_META_CONVERSATION_PATTERNS = [
    r"what\s+(was|is)\s+my\s+(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|previous|latest)\s+question",
    r"what\s+did\s+i\s+(first\s+)?ask(\s+you)?(\s+first)?",
    r"how\s+many\s+questions?\s+have\s+i\s+asked",
    r"what\s+have\s+i\s+asked\s+(you\s+)?(so\s+far|before|earlier|previously|till\s+now|until\s+now)",
    r"repeat\s+my\s+(first|last|previous|second|third)\s+question",
    r"list\s+(all\s+)?my\s+questions",
    r"summarize\s+(our|this|the)\s+conversation",
    r"what\s+have\s+we\s+(discussed|talked\s+about|covered)",
    r"what\s+(was|were)\s+(our|the)\s+(first|last|previous)\s+(topic|question|discussion)",
    r"can\s+you\s+(recall|remember)\s+my\s+(first|previous|earlier)\s+question",
    r"remind\s+me\s+what\s+i\s+asked",
    r"what\s+questions?\s+did\s+i\s+ask",
]

# ── NEW: Follow-up intent patterns ──────────────────────────────────────────
# Matched when user wants to continue/expand on the previous answer.
# These are anchored (^ ... $) so "tell me more about HVAC specs" does NOT match.

_FOLLOW_UP_PATTERNS = [
    r"^\s*tell\s+me\s+more(\s+about\s+(it|that|this))?\s*[!.?]*\s*$",
    r"^\s*(continue|go\s+on|keep\s+going|proceed)\s*[!.?]*\s*$",
    r"^\s*(elaborate|expand)(\s+on\s+(that|this|it))?\s*[!.?]*\s*$",
    r"^\s*more\s+(details?|info(rmation)?|about\s+(it|that|this))\s*[!.?]*\s*$",
    r"^\s*what\s+else\s*[!.?]*\s*$",
    r"^\s*and\s*\?\s*$",
    r"^\s*explain\s+(more|further|in\s+detail)\s*[!.?]*\s*$",
    r"^\s*can\s+you\s+(elaborate|expand|explain\s+more|give\s+more\s+details?)\s*[!.?]*\s*$",
    r"^\s*go\s+(deeper|further)\s*[!.?]*\s*$",
    r"^\s*tell\s+me\s+more\s*$",
]

# ── NEW: Reference-previous-answer patterns ─────────────────────────────────
# Matched when user references something from the previous answer.

_REFERENCE_PREVIOUS_PATTERNS = [
    r"what\s+(size|dimension|measurement|number|value|amount|quantity|specification|material)\s+(you|did\s+you)\s+(mention|say|state|give|provide|specify|note)",
    r"(you|the)\s+(said|mentioned|stated|noted|specified|provided|gave|told\s+me)",
    r"as\s+(you|previously)\s+(mentioned|said|stated|noted)",
    r"(the|that)\s+(one|size|type|value|number)\s+you\s+(mentioned|said|gave)",
    r"what\s+did\s+you\s+(say|mean|mention)\s+(about|regarding)",
    r"(your|the)\s+(previous|last)\s+(answer|response|reply)",
    r"(refer|referring)\s+to\s+(your|the)\s+(previous|last)",
    r"what\s+was\s+(the|that)\s+(size|value|number|measurement|answer)",
    r"you\s+told\s+me\s+(about|that|the)",
]

# ── NEW: Drawing name pattern ────────────────────────────────────────────────
# Matches construction drawing names like A0.01, A912, CG-107, M-101, E1.02, S-501
# Must have at least one letter prefix and at least one digit.

_DRAWING_NAME_PATTERN = re.compile(
    r"\b([A-Z]{1,4}[-.]?\d{1,4}(?:[.-]\d{1,3})?)\b",
    re.IGNORECASE,
)

# ── NEW: Drawing title keywords ──────────────────────────────────────────────
# Common construction drawing titles that appear in drawingTitle metadata.
# These are partial matches (not anchored) so "notes on Partition Schedule" works.

_DRAWING_TITLE_KEYWORDS = [
    "partition schedule", "floor plan", "roof plan", "foundation plan",
    "electrical details", "electrical symbols", "electrical schedules",
    "plumbing symbols", "plumbing roof plan", "plumbing details",
    "mechanical schedules", "mechanical details",
    "structural details", "concrete details", "concrete grade beam",
    "section details", "interior opening details", "exterior plan details",
    "room finish schedule", "material schedule", "general notes",
    "legend and notes", "site plan", "sitework details",
    "roof details", "wall section", "door schedule", "window schedule",
    "fire alarm", "fire protection", "irrigation detail",
    "demolition plan", "reflected ceiling plan", "framing plan",
]

_DRAWING_TITLE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _DRAWING_TITLE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# Compile all patterns once at import time
_COMPILED = {
    "greeting":          [re.compile(p, re.IGNORECASE) for p in _GREETING_PATTERNS],
    "small_talk":        [re.compile(p, re.IGNORECASE) for p in _SMALL_TALK_PATTERNS],
    "thanks":            [re.compile(p, re.IGNORECASE) for p in _THANKS_PATTERNS],
    "farewell":          [re.compile(p, re.IGNORECASE) for p in _FAREWELL_PATTERNS],
    "meta_conversation": [re.compile(p, re.IGNORECASE) for p in _META_CONVERSATION_PATTERNS],
    "follow_up":         [re.compile(p, re.IGNORECASE) for p in _FOLLOW_UP_PATTERNS],
    "reference_previous":[re.compile(p, re.IGNORECASE) for p in _REFERENCE_PREVIOUS_PATTERNS],
}

# ── FRIENDLY RESPONSES ───────────────────────────────────────────────────────

_FRIENDLY_RESPONSES = {
    "greeting": (
        "Hello! I'm your Construction Documentation Assistant. "
        "I can help you find information from project documents, specifications, and drawings. "
        "You can also ask me to search the web for industry standards and regulations. "
        "What would you like to know?"
    ),
    "small_talk": (
        "I'm a Construction Documentation AI assistant. I can search through project documents, "
        "drawings, and specifications to answer your construction-related questions. "
        "I support three search modes: RAG (project documents), Web Search (internet), "
        "and Hybrid (both combined). How can I help you today?"
    ),
    "thanks": (
        "You're welcome! Feel free to ask if you have any more questions about the project documents."
    ),
    "farewell": (
        "Goodbye! Feel free to come back anytime you need help with construction documentation. "
        "Your conversation session will be saved."
    ),
}


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def detect_intent(query: str) -> Tuple[str, str]:
    """
    Classify user query into an intent category.

    Returns:
        Tuple of (intent_type, friendly_response).
        - intent_type: one of "greeting", "small_talk", "thanks", "farewell",
                       "meta_conversation", "follow_up", "reference_previous",
                       "document_query".
        - friendly_response: pre-built response for non-document intents,
                             empty string for document/follow-up intents.
    """
    text = query.strip()
    if not text:
        return ("document_query", "")

    # Check each intent category in priority order
    for intent_type in ("greeting", "small_talk", "thanks", "farewell",
                        "meta_conversation", "follow_up", "reference_previous"):
        for pattern in _COMPILED[intent_type]:
            if pattern.search(text):
                response = _FRIENDLY_RESPONSES.get(intent_type, "")
                return (intent_type, response)

    return ("document_query", "")


def extract_drawing_reference(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract drawing name and/or drawing title reference from a query.

    Returns:
        Tuple of (drawing_name, drawing_title).
        - drawing_name: e.g. "A0.01", "CG-107", "M-101" (or None)
        - drawing_title: e.g. "Partition Schedule", "Floor Plan" (or None)
    """
    drawing_name = None
    drawing_title = None

    # Check for drawing name pattern (e.g., A0.01, CG-107)
    name_match = _DRAWING_NAME_PATTERN.search(query)
    if name_match:
        candidate = name_match.group(1)
        # Must have at least one letter AND one digit to be a drawing name
        if re.search(r"[A-Za-z]", candidate) and re.search(r"\d", candidate):
            drawing_name = candidate.upper()

    # Check for drawing title keyword
    title_match = _DRAWING_TITLE_PATTERN.search(query)
    if title_match:
        drawing_title = title_match.group(1)

    return drawing_name, drawing_title
