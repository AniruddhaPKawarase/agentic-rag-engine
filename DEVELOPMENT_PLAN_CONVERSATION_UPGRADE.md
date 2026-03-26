# RAG Agent — Conversation Intelligence Upgrade

## User Story

**As a construction professional** using the RAG agent through the UI,
**I want** the agent to remember my previous questions and answers, understand follow-up queries like "tell me more" or "what size you mentioned", and find information from specific drawings when I reference them by name,
**so that** I can have a natural multi-turn conversation without repeating context.

---

## Problem Statement (from live testing)

| # | User Query | Expected Behavior | Actual Behavior | Root Cause |
|---|-----------|-------------------|-----------------|------------|
| Q2 | "TELL ME MORE ABOUT IT" | Expand on gap between pipe/conduit (previous answer) | Generic response: "If you have any questions..." | No follow-up intent detection; query treated independently |
| Q3 | "TELL ME MORE ABOUT THE GAP" | Detail about the 1/2 inch gap from Q1 answer | Generic: "If it's related to construction..." | Same — no query augmentation with prior context |
| Q4 | "what size you mentioned" | "The size I mentioned was 1/2 inch" | "I didn't mention a size yet" | Previous answer truncated to 150 chars in context; LLM can't see "1/2 inch" |
| Q5 | "notes present on drawing A0.01" | Retrieve chunks specifically from drawing A0.01 | "documents do not provide information" | No drawingName filtering in FAISS retrieval |
| Q6 | "notes present on Partition Schedule" | Retrieve chunks from that drawing title | "does not provide specific notes" | No drawingTitle filtering in FAISS retrieval |

---

## Design Decisions (Confirmed by User — 2026-03-26)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| Q1 | Follow-up handling | **(A) Augment query + re-retrieve** | User may want deeper info from different chunks |
| Q2 | Drawing name filtering | **(B) Soft filter with boost** | Avoid empty results from OCR metadata inconsistencies |
| Q3 | Previous answer context size | **(C) Full last answer + 300 chars older** | Best accuracy for recent, token-efficient for history |
| Q4 | Project scope | **All projects** | Drawing name filtering works universally |
| Q5 | Conversation token budget | **3000 tokens** (was 2000) | Fuller answer inclusion needs more room |
| Q6 | Intent detection method | **(A) Pure regex (0ms)** | Sufficient for common patterns, zero latency |

---

## Architecture Changes

### 1. Enhanced Intent Detection (`rag/api/intent.py`)

**New query classifications** (regex-based, 0ms):

| Intent Type | Patterns | Action |
|-------------|----------|--------|
| `follow_up` | "tell me more", "continue", "elaborate", "expand on that", "go on", "more details", "what else", "and?", "keep going" | Augment query with previous Q+A context |
| `reference_previous` | "what size you mentioned", "what did you say", "the one you mentioned", "you said", "as you mentioned", "your answer", "you told me" | Include full previous answer in augmented query |
| `drawing_specific` | Regex: `[A-Z]{1,3}[-.]?\d{1,4}[.-]?\d{0,3}` (e.g., A0.01, A912, CG-107, M-101, E1.02) | Extract drawing name → pass to retrieval filter |
| `drawing_title_specific` | "Partition Schedule", "Floor Plan", "Electrical Details" — match against known drawingTitle patterns | Extract title → pass to retrieval filter |
| `meta_conversation` | (existing) "what was my first question", etc. | Use conversation index |
| `greeting`/`small_talk`/`thanks`/`farewell` | (existing) | Instant response |
| `document_query` | (default) all other queries | Standard RAG retrieval |

**Algorithm:**
```
detect_intent(query, session=None):
  1. Check greeting/small_talk/thanks/farewell (existing)
  2. Check meta_conversation (existing)
  3. Check follow_up patterns → return ("follow_up", None)
  4. Check reference_previous patterns → return ("reference_previous", None)
  5. Check drawing_name regex → return ("drawing_specific", extracted_name)
  6. Check drawing_title patterns → return ("drawing_title_specific", extracted_title)
  7. Default → return ("document_query", None)
```

### 2. Query Augmentation for Follow-ups (`rag/api/generation_unified.py`)

**When intent = `follow_up` or `reference_previous`:**

```python
if intent_type in ("follow_up", "reference_previous") and session:
    last_user_msg = session.get_last_user_message()
    last_assistant_msg = session.get_last_assistant_message()

    if intent_type == "follow_up":
        # Augment: "tell me more" → "tell me more about [previous topic]"
        augmented_query = f"{user_query}. Context: The user previously asked '{last_user_msg}' and received this answer: '{last_assistant_msg[:800]}'"

    elif intent_type == "reference_previous":
        # Augment: "what size" → "what size was mentioned in: [previous answer]"
        augmented_query = f"{user_query}. Reference: In the previous response, the answer was: '{last_assistant_msg[:800]}'"

    # Use augmented_query for FAISS retrieval
    retrieval_query = augmented_query
```

### 3. Drawing Name/Title Filtering in Retrieval (`rag/retrieval/engine.py`)

**Soft filter with boost (post-FAISS):**

```python
def retrieve_context(query, ..., filter_drawing_name=None, filter_drawing_title=None):
    # Standard FAISS search (unchanged)
    results = faiss_search(query_vec, k_search)

    # Post-filter: boost matching drawings
    if filter_drawing_name or filter_drawing_title:
        for result in results:
            meta_name = result.get("drawing_name", "").upper()
            meta_title = result.get("drawing_title", "").upper()

            if filter_drawing_name and filter_drawing_name.upper() in meta_name:
                result["similarity"] *= 1.5  # 50% boost
            if filter_drawing_title and filter_drawing_title.upper() in meta_title:
                result["similarity"] *= 1.5  # 50% boost

        # Re-sort by boosted similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
```

### 4. Enhanced Conversation Context (`rag/api/generation_unified.py`)

**Current:** 150 chars per message, last 4 messages
**New:** Full last answer + 300 chars for older messages

```python
def build_conversation_context(session, max_tokens=3000):
    messages = session.messages
    context_parts = []

    # Last Q+A pair: FULL content (no truncation)
    if len(messages) >= 2:
        last_user = get_last_by_role(messages, "user")
        last_assistant = get_last_by_role(messages, "assistant")
        context_parts.append(f"[LAST QUESTION]: {last_user.content}")
        context_parts.append(f"[LAST ANSWER]: {last_assistant.content}")

    # Older messages: 300 chars each
    for msg in older_messages:
        context_parts.append(f"{msg.role}: {msg.content[:300]}")

    # Conversation index (numbered list of all user questions)
    context_parts.append(session.get_conversation_index())

    # Trim to token budget
    return trim_to_tokens(context_parts, max_tokens=3000)
```

### 5. Updated Prompt Instructions (`rag/api/prompts.py`)

Add explicit instructions for follow-up handling:

```
CONVERSATION AWARENESS RULES:
1. When the user says "tell me more", "continue", "elaborate" — expand on your PREVIOUS answer using the LAST ANSWER shown above.
2. When the user references something you said ("what size", "you mentioned") — find the specific detail in your LAST ANSWER.
3. When the user asks about a specific drawing by name (e.g., "drawing A0.01") — prioritize chunks from that drawing.
4. ALWAYS maintain continuity — you are having an ongoing conversation, not answering isolated questions.
5. If the user's query is ambiguous, interpret it in the context of the previous conversation.
```

---

## Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `rag/api/intent.py` | Add follow_up, reference_previous, drawing_specific intent types | New patterns ~lines 57-107 |
| `rag/api/generation_unified.py` | Query augmentation for follow-ups; enhanced conversation context builder | ~lines 169-320 |
| `rag/retrieval/engine.py` | Add filter_drawing_name/filter_drawing_title params with soft boost | ~lines 21-142 |
| `rag/api/prompts.py` | Add conversation awareness rules to all prompt templates | ~lines 54-165 |
| `rag/api/helpers.py` | Add drawing name extraction helper | New function |
| `memory_manager.py` | Add get_last_user_message(), get_last_assistant_message() helpers | New methods |

---

## Testing Scenarios

After implementation, these should all pass:

| # | Query | Expected Response |
|---|-------|-------------------|
| 1 | "How much gap between pipe and conduit?" | "Not more than 1/2 inch" (with source) |
| 2 | "Tell me more about it" | Expands on the gap topic with additional details from related chunks |
| 3 | "What size you mentioned?" | "The size mentioned was 1/2 inch (not more than 1/2 inch gap)" |
| 4 | "What are the notes on drawing A0.01?" | Retrieves chunks specifically from A0.01 drawing |
| 5 | "Notes on Partition Schedule?" | Retrieves chunks from Partition Schedule drawing title |
| 6 | "Continue" | Expands further on the previous answer |
| 7 | "What was my first question?" | Returns the first question from conversation index |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Query augmentation makes retrieval queries too long | Cap augmented query at 500 tokens |
| Drawing name regex matches false positives | Only activate drawing filter if pattern is at least 2 chars with a digit |
| Soft boost gives irrelevant results from wrong drawing | Fall back to unfiltered if boosted results have < 0.2 similarity |
| Increased token budget slows response | 3000 tokens conversation context is within gpt-4o's 128K window |
| Follow-up detection misclassifies normal queries | Regex patterns are specific enough; "tell me more" is unambiguous |
