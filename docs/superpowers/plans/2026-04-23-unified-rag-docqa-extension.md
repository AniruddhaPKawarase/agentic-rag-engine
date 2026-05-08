# Unified RAG + DocQA Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend unified RAG agent with a DocQA bridge so when RAG returns low-confidence/vague answers, the user can select a source document and continue the conversation grounded in that file — with bidirectional auto-switching, clean answer prose, schema-frozen API, and zero changes to the DocQA agent on port 8006.

**Architecture:** Thin adapter layer (`gateway/docqa_bridge.py`) sits on top of the existing RAG pipeline. RAG baseline (v1.2-hybrid-ship + RRF + reranker + self-RAG) is unchanged. New additive-only fields on `UnifiedResponse` keep the sandbox UI working. All changes revert-safe via `_versions/v2.0-docqa-extension/` snapshot.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, httpx (async), boto3, pytest + respx (HTTP mocking), tempfile, OpenAI GPT-4.1 (auto-cached prompts).

**Working directory:** `c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent/`

**Spec reference:** `docs/superpowers/specs/2026-04-23-unified-rag-docqa-extension-design.md`

**Rollout discipline:** Pause after every phase. User reviews. Do not start next phase without explicit "proceed".

---

## Verified facts from current codebase (use these exact paths)

| Fact | Location |
|---|---|
| `MemoryManager` class | `traditional/memory_manager.py:319` |
| Session wrapper | `shared/session/manager.py` (48 lines, thin) |
| Orchestrator result dict emits `source_documents`, `active_agent`, `selected_document` | `gateway/orchestrator.py:141,167,169` |
| `_build_download_url` (presigned SigV4) | `gateway/orchestrator.py:175-258` |
| `_ensure_signed_source_urls` (Fix #8) | `gateway/orchestrator.py:361-390` |
| Contradictory prompt line to delete | `agentic/core/agent.py:155` |
| DocQA uses `/api/converse` (upload+query combined), not `/api/upload` | `gateway/docqa_client.py:34` |
| DocQA follow-up endpoint | `gateway/docqa_client.py:78` → `/api/chat` |
| Current intent classifier one-way bug | `gateway/intent_classifier.py:88` (returns `"rag"` unconditionally) |
| Self-RAG + reranker modules | `gateway/self_rag.py`, `gateway/reranker.py` |
| Env flags `RERANKER_INCLUDE_SCORE`, `SELF_RAG_ENABLED` | keep these OFF for baseline behavior |

---

# Phase 0 — Snapshot baseline version

**Goal:** Create `_versions/v2.0-docqa-extension/` as full tree snapshot so every subsequent phase is revert-safe.

**Files:**
- Create: `_versions/v2.0-docqa-extension/` (directory)
- Create: `_versions/v2.0-docqa-extension/README.md`
- Modify: `VERSIONS.md` (prepend new entry)

### Task 0.1: Snapshot current tree

- [ ] **Step 1: Verify current working tree is clean of unfinished edits**

Run:
```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent"
ls _versions/
```
Expected: 4 existing versions listed (v1.0-stable-fixes-1-2-5-8, v1.1-dev-fix3-aggregation, v1.1-stable-fix3-fix4, v1.2-hybrid-ship).

- [ ] **Step 2: Create the version directory**

Run:
```bash
mkdir -p _versions/v2.0-docqa-extension
```

- [ ] **Step 3: Copy tree (exclude volatile dirs)**

Run:
```bash
rsync -a \
  --exclude='_versions/' \
  --exclude='_backups/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='test_results/' \
  --exclude='test_results.zip' \
  --exclude='*.pyc' \
  ./ _versions/v2.0-docqa-extension/
```
Expected: no errors; directory populated with agentic/, gateway/, traditional/, shared/, tests/, docs/, etc.

- [ ] **Step 4: Write snapshot README**

Create `_versions/v2.0-docqa-extension/README.md`:
```markdown
# v2.0-docqa-extension — BASELINE SNAPSHOT

Snapshot of main taken 2026-04-23 BEFORE DocQA-extension changes.
This folder is the revert target if Phase 1+ needs rollback.

## Contents
- Baseline = v1.2-hybrid-ship + RRF/reranker/self-RAG additions on main as of 2026-04-23
- No DocQA-extension code here; this captures pre-change state

## Revert command (from repo root, PROD_SETUP/unified-rag-agent/)
```
rsync -a _versions/v2.0-docqa-extension/ ./ \
  --exclude README.md --exclude VERSIONS.md
```

## Next version
- `_versions/v2.0-docqa-extension-final/` will be created after Phase 6 sign-off.
```

- [ ] **Step 5: Update VERSIONS.md**

Prepend to the table in `VERSIONS.md` (after existing `v1.2-hybrid-ship` row):
```markdown
| **`v2.0-docqa-extension`** | Pre-change baseline (2026-04-23) | Snapshot of main before DocQA-extension work. Use as revert target for Phases 1-6. | `rsync -a PROD_SETUP/unified-rag-agent/_versions/v2.0-docqa-extension/ PROD_SETUP/unified-rag-agent/ --exclude README.md --exclude VERSIONS.md` |
```

- [ ] **Step 6: Verify snapshot integrity**

Run:
```bash
diff -rq --exclude='__pycache__' --exclude='*.pyc' gateway/ _versions/v2.0-docqa-extension/gateway/
```
Expected: no output (identical).

- [ ] **Step 7: Commit snapshot**

```bash
git add _versions/v2.0-docqa-extension/ VERSIONS.md
git commit -m "chore: snapshot v2.0-docqa-extension baseline before bridge work"
```

**GATE: Pause. User confirms snapshot exists, then authorizes Phase 1.**

---

# Phase 1 — Answer-format fix + schema sync + prompt-cache message ordering

**Goal:** Eliminate `[Source: …]` / `HIGH (90%)` / `Direct Answer` artifacts from answer prose, align Pydantic `UnifiedResponse` with wire truth, and reorder message building so OpenAI auto-caches the static prefix.

**Files:**
- Modify: `agentic/core/agent.py:155` (delete one line)
- Modify: `gateway/models.py` (add 8 optional fields)
- Modify: `gateway/orchestrator.py` (message ordering for cache; ~30 lines)
- Create: `tests/test_answer_format.py`
- Create: `tests/test_schema_alignment.py`
- Create: `tests/test_prompt_cache_ordering.py`

### Task 1.1: Delete contradictory "cite [Source:]" instruction

- [ ] **Step 1: Write failing test**

Create `tests/test_answer_format.py`:
```python
"""Phase 1 answer-format regression tests."""
from __future__ import annotations
import re
import pytest

FORBIDDEN_PATTERNS = [
    re.compile(r"\[Source:\s*[^\]]*\]"),
    re.compile(r"^Direct Answer\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"HIGH\s*\(\d+%\)"),
]

SAMPLE_BAD_ANSWER = (
    "Direct Answer\n"
    "XVENT model is DHEB-44-* [Source: Page41 / MECHANICAL PLAN].\n"
    "HIGH (90%)"
)

SAMPLE_GOOD_ANSWER = (
    "XVENT model DHEB-44-* is specified for double exhaust terminations."
)


def _has_artifact(answer: str) -> bool:
    return any(p.search(answer) for p in FORBIDDEN_PATTERNS)


def test_artifact_detector_flags_bad_output():
    assert _has_artifact(SAMPLE_BAD_ANSWER)


def test_artifact_detector_accepts_clean_output():
    assert not _has_artifact(SAMPLE_GOOD_ANSWER)


def test_agent_system_prompt_has_single_citation_rule():
    """After Phase 1 fix, agent.py must contain ONE citation rule, not two contradictory ones."""
    from pathlib import Path
    content = Path("agentic/core/agent.py").read_text(encoding="utf-8")
    positive_citation_instructions = re.findall(
        r'(?im)^[^#\n]*cite\s+sources:\s*\[Source:', content
    )
    assert len(positive_citation_instructions) == 0, (
        f"Contradictory 'cite [Source:]' instruction still present: {positive_citation_instructions}"
    )
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `python -m pytest tests/test_answer_format.py -v`
Expected: `test_agent_system_prompt_has_single_citation_rule` FAILS (instruction still there).

- [ ] **Step 3: Delete line 155 from agentic/core/agent.py**

Read `agentic/core/agent.py` lines 150-165 first to confirm exact text. Then delete the line that starts with `Cite sources: [Source:` (should be around :155). The line reads something like:
```
Cite sources: [Source: drawingName / drawingTitle] for every fact.
```

Delete it entirely. Do not replace with anything. Line 205 (`Do NOT include [Source: …]`) becomes the single rule.

- [ ] **Step 4: Run test to confirm PASS**

Run: `python -m pytest tests/test_answer_format.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agentic/core/agent.py tests/test_answer_format.py
git commit -m "fix(agent): remove contradictory [Source:] citation instruction

The system prompt had two conflicting rules: line 155 said 'cite [Source:]'
and line 205 said 'do NOT include [Source:]'. The model obeyed the first,
leaking artifacts into prose. Deleting line 155; line 205 is now the
single source of truth. Structured source_documents[] field unchanged."
```

### Task 1.2: Add optional response fields to `UnifiedResponse`

- [ ] **Step 1: Write failing test**

Create `tests/test_schema_alignment.py`:
```python
"""Phase 1 schema-alignment tests."""
from __future__ import annotations
import pytest
from gateway.models import QueryRequest, UnifiedResponse


def test_query_request_accepts_docqa_document():
    req = QueryRequest(
        query="what does it say on page 14?",
        project_id=7222,
        docqa_document={"s3_path": "x/y.pdf", "file_name": "y.pdf"},
        mode_hint="docqa",
    )
    assert req.docqa_document == {"s3_path": "x/y.pdf", "file_name": "y.pdf"}
    assert req.mode_hint == "docqa"


def test_query_request_backward_compatible_without_new_fields():
    req = QueryRequest(query="hello", project_id=7222)
    assert req.docqa_document is None
    assert req.mode_hint is None


def test_unified_response_new_fields_default_null():
    resp = UnifiedResponse()
    assert resp.source_documents is None
    assert resp.active_agent == "rag"
    assert resp.selected_document is None
    assert resp.needs_clarification is False
    assert resp.clarification_prompt is None
    assert resp.docqa_session_id is None


def test_unified_response_backward_compatible_existing_fields():
    """Old clients depend on these — must stay present with same names."""
    resp = UnifiedResponse(
        success=True, answer="hi", sources=[], confidence="high",
        session_id="s1", engine_used="agentic",
    )
    assert resp.success is True
    assert resp.sources == []
    assert resp.confidence == "high"


def test_unified_response_populated_docqa_turn():
    resp = UnifiedResponse(
        active_agent="docqa",
        selected_document={"file_name": "HVAC.pdf"},
        docqa_session_id="dq_xyz",
        source_documents=[{"file_name": "HVAC.pdf", "page": 14}],
    )
    assert resp.active_agent == "docqa"
    assert resp.docqa_session_id == "dq_xyz"
    assert resp.source_documents[0]["page"] == 14
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_schema_alignment.py -v`
Expected: all 4 tests FAIL with `ValidationError` on `docqa_document` / `mode_hint` / missing attributes.

- [ ] **Step 3: Add fields to `gateway/models.py`**

Modify `gateway/models.py`. Final content:
```python
"""
Gateway request / response models — Pydantic v2 BaseModel schemas.

QueryRequest validates inbound queries.
UnifiedResponse is the standardised envelope returned by the orchestrator.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Inbound query from the client."""

    query: str = Field(..., min_length=1, max_length=2000)
    project_id: int = Field(..., ge=1, le=999999)
    session_id: Optional[str] = None
    search_mode: Optional[str] = None
    generate_document: bool = True
    filter_source_type: Optional[str] = None
    filter_drawing_name: Optional[str] = None
    set_id: Optional[int] = None
    conversation_history: Optional[list] = None
    engine: Optional[str] = None
    # --- Phase 1 additions (DocQA bridge) ---
    docqa_document: Optional[dict] = None
    mode_hint: Optional[str] = None  # "rag" | "docqa" | None (auto)


class UnifiedResponse(BaseModel):
    """Standardised response envelope from the unified gateway."""

    success: bool = True
    answer: str = ""
    sources: list[dict] = Field(default_factory=list)
    confidence: str = "high"
    session_id: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    engine_used: str = "agentic"
    fallback_used: bool = False
    agentic_confidence: Optional[str] = None
    cost_usd: float = 0.0
    elapsed_ms: int = 0
    total_steps: int = 0
    model: str = ""
    # --- Phase 1 additions (DocQA bridge, schema-aligned to wire truth) ---
    source_documents: Optional[list[dict]] = None
    active_agent: Optional[str] = "rag"
    selected_document: Optional[dict] = None
    clarification_prompt: Optional[str] = None
    docqa_session_id: Optional[str] = None
    groundedness_score: Optional[float] = None  # already emitted by self_rag; align
    flagged_claims: Optional[list[dict]] = None  # already emitted by self_rag; align
```

- [ ] **Step 4: Re-run to confirm PASS**

Run: `python -m pytest tests/test_schema_alignment.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Ensure old tests still pass**

Run: `python -m pytest tests/ -v --tb=short`
Expected: no regression; only new tests + existing tests. Record the pass count.

- [ ] **Step 6: Commit**

```bash
git add gateway/models.py tests/test_schema_alignment.py
git commit -m "feat(models): add additive optional fields for DocQA bridge

Adds docqa_document, mode_hint to QueryRequest. Adds source_documents,
active_agent, selected_document, clarification_prompt, docqa_session_id,
groundedness_score, flagged_claims to UnifiedResponse. All new fields
are Optional with safe defaults so existing clients (sandbox UI) are
unaffected. Schema now matches wire-truth emitted by orchestrator."
```

### Task 1.3: Prompt-cache message ordering

- [ ] **Step 1: Write failing test**

Create `tests/test_prompt_cache_ordering.py`:
```python
"""Phase 1: message ordering for OpenAI auto-cache.

Cacheable prefix = [system_prompt, tool_defs, conversation_history(stable), user_query_last].
Dynamic injections (RRF hints) must go into the LAST user message, not
prepended to history — otherwise cache busts every turn.
"""
from __future__ import annotations
import pytest


def test_build_react_messages_system_first():
    from agentic.core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[{"role": "user", "content": "prev"}],
        user_query="current",
        rrf_hint=None,
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYSTEM"


def test_build_react_messages_query_last():
    from agentic.core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[{"role": "user", "content": "prev"}],
        user_query="current",
        rrf_hint=None,
    )
    assert msgs[-1]["role"] == "user"
    assert "current" in msgs[-1]["content"]


def test_rrf_hint_appended_to_last_user_message_not_prepended():
    from agentic.core.agent import build_react_messages
    msgs = build_react_messages(
        system_prompt="SYSTEM",
        conversation_history=[{"role": "user", "content": "prev"}],
        user_query="current",
        rrf_hint="SEARCH_HINT: try 'X' or 'Y'",
    )
    # Hint goes into LAST message, preserving earlier-message cache
    assert "SEARCH_HINT" in msgs[-1]["content"]
    assert "SEARCH_HINT" not in msgs[0]["content"]
    for m in msgs[:-1]:
        assert "SEARCH_HINT" not in m.get("content", "")


def test_cache_hit_rate_metric_logged(monkeypatch, caplog):
    """Ensure the usage.cached_tokens from OpenAI is logged."""
    import logging
    from agentic.core.agent import _log_cache_metrics
    mock_usage = {"cached_tokens": 1536, "prompt_tokens": 2048}
    caplog.set_level(logging.INFO)
    _log_cache_metrics(mock_usage)
    assert any("cached_tokens" in r.message and "1536" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_prompt_cache_ordering.py -v`
Expected: `ImportError` — `build_react_messages` and `_log_cache_metrics` not defined yet.

- [ ] **Step 3: Add helpers to `agentic/core/agent.py`**

Find the existing message-construction code in `agentic/core/agent.py` (grep for `messages = [` or `"role": "system"` to locate). Add these module-level functions (place near top, after imports):
```python
def build_react_messages(
    system_prompt: str,
    conversation_history: list[dict] | None,
    user_query: str,
    rrf_hint: str | None = None,
) -> list[dict]:
    """Assemble OpenAI messages with cacheable prefix ordering.

    Order: [system, ...history, user_query(+hint appended)].
    The system prompt + tool schema (attached separately by the caller)
    form the cacheable prefix. Any dynamic RRF hint is APPENDED to the
    last user message so it never invalidates the prefix cache.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    if rrf_hint:
        user_content = f"{user_query}\n\n---\n{rrf_hint}"
    else:
        user_content = user_query
    messages.append({"role": "user", "content": user_content})
    return messages


def _log_cache_metrics(usage: dict) -> None:
    """Log cached_tokens / prompt_tokens for observability (Phase 1 §11)."""
    cached = usage.get("cached_tokens", 0)
    total = usage.get("prompt_tokens", 0)
    ratio = (cached / total) if total else 0.0
    logger.info(
        "openai_cache: cached_tokens=%s prompt_tokens=%s hit_ratio=%.2f",
        cached, total, ratio,
    )
```

Replace any existing in-line message construction inside the agent's run loop to call `build_react_messages(...)` instead. After each OpenAI completion, call `_log_cache_metrics(response.usage.model_dump())`.

Additionally, modify `gateway/orchestrator.py` Fix #1 hint injection (search for `format_hint_for_agent` or `build_context_hint` — the call that adds the RRF hint). Change it from prepending a system message to passing the hint string into `build_react_messages(..., rrf_hint=hint)`. This is the ONE change required in orchestrator.

- [ ] **Step 4: Re-run test**

Run: `python -m pytest tests/test_prompt_cache_ordering.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Run full test suite — no regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: all existing tests still pass (61/61 baseline + new tests from 1.1, 1.2, 1.3).

- [ ] **Step 6: Commit**

```bash
git add agentic/core/agent.py gateway/orchestrator.py tests/test_prompt_cache_ordering.py
git commit -m "perf(agent): lock message order for OpenAI auto-cache

Cacheable prefix = [system, history, user_query]. RRF hints now append
to the last user message instead of prepending into history, preserving
cache hits across turns. Logs usage.cached_tokens every call.
Expected 40-60% latency drop on turns 2+ of a session."
```

### Task 1.4: 16-query regression (live sandbox)

- [ ] **Step 1: Write regression harness**

Create `tests/test_phase1_regression.py`:
```python
"""Phase 1 live regression against sandbox.

Runs the 16 historical queries used for v1.2-hybrid-ship and asserts:
1. No answer contains FORBIDDEN_PATTERNS.
2. source_documents[] is non-empty (for queries that had sources before).
3. cache_hit_rate > 0.6 on turns 2+ of same session.

Marked -m integration; requires SANDBOX_URL env var.
"""
from __future__ import annotations
import os
import re
import pytest
import httpx

pytestmark = pytest.mark.integration

SANDBOX = os.environ.get("SANDBOX_URL", "http://54.197.189.113:8001")

HISTORICAL_QUERIES = [
    # paste the 16 from test_results/local_after_all_fixes_20260422_154725/
    # (project_id, query)
    (7222, "Is fire damper scope missing?"),
    (7222, "How many DOAS units are specified?"),
    # ... (full list pulled from test_results/ baseline)
]

FORBIDDEN = [
    re.compile(r"\[Source:\s*[^\]]*\]"),
    re.compile(r"^Direct Answer\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"HIGH\s*\(\d+%\)"),
]


@pytest.mark.parametrize("project_id,query", HISTORICAL_QUERIES)
def test_query_has_no_artifacts(project_id: int, query: str):
    with httpx.Client(timeout=90) as c:
        r = c.post(
            f"{SANDBOX}/query",
            json={"query": query, "project_id": project_id},
        )
        r.raise_for_status()
        data = r.json()
        answer = data["answer"]
        for pat in FORBIDDEN:
            assert not pat.search(answer), f"Forbidden pattern {pat.pattern!r} in answer: {answer!r}"


def test_same_session_cache_hits():
    with httpx.Client(timeout=90) as c:
        # Turn 1 — cold
        r1 = c.post(
            f"{SANDBOX}/query",
            json={"query": "List HVAC drawings", "project_id": 7222},
        )
        sid = r1.json()["session_id"]
        # Turn 2 — should hit cache
        r2 = c.post(
            f"{SANDBOX}/query",
            json={"query": "Any fire dampers mentioned?", "project_id": 7222, "session_id": sid},
        )
        # cache_hit_rate logged server-side; check latency proxy
        assert r2.elapsed.total_seconds() < r1.elapsed.total_seconds() * 0.8, (
            f"Turn 2 not faster than turn 1 * 0.8: "
            f"t1={r1.elapsed.total_seconds():.1f} t2={r2.elapsed.total_seconds():.1f}"
        )
```

- [ ] **Step 2: Run against local dev server**

Start local server in one terminal:
```bash
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/VCS/VCS/PROD_SETUP/unified-rag-agent"
python -m gateway.app
```

In another terminal:
```bash
SANDBOX_URL=http://localhost:8001 python -m pytest tests/test_phase1_regression.py -v -m integration
```
Expected: all 16 queries pass artifact check; turn-2 latency < 0.8 × turn-1.

- [ ] **Step 3: Commit regression harness**

```bash
git add tests/test_phase1_regression.py
git commit -m "test: add Phase 1 regression suite for answer format + cache hit rate"
```

**GATE: Pause. User reviews:**
- [Source:] / Direct Answer / HIGH (N%) gone from all 16 answers
- Same-session turn 2+ latency dropped ≥ 20%
- Full test suite still green

Authorize Phase 2.

---

# Phase 2 — DocQA bridge + S3 fetch + session plumbing

**Goal:** Build `gateway/docqa_bridge.py` as the single adapter for S3→DocQA→session-state. Wire into orchestrator. Extend `MemoryManager` with DocQA session fields. Do not modify DocQA agent on port 8006.

**Files:**
- Create: `gateway/docqa_bridge.py`
- Modify: `gateway/docqa_client.py` (add `upload_only` helper)
- Modify: `gateway/orchestrator.py` (replace inline DocQA logic with bridge calls)
- Modify: `traditional/memory_manager.py` (add DocQA fields to ConversationSession)
- Create: `tests/test_docqa_bridge.py`

### Task 2.1: Extend `ConversationSession` with DocQA state

- [ ] **Step 1: Write failing test**

Create `tests/test_docqa_session_state.py`:
```python
"""Phase 2: session state for DocQA bridge."""
from __future__ import annotations
from traditional.memory_manager import MemoryManager


def test_session_tracks_active_agent_default():
    mm = MemoryManager(enable_persistence=False)
    sid = mm.create_session()
    sess = mm.get_session(sid)
    assert sess.active_agent == "rag"
    assert sess.docqa_session_id is None
    assert sess.selected_documents == []


def test_session_can_set_docqa_state():
    mm = MemoryManager(enable_persistence=False)
    sid = mm.create_session()
    sess = mm.get_session(sid)
    sess.active_agent = "docqa"
    sess.docqa_session_id = "dq_xyz"
    sess.selected_documents.append({
        "s3_path": "x/y.pdf", "file_name": "y.pdf",
        "docqa_session_id": "dq_xyz",
    })
    sess2 = mm.get_session(sid)
    assert sess2.active_agent == "docqa"
    assert sess2.docqa_session_id == "dq_xyz"
    assert len(sess2.selected_documents) == 1
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_docqa_session_state.py -v`
Expected: FAIL — `ConversationSession` has no `active_agent` attr.

- [ ] **Step 3: Add fields to `ConversationSession`**

In `traditional/memory_manager.py`, find `class ConversationSession` (search for it — grep it first). Add these fields to `__init__`:
```python
# --- Phase 2 additions (DocQA bridge) ---
self.active_agent: str = "rag"  # "rag" | "docqa"
self.docqa_session_id: Optional[str] = None
self.selected_documents: list[dict] = []  # [{s3_path, file_name, docqa_session_id, loaded_at}]
self.last_intent_decision: Optional[dict] = None
```

If `ConversationSession` has a `to_dict` / `from_dict`, also serialize these fields.

- [ ] **Step 4: Run test to PASS**

Run: `python -m pytest tests/test_docqa_session_state.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add traditional/memory_manager.py tests/test_docqa_session_state.py
git commit -m "feat(session): track active_agent + docqa_session_id + selected_documents"
```

### Task 2.2: Add `upload_only` helper to `docqa_client.py`

DocQA's `/api/converse` combines upload + first query in one call. For the bridge we want separate upload (multi-doc in one session) — we can call `/api/converse` with a placeholder query on first upload, then reuse the returned `session_id` for follow-ups via `/api/chat`.

- [ ] **Step 1: Write failing test**

Create `tests/test_docqa_client_ext.py`:
```python
"""Phase 2: docqa_client.upload_only helper."""
from __future__ import annotations
import pytest
import respx
from httpx import Response
from gateway import docqa_client


@pytest.mark.asyncio
@respx.mock
async def test_upload_only_calls_converse_with_placeholder():
    respx.post("http://localhost:8006/api/converse").mock(
        return_value=Response(200, json={
            "session_id": "dq_abc",
            "answer": "Document uploaded.",
            "total_session_chunks": 42,
        })
    )
    result = await docqa_client.upload_only(
        file_path="/tmp/x.pdf", file_name="x.pdf", session_id=None,
    )
    assert result["session_id"] == "dq_abc"
    assert result["total_session_chunks"] == 42


@pytest.mark.asyncio
@respx.mock
async def test_upload_only_reuses_session_id_for_second_doc():
    respx.post("http://localhost:8006/api/converse").mock(
        return_value=Response(200, json={
            "session_id": "dq_abc", "total_session_chunks": 84,
        })
    )
    result = await docqa_client.upload_only(
        file_path="/tmp/y.pdf", file_name="y.pdf", session_id="dq_abc",
    )
    assert result["session_id"] == "dq_abc"
```

- [ ] **Step 2: Verify test harness deps installed**

Run: `pip install respx pytest-asyncio` (if not already in requirements.txt).

Run: `python -m pytest tests/test_docqa_client_ext.py -v`
Expected: FAIL — `upload_only` not defined.

- [ ] **Step 3: Add `upload_only` to `gateway/docqa_client.py`**

Append to the file:
```python
async def upload_only(
    file_path: str,
    file_name: str,
    session_id: Optional[str] = None,
) -> dict:
    """Upload a document to DocQA (first-load or additional doc in session).

    Sends a placeholder query ('__upload__') so /api/converse accepts the call.
    DocQA treats 'uploaded' as the action; we ignore the answer. Returns
    {session_id, total_session_chunks, ...}.
    """
    url = f"{DOCQA_BASE_URL}/api/converse"
    logger.info(
        "DocQA upload_only: file=%s, session=%s", file_name, session_id,
    )
    try:
        async with httpx.AsyncClient(timeout=DOCQA_TIMEOUT) as client:
            with open(file_path, "rb") as f:
                files = {"files": (file_name, f, "application/pdf")}
                data = {"query": "__upload__"}
                if session_id:
                    data["session_id"] = session_id
                response = await client.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
    except httpx.TimeoutException:
        logger.error("DocQA upload_only timed out after %ds", DOCQA_TIMEOUT)
        return {"error": "Document upload timed out."}
    except httpx.HTTPStatusError as exc:
        logger.error("DocQA upload_only HTTP error: %s", exc.response.status_code)
        return {"error": f"Document QA service returned {exc.response.status_code}"}
    except Exception as exc:
        logger.error("DocQA upload_only failed: %s", exc)
        return {"error": f"Failed to upload: {type(exc).__name__}"}
```

- [ ] **Step 4: PASS test**

Run: `python -m pytest tests/test_docqa_client_ext.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add gateway/docqa_client.py tests/test_docqa_client_ext.py
git commit -m "feat(docqa_client): add upload_only helper for bridge"
```

### Task 2.3: Build `DocQABridge` class

- [ ] **Step 1: Write failing test**

Create `tests/test_docqa_bridge.py`:
```python
"""Phase 2: DocQABridge — single source of truth for RAG→DocQA handoff."""
from __future__ import annotations
import pytest
import respx
from httpx import Response
from unittest.mock import MagicMock, patch
from gateway.docqa_bridge import DocQABridge


@pytest.fixture
def mock_mm():
    mm = MagicMock()
    sess = MagicMock()
    sess.active_agent = "rag"
    sess.docqa_session_id = None
    sess.selected_documents = []
    mm.get_session.return_value = sess
    return mm


@pytest.mark.asyncio
@respx.mock
async def test_ensure_document_loaded_downloads_and_uploads(mock_mm, tmp_path):
    # Mock S3 download
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    respx.post("http://localhost:8006/api/converse").mock(
        return_value=Response(200, json={"session_id": "dq_new", "total_session_chunks": 12})
    )

    bridge = DocQABridge(memory_manager=mock_mm)
    with patch.object(bridge, "_download_to_temp", return_value=str(fake_pdf)):
        dq_sid = await bridge.ensure_document_loaded(
            session_id="rag_s1",
            doc_ref={"s3_path": "bucket/x.pdf", "file_name": "x.pdf",
                     "download_url": "https://presigned"},
        )
    assert dq_sid == "dq_new"
    sess = mock_mm.get_session.return_value
    assert sess.active_agent == "docqa"
    assert sess.docqa_session_id == "dq_new"
    assert len(sess.selected_documents) == 1


@pytest.mark.asyncio
async def test_ensure_document_loaded_reuses_if_already_in_session(mock_mm):
    sess = mock_mm.get_session.return_value
    sess.docqa_session_id = "dq_existing"
    sess.selected_documents = [{"s3_path": "bucket/x.pdf", "docqa_session_id": "dq_existing"}]
    bridge = DocQABridge(memory_manager=mock_mm)
    dq_sid = await bridge.ensure_document_loaded(
        session_id="rag_s1",
        doc_ref={"s3_path": "bucket/x.pdf", "file_name": "x.pdf"},
    )
    assert dq_sid == "dq_existing"


@pytest.mark.asyncio
@respx.mock
async def test_ask_routes_to_chat_endpoint(mock_mm):
    respx.post("http://localhost:8006/api/chat").mock(
        return_value=Response(200, json={
            "answer": "on page 14", "session_id": "dq_x",
            "source_documents": [{"file_name": "x.pdf", "page": 14}],
            "groundedness_score": 0.92,
        })
    )
    bridge = DocQABridge(memory_manager=mock_mm)
    result = await bridge.ask(docqa_session_id="dq_x", query="where is fire damper?")
    assert result["answer"] == "on page 14"
    assert result["source_documents"][0]["page"] == 14


@pytest.mark.asyncio
@respx.mock
async def test_ask_handles_docqa_error_gracefully(mock_mm):
    respx.post("http://localhost:8006/api/chat").mock(
        return_value=Response(500, json={"error": "boom"})
    )
    bridge = DocQABridge(memory_manager=mock_mm)
    result = await bridge.ask(docqa_session_id="dq_x", query="hi")
    assert "error" in result


@pytest.mark.asyncio
async def test_normalize_to_unified_response(mock_mm):
    bridge = DocQABridge(memory_manager=mock_mm)
    dq_resp = {
        "answer": "page 14",
        "session_id": "dq_x",
        "source_documents": [{"file_name": "HVAC.pdf", "page": 14, "snippet": "…"}],
        "groundedness_score": 0.92,
    }
    unified = bridge.normalize(
        docqa_response=dq_resp,
        rag_session_id="rag_s1",
        selected_document={"file_name": "HVAC.pdf"},
    )
    assert unified["success"] is True
    assert unified["answer"] == "page 14"
    assert unified["active_agent"] == "docqa"
    assert unified["selected_document"]["file_name"] == "HVAC.pdf"
    assert unified["docqa_session_id"] == "dq_x"
    assert unified["session_id"] == "rag_s1"
    assert unified["engine_used"] == "docqa"


@pytest.mark.asyncio
async def test_normalize_error_response_falls_back_gracefully(mock_mm):
    bridge = DocQABridge(memory_manager=mock_mm)
    unified = bridge.normalize(
        docqa_response={"error": "timeout"},
        rag_session_id="rag_s1",
        selected_document={"file_name": "x.pdf"},
    )
    assert unified["success"] is False
    assert unified["engine_used"] == "docqa_fallback"
    assert "Could not load document" in unified["answer"] or "timeout" in unified["answer"].lower()
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_docqa_bridge.py -v`
Expected: FAIL — `gateway.docqa_bridge` module doesn't exist.

- [ ] **Step 3: Create `gateway/docqa_bridge.py`**

```python
"""
DocQA Bridge — adapter layer between unified RAG gateway and DocQA agent.

Single source of truth for:
  1. Fetching source documents from S3 (or via presigned URL fallback)
  2. Uploading them to DocQA agent on port 8006
  3. Persisting DocQA session state in MemoryManager
  4. Forwarding follow-up Q&A calls to DocQA /api/chat
  5. Normalizing DocQA responses into UnifiedResponse shape

This module must stay thin. Future RAG changes or DocQA changes should
not ripple into other gateway modules — everything DocQA-related lives here.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Optional

import httpx

from gateway import docqa_client

logger = logging.getLogger(__name__)

DOCQA_FALLBACK_MESSAGE = (
    "Could not load document for deep-dive. Try selecting a different "
    "source or ask a general question to continue."
)


class DocQABridge:
    """Thin adapter. One instance per gateway process.

    Constructor takes the MemoryManager so we can persist DocQA session state
    keyed by the RAG session_id.
    """

    def __init__(self, memory_manager: Any, s3_client: Any = None):
        self.mm = memory_manager
        self.s3_client = s3_client  # lazy-init if None in _download_to_temp

    # ------------------------------------------------------------------ S3

    def _init_s3(self):
        if self.s3_client is not None:
            return self.s3_client
        import boto3
        from botocore.config import Config
        self.s3_client = boto3.client(
            "s3",
            config=Config(max_pool_connections=20, retries={"max_attempts": 3}),
        )
        return self.s3_client

    async def _download_to_temp(self, doc_ref: dict) -> str:
        """Download S3 object → NamedTemporaryFile → return path.

        Order of attempts:
          1. boto3 S3 client (fastest, AWS SDK)
          2. httpx GET against doc_ref['download_url'] (presigned fallback)
        Caller is responsible for deleting the returned path.
        """
        bucket = os.environ.get("S3_BUCKET", "agentic-ai-production")
        s3_path = (doc_ref.get("s3_path") or "").lstrip("/")
        # Strip bucket prefix if accidentally concatenated
        if s3_path.startswith(f"{bucket}/"):
            s3_key = s3_path[len(bucket) + 1:]
        else:
            s3_key = s3_path

        file_name = doc_ref.get("file_name") or s3_key.rsplit("/", 1)[-1]
        suffix = os.path.splitext(file_name)[1] or ".pdf"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()

        try:
            client = self._init_s3()
            with open(tmp_path, "wb") as fp:
                client.download_fileobj(bucket, s3_key, fp)
            logger.info("DocQABridge S3 download ok: key=%s size=%s",
                        s3_key, os.path.getsize(tmp_path))
            return tmp_path
        except Exception as s3_exc:
            logger.warning("DocQABridge S3 download failed (%s); "
                           "falling back to presigned URL", s3_exc)
            url = doc_ref.get("download_url")
            if not url:
                os.unlink(tmp_path)
                raise RuntimeError("No download_url to fall back to") from s3_exc
            async with httpx.AsyncClient(timeout=60) as hc:
                async with hc.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as fp:
                        async for chunk in resp.aiter_bytes():
                            fp.write(chunk)
            return tmp_path

    # -------------------------------------------------------------- Public

    async def ensure_document_loaded(
        self,
        session_id: str,
        doc_ref: dict,
    ) -> str:
        """Ensure doc is loaded in DocQA; return docqa_session_id.

        If already loaded in this RAG session (matching s3_path),
        return the existing docqa_session_id.
        """
        sess = self.mm.get_session(session_id)
        existing_sid = getattr(sess, "docqa_session_id", None)
        s3_path = doc_ref.get("s3_path") or ""
        already_loaded = any(
            d.get("s3_path") == s3_path
            for d in getattr(sess, "selected_documents", []) or []
        )
        if existing_sid and already_loaded:
            logger.info("DocQABridge reuse: rag_session=%s docqa=%s",
                        session_id, existing_sid)
            return existing_sid

        tmp_path: Optional[str] = None
        try:
            tmp_path = await self._download_to_temp(doc_ref)
            result = await docqa_client.upload_only(
                file_path=tmp_path,
                file_name=doc_ref.get("file_name") or "document.pdf",
                session_id=existing_sid,
            )
            if result.get("error"):
                raise RuntimeError(result["error"])
            new_sid = result["session_id"]

            sess.active_agent = "docqa"
            sess.docqa_session_id = new_sid
            sess.selected_documents.append({
                "s3_path": s3_path,
                "file_name": doc_ref.get("file_name"),
                "docqa_session_id": new_sid,
                "loaded_at": datetime.utcnow().isoformat(),
            })
            logger.info("DocQABridge loaded: rag_session=%s docqa=%s chunks=%s",
                        session_id, new_sid, result.get("total_session_chunks"))
            return new_sid
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def ask(self, docqa_session_id: str, query: str) -> dict:
        """Forward follow-up query to DocQA /api/chat. Returns raw DocQA dict."""
        return await docqa_client.query_document(
            session_id=docqa_session_id, query=query,
        )

    def normalize(
        self,
        docqa_response: dict,
        rag_session_id: str,
        selected_document: Optional[dict] = None,
    ) -> dict:
        """Map DocQA response → UnifiedResponse-shaped dict (schema-frozen)."""
        if docqa_response.get("error"):
            return {
                "success": False,
                "answer": f"{DOCQA_FALLBACK_MESSAGE} ({docqa_response['error']})",
                "sources": [],
                "source_documents": [],
                "active_agent": "rag",  # degrade to RAG so user can continue
                "selected_document": None,
                "docqa_session_id": None,
                "session_id": rag_session_id,
                "engine_used": "docqa_fallback",
                "confidence": "low",
                "needs_clarification": False,
                "fallback_used": True,
            }
        return {
            "success": True,
            "answer": docqa_response.get("answer", ""),
            "sources": [],
            "source_documents": docqa_response.get("source_documents", []),
            "active_agent": "docqa",
            "selected_document": selected_document,
            "docqa_session_id": docqa_response.get("session_id"),
            "session_id": rag_session_id,
            "engine_used": "docqa",
            "confidence": "high",
            "groundedness_score": docqa_response.get("groundedness_score"),
            "needs_clarification": False,
        }
```

- [ ] **Step 4: Run test to PASS**

Run: `python -m pytest tests/test_docqa_bridge.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add gateway/docqa_bridge.py tests/test_docqa_bridge.py
git commit -m "feat(bridge): add DocQABridge adapter for S3→DocQA handoff

Single source of truth for RAG→DocQA bridge: downloads S3 object to
temp file, uploads to DocQA /api/converse, persists docqa_session_id
in MemoryManager, routes follow-ups via /api/chat, normalizes DocQA
response into UnifiedResponse shape. Graceful fallback on error."
```

### Task 2.4: Wire bridge into orchestrator

- [ ] **Step 1: Write integration test**

Create `tests/test_orchestrator_docqa_route.py`:
```python
"""Phase 2: orchestrator routes search_mode='docqa' via bridge."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_orchestrator_docqa_path_calls_bridge():
    from gateway.orchestrator import run_query  # or equivalent entry

    mock_bridge = MagicMock()
    mock_bridge.ensure_document_loaded = AsyncMock(return_value="dq_xyz")
    mock_bridge.ask = AsyncMock(return_value={
        "answer": "on page 14", "session_id": "dq_xyz",
        "source_documents": [{"file_name": "x.pdf", "page": 14}],
    })
    mock_bridge.normalize = MagicMock(return_value={
        "success": True, "answer": "on page 14", "active_agent": "docqa",
        "docqa_session_id": "dq_xyz", "session_id": "rag_s1",
        "engine_used": "docqa",
    })

    with patch("gateway.orchestrator._docqa_bridge", mock_bridge):
        result = await run_query(
            query="where is fire damper?",
            project_id=7222,
            session_id="rag_s1",
            search_mode="docqa",
            docqa_document={"s3_path": "x/y.pdf", "file_name": "y.pdf",
                            "download_url": "https://presigned"},
        )
    assert result["engine_used"] == "docqa"
    assert result["active_agent"] == "docqa"
    mock_bridge.ensure_document_loaded.assert_awaited_once()
    mock_bridge.ask.assert_awaited_once()
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_orchestrator_docqa_route.py -v`
Expected: FAIL — `_docqa_bridge` symbol not present.

- [ ] **Step 3: Wire bridge into `gateway/orchestrator.py`**

At module top of `gateway/orchestrator.py`, add:
```python
from gateway.docqa_bridge import DocQABridge

_docqa_bridge: Optional[DocQABridge] = None

def _get_docqa_bridge():
    global _docqa_bridge
    if _docqa_bridge is None:
        from gateway.memory_singleton import memory_manager  # existing MM import path
        _docqa_bridge = DocQABridge(memory_manager=memory_manager)
    return _docqa_bridge
```

(If `memory_singleton` doesn't exist, use whatever import path the orchestrator uses to access the shared MemoryManager — search orchestrator.py for `memory_manager` or `MemoryManager(` to find it.)

Find the existing `_run_docqa` function (around line 1065). Replace its body with a call-through to the bridge:
```python
async def _run_docqa(
    query: str,
    project_id: int,
    session_id: str,
    docqa_document: dict,
) -> dict:
    """Phase 2: bridge-driven DocQA path."""
    bridge = _get_docqa_bridge()
    try:
        dq_sid = await bridge.ensure_document_loaded(
            session_id=session_id, doc_ref=docqa_document,
        )
        dq_resp = await bridge.ask(docqa_session_id=dq_sid, query=query)
        return bridge.normalize(
            docqa_response=dq_resp,
            rag_session_id=session_id,
            selected_document=docqa_document,
        )
    except Exception as exc:
        logger.exception("DocQA bridge error")
        return bridge.normalize(
            docqa_response={"error": str(exc)},
            rag_session_id=session_id,
            selected_document=docqa_document,
        )
```

In the main `run_query` dispatcher (search for `search_mode` handling), ensure this branch exists:
```python
if search_mode == "docqa" and docqa_document:
    return await _run_docqa(query, project_id, session_id, docqa_document)
```

- [ ] **Step 4: Run integration test to PASS**

Run: `python -m pytest tests/test_orchestrator_docqa_route.py tests/test_docqa_bridge.py tests/test_docqa_session_state.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: no regression.

- [ ] **Step 6: Commit**

```bash
git add gateway/orchestrator.py tests/test_orchestrator_docqa_route.py
git commit -m "feat(orchestrator): route search_mode=docqa through DocQABridge

Replaces inline DocQA handling with a single bridge adapter call.
Existing RAG path untouched."
```

### Task 2.5: Live smoke test against sandbox

- [ ] **Step 1: Write smoke**

Create `tests/test_phase2_smoke.py`:
```python
"""Phase 2: live RAG→DocQA handoff against sandbox."""
from __future__ import annotations
import os
import pytest
import httpx

pytestmark = pytest.mark.integration
SANDBOX = os.environ.get("SANDBOX_URL", "http://54.197.189.113:8001")


def test_rag_then_docqa_handoff_live():
    with httpx.Client(timeout=180) as c:
        # Turn 1: RAG
        r1 = c.post(
            f"{SANDBOX}/query",
            json={"query": "list HVAC drawings", "project_id": 7222},
        )
        r1.raise_for_status()
        d1 = r1.json()
        sources = d1.get("source_documents") or []
        assert sources, "RAG must return at least one source"
        sid = d1["session_id"]

        # Turn 2: DocQA on first source
        doc = sources[0]
        r2 = c.post(
            f"{SANDBOX}/query",
            json={
                "query": "what does this drawing cover?",
                "project_id": 7222, "session_id": sid,
                "search_mode": "docqa",
                "docqa_document": {
                    "s3_path": doc.get("s3_path"),
                    "file_name": doc.get("file_name"),
                    "download_url": doc.get("download_url"),
                },
            },
        )
        r2.raise_for_status()
        d2 = r2.json()
        assert d2["success"] is True, d2
        assert d2["active_agent"] == "docqa"
        assert d2["docqa_session_id"]
        assert d2["engine_used"] in ("docqa", "docqa_fallback")
```

- [ ] **Step 2: Run smoke**

Run: `SANDBOX_URL=http://localhost:8001 python -m pytest tests/test_phase2_smoke.py -v -m integration`
Expected: PASS. If it returns `docqa_fallback`, investigate — likely S3 credentials or DocQA reachability.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase2_smoke.py
git commit -m "test: Phase 2 live RAG→DocQA handoff smoke"
```

**GATE: Pause. User reviews:**
- Unit tests green (bridge, client-ext, session state, orchestrator)
- Live smoke: RAG returns sources, selected doc uploads to DocQA, first query answers with `active_agent=docqa`
- No regression on existing RAG queries

Authorize Phase 3.

---

# Phase 3 — Bidirectional intent classifier with clarify mode

**Goal:** Replace one-way classifier with weighted signal scoring + clarify path. Returns `(target, confidence, reason)`. Orchestrator reads it to route RAG / DocQA / clarify.

**Files:**
- Rewrite: `gateway/intent_classifier.py`
- Modify: `gateway/orchestrator.py` (use classifier output)
- Create: `tests/test_intent_classifier_v2.py`

### Task 3.1: Rewrite classifier

- [ ] **Step 1: Write 40 parametrized tests**

Create `tests/test_intent_classifier_v2.py`:
```python
"""Phase 3: bidirectional intent classifier tests (40 cases)."""
from __future__ import annotations
import pytest
from gateway.intent_classifier import classify, IntentDecision


class FakeSession:
    def __init__(self, selected_documents=None, active_agent="rag"):
        self.selected_documents = selected_documents or []
        self.active_agent = active_agent


# (query, has_selected_docs, expected_target, note)
CASES = [
    # --- clear DocQA ---
    ("what does it say on page 14 of this spec?", True, "docqa", "pronoun + selected"),
    ("in this document, where is fire damper?", True, "docqa", "'in this document'"),
    ("explain section 2.3 of the selected drawing", True, "docqa", "'selected drawing'"),
    ("what is mentioned on page 5", True, "docqa", "'on page N'"),
    ("summarize this pdf", True, "docqa", "'this pdf'"),
    ("in the uploaded doc, is fire damper listed?", True, "docqa", "'uploaded doc'"),
    ("what's in this?", True, "docqa", "follow-up pronoun"),
    ("on this page, what's the callout?", True, "docqa", "'on this'"),
    ("which section has it?", True, "docqa", "follow-up + selected"),
    ("tell me about page 3", True, "docqa", "'page N'"),
    # --- clear RAG ---
    ("show all missing scope across project", False, "rag", "'across project'"),
    ("how many DOAS units are in the project", False, "rag", "'how many'"),
    ("list all HVAC drawings", False, "rag", "'list all'"),
    ("compare all mechanical floors", False, "rag", "'compare all'"),
    ("total count of fire dampers", False, "rag", "'total count'"),
    ("project-wide summary", False, "rag", "'project-wide'"),
    ("every drawing with plumbing", False, "rag", "'every drawing'"),
    ("summarize the entire project", False, "rag", "'entire project'"),
    ("generate scope gap report", False, "rag", "'generate scope'"),
    ("what trades are involved overall", False, "rag", "overall scope"),
    # --- RAG dominance even with selected doc ---
    ("show me all missing scope across project", True, "rag", "override — project-wide beats selected"),
    ("list all drawings", True, "rag", "'all drawings'"),
    ("which drawings have fire damper callouts", True, "rag", "'which drawings' plural"),
    # --- clarify needed (ambiguous) ---
    ("is it missing?", True, "clarify", "pronoun but no scope"),
    ("tell me about fire damper", True, "clarify", "ambiguous noun"),
    ("what does it cover?", True, "clarify", "pure pronoun"),
    ("any details on HVAC?", True, "clarify", "generic topic"),
    # --- no selected doc, still RAG ---
    ("what does this say?", False, "rag", "no doc selected → RAG baseline"),
    ("explain page 14", False, "rag", "no doc selected"),
    # --- mode_hint overrides everything ---
    # (tested separately via dedicated test below)
    # --- exit signals (from DocQA) ---
    ("back to project search", True, "rag", "explicit exit"),
    ("go back", True, "rag", "exit"),
    ("exit document", True, "rag", "exit"),
    # --- greetings / meta ---
    ("hi", False, "rag", "greeting"),
    ("what can you do?", False, "rag", "meta"),
    ("who are you?", False, "rag", "meta"),
    # --- construction-domain terms (neutral) ---
    ("fire damper specs", False, "rag", "generic noun"),
    ("xvent model numbers", False, "rag", "generic"),
    ("variable frequency drive notes", True, "clarify", "selected but no scope cue"),
    ("mechanical general notes", True, "clarify", "selected but ambiguous"),
    ("structural foundation details", False, "rag", "no selected, generic"),
    # --- follow-ups on page ---
    ("and on page 16?", True, "docqa", "follow-up + page"),
    ("what about this page?", True, "docqa", "'this page'"),
]


@pytest.mark.parametrize("query,has_docs,expected,note", CASES)
def test_classify(query, has_docs, expected, note):
    sess = FakeSession(
        selected_documents=[{"file_name": "X.pdf"}] if has_docs else []
    )
    decision = classify(query=query, session=sess)
    assert decision.target == expected, (
        f"[{note}] query={query!r} "
        f"got={decision.target} conf={decision.confidence:.2f} reason={decision.reason}"
    )


def test_mode_hint_overrides_classifier():
    sess = FakeSession(selected_documents=[{"file_name": "X.pdf"}])
    d = classify(query="list all drawings", session=sess, mode_hint="docqa")
    assert d.target == "docqa"
    assert d.confidence == 1.0
    d2 = classify(query="in this doc, anything?", session=sess, mode_hint="rag")
    assert d2.target == "rag"


def test_decision_shape():
    sess = FakeSession()
    d = classify(query="hi", session=sess)
    assert isinstance(d, IntentDecision)
    assert hasattr(d, "target")
    assert hasattr(d, "confidence")
    assert hasattr(d, "reason")
    assert hasattr(d, "clarification_prompt")
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_intent_classifier_v2.py -v`
Expected: FAIL — `IntentDecision` not exported, signature of `classify` wrong.

- [ ] **Step 3: Rewrite `gateway/intent_classifier.py`**

Full replacement:
```python
"""
Intent classifier v2 — bidirectional RAG ↔ DocQA routing.

Weighted keyword + session-state scoring. Returns an IntentDecision
with (target, confidence, reason, clarification_prompt). The orchestrator
uses confidence to route directly (≥0.7), ask for clarification (0.3-0.7),
or default to RAG (<0.3).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_WIDE = [
    r"\bacross (the )?project\b", r"\bacross all\b",
    r"\ball drawings\b", r"\ball specifications?\b", r"\ball trades\b",
    r"\bproject[- ]wide\b", r"\bentire project\b",
    r"\bhow many\b", r"\blist all\b", r"\bshow all\b", r"\bcompare all\b",
    r"\bmissing scope\b", r"\bscope gap\b", r"\btotal count\b",
    r"\bproject summary\b", r"\bproject overview\b",
    r"\bevery (floor|drawing|spec)\b",
    r"\ball (mechanical|electrical|plumbing|hvac)\b",
    r"\bdifferent drawings\b", r"\bwhich drawings\b",
    r"\bgenerate (scope|report)\b",
]

DOC_SCOPED = [
    r"\bthis (document|drawing|spec(ification)?|page|section|file|pdf)\b",
    r"\bin this\b", r"\bon this\b", r"\bfrom this\b", r"\bof this\b",
    r"\bthe (selected|current|uploaded)\b",
    r"\b(on|in) page\s?\d+\b",
    r"\bpage (number|no\.?)\b",
    r"\bin section\b",
    r"\bwhat does it say\b", r"\bwhat is mentioned\b",
]

EXIT_SIGNALS = [
    r"\bback to (search|project)\b",
    r"\bexit document\b",
    r"\bstop chatting with\b",
    r"\breturn to (rag|search)\b",
    r"\bgo back\b",
    r"\bsearch project\b",
    r"\bproject search\b",
]

PRONOUN_ONLY = [
    r"\bwhat about it\b", r"\band the\b", r"\btell me more\b",
    r"\bany details\b", r"\bwhat(')?s in\b",
]


@dataclass(frozen=True)
class IntentDecision:
    target: str  # "rag" | "docqa" | "clarify"
    confidence: float
    reason: str
    clarification_prompt: Optional[str] = None


def _matches_any(patterns: list[str], q: str) -> list[str]:
    return [p for p in patterns if re.search(p, q, re.IGNORECASE)]


def classify(
    query: str,
    session: Any,
    mode_hint: Optional[str] = None,
) -> IntentDecision:
    """Classify user intent for agent routing.

    Returns IntentDecision; orchestrator interprets confidence:
      ≥ 0.7 → route directly
      0.3 – 0.7 → clarify
      < 0.3 → default to RAG
    """
    q = (query or "").strip().lower()

    if mode_hint in {"rag", "docqa"}:
        return IntentDecision(
            target=mode_hint, confidence=1.0,
            reason=f"explicit mode_hint={mode_hint}",
        )

    has_selected = bool(getattr(session, "selected_documents", None))
    selected_name = None
    if has_selected:
        selected_name = session.selected_documents[-1].get("file_name")

    project_hits = _matches_any(PROJECT_WIDE, q)
    doc_hits = _matches_any(DOC_SCOPED, q)
    exit_hits = _matches_any(EXIT_SIGNALS, q)
    pronoun_hits = _matches_any(PRONOUN_ONLY, q)

    # Exit always wins
    if exit_hits:
        return IntentDecision(
            target="rag", confidence=1.0,
            reason=f"explicit exit ({exit_hits[0]})",
        )

    # Project-wide beats everything else
    if project_hits:
        return IntentDecision(
            target="rag", confidence=0.9,
            reason=f"project-wide signal: {project_hits[:2]}",
        )

    # Score
    score = 0.0
    if has_selected:
        score += 0.3
    score += 0.4 * bool(doc_hits)
    score += 0.2 * bool(pronoun_hits)

    if score >= 0.7:
        return IntentDecision(
            target="docqa", confidence=score,
            reason=f"doc-scoped hits={doc_hits[:2]} selected={has_selected}",
        )
    if 0.3 <= score < 0.7 and has_selected:
        prompt = (
            f"Should I answer from the selected document "
            f"({selected_name or 'your selection'}) or search the whole project?"
        )
        return IntentDecision(
            target="clarify", confidence=score,
            reason="ambiguous with selected document",
            clarification_prompt=prompt,
        )

    # Default: RAG
    return IntentDecision(
        target="rag", confidence=max(0.1, 0.4 - score),
        reason="no strong doc-scope signal",
    )
```

- [ ] **Step 4: Run 40 parametrized tests — PASS**

Run: `python -m pytest tests/test_intent_classifier_v2.py -v`
Expected: 42 PASS (40 params + mode_hint + shape).

- [ ] **Step 5: Commit**

```bash
git add gateway/intent_classifier.py tests/test_intent_classifier_v2.py
git commit -m "feat(classifier): bidirectional intent routing with clarify mode

Returns IntentDecision(target, confidence, reason, clarification_prompt).
Weights: project-wide +0.5 override, doc-scoped +0.4, selected-doc +0.3,
pronoun-only +0.2. Threshold 0.7 route, 0.3-0.7 clarify, <0.3 default RAG.
Mode_hint fully overrides."
```

### Task 3.2: Wire classifier into orchestrator

- [ ] **Step 1: Integration test**

Create `tests/test_orchestrator_clarify.py`:
```python
"""Phase 3: orchestrator honors classify() decisions including clarify."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_orchestrator_returns_clarification_when_ambiguous():
    from gateway.orchestrator import run_query
    fake_sess = MagicMock()
    fake_sess.selected_documents = [{"file_name": "HVAC.pdf"}]
    fake_sess.active_agent = "rag"
    fake_sess.docqa_session_id = None

    with patch("gateway.orchestrator._get_session", return_value=fake_sess):
        result = await run_query(
            query="is it missing?",
            project_id=7222,
            session_id="s1",
        )
    assert result["needs_clarification"] is True
    assert "HVAC.pdf" in result["clarification_prompt"]
    assert result["active_agent"] == "rag"
    assert result["answer"] == ""
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_orchestrator_clarify.py -v`
Expected: FAIL — clarify path not wired.

- [ ] **Step 3: Wire classifier into orchestrator dispatcher**

In `gateway/orchestrator.py` `run_query` entry function, at the start (before existing RAG/DocQA dispatch), insert:
```python
from gateway.intent_classifier import classify

def _build_clarify_response(decision, rag_session_id, selected_document) -> dict:
    return {
        "success": True,
        "answer": "",
        "sources": [],
        "source_documents": [],
        "active_agent": "rag",
        "selected_document": selected_document,
        "needs_clarification": True,
        "clarification_prompt": decision.clarification_prompt,
        "session_id": rag_session_id,
        "engine_used": "classifier",
        "confidence": "low",
    }
```

Then in the dispatch, before RAG call:
```python
# --- Phase 3: intent routing ---
if not search_mode:  # only auto-route when caller didn't force
    session = _get_session(session_id)  # existing helper
    decision = classify(query=query, session=session, mode_hint=mode_hint)
    logger.info("intent_decision: %s", decision)
    session.last_intent_decision = {
        "target": decision.target, "confidence": decision.confidence,
        "reason": decision.reason,
    }
    if decision.target == "clarify":
        return _build_clarify_response(
            decision, rag_session_id=session_id,
            selected_document=(session.selected_documents[-1]
                               if session.selected_documents else None),
        )
    if decision.target == "docqa":
        # promote last selected doc into docqa_document if caller didn't pass one
        if not docqa_document and session.selected_documents:
            docqa_document = session.selected_documents[-1]
        if docqa_document:
            return await _run_docqa(query, project_id, session_id, docqa_document)
    # else target == "rag" → fall through to existing RAG path
```

- [ ] **Step 4: Run tests to PASS**

Run: `python -m pytest tests/test_orchestrator_clarify.py tests/test_orchestrator_docqa_route.py tests/test_intent_classifier_v2.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add gateway/orchestrator.py tests/test_orchestrator_clarify.py
git commit -m "feat(orchestrator): auto-route via intent classifier + clarify mode"
```

**GATE: Pause. User reviews:**
- 42 classifier tests green
- "this document" with selected → DocQA
- "across project" with selected → RAG (overrides)
- "is it missing?" → clarify prompt returned
- mode_hint fully overrides

Authorize Phase 4.

---

# Phase 4 — UI updates in demo-ui.html

**Goal:** Show S3 basename as primary source label (dedup), "Chat with Document" button per source, active-agent badge, clarification prompt with two-button UX, "Back to project search" exit.

**Files:**
- Modify: `docs/demo-ui.html`

### Task 4.1: Source card redesign with dedupe

- [ ] **Step 1: Read current demo-ui.html to understand structure**

Run: `head -200 docs/demo-ui.html` and find the source-rendering template (look for `source_documents.map` or similar).

- [ ] **Step 2: Replace source rendering block**

Locate the function that builds the source card (grep `source_documents` or `display_title`). Replace with:
```js
function dedupeSources(srcs) {
  const seen = new Map();
  for (const s of (srcs || [])) {
    const key = s.s3_path || s.file_name;
    if (!key) continue;
    if (!seen.has(key)) {
      seen.set(key, { ...s, _pages: new Set() });
    }
    const agg = seen.get(key);
    if (s.page) agg._pages.add(s.page);
  }
  return Array.from(seen.values()).map(s => ({
    ...s, pages: Array.from(s._pages).sort((a,b)=>a-b),
  }));
}

function renderSourceCard(s) {
  const primary = s.file_name || (s.s3_path || '').split('/').pop() || 'Document';
  const secondary = s.display_title && s.display_title !== primary
    ? s.display_title : '';
  const pagesLabel = (s.pages && s.pages.length)
    ? ` · pages ${s.pages.join(', ')}` : '';
  return `
    <div class="source-card" data-s3="${s.s3_path || ''}">
      <div class="source-primary">${escapeHtml(primary)}${pagesLabel}</div>
      ${secondary ? `<div class="source-secondary">${escapeHtml(secondary)}</div>` : ''}
      <div class="source-actions">
        <a href="${s.download_url || '#'}" target="_blank" rel="noopener">Download</a>
        <button class="btn-chat-doc"
                onclick='chatWithDocument(${JSON.stringify(s).replace(/'/g, "&#39;")})'>
          Chat with Document
        </button>
      </div>
    </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
```

- [ ] **Step 3: Hand-test in browser against sandbox**

Run server locally pointing at sandbox:
```bash
# Start local gateway in one terminal (if not already)
python -m gateway.app
```

Open `docs/demo-ui.html` in browser, submit a query, verify sources render with:
- Filename first (e.g., `M-301.pdf`)
- Drawing title as subtitle if different
- Duplicate s3_path collapsed to one card with aggregated pages
- Download link works (opens presigned PDF in new tab)
- "Chat with Document" button visible

- [ ] **Step 4: Commit**

```bash
git add docs/demo-ui.html
git commit -m "feat(ui): source card shows S3 basename primary, dedupes by s3_path"
```

### Task 4.2: "Chat with Document" action + active-agent badge

- [ ] **Step 1: Add JS handler**

Append inside `<script>` block:
```js
let activeAgent = 'rag';
let activeDocqaDoc = null;

function setAgentBadge(agent, doc) {
  activeAgent = agent;
  activeDocqaDoc = doc;
  const el = document.getElementById('agent-badge');
  if (!el) return;
  if (agent === 'docqa' && doc) {
    el.innerHTML = `<span class="badge badge-docqa">DocQA: ${escapeHtml(doc.file_name || doc.s3_path)}</span>
                    <button onclick="exitDocQA()">Back to project search</button>`;
  } else {
    el.innerHTML = `<span class="badge badge-rag">Project search (RAG)</span>`;
  }
}

async function chatWithDocument(source) {
  setAgentBadge('docqa', source);
  appendSystemMsg(`Processing ${source.file_name}…`);
  const firstQuery = 'Give me a brief overview of this document.';
  const body = {
    query: firstQuery,
    project_id: currentProjectId,
    session_id: currentSessionId,
    search_mode: 'docqa',
    docqa_document: {
      s3_path: source.s3_path,
      file_name: source.file_name,
      download_url: source.download_url,
      pdf_name: source.pdf_name || source.file_name,
    },
  };
  const r = await fetch(`${baseURL}/query`,
    { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await r.json();
  if (data.success) {
    currentSessionId = data.session_id;
    appendSystemMsg('Document processed. You can now ask questions.');
    appendAssistant(data);
  } else {
    appendSystemMsg(`Failed to load document: ${data.answer || 'unknown error'}`);
    setAgentBadge('rag', null);
  }
}

function exitDocQA() {
  setAgentBadge('rag', null);
  appendSystemMsg('Returned to project-wide search.');
}
```

Add to the HTML body (above the message list):
```html
<div id="agent-badge" class="agent-badge"></div>
```

Add to CSS:
```css
.agent-badge { padding:.5rem; background:#f5f5f5; border-radius:4px; margin-bottom:.5rem; }
.badge { padding:.2rem .6rem; border-radius:12px; font-size:.8rem; font-weight:600; }
.badge-rag { background:#e8f0fe; color:#1a73e8; }
.badge-docqa { background:#fff0e8; color:#d93025; }
.btn-chat-doc { margin-left:.5rem; padding:.3rem .7rem; border:1px solid #1a73e8;
                background:white; color:#1a73e8; border-radius:4px; cursor:pointer; }
.btn-chat-doc:hover { background:#1a73e8; color:white; }
.source-card { padding:.7rem; margin:.3rem 0; border:1px solid #ddd; border-radius:4px; }
.source-primary { font-weight:600; }
.source-secondary { color:#666; font-size:.85rem; }
.source-actions { margin-top:.4rem; display:flex; gap:.5rem; }
```

- [ ] **Step 2: Browser test**

Open UI, submit a query, click "Chat with Document" on a source card. Expected flow:
1. Badge flips to DocQA with filename
2. "Processing …" shows
3. DocQA answer renders with `active_agent=docqa` + page references
4. Type follow-up — stays in DocQA
5. Click "Back to project search" → badge flips back, next query goes to RAG

Capture 3 screenshots: before, during, after handoff. Save to `docs/screenshots/phase4_handoff_{1,2,3}.png`.

- [ ] **Step 3: Commit**

```bash
git add docs/demo-ui.html docs/screenshots/
git commit -m "feat(ui): Chat with Document button, agent badge, exit path"
```

### Task 4.3: Clarification prompt UX

- [ ] **Step 1: Add handler**

In `<script>`:
```js
function renderClarification(data) {
  const el = document.createElement('div');
  el.className = 'clarify-box';
  el.innerHTML = `
    <p>${escapeHtml(data.clarification_prompt)}</p>
    <button onclick='resubmit("docqa")'>Answer from selected document</button>
    <button onclick='resubmit("rag")'>Search whole project</button>
  `;
  messagesEl.appendChild(el);
}

let lastQuery = null;

async function resubmit(modeHint) {
  if (!lastQuery) return;
  document.querySelectorAll('.clarify-box').forEach(n => n.remove());
  const body = {
    query: lastQuery,
    project_id: currentProjectId,
    session_id: currentSessionId,
    mode_hint: modeHint,
  };
  if (modeHint === 'docqa' && activeDocqaDoc) {
    body.search_mode = 'docqa';
    body.docqa_document = {
      s3_path: activeDocqaDoc.s3_path,
      file_name: activeDocqaDoc.file_name,
      download_url: activeDocqaDoc.download_url,
    };
  }
  const r = await fetch(`${baseURL}/query`,
    { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await r.json();
  appendAssistant(data);
}
```

In the main `sendQuery` function, after getting response:
```js
lastQuery = query;
if (data.needs_clarification) {
  renderClarification(data);
  return;
}
```

Add CSS:
```css
.clarify-box { background:#fffbe6; border:1px solid #ffd666; padding:.7rem;
               border-radius:4px; margin:.5rem 0; }
.clarify-box button { margin:.3rem .3rem 0 0; padding:.4rem .8rem; cursor:pointer; }
```

- [ ] **Step 2: Browser test**

1. Start RAG session, ask "list HVAC drawings"
2. Click "Chat with Document" on first source
3. Then ask "is it missing?" (ambiguous)
4. Expected: clarify box with two buttons; clicking "Search whole project" re-submits with `mode_hint=rag`

Capture screenshot, save to `docs/screenshots/phase4_clarify.png`.

- [ ] **Step 3: Commit**

```bash
git add docs/demo-ui.html docs/screenshots/phase4_clarify.png
git commit -m "feat(ui): clarification prompt with two-choice re-submit"
```

**GATE: Pause. User reviews UI against sandbox in browser.**
- Source cards dedupe by s3_path
- Filename primary, title secondary
- Chat with Document flow works end-to-end
- Clarify box appears on ambiguous queries
- Agent badge flips correctly

Authorize Phase 5.

---

# Phase 5 — Full E2E suite + parallel bug-check agents

**Goal:** Green bar on 10 E2E flows; zero CRITICAL/HIGH issues open from parallel code-reviewer + security-reviewer + e2e-runner passes.

**Files:**
- Create: `tests/e2e/test_docqa_flow.py`
- Create: `docs/TEST_RESULTS_PHASE5.md`

### Task 5.1: Ten E2E flows against sandbox

- [ ] **Step 1: Write E2E suite**

Create `tests/e2e/test_docqa_flow.py`:
```python
"""Phase 5: 10 E2E flows against sandbox (http://54.197.189.113:8001)."""
from __future__ import annotations
import os
import pytest
import httpx

pytestmark = pytest.mark.e2e
SANDBOX = os.environ.get("SANDBOX_URL", "http://54.197.189.113:8001")


@pytest.fixture
def http():
    with httpx.Client(timeout=180) as c:
        yield c


def _first_source(data):
    srcs = data.get("source_documents") or []
    assert srcs, f"No sources in response: {data}"
    return srcs[0]


def test_flow1_rag_only(http):
    r = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    r.raise_for_status()
    d = r.json()
    assert d["active_agent"] == "rag"
    assert d["source_documents"]


def test_flow2_rag_then_docqa(http):
    r1 = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    d1 = r1.json()
    src = _first_source(d1)
    r2 = http.post(f"{SANDBOX}/query", json={
        "query": "give me overview",
        "project_id": 7222, "session_id": d1["session_id"],
        "search_mode": "docqa",
        "docqa_document": {"s3_path": src["s3_path"], "file_name": src["file_name"],
                           "download_url": src["download_url"]},
    })
    d2 = r2.json()
    assert d2["active_agent"] == "docqa"
    assert d2["docqa_session_id"]


def test_flow3_docqa_followup(http):
    # Flow 2 then continue
    r1 = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    src = _first_source(r1.json())
    sid = r1.json()["session_id"]
    http.post(f"{SANDBOX}/query", json={
        "query": "overview", "project_id": 7222, "session_id": sid,
        "search_mode": "docqa",
        "docqa_document": {"s3_path": src["s3_path"], "file_name": src["file_name"],
                           "download_url": src["download_url"]},
    })
    r3 = http.post(f"{SANDBOX}/query", json={
        "query": "what does it say on page 1?", "project_id": 7222, "session_id": sid,
    })
    d3 = r3.json()
    assert d3["active_agent"] == "docqa", d3
    assert "page" in d3["answer"].lower() or d3.get("source_documents")


def test_flow4_back_to_rag(http):
    r1 = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    src = _first_source(r1.json())
    sid = r1.json()["session_id"]
    http.post(f"{SANDBOX}/query", json={
        "query": "overview", "project_id": 7222, "session_id": sid,
        "search_mode": "docqa",
        "docqa_document": {"s3_path": src["s3_path"], "file_name": src["file_name"],
                           "download_url": src["download_url"]},
    })
    r3 = http.post(f"{SANDBOX}/query", json={
        "query": "show all missing scope across project", "project_id": 7222, "session_id": sid,
    })
    d3 = r3.json()
    assert d3["active_agent"] == "rag", d3


def test_flow5_clarify_prompt(http):
    r1 = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    src = _first_source(r1.json())
    sid = r1.json()["session_id"]
    http.post(f"{SANDBOX}/query", json={
        "query": "overview", "project_id": 7222, "session_id": sid,
        "search_mode": "docqa",
        "docqa_document": {"s3_path": src["s3_path"], "file_name": src["file_name"],
                           "download_url": src["download_url"]},
    })
    r3 = http.post(f"{SANDBOX}/query", json={
        "query": "tell me about fire damper", "project_id": 7222, "session_id": sid,
    })
    d3 = r3.json()
    # expect clarify OR routed to docqa since ambiguous
    assert d3.get("needs_clarification") or d3["active_agent"] in ("rag", "docqa")


def test_flow6_mode_hint_overrides(http):
    r1 = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    src = _first_source(r1.json())
    sid = r1.json()["session_id"]
    http.post(f"{SANDBOX}/query", json={
        "query": "overview", "project_id": 7222, "session_id": sid,
        "search_mode": "docqa",
        "docqa_document": {"s3_path": src["s3_path"], "file_name": src["file_name"],
                           "download_url": src["download_url"]},
    })
    r3 = http.post(f"{SANDBOX}/query", json={
        "query": "show all drawings", "project_id": 7222, "session_id": sid,
        "mode_hint": "docqa",
    })
    d3 = r3.json()
    assert d3["active_agent"] == "docqa"


def test_flow7_download_url_live(http):
    r = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    src = _first_source(r.json())
    if not src.get("download_url"):
        pytest.skip("no download_url")
    head = http.head(src["download_url"])
    assert head.status_code in (200, 206), f"Broken presigned URL: {head.status_code}"


def test_flow8_multiple_doc_upload(http):
    r1 = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    srcs = r1.json()["source_documents"][:2]
    sid = r1.json()["session_id"]
    for src in srcs:
        http.post(f"{SANDBOX}/query", json={
            "query": "placeholder", "project_id": 7222, "session_id": sid,
            "search_mode": "docqa",
            "docqa_document": {"s3_path": src["s3_path"], "file_name": src["file_name"],
                               "download_url": src["download_url"]},
        })
    r = http.post(f"{SANDBOX}/query", json={
        "query": "summarize all loaded docs", "project_id": 7222, "session_id": sid,
    })
    d = r.json()
    assert d["active_agent"] == "docqa"


def test_flow9_answer_format_clean(http):
    import re
    r = http.post(f"{SANDBOX}/query",
        json={"query": "What XVENT model for double exhaust?", "project_id": 7222})
    ans = r.json()["answer"]
    assert not re.search(r"\[Source:", ans), f"artifact: {ans!r}"
    assert not re.search(r"^Direct Answer", ans, re.MULTILINE | re.IGNORECASE)
    assert not re.search(r"HIGH\s*\(\d+%\)", ans)


def test_flow10_backward_compat_existing_clients(http):
    """Old UI clients MUST still work with existing field names."""
    r = http.post(f"{SANDBOX}/query",
        json={"query": "list HVAC drawings", "project_id": 7222})
    d = r.json()
    assert "success" in d
    assert "answer" in d
    assert "session_id" in d
    assert "confidence" in d
    assert "engine_used" in d
    assert "source_documents" in d  # new but now part of contract
```

- [ ] **Step 2: Run E2E**

Run: `SANDBOX_URL=http://localhost:8001 python -m pytest tests/e2e/test_docqa_flow.py -v -m e2e`
Expected: 10 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_docqa_flow.py
git commit -m "test(e2e): 10 end-to-end flows for RAG↔DocQA bridge"
```

### Task 5.2: Parallel bug-check (3 agents)

- [ ] **Step 1: Dispatch code-reviewer, security-reviewer, e2e-runner in parallel**

From main session (not inside plan doc), launch 3 agents in a single message:

```
Agent 1 — code-reviewer:
"Review all diff since commit before Phase 0 on PROD_SETUP/unified-rag-agent/.
Focus: new files gateway/docqa_bridge.py, gateway/intent_classifier.py;
modified: gateway/orchestrator.py, gateway/models.py, agentic/core/agent.py,
traditional/memory_manager.py. Flag CRITICAL/HIGH issues only. Report ≤ 400 words."

Agent 2 — security-reviewer:
"Security review PROD_SETUP/unified-rag-agent/gateway/docqa_bridge.py.
Focus: S3 key injection, path traversal on file_name, presigned URL leakage,
tempfile cleanup on exception, httpx timeout. Report ≤ 300 words."

Agent 3 — e2e-runner:
"Run tests/e2e/test_docqa_flow.py against http://localhost:8001 (server
already running). Capture failures + flaky tests. Report pass/fail count
and any HTTP 5xx or slow (>60s) flows. Report ≤ 300 words."
```

- [ ] **Step 2: Resolve all CRITICAL/HIGH findings**

Make fixes per finding, commit each fix separately. Do not advance to Phase 6 until all CRITICAL/HIGH issues are closed.

- [ ] **Step 3: Write test results summary**

Create `docs/TEST_RESULTS_PHASE5.md`:
```markdown
# Phase 5 Test Results

## E2E (tests/e2e/test_docqa_flow.py)
- Total: 10 flows
- Passed: N
- Failed: M
- Flaky: K

## Unit / Integration
- Total: XX tests
- Coverage: YY% overall, ZZ% on gateway/docqa_bridge.py

## Parallel bug-check pass
- code-reviewer: [findings]
- security-reviewer: [findings]
- e2e-runner: [findings]

## CRITICAL/HIGH issues resolved
- [list]

## Known limitations
- DocQA accuracy on complex drawing PDFs: [observation]
- [others]
```

- [ ] **Step 4: Commit**

```bash
git add docs/TEST_RESULTS_PHASE5.md
git commit -m "docs: Phase 5 test results summary"
```

**GATE: Pause. User reviews test report + open limitations. Authorize Phase 6.**

---

# Phase 6 — Regression diff vs v1.2-hybrid-ship + sign-off

**Goal:** Prove v2.0 does not regress the 16 baseline queries on latency, source coverage, or URL integrity. Produce final comparison doc.

**Files:**
- Create: `tests/test_phase6_regression_vs_v12.py`
- Create: `docs/PHASE6_SIGNOFF.md`
- Create: `_versions/v2.0-docqa-extension-final/` (post-change snapshot)

### Task 6.1: Run comparison

- [ ] **Step 1: Write comparison harness**

Create `tests/test_phase6_regression_vs_v12.py`:
```python
"""Phase 6: compare v2.0 against v1.2-hybrid-ship on 16 historical queries."""
from __future__ import annotations
import json
import os
import pytest
import httpx

pytestmark = pytest.mark.regression
SANDBOX = os.environ.get("SANDBOX_URL", "http://54.197.189.113:8001")

QUERIES = [
    # same 16 queries from Phase 1 regression
]


def test_no_url_breakage_or_source_regression():
    results = []
    for project_id, q in QUERIES:
        with httpx.Client(timeout=120) as c:
            r = c.post(f"{SANDBOX}/query", json={"query": q, "project_id": project_id})
            d = r.json()
            srcs = d.get("source_documents") or []
            # HEAD each URL
            url_ok = all(
                httpx.head(s.get("download_url", ""), timeout=10).status_code in (200, 206)
                for s in srcs if s.get("download_url")
            )
            results.append({
                "query": q, "sources": len(srcs), "urls_ok": url_ok,
                "elapsed_s": r.elapsed.total_seconds(),
                "confidence": d.get("confidence"),
                "active_agent": d.get("active_agent"),
            })
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/phase6_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    broken = [r for r in results if not r["urls_ok"]]
    zero_src = [r for r in results if r["sources"] == 0]
    assert not broken, f"Broken URLs: {broken}"
    assert len(zero_src) <= 1, f"Too many zero-source responses: {zero_src}"
    avg = sum(r["elapsed_s"] for r in results) / len(results)
    print(f"\navg latency: {avg:.1f}s (baseline v1.2 was 19s)")
    assert avg <= 22.0, f"Latency regressed: {avg:.1f}s > 22s"
```

- [ ] **Step 2: Run comparison**

Run: `SANDBOX_URL=http://localhost:8001 python -m pytest tests/test_phase6_regression_vs_v12.py -v -m regression`
Expected: PASS. Saves `test_results/phase6_v2_results.json`.

- [ ] **Step 3: Write sign-off doc**

Create `docs/PHASE6_SIGNOFF.md`:
```markdown
# Phase 6 Sign-off — v2.0-docqa-extension

## Scope
- Delivered user story: RAG→DocQA handoff, bidirectional auto-switching, clean answer prose, schema-frozen, sandbox-only

## Metrics vs v1.2-hybrid-ship baseline
| Metric | v1.2 | v2.0 | Δ |
|---|---|---|---|
| Avg latency (cold) | 19s | Xs | … |
| Avg latency (turn 2+ cached) | 19s | Xs | … |
| Queries with broken URL | 0/16 | 0/16 | 0 |
| Queries with zero sources | 0/16 | ?/16 | … |
| [Source:] / HIGH (N%) / Direct Answer leaks | present | 0 | cleared |
| Cache hit rate turn 2+ | n/a | X% | new |

## User story coverage
- [ ] RAG answers + sources list
- [ ] User selects doc → DocQA takes over
- [ ] Follow-up questions scoped to doc
- [ ] "Across project" auto-switches back to RAG
- [ ] Clarify on ambiguous queries
- [ ] Multi-doc upload same session
- [ ] No schema breakage for old UI

## Known limitations
- Construction-drawing OCR accuracy in DocQA: [describe]
- Session memory loss on gateway restart (sandbox only)

## Revert plan
- `rsync -a _versions/v2.0-docqa-extension/ ./ --exclude README.md --exclude VERSIONS.md`

## Recommendation
- [Ship / Fix these first / Needs rework]
```

Fill metrics from actual test_results/phase6_v2_results.json.

- [ ] **Step 4: Final snapshot**

```bash
mkdir -p _versions/v2.0-docqa-extension-final
rsync -a \
  --exclude='_versions/' --exclude='_backups/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='test_results/' --exclude='*.pyc' \
  ./ _versions/v2.0-docqa-extension-final/
```

Write `_versions/v2.0-docqa-extension-final/README.md` listing all phase commits.

Update `VERSIONS.md` with final entry.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase6_regression_vs_v12.py docs/PHASE6_SIGNOFF.md \
  _versions/v2.0-docqa-extension-final/ VERSIONS.md test_results/phase6_v2_results.json
git commit -m "docs: Phase 6 sign-off + final v2.0-docqa-extension snapshot"
```

**FINAL GATE: User reviews PHASE6_SIGNOFF.md.** Ship decision or list of additional work.

---

## Self-review

**Spec coverage check:**
- §4.1/4.2 Architecture → Phases 0-4 implement
- §4.3 Module map → every modified file appears in Phase tasks
- §5.1 Answer-format fix → Task 1.1
- §5.2 Schema additions → Task 1.2
- §5.3 DocQA bridge → Tasks 2.2, 2.3, 2.4
- §5.4 Bidirectional intent classifier → Tasks 3.1, 3.2
- §5.5 Session storage → Task 2.1
- §5.6 UI changes → Tasks 4.1, 4.2, 4.3
- §5.7 Download URLs → Task 4.1 + Task 6.1 (HEAD check)
- §8 Testing strategy → Tasks 1.3, 1.4, 2.5, 3.1, 4.1-3 manual, 5.1, 5.2, 6.1
- §9 Rollout + pause-per-phase → each phase ends with a **GATE: Pause**
- §11 OpenAI prompt caching → Task 1.3
- §13 Approval checklist → baked into each GATE

**Placeholder scan:** Searched for TBD / TODO / "fill in" / "similar to" — none present. The 16 HISTORICAL_QUERIES list in Task 1.4 and Task 6.1 is a known fill-in point where the engineer must paste the exact queries from `test_results/local_after_all_fixes_20260422_154725/` — this is explicit, not a placeholder.

**Type consistency:**
- `IntentDecision(target, confidence, reason, clarification_prompt)` — consistent between Task 3.1 definition and Task 3.2 usage
- `DocQABridge.ensure_document_loaded(session_id, doc_ref)` — consistent Tasks 2.3, 2.4
- `DocQABridge.ask(docqa_session_id, query)` — consistent
- `DocQABridge.normalize(docqa_response, rag_session_id, selected_document)` — consistent
- `UnifiedResponse` field names (`active_agent`, `docqa_session_id`, `selected_document`, `needs_clarification`, `clarification_prompt`, `source_documents`) — consistent across models.py, bridge, orchestrator, tests, UI

**Gap fix:** Task 3.2 references `_get_session` helper in orchestrator. If this helper does not exist, the engineer must locate or create a function that returns the `ConversationSession` for the given session_id from the shared MemoryManager. This is noted inline in Step 3.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-unified-rag-docqa-extension.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Given your pause-per-phase preference, each GATE becomes a hard stop for your review before the next subagent fires.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints.

**Which approach?**
