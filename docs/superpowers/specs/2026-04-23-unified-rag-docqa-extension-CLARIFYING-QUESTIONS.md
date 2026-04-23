# Clarifying Questions — `unified_rag_agent_with_docqa_extension`

**Date:** 2026-04-23
**Target version:** `unified_rag_agent_with_docqa_extension` (new, local-only, sandbox-tested, NOT deployed to VM)
**Baseline:** `_versions/v1.2-hybrid-ship/` (19s latency, clean schema, SELF_RAG off, RERANKER_INCLUDE_SCORE off)

Please answer each question (A/B/C or free text). I will NOT touch code until every question is answered and you approve the design that follows.

---

## AUDIT RECAP (context for the questions below)

Current state vs. user story: **~55% wired**. Gaps:

1. **Answer format bug (CRITICAL)** — `agentic/core/agent.py:155` says *"cite [Source: drawingName / drawingTitle] for every fact"* but `:205` says *"do NOT include [Source: …]"*. Model obeys the first. No post-processor. → artifacts like `[Source: Page40 / …]`, `HIGH (90%)`, `Direct Answer` header leak to UI.
2. **Schema mismatch** — `QueryRequest` has no `docqa_document`; `UnifiedResponse` has no `active_agent`/`selected_document`/`source_documents`/`download_url` even though orchestrator returns them.
3. **S3 download missing** — `docqa_client.upload_and_query` expects a local path; `_run_docqa` has no fetch-from-S3 step.
4. **One-way intent classifier** — only classifies in DocQA mode; in RAG mode returns `"rag"` unconditionally. "this document" never auto-routes from RAG → DocQA.
5. **DocQA prefix** — client hits `/api/converse` directly on 8006; story assumes `/docqa/*` via Nginx.
6. **UI** — renders sources by `display_title`, no `download_url` button, points at sandbox (correct) but lacks DocQA UX.
7. **No unit tests** for `docqa_client.py` or `intent_classifier.py`.

---

## QUESTIONS

### Q1 — S3 document fetch strategy

To hand a cited document from RAG → DocQA, pick one:

- **(A)** Gateway-side fetch (S3 → temp file → multipart upload to DocQA `/api/upload`). **Zero changes to DocQA agent.** Matches `PROD_SETUP/CLAUDE.md` rule "NEVER modify agent code". Double-hop adds ~1-3s per PDF inside us-east-1. **← my recommendation**
- **(B)** Extend DocQA with new endpoint accepting `{s3_key, presigned_url}` — DocQA fetches itself. Faster. **Breaks the "never modify agent code" rule.**
- **(C)** Reuse the existing presigned `download_url` — DocQA accepts URL. Same rule violation as B.
- **(D)** Other — describe.

**Your choice:** option A

---

### Q2 — Answer-format cleanup (the user's top complaint)

The user wants clean prose answers. No `[Source: …]`, no `HIGH (90%)`, no `Direct Answer` header. Two levers:

- **(A)** Fix the contradictory system prompt in `agentic/core/agent.py:155` (delete the "cite [Source: …]" instruction). Cheapest, most deterministic. **← recommended primary fix.**
- **(B)** Add a post-processor regex-strip stage after the LLM returns (`\[Source:[^\]]*\]`, `^Direct Answer.*$`, `HIGH \(\d+%\)\s*$`). Belt-and-suspenders safety net.
- **(C)** Both.

Source citations still need to be returned — but as structured data in `source_documents[]` (separate field the UI renders), **not** embedded in the answer text.

**Your choice:** option A 

Also: **Should the `confidence` label still appear anywhere in the API response?**
- (i) Keep it inside `UnifiedResponse.confidence` field but strip from answer text. ← recommended
- (ii) Remove entirely.

**Your choice:** i 

---

### Q3 — Auto-switch sensitivity (RAG ↔ DocQA)

Bidirectional intent classifier needs clear rules. When should the system auto-switch?

- **(A) Aggressive** — any match of *"this document/spec/drawing", "in this PDF", "open it", "on page X"* + active selected_document → switch to DocQA. Any match of *"across project", "all X", "missing scope", "summarize project"* → switch to RAG.
- **(B) Conservative** — only switch when explicit (user clicks "Chat with Document" button, or types `/docqa` / `/rag` prefix). No keyword inference.
- **(C) Hybrid** — keyword detection, but **ask a one-line clarification** if confidence < 0.7 ("Should I answer from the selected spec or search the whole project?"). **← recommended.**

**Your choice:** option C

---

### Q4 — Multi-document DocQA

User story mentions "select multiple documents, ask combined questions". DocQA agent currently accepts one file per upload. Options:

- **(A)** Upload each selected doc separately → DocQA stores them in one session → user asks; DocQA answers across all in-session docs. Zero DocQA changes; just call `/api/upload` N times with same `docqa_session_id`. **← recommended.**
- **(B)** Defer multi-doc to Phase 2, ship single-doc first.
- **(C)** Build a gateway-side merged-context layer (download all, concat, pass to RAG's own LLM instead of DocQA).

**Your choice:** option A for now but we need to add this multidocument functionality in docqa agent later

---

### Q5 — Versioning & revert strategy

You asked for a new named version `unified_rag_agent_with_docqa_extension`. Existing convention is `_versions/vX.Y-name/`. Pick format:

- **(A)** `_versions/v2.0-docqa-extension/` — matches existing `vX.Y` naming, two-digit bump signals feature addition. **← recommended.**
- **(B)** `_versions/unified_rag_agent_with_docqa_extension/` — literal name you asked for.
- **(C)** `_versions/v1.3-docqa-bridge/` — minor bump.

Plus: **Do you want a full file snapshot (rsync the whole tree) or only a diff-patch to reduce disk?**
- (i) Full snapshot (matches existing versions, ~50-80 MB, easy revert). ← recommended
- (ii) Diff-patch only.

**Your choice:** option A version format + snapshot mode.

---

### Q6 — `download_url` / SSL follow-up

You mentioned the RAG response includes `download_url` and once S3 credentials arrive you want an SSL cert for those URLs (CloudFront or a subdomain).

- **(A)** For now, return **presigned S3 URLs** (existing `_build_download_url` in orchestrator:175-258), TTL 1 hour. When you get credentials we add CloudFront in front with Let's Encrypt via ACM. **← recommended as interim.**
- **(B)** Leave `download_url` empty until CloudFront is live.
- **(C)** Return raw `s3://` keys and let the UI fetch via a proxy endpoint (`/rag/file?s3_key=…`) that gateway streams.

**Your choice:** option A and in the current system we get the presigned s3 urls right?

Also: **SSL for CloudFront — CNAME from `docs.ai5.ifieldsmart.com` (or similar) to CloudFront distribution, with ACM cert. Sound right, or do you want a different subdomain?** we will discuss this later

---

### Q7 — UI file naming / display

Today the UI renders the source card with whatever title the agent returns. The user says "the name should be the file name which is located at s3 bucket".

- **(A)** Show the **S3 object key basename** (e.g., `HVAC_SPEC_v3.pdf`) as the primary label, with `display_title` (drawing title) as a secondary subtitle. ← recommended
- **(B)** Only the S3 basename.
- **(C)** Only the `display_title`.

**Your choice:** Option A, but make sure there will be no duplicates

---

### Q8 — Session storage backend

DocQA session + RAG session need to share state (active_agent, selected_document, docqa_session_id). Options:

- **(A)** Extend existing in-memory `MemoryManager` (already used for `pin_document`) with `active_agent` / `docqa_session_id` fields. Zero new infra. **← recommended for sandbox.**
- **(B)** MongoDB-backed sessions (survives restart). Higher complexity.
- **(C)** Redis. New dependency.

Sandbox won't restart often; memory is fine. Production we can migrate later.

**Your choice:** option A 

---

### Q9 — Testing depth before you approve

User said "rigorous testing… do not stop until all issues resolved… parallel agents… every method". How deep?

- **(A) Minimum**: unit + integration (mocked DocQA) + smoke E2E against sandbox. ~2-3 hrs. No DocQA server needed locally.
- **(B) Standard**: unit + integration + full E2E against sandbox DocQA + UI manual walk-through + bug-check subagent pass. ~6-8 hrs. **← recommended.**
- **(C) Exhaustive**: B + load test + adversarial prompt-injection test + cross-browser UI test + regression vs v1.2-hybrid-ship on all 16 historical queries. ~12-15 hrs.

**Your choice:** option B

---

### Q10 — Deployment scope for this round

You said "do not deploy anything on vm". Confirm scope:

- **(A)** Build new version in `_versions/`, run locally, test against **sandbox APIs only** (54.197.189.113 + DocQA at whichever port/URL sandbox exposes). Do NOT touch prod VM (13.217.22.125). No systemd. No nginx change. **← my reading.**
- **(B)** Also stage the artifacts on sandbox VM's filesystem but don't start the service.
- **(C)** Sandbox VM full deploy (service runs on sandbox).

**Your choice:** option A 

---

### Q11 — Rollout phase granularity

After approval you want phase-wise construct. Preview:

- **Phase 0**: snapshot baseline `v1.2-hybrid-ship` → create `_versions/v2.0-docqa-extension/`
- **Phase 1**: answer-format fix (prompt + post-processor) + schema alignment (Q1 + Q2)
- **Phase 2**: S3 fetch + DocQA upload bridge + session plumbing (Q1, Q4, Q8)
- **Phase 3**: bidirectional intent classifier (Q3)
- **Phase 4**: UI — "Chat with Document" button, filename display, download links (Q6, Q7)
- **Phase 5**: tests (unit + integration + E2E) at chosen depth (Q9)
- **Phase 6**: final regression vs baseline + sign-off doc

Is this the right granularity, or do you want fewer/larger phases? Also: **after each phase, should I pause for your approval, or run all phases then present a single result?**

**Your choice:** granularity

---

### Q12 — Anything else I should know?

- Open bugs in `docs/project Q&A bugs.docx` you want folded in?
- Any of the 8 manager-reported failures still relevant that I should NOT regress?
- Any hard deadline?

**Your answer:** Before starting the development make sure that we should mix these two agents completely. Its like therre are two separate branches of a tree, one is rag and another one is docqa.So first check the aligne ment first of these two agents. Yesterday we made additional changes add new techniques in rag agents, make sure this should be keep as it is. this docqa agent layer will be on top of it. In simple terms of user story is "If our rag agent is getting hallucinating, or giving vague/incorrect answers or completely fail to answer user's particular question but the agent is giving the list of source documents and the user knows that one of these files containing that answer then our  docqa agent comes into the picture, user will select the document pass it to the docqa agent and start the conversation with that document, and the flows goes on." In that way  we can satisfy the user story. Now for developing this story we need to make sure these things:
1) As the requirement goes on, we need to add additional features in both the agents separately , whether it might be related to both of separate so this user story should not be affected because of this.
2) Our rag agent is developing rapid while adding addvanced retrieval techniques such as RRF, and other, so docqa agent will ost like be used after the answer generated by rag agent, not before.
3) Most of the times the pdf files or source documents which user will pass/upload to docqa agent will be construction based drawings, so make sure docqa agent should be handle such complex drawing data.
4) MAke sure all this backend development should not change the outtput schema of apis, otherwise it should get very messy on UI side to integrate. Make sure not to changes the schema, output variable names and other stuff. Try to keep it constant.
5) Also we already integrated our old apis to sandbox UI, so make sure do not disrupt the schema of old api until its non-negotiable because of this additional features, and new apis should be aligned or related whatever the reson with old apis so that suer story should satisfy.


Apart from that if you have any queries then ask.

---

## NEXT STEP

Paste your answers inline in this file (or reply in chat). Once all 12 are answered, I will:

1. Write the design doc → `docs/superpowers/specs/2026-04-23-unified-rag-docqa-extension-design.md`
2. Get your review on the spec
3. Then invoke `writing-plans` to produce the phase-by-phase implementation plan
4. Wait for your go-ahead before touching any code

No code, no snapshots, no version folders created until you approve.
