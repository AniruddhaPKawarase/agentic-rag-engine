# Production Review — Clarifying Questions

**Project:** Unified RAG Agent — Ask AI from Document + Intelligent Follow-Up
**Date:** 2026-04-15
**Status:** AWAITING ANSWERS — All questions must be answered before development begins

---

## Instructions

Answer each question below. Short answers are fine. Write "N/A" if not applicable.
Questions are organized by the 12 production review dimensions + additional critical areas.

---

## 1. Scaling

**Q1.1:** What is the expected concurrent user count for this system?
- Current: _20 to 50__
- 6 months: 250 - 500___
- 1 year: __500+_

**Q1.2:** How many projects will run simultaneously? The drawing title aggregation query runs per-project. Do we expect 10 projects or 1000?
- Answer: __currently only around 10 may be less but in future it will ofcourse increase_

**Q1.3:** The `drawing` collection has 2.8M fragments. For a single project, what is the typical fragment count? (Affects aggregation speed for listing unique drawingTitles)
- Answer: ___around 50k

**Q1.4:** Should the drawing title list cache be in-memory (per-worker, simple, fast) or Redis (shared across workers, survives restart)?
- Answer: ___in memory

**Q1.5:** Is horizontal scaling needed? Currently `--workers 1`. Should we plan for multi-worker (2-4 workers with shared state)?
- Answer: ___keep it at most 3

---

## 2. Optimization

**Q2.1:** The drawing title aggregation runs on every "no results" scenario. Should we pre-compute and cache title lists at session creation time (eager) or only when needed (lazy)?
- Answer: ___yes pre-compute

**Q2.2:** Should the FAISS traditional fallback still run alongside document discovery, or should document discovery REPLACE the fallback entirely?
- Answer: ___REPLACE it

**Q2.3:** Query enhancement (Story I, no-results rephrasing) requires an additional LLM call. Is this acceptable cost-wise, or should we use a cheaper model (gpt-4.1-mini) for enhancement?
- Answer: ___acceptable

**Q2.4:** Should scoped queries skip the traditional FAISS fallback entirely? (Since scope is MongoDB-based, FAISS may not have matching scope filters)
- Answer: ___Yes skip it

**Q2.5:** The current FAISS lazy load saves ~2GB RAM. With drawingVision removed, does the memory profile change? Any new memory constraints?
- Answer: ___not now

---

## 3. Performance Metrics

**Q3.1:** What is the acceptable latency for document discovery (listing drawing titles)?
- Target: ___<2ms (current FAISS fallback is < 5s cold, < 3s warm)

**Q3.2:** What is the acceptable latency for a scoped query (query within a specific drawingTitle)?
- Target: ___<5 ms

**Q3.3:** Should we add a Prometheus metric for scope usage? (e.g., `scoped_queries_total`, `discovery_requests_total`, `scope_set_total`)
- Answer: ___Not now we will fix it later make the arrangements for it.

**Q3.4:** What SLA do we target for this feature? (e.g., 99.5% success rate on scoped queries)
- Answer: ___99.5% success rate on scoped queries and related .

**Q3.5:** Should we track "query improvement rate" — how often users refine their query using suggested improvements vs abandoning?
- Answer: ___Yes

---

## 4. Request Handling

**Q4.1:** When user is in scoped mode and sends a query that's clearly outside the document (e.g., "What's the weather?"), should the system:
- A) Answer from scope (will return empty) then suggest unscoping
- B) Detect off-topic and auto-unscope
- C) Detect off-topic and ask user "Did you mean to search all documents?"
- Answer: ___ Detect off-topic and send the answer, question is out of scope, please ask regarding project.

**Q4.2:** Should document discovery be a separate API endpoint (e.g., `GET /projects/{id}/documents`) or embedded in the `/query` response?
- Answer: ___Now we need to integrate this agent in UI which is developed in Angular so make a call according to it.

**Q4.3:** For streaming (`/query/stream`), when document discovery triggers, should we:
- A) Stream the partial agentic answer first, then append document list
- B) Return document list as a single SSE event (no streaming)
- Answer: ___Stream the partial agentic answer first, then append document list

**Q4.4:** Rate limiting: should scoped queries have a different rate limit than unscoped? (Scoped queries hit fewer documents, cheaper)
- Answer: ___No

**Q4.5:** Should the `engine` override field still work in scoped mode? (e.g., `engine="traditional"` + scope = traditional with pinned docs, `engine="agentic"` + scope = agentic with DB filter)
- Answer: ___ No, as it will be completely on Mongodb side.

---

## 5. Vulnerability / Security

**Q5.1:** The drawing_title filter uses regex matching. Should we sanitize for regex injection? (e.g., user sends `drawing_title=".*"` to match everything)
- Answer: ___No list down the drawing titles which are in the project  id

**Q5.2:** Can a user scope to a drawing from a DIFFERENT project than their session's projectId? (Cross-project access control)
- Answer: ___No, never.

**Q5.3:** Should document discovery expose ALL drawing titles, or should there be access control per user/role?
- Answer: ___ Expose the drawing title which are only available in the respective project id. Do not show all

**Q5.4:** The agentic agent sees the scope via system prompt. Can a prompt injection attack in the user query override the scope? (We have DB-level enforcement as backup, but should we also sanitize?)
- Answer: ___Yes there is a possibility sanitize it

**Q5.5:** Are there any documents (by drawingTitle) that should be excluded from discovery? (Confidential/restricted drawings)
- Answer: ___No.

---

## 6. SDLC Parameters

**Q6.1:** Target test coverage for new code?
- Current: __85_ % | Target: _99.5__ %

**Q6.2:** Should we write tests before implementation (TDD) or after?
- Answer: ___Yes

**Q6.3:** Branching strategy: feature branch per phase, or single branch for all phases?
- Answer: ___Featue branch

**Q6.4:** Code review process: self-review via code-reviewer agent, or human review required?
- Answer: ___Self review,as agent have an access.

**Q6.5:** Should we create a separate PR per phase or bundle phases?
- Answer: ___ Yes.

**Q6.6:** Are there any CI/CD pipelines to integrate with? (GitHub Actions, Jenkins, etc.)
- Answer: ___Now now, bbut we will add , so make arrangements for it

---

## 7. Compliance

**Q7.1:** Are there any data residency requirements for the MongoDB queries? (e.g., construction data must stay in US region)
- Answer: ___make a call on yourself.

**Q7.2:** Does the audit trail need to include document scope events? (Who scoped to what document, when)
- Answer: ___No, only mention the projectid as an identifier.

**Q7.3:** Are there any SOC 2 / ISO 27001 requirements for this system?
- Answer: ___It will manage later, make arrangements.

**Q7.4:** Do we need to log which specific documents a user accessed via scoped queries? (For compliance audit)
- Answer: ___yes.

**Q7.5:** GDPR / data protection: does the follow-up question generation need to avoid including PII from documents?
- Answer: ___yes

---

## 8. Disaster Recovery & Backup

**Q8.1:** If MongoDB is down, should document discovery gracefully degrade? (Return "service unavailable" or fall through to FAISS without scope)
- Answer: ___Yes

**Q8.2:** If the drawing title cache is stale (new drawings added to project), what's the acceptable staleness? (5 min, 1 hr, session-lifetime)
- Answer: ___1 hr

**Q8.3:** Should session scope state be backed up to S3 (like existing sessions), or is in-memory sufficient?
- Answer: ___yes, it should

**Q8.4:** If the OpenAI API is down during query enhancement (Story I), should we:
- A) Skip enhancement and return raw "no results"
- B) Use a fallback/cached set of generic improvement tips
- Answer: ___se a fallback/cached set of generic improvement tips

**Q8.5:** What is the RPO (Recovery Point Objective) for session data?
- Answer: ___ MAke it general.

---

## 9. Support & Helpdesk Framework

**Q9.1:** Should the `/debug-pipeline` endpoint include document scope state? (For support debugging)
- Answer: ___yes

**Q9.2:** When a user reports "wrong answer in scoped mode", what debug info should be logged? (Scope title, query, tool calls, results)
- Answer: ___All of it

**Q9.3:** Should there be an admin endpoint to list all active scoped sessions? (For monitoring user behavior)
- Answer: ___yes

**Q9.4:** Error messages to users: should they be technical ("MongoDB aggregation timeout") or user-friendly ("Please try again in a moment")?
- Answer: ___user-friendly

**Q9.5:** Should the system detect and warn about projects with very few drawings (< 5) where document scoping may not add value?
- Answer: ___yes

---

## 10. System Maintenance

**Q10.1:** When new drawings are added to a project in MongoDB, should the title cache auto-invalidate? Or manual cache clear?
- Answer: ___yes auto

**Q10.2:** Should there be a maintenance endpoint to rebuild/refresh the drawing title cache?
- Answer: ___yes

**Q10.3:** Is there a process for adding new MongoDB indexes? The `drawingTitle` + `projectId` compound index would significantly speed up aggregation.
- Answer: ___not now.

**Q10.4:** How often do the drawing/specification collections get updated? (Real-time, daily batch, on upload)
- Answer: ___daily-batch

**Q10.5:** When a project is archived/deleted, should cached title lists be cleaned up?
- Answer: ___yes.

---

## 11. Network & Security Requirements

**Q11.1:** Is the MongoDB connection TLS-encrypted? (Currently `mongodb+srv://` suggests yes)
- Answer: ___yes

**Q11.2:** Should the document discovery endpoint require the same API key authentication as `/query`?
- Answer: ___yes

**Q11.3:** CORS: any changes needed for the scope-related UI interactions?
- Answer: ___No

**Q11.4:** Should scope state be included in the bearer token / session token, or kept server-side only?
- Answer: ___Kept server side only

**Q11.5:** Are there firewall rules that need updating for any new endpoints?
- Answer: ___Not now.

---

## 12. Resource Management: Efficiency through Automation

**Q12.1:** Should document discovery results be pre-computed for all active projects on startup? (Saves first-query latency at cost of startup time)
- Answer: ___Yes

**Q12.2:** The agent currently has a $0.50/request cost cap. Should scoped queries have a lower cap? (They should be cheaper since they search less data)
- Answer: ___Yes, it can be work, but if results are not as relevant then use high cost cap infra.

**Q12.3:** Daily budget of $50. With query enhancement adding LLM calls on failure, should the daily budget increase or should enhancement respect the existing budget?
- Answer: ___The daily budget should increase, quality is important.

**Q12.4:** Should idle scoped sessions auto-unscope after inactivity? (e.g., 30 min idle → clear scope)
- Answer: ___yes

**Q12.5:** Memory management: should we cap the number of cached drawing title lists? (e.g., LRU eviction after 100 projects)
- Answer: ___yes

---

## 13. Additional Critical Questions

### Data Quality

**Q13.1:** Are drawingTitle values consistent in the database, or are there duplicates/typos? (e.g., "Mechanical Floor Plan" vs "MECHANICAL FLOOR PLAN" vs "Mech Floor Plan")
- Answer: ___There are chances.

**Q13.2:** Do ALL drawings have a drawingTitle? Or are some null/empty? How should we handle drawings without titles?
- Answer: ___it is the possibility but they might be very few, use alternatives in such cases.

**Q13.3:** Are there drawings with the same drawingTitle but different drawingName? (e.g., "Floor Plan" appears 5 times with different drawing numbers)
- Answer: ___May be yes, use alternatives in such cases.

### User Experience

**Q13.4:** When presenting available_documents, should we group by:
- A) Trade (Mechanical, Electrical, Plumbing...)
- B) Document type (Drawing, Specification)
- C) Alphabetical by title
- D) Relevance to the failed query (smart ranking)
- Answer: ___  Document type, Trade , Relevance to the failed query (smart ranking)

**Q13.5:** Maximum number of document suggestions to show? (All unique titles, or top 10/20/50?)
- Answer: ___All unique titles

**Q13.6:** Should the UI remember which documents the user has previously scoped to in this session? (Quick-access list)
- Answer: ___Yes

### Business Logic

**Q13.7:** When user selects a specification (not a drawing), should the scope filter use `sectionTitle` or `pdfName`?
- Answer: ___Yes

**Q13.8:** Should drawing title scoping work across BOTH the `drawing` and `specification` collections simultaneously? (e.g., scope to "HVAC" filters both drawings AND specs)
- Answer: ___Yes

**Q13.9:** For the traditional engine (FAISS fallback), should the existing `pin-document` API remain for advanced users, or be fully replaced by the new scope feature?
- Answer: ___fully replaced by the new scope feature

**Q13.10:** Should the system learn from user behavior? (e.g., if users frequently scope to "Mechanical" drawings, rank them higher in discovery)
- Answer: ___ Yes, everytime.

---

## Summary Checklist

| # | Section | Questions | Answered |
|---|---------|-----------|----------|
| 1 | Scaling | 5 | [ ] |
| 2 | Optimization | 5 | [ ] |
| 3 | Performance Metrics | 5 | [ ] |
| 4 | Request Handling | 5 | [ ] |
| 5 | Vulnerability / Security | 5 | [ ] |
| 6 | SDLC Parameters | 6 | [ ] |
| 7 | Compliance | 5 | [ ] |
| 8 | Disaster Recovery | 5 | [ ] |
| 9 | Support & Helpdesk | 5 | [ ] |
| 10 | System Maintenance | 5 | [ ] |
| 11 | Network & Security | 5 | [ ] |
| 12 | Resource Management | 5 | [ ] |
| 13 | Additional Critical | 10 | [ ] |
| **TOTAL** | | **71** | |

---

**Next Step:** Once ALL questions are answered, we proceed to:
1. Finalize the roadmap with answers incorporated
2. Create detailed development documentation
3. Begin Phase 0 (cleanup) after user confirmation
