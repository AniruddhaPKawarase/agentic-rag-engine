"""
Gateway request / response models — Pydantic v2 BaseModel schemas.

QueryRequest validates inbound queries.
UnifiedResponse is the standardised envelope returned by the orchestrator.

Phase 1 additions: fields needed for the DocQA bridge and to align the
Pydantic model with the wire payload the orchestrator actually emits.
All new fields are Optional with safe defaults — existing clients that
do not know about these fields are unaffected (forward compatibility).
"""

from __future__ import annotations

from typing import Optional

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
    # --- Phase 1 additions (schema-aligned to wire truth + DocQA bridge) ---
    source_documents: Optional[list[dict]] = None
    active_agent: Optional[str] = "rag"
    selected_document: Optional[dict] = None
    clarification_prompt: Optional[str] = None
    docqa_session_id: Optional[str] = None
    groundedness_score: Optional[float] = None
    flagged_claims: Optional[list[dict]] = None
