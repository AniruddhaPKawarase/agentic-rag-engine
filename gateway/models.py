"""
Gateway request / response models — Pydantic v2 BaseModel schemas.

QueryRequest validates inbound queries.
UnifiedResponse is the standardised envelope returned by the orchestrator.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=10000)


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
    conversation_history: Optional[list[ConversationMessage]] = None
    engine: Optional[str] = None


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
    needs_document_selection: bool = False
    available_documents: list[dict] = Field(default_factory=list)
    improved_queries: list[str] = Field(default_factory=list)
    query_tips: list[str] = Field(default_factory=list)
    scoped_to: Optional[str] = None
