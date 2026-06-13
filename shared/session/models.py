"""
Session data models — EngineUsage and UnifiedSessionMeta.

These are intentionally *mutable* dataclasses (not frozen) because session
state is accumulated over time via ``record()`` calls. Immutability is
enforced at the module boundary — callers receive copies via ``to_dict()``.
"""

import time
from dataclasses import dataclass, field


@dataclass
class EngineUsage:
    """Tracks how many times each engine was invoked in a session."""

    agentic: int = 0
    traditional: int = 0
    fallback: int = 0

    def record(self, engine: str) -> None:
        """Increment the counter for the given engine name."""
        if engine == "agentic":
            self.agentic += 1
        elif engine == "traditional":
            self.traditional += 1
        elif engine == "fallback":
            self.fallback += 1

    def to_dict(self) -> dict:
        """Return a plain dict snapshot of current counts."""
        return {
            "agentic": self.agentic,
            "traditional": self.traditional,
            "fallback": self.fallback,
        }


@dataclass
class DocumentScope:
    """Tracks the active document scope for a session.

    State machine: NO_SCOPE -> SCOPED (user selects doc) -> NO_SCOPE (user unscopes)
    Auto-unscope after 30 min idle.
    """

    drawing_title: str = ""
    drawing_name: str = ""
    document_type: str = ""  # "drawing" or "specification"
    section_title: str = ""  # for specifications
    pdf_name: str = ""
    activated_at: float = 0.0
    last_query_at: float = 0.0

    @property
    def is_active(self) -> bool:
        """True if a document scope is currently set and not idle-expired."""
        if not self.drawing_title and not self.section_title:
            return False
        # Auto-unscope after 30 min idle
        if self.last_query_at and (time.time() - self.last_query_at) > 1800:
            return False
        return True

    def activate(
        self,
        drawing_title: str = "",
        drawing_name: str = "",
        document_type: str = "drawing",
        section_title: str = "",
        pdf_name: str = "",
    ) -> None:
        """Enter scoped mode for a specific document."""
        self.drawing_title = drawing_title
        self.drawing_name = drawing_name
        self.document_type = document_type
        self.section_title = section_title
        self.pdf_name = pdf_name
        self.activated_at = time.time()
        self.last_query_at = time.time()

    def touch(self) -> None:
        """Update last_query_at to prevent idle expiry."""
        self.last_query_at = time.time()

    def clear(self) -> None:
        """Exit scoped mode."""
        self.drawing_title = ""
        self.drawing_name = ""
        self.document_type = ""
        self.section_title = ""
        self.pdf_name = ""
        self.activated_at = 0.0
        self.last_query_at = 0.0

    def to_dict(self) -> dict:
        """Return scope state for API responses."""
        return {
            "is_active": self.is_active,
            "drawing_title": self.drawing_title,
            "drawing_name": self.drawing_name,
            "document_type": self.document_type,
            "section_title": self.section_title,
            "pdf_name": self.pdf_name,
        }


@dataclass
class UnifiedSessionMeta:
    """Extended metadata attached to each unified session."""

    engine_usage: EngineUsage = field(default_factory=EngineUsage)
    last_engine: str = ""
    total_cost_usd: float = 0.0
    scope: DocumentScope = field(default_factory=DocumentScope)
    previously_scoped: list = field(default_factory=list)  # quick-access history
