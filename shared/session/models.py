"""
Session data models — EngineUsage and UnifiedSessionMeta.

These are intentionally *mutable* dataclasses (not frozen) because session
state is accumulated over time via ``record()`` calls. Immutability is
enforced at the module boundary — callers receive copies via ``to_dict()``.
"""

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
class UnifiedSessionMeta:
    """Extended metadata attached to each unified session."""

    engine_usage: EngineUsage = field(default_factory=EngineUsage)
    last_engine: str = ""
    total_cost_usd: float = 0.0
