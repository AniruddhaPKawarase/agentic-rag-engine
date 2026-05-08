"""Phase 1 answer-format regression tests."""
from __future__ import annotations
import re
from pathlib import Path

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


def test_agent_system_prompt_has_no_citation_rule():
    """After Phase 1 fix, agent.py must NOT contain 'Cite sources: [Source:...]' instruction."""
    content = Path("agentic/core/agent.py").read_text(encoding="utf-8")
    positive_citation_instructions = re.findall(
        r'(?im)^[^#\n]*cite\s+sources:\s*\[Source:', content
    )
    assert len(positive_citation_instructions) == 0, (
        f"Contradictory 'Cite sources: [Source:]' instruction still present: "
        f"{positive_citation_instructions}"
    )
