"""Tests for `agentic.generation.text_normalizer`.

Validates that the cp1252-mojibake-fix layer correctly replaces typographic
Unicode characters with ASCII equivalents and is idempotent.
"""

from __future__ import annotations

from agentic.generation.text_normalizer import normalize_chunks, normalize_output


def test_normalize_strips_bullet():
    assert normalize_output("• Resilient tile flooring") == "- Resilient tile flooring"


def test_normalize_strips_smart_quotes():
    assert normalize_output("“AHU-1”") == '"AHU-1"'
    assert normalize_output("‘it’s’") == "'it's'"


def test_normalize_strips_em_dash_and_en_dash():
    assert normalize_output("Level 1 – Level 5") == "Level 1 - Level 5"
    assert normalize_output("DOAS—main floor") == "DOAS-main floor"


def test_normalize_strips_ellipsis():
    assert normalize_output("Loading…") == "Loading..."


def test_normalize_strips_nbsp():
    assert normalize_output("12 CFM") == "12 CFM"


def test_normalize_handles_double_encoded_mojibake():
    """Defensive: if upstream already produced mojibake, fix it on the way out."""
    assert normalize_output("â€¢ item one") == "- item one"
    assert normalize_output("â€“ dash") == "- dash"


def test_normalize_is_idempotent():
    s = "• — “Quoted” …"
    once = normalize_output(s)
    twice = normalize_output(once)
    assert once == twice


def test_normalize_empty_input():
    assert normalize_output("") == ""
    assert normalize_output(None) is None  # type: ignore[arg-type]


def test_normalize_no_bullets_passthrough():
    s = "Plain ASCII answer with no fancy chars."
    assert normalize_output(s) == s


def test_normalize_chunks_streams_correctly():
    chunks = ["First — ", "second “third” ", "fourth…"]
    out = list(normalize_chunks(chunks))
    assert out == ["First - ", 'second "third" ', "fourth..."]


def test_normalize_chunks_handles_non_string():
    """Non-string chunks pass through unchanged (defensive)."""
    out = list(normalize_chunks(["text", None, "more"]))
    assert out == ["text", None, "more"]
