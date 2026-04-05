"""Unit tests for tools/validation.py — input validation functions."""

import pytest

from tools.validation import (
    validate_drawing_id,
    validate_limit,
    validate_project_id,
    validate_search_text,
    validate_source_file,
)


# ── validate_project_id ──────────────────────────────────────────────


class TestValidateProjectId:
    """Tests for validate_project_id."""

    def test_validate_project_id_valid(self) -> None:
        """Normal positive integer passes validation and is returned."""
        assert validate_project_id(7166) == 7166

    def test_validate_project_id_valid_boundary(self) -> None:
        """Boundary value 1 is accepted."""
        assert validate_project_id(1) == 1

    def test_validate_project_id_max_boundary(self) -> None:
        """Boundary value 999999 is accepted."""
        assert validate_project_id(999999) == 999999

    def test_validate_project_id_zero(self) -> None:
        """Zero raises ValueError (must be positive)."""
        with pytest.raises(ValueError, match="out of range"):
            validate_project_id(0)

    def test_validate_project_id_negative(self) -> None:
        """Negative value raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_project_id(-1)

    def test_validate_project_id_too_large(self) -> None:
        """Value exceeding 999999 raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_project_id(1_000_000)

    def test_validate_project_id_float_coercion(self) -> None:
        """Float value is coerced to int."""
        assert validate_project_id(42.9) == 42

    def test_validate_project_id_string_raises(self) -> None:
        """String input raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_project_id("abc")  # type: ignore[arg-type]


# ── validate_limit ───────────────────────────────────────────────────


class TestValidateLimit:
    """Tests for validate_limit."""

    def test_validate_limit_default_on_invalid_type(self) -> None:
        """Non-numeric input returns default value 10."""
        assert validate_limit("invalid") == 10  # type: ignore[arg-type]

    def test_validate_limit_default_on_zero(self) -> None:
        """Zero returns default value 10."""
        assert validate_limit(0) == 10

    def test_validate_limit_default_on_negative(self) -> None:
        """Negative returns default value 10."""
        assert validate_limit(-5) == 10

    def test_validate_limit_within_max(self) -> None:
        """Value under max_limit is returned as-is."""
        assert validate_limit(25) == 25

    def test_validate_limit_capped_to_max(self) -> None:
        """Value exceeding max_limit is capped to 50."""
        assert validate_limit(100) == 50

    def test_validate_limit_exactly_max(self) -> None:
        """Value equal to max_limit is returned as-is."""
        assert validate_limit(50) == 50

    def test_validate_limit_custom_max(self) -> None:
        """Custom max_limit is respected."""
        assert validate_limit(30, max_limit=20) == 20


# ── validate_search_text ─────────────────────────────────────────────


class TestValidateSearchText:
    """Tests for validate_search_text."""

    def test_validate_search_text_valid(self) -> None:
        """Normal text passes validation."""
        assert validate_search_text("electrical drawings") == "electrical drawings"

    def test_validate_search_text_empty_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="search_text is required"):
            validate_search_text("")

    def test_validate_search_text_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="search_text is required"):
            validate_search_text(None)  # type: ignore[arg-type]

    def test_validate_search_text_too_long_truncated(self) -> None:
        """Text exceeding max_length is truncated (not rejected)."""
        long_text = "a" * 600
        result = validate_search_text(long_text)
        assert len(result) == 500

    def test_validate_search_text_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped from result."""
        assert validate_search_text("  hello  ") == "hello"


# ── validate_source_file ─────────────────────────────────────────────


class TestValidateSourceFile:
    """Tests for validate_source_file."""

    def test_validate_source_file_valid(self) -> None:
        """Normal filename passes validation."""
        assert validate_source_file("report.pdf") == "report.pdf"

    def test_validate_source_file_path_traversal_dotdot(self) -> None:
        """Path traversal with .. raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source_file path"):
            validate_source_file("../etc/passwd")

    def test_validate_source_file_path_traversal_forward_slash(self) -> None:
        """Forward slash in path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source_file path"):
            validate_source_file("etc/passwd")

    def test_validate_source_file_path_traversal_backslash(self) -> None:
        """Backslash in path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source_file path"):
            validate_source_file("etc\\passwd")

    def test_validate_source_file_empty_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="source_file is required"):
            validate_source_file("")

    def test_validate_source_file_truncated_at_500(self) -> None:
        """Long filename is truncated to 500 chars."""
        long_name = "x" * 600
        result = validate_source_file(long_name)
        assert len(result) == 500


# ── validate_drawing_id ──────────────────────────────────────────────


class TestValidateDrawingId:
    """Tests for validate_drawing_id."""

    def test_validate_drawing_id_valid(self) -> None:
        """Positive integer passes validation."""
        assert validate_drawing_id(42) == 42

    def test_validate_drawing_id_negative_raises(self) -> None:
        """Negative value raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_drawing_id(-1)

    def test_validate_drawing_id_zero_raises(self) -> None:
        """Zero raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_drawing_id(0)

    def test_validate_drawing_id_string_raises(self) -> None:
        """String input raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_drawing_id("abc")  # type: ignore[arg-type]

    def test_validate_drawing_id_float_coercion(self) -> None:
        """Float is coerced to int."""
        assert validate_drawing_id(10.7) == 10
