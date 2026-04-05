"""Unit tests for core/text_reconstruction.py — spatial OCR fragment reconstruction."""

from core.text_reconstruction import reconstruct_drawing_text


class TestReconstructDrawingText:
    """Tests for the reconstruct_drawing_text pure function."""

    def test_reconstruct_empty_fragments(self) -> None:
        """Empty fragment list returns empty string."""
        assert reconstruct_drawing_text([]) == ""

    def test_reconstruct_single_fragment(self) -> None:
        """Single fragment returns its text."""
        fragments = [{"text": "ELECTRICAL PANEL", "x": 0, "y": 0, "page": 1}]
        result = reconstruct_drawing_text(fragments)
        assert result == "ELECTRICAL PANEL"

    def test_reconstruct_same_line(self) -> None:
        """Two fragments at the same y-position are joined with a space."""
        fragments = [
            {"text": "DRAWING", "x": 0, "y": 100, "page": 1},
            {"text": "TITLE", "x": 200, "y": 100, "page": 1},
        ]
        result = reconstruct_drawing_text(fragments)
        assert result == "DRAWING TITLE"

    def test_reconstruct_different_lines(self) -> None:
        """Fragments at different y-positions produce separate lines."""
        fragments = [
            {"text": "Line One", "x": 0, "y": 50, "page": 1},
            {"text": "Line Two", "x": 0, "y": 200, "page": 1},
        ]
        result = reconstruct_drawing_text(fragments)
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "Line One"
        assert lines[1] == "Line Two"

    def test_reconstruct_sorting_by_y_then_x(self) -> None:
        """Fragments out of order are sorted by y then x."""
        fragments = [
            {"text": "BOTTOM", "x": 0, "y": 300, "page": 1},
            {"text": "TOP-RIGHT", "x": 200, "y": 50, "page": 1},
            {"text": "TOP-LEFT", "x": 0, "y": 50, "page": 1},
        ]
        result = reconstruct_drawing_text(fragments)
        lines = result.split("\n")
        assert lines[0] == "TOP-LEFT TOP-RIGHT"
        assert lines[1] == "BOTTOM"

    def test_reconstruct_max_chars_truncation(self) -> None:
        """Output exceeding max_chars is truncated with marker."""
        fragments = [
            {"text": "A" * 100, "x": 0, "y": i * 100, "page": 1}
            for i in range(10)
        ]
        result = reconstruct_drawing_text(fragments, max_chars=250)
        assert "[... truncated ...]" in result

    def test_reconstruct_empty_text_fragments_skipped(self) -> None:
        """Fragments with empty or whitespace-only text are skipped."""
        fragments = [
            {"text": "", "x": 0, "y": 0, "page": 1},
            {"text": "   ", "x": 50, "y": 0, "page": 1},
            {"text": "VALID", "x": 100, "y": 0, "page": 1},
        ]
        result = reconstruct_drawing_text(fragments)
        assert result == "VALID"

    def test_reconstruct_multi_page(self) -> None:
        """Fragments on different pages produce separate lines."""
        fragments = [
            {"text": "Page 1 content", "x": 0, "y": 0, "page": 1},
            {"text": "Page 2 content", "x": 0, "y": 0, "page": 2},
        ]
        result = reconstruct_drawing_text(fragments)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "Page 1" in lines[0]
        assert "Page 2" in lines[1]

    def test_reconstruct_line_merge_threshold(self) -> None:
        """Fragments within LINE_MERGE_THRESHOLD (15px) are on the same line."""
        fragments = [
            {"text": "SAME", "x": 0, "y": 100, "page": 1},
            {"text": "LINE", "x": 100, "y": 110, "page": 1},  # within 15px
        ]
        result = reconstruct_drawing_text(fragments)
        assert result == "SAME LINE"

    def test_reconstruct_pre_sorted_skips_sort(self) -> None:
        """pre_sorted=True uses fragments in given order."""
        fragments = [
            {"text": "FIRST", "x": 0, "y": 0, "page": 1},
            {"text": "SECOND", "x": 100, "y": 0, "page": 1},
        ]
        result = reconstruct_drawing_text(fragments, pre_sorted=True)
        assert result == "FIRST SECOND"
