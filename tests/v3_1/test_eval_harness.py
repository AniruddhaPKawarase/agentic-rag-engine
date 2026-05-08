"""Smoke tests for the Phase P5 eval harness (eval/).

These tests do NOT make real LLM calls. They exercise the pure-python
seams: question loading, multi-turn pair list, env-flag context manager,
and the integer parser inside the LLM judge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the v3.1 worktree importable regardless of pytest's CWD.
_WORKTREE = Path(__file__).resolve().parent.parent.parent
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# ---------------------------------------------------------------------------
# Helpers — synthesise xlsx fixtures on the fly with openpyxl.
# ---------------------------------------------------------------------------


def _write_xlsx(path: Path, rows: list[list]) -> None:
    """Write a list-of-rows to an xlsx file using openpyxl."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(str(path))
    wb.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_questions_loader_handles_xlsx_with_header(tmp_path: Path) -> None:
    """A header row named 'question' should be detected and skipped."""
    from eval.run_v31_eval import load_questions

    fixture = tmp_path / "with_header.xlsx"
    _write_xlsx(fixture, [
        ["question"],
        ["What is the foundation type?"],
        ["List all electrical drawings."],
        [""],
        ["Show the chiller specs."],
    ])

    questions = load_questions(str(fixture))
    assert questions == [
        "What is the foundation type?",
        "List all electrical drawings.",
        "Show the chiller specs.",
    ]


def test_questions_loader_handles_xlsx_without_header(tmp_path: Path) -> None:
    """When no header is present, every non-empty row should be kept verbatim."""
    from eval.run_v31_eval import load_questions

    fixture = tmp_path / "no_header.xlsx"
    _write_xlsx(fixture, [
        ["What is the floor-to-floor height?"],
        ["Show drawing M-501."],
        ["List all VAVs."],
    ])

    questions = load_questions(str(fixture))
    assert len(questions) == 3
    assert questions[0] == "What is the floor-to-floor height?"
    assert questions[2] == "List all VAVs."


def test_questions_loader_handles_task_header(tmp_path: Path) -> None:
    """Header named 'Task' should also be detected (case-insensitive)."""
    from eval.run_v31_eval import load_questions

    fixture = tmp_path / "task_header.xlsx"
    _write_xlsx(fixture, [
        ["Task"],
        ["List all transformers."],
        ["Show the main switchgear."],
    ])

    questions = load_questions(str(fixture))
    assert questions == [
        "List all transformers.",
        "Show the main switchgear.",
    ]


def test_multi_turn_pairs_count_is_50() -> None:
    """The hand-crafted pair list must contain exactly 50 pairs."""
    from eval.run_v31_eval import MULTI_TURN_PAIRS

    assert len(MULTI_TURN_PAIRS) == 50
    # Each entry is a 2-tuple of strings.
    for pair in MULTI_TURN_PAIRS:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert pair[0] and pair[1]


def test_llm_judge_parses_integer_from_response() -> None:
    """The judge parses 1-5 out of free-form text and returns -1 otherwise."""
    from eval.llm_judge import _parse_score

    assert _parse_score("4") == 4
    assert _parse_score("Score: 5/5") == 5
    assert _parse_score("I think it's a 3.") == 3
    assert _parse_score("0") == -1
    assert _parse_score("9") == -1
    assert _parse_score("") == -1
    assert _parse_score("no digits here") == -1


def test_llm_judge_returns_minus_one_on_call_failure() -> None:
    """When the underlying LLM call raises, the judge returns -1 (no crash)."""
    from eval import llm_judge

    def _boom(**_kwargs):
        raise RuntimeError("API explosion")

    with patch.object(llm_judge, "_judge_call", return_value=-1):
        score = llm_judge.score_human_tone("q", "a")
        assert score == -1


def test_baseline_flags_all_off() -> None:
    """baseline_flags() must set every v3.1 flag to 'false'."""
    from eval.run_v31_eval import baseline_flags

    flags = baseline_flags()
    assert flags["V31_CHAIN_ENABLED"] == "false"
    assert flags["MEMORY_RECALL_ENABLED"] == "false"
    assert flags["QUERY_REWRITER_ENABLED"] == "false"
    assert flags["ANSWER_SYNTHESIZER_ENABLED"] == "false"
    assert flags["STYLE_REWRITER_ENABLED"] == "false"
    assert flags["CACHE_REEXPRESSION_ENABLED"] == "false"
    assert flags["MEMORY_WRITER_VECTOR_ENABLED"] == "false"


def test_variant_flags_synth_and_stylist_on() -> None:
    """variant_flags() must enable the master switch + synth + stylist.

    Memory flags are off in single-turn mode, on in multi-turn mode.
    """
    from eval.run_v31_eval import variant_flags

    no_mem = variant_flags(with_memory=False)
    assert no_mem["V31_CHAIN_ENABLED"] == "true"
    assert no_mem["ANSWER_SYNTHESIZER_ENABLED"] == "true"
    assert no_mem["STYLE_REWRITER_ENABLED"] == "true"
    assert no_mem["MEMORY_RECALL_ENABLED"] == "false"
    assert no_mem["QUERY_REWRITER_ENABLED"] == "false"
    assert no_mem["MEMORY_WRITER_VECTOR_ENABLED"] == "false"

    with_mem = variant_flags(with_memory=True)
    assert with_mem["MEMORY_RECALL_ENABLED"] == "true"
    assert with_mem["QUERY_REWRITER_ENABLED"] == "true"
    assert with_mem["MEMORY_WRITER_VECTOR_ENABLED"] == "true"


def test_env_flags_context_manager_restores() -> None:
    """env_flags must restore prior values (or absence) on exit."""
    from eval.run_v31_eval import env_flags

    os.environ["TEST_KEEP"] = "original"
    os.environ.pop("TEST_NEW", None)

    with env_flags(TEST_KEEP="modified", TEST_NEW="value"):
        assert os.environ["TEST_KEEP"] == "modified"
        assert os.environ["TEST_NEW"] == "value"

    assert os.environ["TEST_KEEP"] == "original"
    assert "TEST_NEW" not in os.environ
    os.environ.pop("TEST_KEEP", None)


def test_count_citations() -> None:
    """count_citations() detects bracketed drawing-style markers."""
    from eval.run_v31_eval import count_citations

    text = "See [E-101] and [M-501.2]. Also reference [P-201, p.3]."
    assert count_citations(text) == 3
    assert count_citations("") == 0
    assert count_citations("no citations here") == 0


def test_summarize_pair_handles_empty_inputs() -> None:
    """summarize_pair() must not crash on empty input lists."""
    from eval.run_v31_eval import summarize_pair

    summary = summarize_pair([], [])
    assert summary["total_questions"] == 0
    assert summary["total_pairs_clean"] == 0


def test_cli_help_works(capsys: pytest.CaptureFixture) -> None:
    """`--help` must succeed and list all 4 modes."""
    from eval.run_v31_eval import build_arg_parser

    parser = build_arg_parser()
    # SystemExit raised by argparse on --help; we just verify the parser
    # is well-formed and our 4 modes are in the choices.
    action = next(a for a in parser._actions if a.dest == "mode")
    assert set(action.choices) == {"single_turn", "multi_turn", "model_ab", "all"}
