"""Unit tests for Fix #3 — ``agentic.tools.aggregation_tools``.

MongoDB is stubbed via a fake collection so the tests run in <1s and don't
require live creds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))


class _FakeCursor:
    """Stub PyMongo cursor with chainable limit()."""

    def __init__(self, docs: list[dict]) -> None:
        self._docs = list(docs)

    def limit(self, n: int) -> "_FakeCursor":
        self._docs = self._docs[: n]
        return self

    def __iter__(self):
        return iter(self._docs)


class _FakeCollection:
    def __init__(self, find_result: list[dict] | None = None, aggregate_result: list[dict] | None = None) -> None:
        self._find = find_result or []
        self._agg = aggregate_result or []
        self.find_calls: list[tuple] = []
        self.agg_calls: list[tuple] = []

    def find(self, query, projection=None):
        self.find_calls.append((query, projection))
        return _FakeCursor(self._find)

    def aggregate(self, pipeline, **kwargs):
        self.agg_calls.append((pipeline, kwargs))
        return _FakeCursor(self._agg)


def _install_collection(monkeypatch, name: str, coll: _FakeCollection) -> None:
    """Patch ``agentic.tools.aggregation_tools.get_collection`` to return the fake."""
    from agentic.tools import aggregation_tools

    def fake_get(collection_name: str):
        return coll  # always the same fake
    monkeypatch.setattr(aggregation_tools, "get_collection", fake_get)


# ---------------------------------------------------------------------------
# count_equipment_tags
# ---------------------------------------------------------------------------

def test_count_equipment_extracts_unique_doas_tags(monkeypatch) -> None:
    from agentic.tools.aggregation_tools import count_equipment_tags

    fragments = [
        {"text": "OUTSIDE AIR DUCT TO DOAS-1 ON ROOF", "drawingTitle": "MECH ROOF PLAN", "drawingName": "M-ROOF", "pdfName": "M-ROOF.pdf"},
        {"text": "DOAS-1 schedule reference", "drawingTitle": "HVAC SCHEDULES", "drawingName": "M-201"},
        {"text": "DOAS-2 located in east mechanical room", "drawingTitle": "MECH 2ND FLOOR PLAN", "drawingName": "M-2"},
        {"text": "see DOAS-1 and DOAS-2 for outside-air requirements", "drawingTitle": "HVAC SCHEDULES", "drawingName": "M-201"},
    ]
    coll = _FakeCollection(find_result=fragments)
    _install_collection(monkeypatch, "drawing", coll)

    r = count_equipment_tags(project_id=7222, keywords=["DOAS"])
    assert r["total_unique_tags"] == 2
    assert set(r["unique_tags"]) == {"DOAS-1", "DOAS-2"}
    # Samples are attached per tag
    assert "DOAS-1" in r["samples"]
    # Per-drawing mentions are recorded
    assert any("SCHEDULES" in d["drawing"].upper() for d in r["by_drawing"])


def test_count_equipment_respects_level_filter(monkeypatch) -> None:
    from agentic.tools.aggregation_tools import count_equipment_tags

    fragments = [
        {"text": "VAV-201", "drawingTitle": "2ND FLOOR PLAN", "drawingName": "M-2"},
        {"text": "VAV-301", "drawingTitle": "3RD FLOOR PLAN", "drawingName": "M-3"},
    ]
    coll = _FakeCollection(find_result=fragments)
    _install_collection(monkeypatch, "drawing", coll)

    only_lvl2 = count_equipment_tags(project_id=1, keywords=["VAV"], level_filter=2)
    assert set(only_lvl2["unique_tags"]) == {"VAV-201"}
    # level_filter recorded in result
    assert only_lvl2["level_filter_applied"] == 2


def test_count_equipment_ignores_non_matching_prefix(monkeypatch) -> None:
    from agentic.tools.aggregation_tools import count_equipment_tags

    fragments = [
        {"text": "DOAS-1 unit and also VAV-103 and AHU-7 nearby", "drawingTitle": "M-01"},
    ]
    coll = _FakeCollection(find_result=fragments)
    _install_collection(monkeypatch, "drawing", coll)

    r = count_equipment_tags(project_id=1, keywords=["DOAS"])
    # Only DOAS-1 counts; VAV-103 and AHU-7 ignored even though in same text.
    assert r["unique_tags"] == ["DOAS-1"]


def test_count_equipment_raises_without_keywords() -> None:
    from agentic.tools.aggregation_tools import count_equipment_tags

    with pytest.raises(ValueError):
        count_equipment_tags(project_id=1, keywords=[])


# ---------------------------------------------------------------------------
# find_typical_levels
# ---------------------------------------------------------------------------

def test_find_typical_levels_clusters_floors_with_same_plan(monkeypatch) -> None:
    from agentic.tools.aggregation_tools import find_typical_levels

    aggregated_drawings = [
        {"_id": {"title": "3RD FLOOR PLAN", "name": "A-301"}, "pdfName": "A-301.pdf", "fragment_count": 40},
        {"_id": {"title": "4TH FLOOR PLAN", "name": "A-401"}, "pdfName": "A-401.pdf", "fragment_count": 39},
        {"_id": {"title": "5TH FLOOR PLAN", "name": "A-501"}, "pdfName": "A-501.pdf", "fragment_count": 41},
        {"_id": {"title": "6TH FLOOR PLAN", "name": "A-601"}, "pdfName": "A-601.pdf", "fragment_count": 40},
        {"_id": {"title": "LOBBY PLAN",      "name": "A-101"}, "pdfName": "A-101.pdf", "fragment_count": 90},
        {"_id": {"title": "ROOF PLAN",       "name": "A-ROOF"},"pdfName": "A-ROOF.pdf","fragment_count": 15},
    ]
    coll = _FakeCollection(aggregate_result=aggregated_drawings, find_result=[])
    _install_collection(monkeypatch, "drawing", coll)

    r = find_typical_levels(project_id=1)
    # The 4 floor-plan sheets cluster into one typical group
    assert len(r["typical_groups"]) == 1
    group = r["typical_groups"][0]
    assert set(group["levels"]) == {3, 4, 5, 6}


def test_find_typical_levels_returns_standalones_when_no_cluster(monkeypatch) -> None:
    from agentic.tools.aggregation_tools import find_typical_levels

    agg = [
        {"_id": {"title": "2ND FLOOR FRAMING PLAN", "name": "S-2"}, "pdfName": "S-2.pdf", "fragment_count": 10},
        {"_id": {"title": "10TH FLOOR ROOF PLAN",   "name": "S-10"},"pdfName": "S-10.pdf","fragment_count": 10},
    ]
    coll = _FakeCollection(aggregate_result=agg, find_result=[])
    _install_collection(monkeypatch, "drawing", coll)

    r = find_typical_levels(project_id=1)
    # Each is a unique title pattern → no typical groups (different "FRAMING" vs "ROOF")
    assert r["typical_groups"] == []
    assert set(r["standalone_levels"]) == {2, 10}


def test_find_typical_levels_captures_explicit_hints(monkeypatch) -> None:
    from agentic.tools.aggregation_tools import find_typical_levels

    agg = [
        {"_id": {"title": "3RD FLOOR PLAN", "name": "A-3"}, "pdfName": "A-3.pdf", "fragment_count": 1},
    ]
    # Fragments containing explicit typical hints
    fragments = [
        {"text": "Levels 3 THRU 6 are typical guestroom floors"},
        {"text": "TYPICAL LEVELS 7-10"},
        {"text": "random junk"},
    ]
    coll = _FakeCollection(aggregate_result=agg, find_result=fragments)
    _install_collection(monkeypatch, "drawing", coll)

    r = find_typical_levels(project_id=1)
    hint_ranges = {(h["range_start"], h["range_end"]) for h in r["explicit_typical_hints"]}
    assert (3, 6) in hint_ranges
    assert (7, 10) in hint_ranges


# ---------------------------------------------------------------------------
# list_schedule_entries
# ---------------------------------------------------------------------------

def test_list_schedule_entries_uses_vision_elements(monkeypatch) -> None:
    from agentic.tools import aggregation_tools
    from agentic.tools.aggregation_tools import list_schedule_entries

    vision_doc = {
        "sourceFile": "M-ROOF",
        "pages": [{
            "sheet_number": "M-ROOF",
            "page_summary": "Roof mechanical plan showing DOAS-1 and DOAS-2",
            "vision_elements": [
                {"label": "DOAS-1", "cfm": 5000, "kw": 25},
                {"label": "DOAS-2", "cfm": 4200, "kw": 22},
                {"label": "EF-1", "cfm": 600},  # shouldn't match DOAS
            ],
        }],
    }
    vision_coll = _FakeCollection(find_result=[vision_doc])
    drawing_coll = _FakeCollection()

    def fake_get(name: str):
        return vision_coll if name == "drawingVision" else drawing_coll
    monkeypatch.setattr(aggregation_tools, "get_collection", fake_get)

    r = list_schedule_entries(project_id=1, schedule_type="doas")
    assert r["source"] == "vision"
    labels = [row["label"] for row in r["vision_rows"]]
    assert any("DOAS-1" in l for l in labels)
    assert any("DOAS-2" in l for l in labels)
    # EF-1 should not be in the filtered list
    assert not any("EF-1" in l for l in labels)
    # Fallback should be None since vision path produced rows
    assert r["fallback_tag_extraction"] is None


def test_list_schedule_entries_falls_back_to_drawing_ocr(monkeypatch) -> None:
    """When vision returns nothing, fall back to drawing tag extraction."""
    from agentic.tools import aggregation_tools
    from agentic.tools.aggregation_tools import list_schedule_entries

    vision_coll = _FakeCollection(find_result=[])  # no vision docs
    drawing_coll = _FakeCollection(find_result=[
        {"text": "schedule references DOAS-1 only on this sheet", "drawingTitle": "MECH SCHEDULE"},
    ])

    def fake_get(name: str):
        return vision_coll if name == "drawingVision" else drawing_coll
    monkeypatch.setattr(aggregation_tools, "get_collection", fake_get)

    r = list_schedule_entries(project_id=1, schedule_type="doas")
    assert r["source"] == "drawing_ocr_fallback"
    assert r["fallback_tag_extraction"] is not None
    assert r["fallback_tag_extraction"]["unique_tags"] == ["DOAS-1"]


def test_list_schedule_entries_raw_keyword_type(monkeypatch) -> None:
    """Unknown schedule_type uses the raw string as a keyword."""
    from agentic.tools import aggregation_tools
    from agentic.tools.aggregation_tools import list_schedule_entries

    vision_coll = _FakeCollection(find_result=[])
    drawing_coll = _FakeCollection(find_result=[{"text": "XYZ-1 custom"}])

    def fake_get(name: str):
        return vision_coll if name == "drawingVision" else drawing_coll
    monkeypatch.setattr(aggregation_tools, "get_collection", fake_get)

    r = list_schedule_entries(project_id=1, schedule_type="XYZ")
    assert r["keywords_used"] == ["XYZ"]
