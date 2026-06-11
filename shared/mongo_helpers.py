"""shared.mongo_helpers — single source of truth for Mongo field-name conventions.

The iField Atlas drawings_v3 / specifications_v3 / drawings_v3_blocks collections
use **camelCase** field names (Mongo / Mongoose convention from the source ingest):

    projectId           ← camelCase, INT
    sheetNumber         ← camelCase
    drawingTitle        ← camelCase
    drawingName         ← camelCase

Python variables here are conventionally snake_case (project_id, sheet_number).
The mismatch causes recurring bugs when probes / queries use the wrong field.
This module centralises the conversion. **All Mongo-query construction code MUST
use project_filter() instead of building {"project_id": pid} inline.**

The legacy `drawing` and `drawingTexts` collections are READ-ONLY history, do not
touch. The OFF-LIMITS `drawings_v2` / `specifications_v2` are protected by HARD
RULE (see [[feedback_v2_collection_offlimits]]) — never query them.
"""
from __future__ import annotations
from typing import Any, Dict, Union

# The single source of truth — DO NOT inline {"projectId": ...} anywhere else
MONGO_PROJECT_ID_FIELD = "projectId"

# Whitelisted v3 collections this helper covers
V3_COLLECTIONS = (
    "drawings_v3",
    "drawings_v3_blocks",
    "specifications_v3",
    "specifications_v3_blocks",
)

# OFF-LIMITS per HARD RULE — never query these
FORBIDDEN_COLLECTIONS = (
    "drawings_v2",
    "specifications_v2",
)


def _coerce_project_id(pid: Union[int, str, None]) -> int:
    """Coerce to int. The Mongo field is stored as INT. String→int conversion."""
    if pid is None:
        raise ValueError("project_id required (got None)")
    if isinstance(pid, int):
        return pid
    if isinstance(pid, str):
        if not pid.strip():
            raise ValueError("project_id required (got empty string)")
        try:
            return int(pid)
        except ValueError as e:
            raise ValueError(f"project_id must be int-coercible (got {pid!r})") from e
    raise TypeError(f"project_id must be int or str (got {type(pid).__name__})")


def project_filter(project_id: Union[int, str]) -> Dict[str, Any]:
    """Return the canonical {"projectId": <int>} filter for any v3 collection.

    Usage::

        from shared.mongo_helpers import project_filter
        coll.find({**project_filter(project_id), "sheetNumber": "A-211"})

    Never do {"project_id": project_id} — the field is camelCase in Mongo.
    """
    return {MONGO_PROJECT_ID_FIELD: _coerce_project_id(project_id)}


def project_filter_strict(project_id: Union[int, str], collection_name: str) -> Dict[str, Any]:
    """Same as project_filter() but raises if the collection is OFF-LIMITS.

    Use this in any code path that takes a dynamic collection name.
    """
    if collection_name in FORBIDDEN_COLLECTIONS:
        raise PermissionError(
            f"collection {collection_name!r} is OFF-LIMITS per HARD RULE — never query"
        )
    return project_filter(project_id)


# Common compound filters — saves repetition
def project_and_sheet_filter(project_id: Union[int, str], sheet_number: str) -> Dict[str, Any]:
    """Filter by (projectId, sheetNumber). Both fields are camelCase in Mongo."""
    return {
        MONGO_PROJECT_ID_FIELD: _coerce_project_id(project_id),
        "sheetNumber": sheet_number,
    }


def project_and_text_filter(project_id: Union[int, str], text_query: str) -> Dict[str, Any]:
    """Filter by (projectId, $text). Caller is responsible for the $text index."""
    return {
        MONGO_PROJECT_ID_FIELD: _coerce_project_id(project_id),
        "$text": {"$search": text_query},
    }


# Doc helpers — for reading projectId from a returned doc, accepting either field
# name (defensive — older inserts may use snake_case in legacy data).
def read_project_id(doc: Dict[str, Any]) -> Any:
    """Read projectId from a doc, accepting camelCase or snake_case (legacy fallback)."""
    return doc.get(MONGO_PROJECT_ID_FIELD) if MONGO_PROJECT_ID_FIELD in doc else doc.get("project_id")


__all__ = [
    "MONGO_PROJECT_ID_FIELD",
    "V3_COLLECTIONS",
    "FORBIDDEN_COLLECTIONS",
    "project_filter",
    "project_filter_strict",
    "project_and_sheet_filter",
    "project_and_text_filter",
    "read_project_id",
]
