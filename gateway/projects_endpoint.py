"""Projects listing endpoint — enumerate every project that has data available
in the ``drawings_v3`` collection.

A drawing-set is considered *processed / available* when at least one document
for that ``projectId`` exists in ``drawings_v3``. This endpoint aggregates the
collection by ``projectId`` and returns the distinct projects plus lightweight
per-project counts so a UI can populate a project picker without scanning the
whole collection client-side.

Design notes
------------
- Field convention is camelCase (``projectId`` is an INT) — see
  ``shared.mongo_helpers``. We never inline ``{"project_id": ...}``.
- Read-only. Touches only ``drawings_v3`` (never the OFF-LIMITS v2 collections).
- Additive: mounted as a sub-router via ``router.include_router`` with the same
  guarded-import pattern as the session-history endpoint, so a failure here can
  never take down ``/query``.
- Two modes: light (default — projectId + drawing_count, one ``$group``) and
  ``detail=true`` (adds disciplines, set ids, distinct sheet count).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Query

logger = logging.getLogger("agentic_rag.projects_endpoint")

router = APIRouter()

COLLECTION = "drawings_v3"


def _get_drawings_collection():
    """Open the drawings_v3 collection using the app's shared Mongo accessor."""
    from core.db import get_collection  # lazy: keeps import failures local
    return get_collection(COLLECTION)


def _build_pipeline(detail: bool, min_drawings: int) -> List[Dict[str, Any]]:
    """Aggregation pipeline grouping drawings_v3 by projectId.

    Light mode groups + counts only. Detail mode additionally collects the
    distinct disciplines, set ids, and a distinct sheet count per project.
    Documents with a null/missing projectId are excluded.
    """
    group: Dict[str, Any] = {"_id": "$projectId", "drawing_count": {"$sum": 1}}
    project: Dict[str, Any] = {
        "_id": 0,
        "project_id": "$_id",
        "drawing_count": 1,
    }
    if detail:
        group["disciplines"] = {"$addToSet": "$discipline"}
        group["set_ids"] = {"$addToSet": "$setId"}
        group["_sheet_numbers"] = {"$addToSet": "$sheetNumber"}
        project["disciplines"] = {
            "$sortArray": {
                "input": {
                    "$filter": {
                        "input": "$disciplines",
                        "as": "d",
                        "cond": {"$ne": ["$$d", None]},
                    }
                },
                "sortBy": 1,
            }
        }
        project["set_ids"] = {
            "$filter": {
                "input": "$set_ids",
                "as": "s",
                "cond": {"$ne": ["$$s", None]},
            }
        }
        project["sheet_count"] = {"$size": "$_sheet_numbers"}

    pipeline: List[Dict[str, Any]] = [
        {"$match": {"projectId": {"$ne": None}}},
        {"$group": group},
    ]
    if min_drawings > 1:
        pipeline.append({"$match": {"drawing_count": {"$gte": min_drawings}}})
    pipeline.append({"$project": project})
    pipeline.append({"$sort": {"project_id": 1}})
    return pipeline


@router.get("/projects")
async def list_projects(
    detail: bool = Query(
        False,
        description="Include per-project disciplines, set ids, and distinct sheet count.",
    ),
    min_drawings: int = Query(
        1,
        ge=1,
        description="Only return projects with at least this many drawings.",
    ),
) -> Dict[str, Any]:
    """List every project that has processed data available in drawings_v3.

    Returns
    -------
    {
        "success": true,
        "collection": "drawings_v3",
        "count": <int>,                       # number of distinct projects
        "projects": [
            {"project_id": 7201, "drawing_count": 2361},   # light
            ...                                            # + disciplines/set_ids/sheet_count when detail=true
        ]
    }
    """
    try:
        coll = _get_drawings_collection()
    except Exception as exc:  # noqa: BLE001
        logger.error("[projects] cannot open %s: %s", COLLECTION, exc)
        return {"success": False, "collection": COLLECTION, "error": str(exc),
                "count": 0, "projects": []}

    pipeline = _build_pipeline(detail=detail, min_drawings=int(min_drawings))
    try:
        projects = list(coll.aggregate(pipeline, allowDiskUse=True))
    except Exception as exc:  # noqa: BLE001
        logger.error("[projects] aggregation failed: %s", exc)
        return {"success": False, "collection": COLLECTION, "error": str(exc),
                "count": 0, "projects": []}

    logger.info("[projects] returned %d projects (detail=%s, min_drawings=%d)",
                len(projects), detail, min_drawings)
    return {
        "success": True,
        "collection": COLLECTION,
        "count": len(projects),
        "projects": projects,
    }
