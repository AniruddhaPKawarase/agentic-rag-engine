"""Project diagnostics and retrieval test helpers."""
from __future__ import annotations

from typing import Dict, List, Optional

from .engine import retrieve_context
from .state import PROJECTS

def list_projects() -> List[Dict]:
    """Return status of all configured projects."""
    return [
        {
            "project_id":       pid,
            "loaded":           cfg.loaded,
            "vectors":          cfg.index.ntotal if cfg.index else 0,
            "metadata_records": len(cfg.metadata),
        }
        for pid, cfg in PROJECTS.items()
    ]


def get_project_stats(project_id: int) -> Dict:
    """Return detailed statistics for a single project."""
    if project_id not in PROJECTS:
        return {"error": f"Project {project_id} not in registry."}

    config = PROJECTS[project_id]
    if not config.loaded:
        return {"error": f"Project {project_id} not loaded."}

    source_types: Dict[str, int] = {}
    for record in config.metadata:
        st = record.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1

    return {
        "project_id":       project_id,
        "vectors":          config.index.ntotal if config.index else 0,
        "metadata_records": len(config.metadata),
        "source_types":     source_types,
        "loaded":           config.loaded,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST  (python retrieve.py --test --project 7222)
# ─────────────────────────────────────────────────────────────────────────────

_TEST_QUERIES: Dict[int, List[str]] = {
    7212: [
        "What compliance is required for GAS FIREPLACES?",
        "Instructions for DOOR FRAME OPENINGS",
        "Thickness of wall for WATER TANK SUPPORT",
    ],
    7223: [
        "SOVENT FITTING DETAIL",
        "Field quality control requirements for Interior Lighting",
        "Submittals required for SURGE PROTECTIVE DEVICES",
    ],
}


def test_retrieval(project_id: Optional[int] = None) -> None:
    """Run test queries and print results to stdout."""
    targets = [project_id] if project_id else list(PROJECTS.keys())

    for pid in targets:
        if pid not in PROJECTS or not PROJECTS[pid].loaded:
            print(f"[test] Project {pid} not loaded — skipping.")
            continue

        queries = _TEST_QUERIES.get(pid, ["What are the key specifications?"])
        print(f"\n{'='*70}\nTEST — Project {pid}\n{'='*70}")

        for query in queries:
            print(f"\nQuery: {query!r}")
            results = retrieve_context(query, top_k=3, min_score=0.1, filter_project_id=pid)

            if not results:
                print("  (no results)")
                continue

            for i, r in enumerate(results):
                print(f"  [{i+1}] {r['source_type'].upper()}  score={r['similarity']:.3f}")
                print(f"       display_title : {r['display_title']}")
                print(f"       drawing_title : {r['drawing_title']}")
                print(f"       download_url  : {r['download_url']}")
                print(f"       page          : {r['page']}")
                print(f"       text preview  : {r['text'][:120]}…")
