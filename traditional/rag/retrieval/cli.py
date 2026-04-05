"""CLI entrypoint for retrieval module."""
from __future__ import annotations

import argparse

from .diagnostics import get_project_stats, list_projects, test_retrieval
from .engine import retrieve_context
from .loaders import initialize_index
from .state import PROJECTS


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS retrieval system")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--interactive", action="store_true", help="Interactive query mode")
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--source-type", type=str, help="drawing / specification")
    parser.add_argument("--project", type=int, help="Project ID")
    parser.add_argument("--list-projects", action="store_true", help="List configured projects")
    parser.add_argument("--stats", action="store_true", help="Show project statistics")
    args = parser.parse_args()

    initialize_index(args.project)

    if args.list_projects:
        print("\nConfigured projects:")
        for info in list_projects():
            print(
                f"  {info['project_id']}: loaded={info['loaded']}, "
                f"vectors={info['vectors']}, metadata={info['metadata_records']}"
            )

    elif args.stats:
        pids = [args.project] if args.project else list(PROJECTS.keys())
        for pid in pids:
            stats = get_project_stats(pid)
            print(f"\nProject {pid}:", stats)

    elif args.test:
        test_retrieval(args.project)

    elif args.interactive:
        pid = args.project or next(iter(PROJECTS))
        print(f"\nInteractive mode ??? project {pid}. Type 'quit' to exit.")
        while True:
            try:
                q = input("\nQuery: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            for i, r in enumerate(retrieve_context(q, top_k=5, filter_project_id=pid)):
                print(f"  [{i+1}] {r['display_title']}  (score={r['similarity']:.3f})")
                print(f"       download: {r['download_url']}")
                print(f"       {r['text'][:200]}")

    elif args.query:
        pid = args.project or 7212
        results = retrieve_context(
            args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            filter_source_type=args.source_type,
            filter_project_id=pid,
        )
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r['source_type'].upper()}  score={r['similarity']:.3f}")
            print(f"  display_title : {r['display_title']}")
            print(f"  download_url  : {r['download_url']}")
            print(f"  text: {r['text'][:300]}")

    else:
        test_retrieval()


if __name__ == "__main__":
    main()
