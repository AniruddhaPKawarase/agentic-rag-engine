"""Backward-compatible retrieval entrypoint.

The implementation lives in `rag.retrieval.*` modules.
"""

from rag import retrieval as _retrieval
from rag.retrieval import *  # noqa: F401,F403
from rag.retrieval.cli import main as _main
from rag.retrieval import state as _state


def __getattr__(name: str):
    if hasattr(_state, name):
        return getattr(_state, name)
    if hasattr(_retrieval, name):
        return getattr(_retrieval, name)
    raise AttributeError(f"module 'retrieve' has no attribute {name!r}")


if __name__ == "__main__":
    _main()
