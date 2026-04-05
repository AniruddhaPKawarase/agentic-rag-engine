"""RAG API package exports."""

from .state import app

# Import routes for side-effect route registration on `app`.
from . import routes  # noqa: F401

__all__ = ["app"]
