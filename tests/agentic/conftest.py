"""Shared pytest fixtures for AgenticRAG tests."""

import sys
from pathlib import Path

# Ensure the project root is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

# Ensure agentic subpackages (core/, tools/) are importable by short name
sys.path.insert(0, str(_project_root / "agentic"))
