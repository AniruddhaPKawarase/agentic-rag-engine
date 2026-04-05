"""
Import shim for AgenticRAG engine.

Adds the unified-rag-agent root AND the ``agentic/`` sub-package directory
to ``sys.path`` so that internal imports like ``from core.agent import ...``
resolve correctly when running from the unified project root.
"""

import os
import sys

_AGENTIC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AGENTIC_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if _AGENTIC_DIR not in sys.path:
    sys.path.insert(0, _AGENTIC_DIR)
