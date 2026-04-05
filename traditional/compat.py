"""
Import shim for Traditional RAG engine.

Adds the unified-rag-agent root AND the ``traditional/`` sub-package directory
to ``sys.path`` so that internal imports resolve correctly.

Also redirects bare ``s3_utils`` imports to ``shared.s3_utils`` so that legacy
code like ``from s3_utils.client import get_s3_client`` keeps working.
"""

import importlib
import os
import sys

_TRADITIONAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TRADITIONAL_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if _TRADITIONAL_DIR not in sys.path:
    sys.path.insert(0, _TRADITIONAL_DIR)

# --- Redirect bare s3_utils → shared.s3_utils ---
import shared.s3_utils as _s3  # noqa: E402

sys.modules["s3_utils"] = _s3
sys.modules["s3_utils.client"] = importlib.import_module("shared.s3_utils.client")
sys.modules["s3_utils.config"] = importlib.import_module("shared.s3_utils.config")
sys.modules["s3_utils.helpers"] = importlib.import_module("shared.s3_utils.helpers")
sys.modules["s3_utils.operations"] = importlib.import_module(
    "shared.s3_utils.operations"
)
