"""Shared retrieval state and project registry."""
from __future__ import annotations

import functools
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DIMENSION       = 1536

client = OpenAI()

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT REGISTRY
# Edit this block (or use env vars) to add / remove projects.
# ─────────────────────────────────────────────────────────────────────────────
_INDEX_ROOT = Path(os.getenv(
    "INDEX_ROOT",
    "/home/ubuntu/chatbot/aniruddha/Agentic_AI/PROD/index"
))


@dataclass
class ProjectConfig:
    """Per-project FAISS index + metadata state."""
    project_id:    int
    index_path:    Path
    metadata_path: Path
    loaded:        bool            = False
    index:         Any             = None
    metadata:      List[Dict]      = field(default_factory=list)
    metadata_dict: Dict[int, Dict] = field(default_factory=dict)


def _make_config(pid: int) -> ProjectConfig:
    return ProjectConfig(
        project_id    = pid,
        index_path    = _INDEX_ROOT / f"faiss_index_{pid}.bin",
        metadata_path = _INDEX_ROOT / f"metadata_{pid}.jsonl",
    )


PROJECTS: Dict[int, ProjectConfig] = {
    pid: _make_config(pid) for pid in [7166, 7201, 7212, 7222, 7223, 7277, 7292, 7325]
}

# Backward-compat for code that reads _current_project_id
_current_project_id: Optional[int] = None
