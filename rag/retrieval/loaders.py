"""Project index loading utilities."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import faiss

from . import state
from .state import PROJECTS, ProjectConfig

def _load_project(config: ProjectConfig) -> None:
    """Load FAISS index + metadata for one project. Internal use only."""
    print(f"[retrieve] Loading project {config.project_id} …")

    # --- S3 FALLBACK: download index from S3 if local file missing ---
    if not config.index_path.exists() and os.getenv("STORAGE_BACKEND", "local") == "s3":
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from s3_utils.operations import download_file
            from s3_utils.helpers import faiss_index_key
            idx_key = faiss_index_key(config.index_path.name)
            print(f"[retrieve] Downloading {config.index_path.name} from S3...")
            download_file(idx_key, str(config.index_path))
            meta_key = faiss_index_key(config.metadata_path.name)
            download_file(meta_key, str(config.metadata_path))
        except Exception as e:
            print(f"[retrieve] S3 download failed: {e}")
    # --- END S3 FALLBACK ---

    if not config.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {config.index_path}")

    config.index = faiss.read_index(str(config.index_path))
    print(f"[retrieve] Project {config.project_id}: {config.index.ntotal} vectors (dim={config.index.d})")

    if not config.metadata_path.exists():
        warnings.warn(f"Metadata not found: {config.metadata_path} — retrieval will have no metadata.")
        config.loaded = True
        return

    meta_list: List[Dict]      = []
    meta_dict: Dict[int, Dict] = {}

    with open(config.metadata_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                warnings.warn(f"[retrieve] Bad JSON on line {i}: {exc}")
                continue
            meta_list.append(record)
            meta_dict[i] = record

    config.metadata      = meta_list
    config.metadata_dict = meta_dict
    config.loaded        = True

    print(f"[retrieve] Project {config.project_id}: {len(meta_list)} metadata records loaded.")

    # Show field inventory for first record (helps during debugging)
    if meta_list:
        sample = meta_list[0]
        print(f"[retrieve]   Fields: {list(sample.keys())}")
        print(f"[retrieve]   source_type: {sample.get('source_type', 'N/A')}")


def initialize_index(project_id: Optional[int] = None) -> None:
    """
    Load FAISS index + metadata.

    Args:
        project_id: Load a specific project, or None to load ALL projects.
    """
    
    if project_id is not None:
        if project_id not in PROJECTS:
            raise ValueError(f"Project {project_id} not in registry. "
                             f"Add it to PROJECTS in retrieve.py.")
        config = PROJECTS[project_id]
        if not config.loaded:
            _load_project(config)
        state._current_project_id = project_id
    else:
        for config in PROJECTS.values():
            if not config.loaded:
                try:
                    _load_project(config)
                except Exception as exc:
                    warnings.warn(f"[retrieve] Failed to load project {config.project_id}: {exc}")


def _get_project_config(project_id: int) -> ProjectConfig:
    """Return a loaded ProjectConfig, loading it if necessary."""
    if project_id not in PROJECTS:
        raise ValueError(f"Project {project_id} not in registry.")
    config = PROJECTS[project_id]
    if not config.loaded:
        _load_project(config)
    return config
