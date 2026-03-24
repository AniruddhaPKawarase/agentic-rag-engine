"""Public retrieval API."""

from .diagnostics import _TEST_QUERIES, get_project_stats, list_projects, test_retrieval
from .embeddings import _cached_embedding, convert_l2_to_similarity, get_embedding
from .engine import retrieve_context, retrieve_context_with_session
from .loaders import _get_project_config, _load_project, initialize_index
from .metadata import (
    _NAME_KEYS,
    _TITLE_KEYS,
    build_pdf_download_url,
    extract_field,
    get_display_title,
    get_document_name,
    get_drawing_title,
    get_page_number,
    get_s3_path,
)
from .state import (
    DIMENSION,
    EMBEDDING_MODEL,
    PROJECTS,
    ProjectConfig,
    _INDEX_ROOT,
    _make_config,
    client,
)

__all__ = [
    "EMBEDDING_MODEL",
    "DIMENSION",
    "client",
    "ProjectConfig",
    "_INDEX_ROOT",
    "_make_config",
    "PROJECTS",
    "_load_project",
    "initialize_index",
    "_get_project_config",
    "_TITLE_KEYS",
    "_NAME_KEYS",
    "get_drawing_title",
    "get_display_title",
    "build_pdf_download_url",
    "extract_field",
    "get_document_name",
    "get_page_number",
    "get_s3_path",
    "_cached_embedding",
    "get_embedding",
    "convert_l2_to_similarity",
    "retrieve_context",
    "retrieve_context_with_session",
    "list_projects",
    "get_project_stats",
    "_TEST_QUERIES",
    "test_retrieval",
]
