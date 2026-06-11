"""Input validation for all tool parameters."""


def validate_project_id(project_id: int) -> int:
    """Validate project_id is a positive integer."""
    if not isinstance(project_id, (int, float)):
        raise ValueError(f"project_id must be an integer, got {type(project_id).__name__}")
    project_id = int(project_id)
    if project_id <= 0 or project_id > 999999:
        raise ValueError(f"project_id out of range: {project_id}")
    return project_id


def validate_limit(limit: int, max_limit: int = 50) -> int:
    """Validate and cap the limit parameter."""
    if not isinstance(limit, (int, float)):
        return 10
    limit = int(limit)
    if limit <= 0:
        return 10
    return min(limit, max_limit)


def validate_search_text(text: str, max_length: int = 500) -> str:
    """Validate search text is a non-empty string within length limits."""
    if not text or not isinstance(text, str):
        raise ValueError("search_text is required and must be a string")
    return text[:max_length].strip()


def validate_source_file(source_file: str) -> str:
    """Validate source_file is a safe string (no path traversal)."""
    if not source_file or not isinstance(source_file, str):
        raise ValueError("source_file is required")
    if ".." in source_file or "/" in source_file or "\\" in source_file:
        raise ValueError("Invalid source_file path")
    return source_file[:500]


def validate_drawing_id(drawing_id) -> int:
    """Validate drawing_id is a positive integer.

    Accepts int, float, or numeric string (LLMs sometimes pass quoted IDs).
    Raises ValueError with a clear, agent-readable message if the value is
    None or unparseable — the agent's tool-error handler turns that into
    a recoverable observation so the ReAct loop can try a different tool
    instead of dying mid-step.
    """
    if drawing_id is None:
        raise ValueError(
            "drawing_id is required but was null. "
            "Call legacy_list_drawings or legacy_search_text first to find "
            "a valid drawing_id, then retry with that integer."
        )
    if isinstance(drawing_id, str):
        s = drawing_id.strip()
        if not s.isdigit():
            raise ValueError(
                f"drawing_id must be an integer, got string {drawing_id!r}. "
                "Use legacy_list_drawings to discover the numeric drawing_id "
                "for a sheet name like 'M-501'."
            )
        drawing_id = int(s)
    elif not isinstance(drawing_id, (int, float)):
        raise ValueError(
            f"drawing_id must be an integer, got {type(drawing_id).__name__}. "
            "Use legacy_list_drawings to discover the numeric drawing_id."
        )
    drawing_id = int(drawing_id)
    if drawing_id <= 0:
        raise ValueError(f"drawing_id out of range: {drawing_id}")
    return drawing_id
