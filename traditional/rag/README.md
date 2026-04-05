# RAG Module Layout

This folder contains the refactored RAG flow split into two sub-packages:

- `rag/retrieval/`:
  - `state.py`: shared retrieval constants, OpenAI client, project registry.
  - `loaders.py`: FAISS index + metadata loading lifecycle.
  - `metadata.py`: metadata normalization/title/url helpers.
  - `embeddings.py`: embedding + similarity helpers.
  - `engine.py`: core retrieval functions.
  - `diagnostics.py`: project stats and retrieval test helpers.
  - `cli.py`: retrieval CLI entrypoint.

- `rag/api/`:
  - `state.py`: FastAPI app bootstrap + dependency initialization.
  - `models.py`: request/response schemas.
  - `helpers.py`: context formatting and utility helpers.
  - `prompts.py`: prompt builders for `rag`, `web`, `hybrid` modes.
  - `generation_web.py`: web-search generation flow.
  - `generation_unified.py`: unified RAG/web/hybrid generation flow.
  - `generation.py`: compatibility export for generation functions.
  - `routes.py`: API endpoints, handlers, and startup hooks.

Backward-compatible entrypoints remain at project root:

- `retrieve.py`: imports/re-exports from `rag.retrieval.*`, keeps CLI behavior.
- `generate.py`: imports/re-exports from `rag.api.*`, keeps API startup behavior.

