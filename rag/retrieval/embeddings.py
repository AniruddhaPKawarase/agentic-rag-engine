"""Embedding generation and similarity helpers."""
from __future__ import annotations

import functools

import numpy as np

from . import state

client = state.client
EMBEDDING_MODEL = state.EMBEDDING_MODEL

@functools.lru_cache(maxsize=512)
def _cached_embedding(text: str) -> tuple:
    """Return embedding as a plain tuple (hashable → LRU-safe)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return tuple(response.data[0].embedding)


def get_embedding(text: str) -> np.ndarray:
    return np.array(_cached_embedding(text), dtype=np.float32)


def convert_l2_to_similarity(l2_distance: float) -> float:
    """Map L2 distance to a [0, 1] similarity score."""
    return max(0.0, min(1.0, 1.0 - (l2_distance ** 2) / 4.0))
