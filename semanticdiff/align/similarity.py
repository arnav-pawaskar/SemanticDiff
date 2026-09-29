"""Similarity signals used for provision alignment."""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist


def embed(texts: list[str], encoder, batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype=np.float32)
    return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Embeddings are L2-normalised, so the dot product is the cosine."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)


def token_matrix(a: list[str], b: list[str]) -> np.ndarray:
    """Order-insensitive token overlap in [0, 1] (rapidfuzz token_sort_ratio)."""
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return cdist([t.lower() for t in a], [t.lower() for t in b],
                 scorer=fuzz.token_sort_ratio, dtype=np.float32) / 100.0


def fused_matrix(cos: np.ndarray, tok: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * cos + (1.0 - alpha) * tok
