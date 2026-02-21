"""
defenx_nlp.utils — Shared mathematical and embedding utilities.

All functions are pure NumPy (no torch dependency) and safe to use
from any thread without locks.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ── Similarity ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D float arrays.

    Parameters
    ----------
    a, b : np.ndarray  — shape (D,)

    Returns
    -------
    float in [-1, 1]

    Examples
    --------
    >>> cosine_similarity(np.array([1, 0]), np.array([0, 1]))
    0.0
    >>> cosine_similarity(np.array([1, 0]), np.array([1, 0]))
    1.0
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a query vector and every row of a matrix.

    Parameters
    ----------
    query  : np.ndarray of shape (D,)
    matrix : np.ndarray of shape (N, D)

    Returns
    -------
    np.ndarray of shape (N,) — similarity scores in [-1, 1]

    Examples
    --------
    >>> q = np.array([1, 0], dtype=np.float32)
    >>> m = np.eye(2, dtype=np.float32)
    >>> batch_cosine_similarity(q, m)
    array([1., 0.], dtype=float32)
    """
    q = (query / (np.linalg.norm(query) + 1e-8)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normed = (matrix / norms).astype(np.float32)
    return normed @ q


# ── Retrieval ─────────────────────────────────────────────────────────────────

def top_k_similar(
    query: np.ndarray,
    corpus: List[np.ndarray],
    k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find the k most similar embeddings to a query.

    Parameters
    ----------
    query  : np.ndarray of shape (D,)
    corpus : list of np.ndarray, each shape (D,)
    k      : int — number of results to return (default 5)

    Returns
    -------
    list of (index, score) sorted by score descending

    Examples
    --------
    >>> corpus = [np.array([1,0], dtype=np.float32),
    ...           np.array([0,1], dtype=np.float32)]
    >>> top_k_similar(np.array([1,0], dtype=np.float32), corpus, k=1)
    [(0, 1.0)]
    """
    if not corpus:
        return []
    matrix = np.stack(corpus, axis=0)          # (N, D)
    scores = batch_cosine_similarity(query, matrix)
    k = min(k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(int(i), float(scores[i])) for i in top_idx]


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """
    L2-normalise an embedding vector to unit length.

    Parameters
    ----------
    emb : np.ndarray of shape (D,)

    Returns
    -------
    np.ndarray of shape (D,), dtype float32, unit length

    Examples
    --------
    >>> v = normalize_embedding(np.array([3, 4], dtype=np.float32))
    >>> round(np.linalg.norm(v), 5)
    1.0
    """
    norm = np.linalg.norm(emb) + 1e-8
    return (emb / norm).astype(np.float32)


def normalize_batch(matrix: np.ndarray) -> np.ndarray:
    """
    Row-wise L2-normalisation of a 2-D embedding matrix.

    Parameters
    ----------
    matrix : np.ndarray of shape (N, D)

    Returns
    -------
    np.ndarray of shape (N, D), dtype float32
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return (matrix / norms).astype(np.float32)


# ── Dimensionality ────────────────────────────────────────────────────────────

def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean pooling over token embeddings weighted by the attention mask.

    Useful when working directly with raw transformer hidden states.

    Parameters
    ----------
    token_embeddings : np.ndarray of shape (seq_len, hidden_dim)
    attention_mask   : np.ndarray of shape (seq_len,), values 0 or 1

    Returns
    -------
    np.ndarray of shape (hidden_dim,), dtype float32
    """
    mask = attention_mask.astype(np.float32).reshape(-1, 1)
    summed = (token_embeddings * mask).sum(axis=0)
    count  = mask.sum() + 1e-8
    return (summed / count).astype(np.float32)
