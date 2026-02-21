"""
defenx_nlp.interfaces — Abstract base classes and protocols.

Define contracts that every encoder or inference component must satisfy.
Third-party integrations should subclass these to guarantee compatibility
with the rest of the defenx-nlp ecosystem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class BaseEncoder(ABC):
    """
    Abstract base class for all text encoders.

    Subclass this when building custom encoder backends (e.g. OpenAI
    embeddings, custom fine-tuned models, or mock encoders for testing).

    Minimum contract
    ----------------
    - ``encode(text)``       → (embedding_dim,) float32 numpy array
    - ``encode_batch(texts)``→ (N, embedding_dim) float32 numpy array
    - ``embedding_dim``      → int property
    - ``device``             → torch.device property

    Examples
    --------
    >>> class MyEncoder(BaseEncoder):
    ...     def encode(self, text): return np.zeros(384, dtype=np.float32)
    ...     def encode_batch(self, texts): return np.zeros((len(texts), 384), np.float32)
    ...     @property
    ...     def embedding_dim(self): return 384
    ...     @property
    ...     def device(self): return torch.device("cpu")
    """

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single sentence.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray of shape (embedding_dim,), dtype float32
        """

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of sentences.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), dtype float32
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Torch device the model runs on."""

    # ── Optional helpers with default implementations ─────────────────────────

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two embeddings produced by this encoder.
        Override for GPU-accelerated similarity if needed.
        """
        from .utils import cosine_similarity
        return cosine_similarity(a, b)

    def warmup(self) -> None:
        """
        Run a no-op encode to initialise CUDA kernels.
        Call once after construction when CUDA is used to avoid cold-start
        latency on the first real inference.
        """
        self.encode("warmup")


class BaseInferenceEngine(ABC):
    """
    Abstract base class for any classification / scoring model that sits
    downstream of a ``BaseEncoder``.

    Typical pattern
    ---------------
    embedding = encoder.encode(text)
    result    = engine.infer(embedding)
    """

    @abstractmethod
    def infer(self, embedding: np.ndarray) -> np.ndarray:
        """
        Run inference on a pre-computed embedding.

        Parameters
        ----------
        embedding : np.ndarray of shape (embedding_dim,)

        Returns
        -------
        np.ndarray — model output (probabilities, scores, logits, …)
        """

    @abstractmethod
    def infer_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of pre-computed embeddings.

        Parameters
        ----------
        embeddings : np.ndarray of shape (N, embedding_dim)

        Returns
        -------
        np.ndarray of shape (N, output_dim)
        """
