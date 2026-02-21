"""
defenx-nlp — Semantic NLP Intelligence Toolkit
===============================================

A domain-agnostic Python library for semantic sentence encoding,
embedding generation, and GPU/CPU-aware inference interfaces.

Designed to be extracted from the DEFENX NeuroSight Artificial Brain
Cognitive Simulator and used independently in:

  * NLP classification systems
  * Anomaly detection pipelines
  * Log intelligence engines
  * Behavioural analytics platforms

Quick start
-----------
    from defenx_nlp import SemanticEncoder

    enc = SemanticEncoder()                         # lazy, auto device
    emb = enc.encode("Neural networks are great.")  # (384,) float32
    embs = enc.encode_batch(["Hello", "World"])     # (2, 384) float32

Public API surface
------------------
    SemanticEncoder      — main encoder class
    BaseEncoder          — abstract base for custom encoders
    BaseInferenceEngine  — abstract base for downstream models
    get_device           — resolve "auto" / "cuda" / "cpu" → torch.device
    device_info          — hardware diagnostic dictionary
    clean_text           — configurable single-text cleaner
    batch_clean          — apply clean_text to a list
    truncate             — hard-truncate text
    cosine_similarity    — cosine sim between two embeddings
    batch_cosine_similarity — query vs matrix, vectorised
    top_k_similar        — retrieve top-k similar embeddings
    normalize_embedding  — L2-normalise a single embedding
    normalize_batch      — L2-normalise a (N, D) matrix
"""

from .device import device_info, get_device
from .encoder import SemanticEncoder
from .interfaces import BaseEncoder, BaseInferenceEngine
from .preprocessing import batch_clean, clean_text, truncate
from .utils import (
    batch_cosine_similarity,
    cosine_similarity,
    normalize_batch,
    normalize_embedding,
    top_k_similar,
)

__version__  = "0.1.0"
__author__   = "DEFENX"
__email__    = "contact@defenx.ai"
__license__  = "MIT"

__all__ = [
    # Core
    "SemanticEncoder",
    # Abstracts
    "BaseEncoder",
    "BaseInferenceEngine",
    # Device
    "get_device",
    "device_info",
    # Preprocessing
    "clean_text",
    "batch_clean",
    "truncate",
    # Similarity / retrieval
    "cosine_similarity",
    "batch_cosine_similarity",
    "top_k_similar",
    "normalize_embedding",
    "normalize_batch",
]
