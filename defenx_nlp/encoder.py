"""
defenx_nlp.encoder — SemanticEncoder: the primary public interface.

Architecture decisions
----------------------
* **Lazy loading** — the SentenceTransformer model is NOT loaded at import
  time or even at ``__init__`` time (when ``lazy=True``).  The first call to
  ``encode()`` or ``encode_batch()`` triggers the download/cache load.
  This keeps ``import defenx_nlp`` instant and avoids loading heavy weights
  in processes that only need the preprocessing utilities.

* **Thread safety** — a ``threading.Lock`` guards ``_load_model()`` so that
  concurrent first-calls from multiple threads do not double-load the model.

* **Device-agnostic output** — ``encode()`` always returns a CPU float32
  NumPy array regardless of whether the model runs on CUDA, MPS, or CPU.
  Downstream code never needs to call ``.cpu()`` or check the device.

* **Warmup helper** — ``warmup()`` runs a dummy encode during application
  startup so that CuDNN kernels are initialised before the first real user
  request (eliminates the 1–3 s cold-start spike on CUDA devices).
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from typing import List, Optional

import numpy as np

from .device import get_device
from .interfaces import BaseEncoder

logger = logging.getLogger(__name__)

# ── Suppress HuggingFace startup noise ───────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
for _log in ("sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(_log).setLevel(logging.ERROR)


class SemanticEncoder(BaseEncoder):
    """
    Thread-safe sentence encoder backed by a SentenceTransformer model.

    By default uses ``all-MiniLM-L6-v2`` (22 M params, 384-dim output,
    ~90 MB cached, ~15 ms / sentence on CUDA RTX 3050).

    Parameters
    ----------
    model_name : str
        Any model from the sentence-transformers hub.
        Default: ``"all-MiniLM-L6-v2"``.
    device : str
        ``"auto"`` (default), ``"cuda"``, ``"cpu"``, or ``"mps"``.
        Passed to :func:`defenx_nlp.device.get_device`.
    lazy : bool
        If ``True`` (default), the model loads on first encode call.
        If ``False``, the model is loaded immediately in ``__init__``.

    Examples
    --------
    Basic usage::

        from defenx_nlp import SemanticEncoder
        enc = SemanticEncoder()
        emb = enc.encode("Neural networks are universal approximators.")
        print(emb.shape)   # (384,)

    Explicit device::

        enc = SemanticEncoder(device="cuda")

    Eager loading (e.g. in a web server startup hook)::

        enc = SemanticEncoder(lazy=False)

    Different model::

        enc = SemanticEncoder("all-mpnet-base-v2")   # 768-dim
    """

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        lazy: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device     = get_device(device)
        self._model      = None          # loaded lazily
        self._lock       = threading.Lock()
        self._dim: Optional[int] = None  # resolved on first encode

        if not lazy:
            self._load_model()

    # ── private ───────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the SentenceTransformer model. Idempotent — safe to call N times."""
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        logger.debug("Loading SentenceTransformer '%s' on %s", self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name, device=str(self._device))

        # Resolve embedding dimension from a dummy forward pass
        with torch.inference_mode():
            dummy = self._model.encode("probe", convert_to_numpy=True, show_progress_bar=False)
        self._dim = int(dummy.shape[-1])
        logger.debug("Model loaded — embedding_dim=%d", self._dim)

    def _ensure_loaded(self) -> None:
        """Thread-safe model-load gate."""
        if self._model is None:
            with self._lock:
                self._load_model()   # double-checked inside lock

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single sentence.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray of shape (embedding_dim,), dtype float32 (always on CPU)

        Examples
        --------
        >>> enc = SemanticEncoder()
        >>> enc.encode("hello world").shape
        (384,)
        """
        import torch
        self._ensure_loaded()
        with torch.inference_mode():
            emb = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        return emb.astype(np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of sentences in one batched forward pass.

        Significantly faster than calling ``encode()`` in a loop for N > 5.

        Parameters
        ----------
        texts         : list of str
        batch_size    : int  — sentences per GPU batch (default 32)
        show_progress : bool — display tqdm progress bar (default False)

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), dtype float32

        Examples
        --------
        >>> enc = SemanticEncoder()
        >>> embs = enc.encode_batch(["hello", "world"])
        >>> embs.shape
        (2, 384)
        """
        import torch
        self._ensure_loaded()
        with torch.inference_mode():
            embs = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=False,
            )
        return embs.astype(np.float32)

    def warmup(self) -> None:
        """
        Initialise CUDA kernels by running a dummy encode.

        Call once during application startup on CUDA devices to avoid the
        first-inference cold-start latency spike (typically 1–3 seconds).

        Safe to call on CPU devices — it just runs a normal encode.

        Examples
        --------
        >>> enc = SemanticEncoder(lazy=False)
        >>> enc.warmup()  # now the first real encode() will be fast
        """
        self._ensure_loaded()
        _ = self.encode("warmup")
        logger.debug("Warmup complete on %s", self._device)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """
        Dimensionality of the embedding vectors produced by this encoder.

        Triggers a model load on first access if ``lazy=True``.
        """
        if self._dim is None:
            self._ensure_loaded()
        return self._dim  # type: ignore[return-value]

    @property
    def device(self):
        """The torch.device the model is running on."""
        return self._device

    @property
    def model_name(self) -> str:
        """Name of the underlying SentenceTransformer model."""
        return self._model_name

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "lazy (not yet loaded)"
        return (
            f"SemanticEncoder(model='{self._model_name}', "
            f"device={self._device}, status={status})"
        )
