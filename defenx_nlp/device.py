"""
defenx_nlp.device — CUDA / CPU device resolution.

Centralises all hardware-detection logic so the rest of the package
never imports torch conditionally or repeats availability checks.
"""

from __future__ import annotations

from typing import Dict, Union

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """
    Resolve a compute device from a human-readable preference string.

    Parameters
    ----------
    preferred : str
        ``"auto"``   — use CUDA if available, else CPU  (default)
        ``"cuda"``   — require CUDA; raises RuntimeError if absent
        ``"cpu"``    — force CPU regardless of GPU presence
        ``"mps"``    — Apple Silicon MPS (falls back to CPU if unavailable)
        Any string accepted by ``torch.device()`` is also valid.

    Returns
    -------
    torch.device

    Raises
    ------
    RuntimeError
        When ``preferred="cuda"`` but CUDA is not available.

    Examples
    --------
    >>> device = get_device()          # auto
    >>> device = get_device("cuda")    # force GPU
    >>> device = get_device("cpu")     # force CPU
    """
    p = preferred.lower().strip()

    if p == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if p == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch reports no CUDA-capable GPU. "
                "Install a CUDA-enabled torch build or use preferred='auto'."
            )
        return torch.device("cuda")

    if p == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Pass-through for "cpu", "cuda:0", "cuda:1", etc.
    return torch.device(preferred)


def device_info() -> Dict[str, Union[str, int, float, bool]]:
    """
    Return a diagnostic dictionary about the available compute hardware.

    Useful for logging, debugging, and environment verification.

    Returns
    -------
    dict with keys:
        cuda_available  : bool
        device_count    : int
        active_device   : str   — resolved "auto" device name
        device_name     : str   — GPU name (if CUDA), else "N/A"
        vram_gb         : float — total VRAM in GiB (if CUDA), else 0.0
        torch_version   : str
        cuda_version    : str   — CUDA runtime version (if available), else "N/A"

    Examples
    --------
    >>> from defenx_nlp import device_info
    >>> info = device_info()
    >>> print(info["device_name"])
    """
    cuda_ok = torch.cuda.is_available()
    info: Dict[str, Union[str, int, float, bool]] = {
        "cuda_available": cuda_ok,
        "device_count":   torch.cuda.device_count() if cuda_ok else 0,
        "active_device":  str(get_device("auto")),
        "device_name":    torch.cuda.get_device_name(0) if cuda_ok else "N/A",
        "vram_gb":        (
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if cuda_ok else 0.0
        ),
        "torch_version":  torch.__version__,
        "cuda_version":   torch.version.cuda if cuda_ok else "N/A",
    }
    return info
