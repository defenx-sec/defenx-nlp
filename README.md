# defenx-nlp

**Semantic NLP Intelligence Toolkit**

> A domain-agnostic library for semantic sentence encoding, embedding generation,
> GPU/CPU-aware device handling, and reusable inference interfaces.

[![PyPI version](https://img.shields.io/pypi/v/defenx-nlp)](https://pypi.org/project/defenx-nlp/)
[![Python](https://img.shields.io/pypi/pyversions/defenx-nlp)](https://pypi.org/project/defenx-nlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`defenx-nlp` is a standalone, pip-installable semantic NLP library. It is designed to be **domain-agnostic** so
the same encoder that understands human chat intent can be repurposed for:

| Use case | What you embed |
|---|---|
| NLP classification | User sentences → intent labels |
| Anomaly detection | System log lines → outlier scores |
| Log intelligence | Server events → semantic clusters |
| Behavioural analytics | User actions → behavioural patterns |
| Semantic search | Documents → retrieval ranking |

---

## Installation

### Standard (CPU)

```bash
pip install defenx-nlp
```

### With CUDA 12 (RTX 30/40 series, recommended)

```bash
pip install defenx-nlp
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Development install (editable + test tools)

```bash
git clone https://github.com/defenx-sec/defenx-nlp.git
cd defenx-nlp
pip install -e ".[dev]"
```

---

## Quick Start

```python
from defenx_nlp import SemanticEncoder

# Auto-detects CUDA — falls back to CPU silently
enc = SemanticEncoder()

# Encode a single sentence → (384,) float32 numpy array
embedding = enc.encode("Neural networks are universal approximators.")
print(embedding.shape)   # (384,)
print(embedding.dtype)   # float32

# Batch encode — much faster than looping
embeddings = enc.encode_batch(["Hello", "Goodbye", "Help me please"])
print(embeddings.shape)  # (3, 384)
```

### Semantic similarity

```python
from defenx_nlp import SemanticEncoder, cosine_similarity

enc = SemanticEncoder()
e1 = enc.encode("I love machine learning")
e2 = enc.encode("I enjoy deep learning")

sim = cosine_similarity(e1, e2)
print(f"Similarity: {sim:.3f}")   # ~0.87
```

### Top-k retrieval

```python
from defenx_nlp import SemanticEncoder, top_k_similar

enc = SemanticEncoder()
corpus = ["Help me", "Goodbye", "Great job!", "What is AI?"]
query  = "Can you assist me?"

c_embs = [enc.encode(t) for t in corpus]
q_emb  = enc.encode(query)

results = top_k_similar(q_emb, c_embs, k=1)
print(corpus[results[0][0]])   # "Help me"
```

### Text preprocessing

```python
from defenx_nlp import clean_text, batch_clean

text = clean_text("  HELLO  WORLD!  ", lowercase=True)
# → "hello world!"

texts = batch_clean(["  A  ", " B  "], lowercase=True)
# → ["a", "b"]
```

### CUDA warmup (for production services)

```python
enc = SemanticEncoder(lazy=False)
enc.warmup()   # initialise CuDNN kernels at startup, not first request
```

---

## API Summary

| Symbol | Description |
|---|---|
| `SemanticEncoder` | Main encoder class — lazy, thread-safe, CUDA-aware |
| `BaseEncoder` | Abstract base for custom encoder backends |
| `BaseInferenceEngine` | Abstract base for downstream classifiers |
| `get_device(preferred)` | Resolve `"auto"/"cuda"/"cpu"/"mps"` → `torch.device` |
| `device_info()` | Hardware diagnostic dictionary |
| `clean_text(text, **opts)` | Configurable single-text cleaner |
| `batch_clean(texts, **opts)` | Apply `clean_text` to a list |
| `truncate(text, max_chars)` | Hard-truncate with optional ellipsis |
| `cosine_similarity(a, b)` | Scalar cosine similarity in `[-1, 1]` |
| `batch_cosine_similarity(q, M)` | Vectorised query-vs-matrix similarity `(N,)` |
| `top_k_similar(q, corpus, k)` | Top-k retrieval → `[(idx, score)]` |
| `normalize_embedding(v)` | L2-normalise a single embedding |
| `normalize_batch(M)` | Row-wise L2-normalise `(N, D)` matrix |

Full API docs: [`docs/api_reference.md`](docs/api_reference.md)

---

## Hardware Requirements

### Minimum
| Component | Requirement |
|---|---|
| CPU | Dual-core, 64-bit |
| RAM | 4 GB |
| Disk | 500 MB (model cache) |
| GPU | None (CPU mode) |
| Python | 3.9+ |

### Recommended
| Component | Requirement |
|---|---|
| CPU | 6+ cores (AMD Ryzen 7 / Intel Core i7+) |
| RAM | 16 GB |
| GPU | NVIDIA RTX 20-series or newer |
| VRAM | 4+ GB |
| CUDA | 11.8 or 12.x |
| Python | 3.11+ |

> **Tested on:** AMD Ryzen 7 4800H + NVIDIA RTX 3050 6 GB (CUDA 12.8) on Kali Linux (WSL2).
> Average inference latency: **~15 ms/sentence on CUDA**, **~80 ms on CPU**.

---

## Supported Operating Systems

| OS | CPU mode | CUDA mode | Notes |
|---|---|---|---|
| **Linux** (Ubuntu 20.04+, Debian 11+, Kali) | ✅ | ✅ | Fully tested |
| **Windows 10 / 11** | ✅ | ✅ | Use WSL2 for CUDA in WSL |
| **macOS 12+** (Intel) | ✅ | — | No NVIDIA CUDA support |
| **macOS 12+** (Apple Silicon M1/M2/M3) | ✅ | MPS | Use `device="mps"` |

---

## Extending the Library

### Custom encoder backend

```python
import numpy as np
import torch
from defenx_nlp import BaseEncoder

class OpenAIEncoder(BaseEncoder):
    """Drop-in encoder using OpenAI embeddings API."""

    def __init__(self, api_key: str):
        import openai
        openai.api_key = api_key
        self._client = openai.OpenAI()

    def encode(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def encode_batch(self, texts):
        resp = self._client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    @property
    def embedding_dim(self) -> int: return 1536

    @property
    def device(self) -> torch.device: return torch.device("cpu")
```

---

## Running Tests

```bash
# Install dev extras first
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=defenx_nlp --cov-report=term-missing
```

Expected output:
```
tests/test_encoder.py::TestSemanticEncoder::test_encode_shape          PASSED
tests/test_encoder.py::TestSemanticEncoder::test_embedding_dim_property PASSED
...
13 passed in 42.3s
```

---

## Running Examples

```bash
# Basic single-sentence usage + similarity + retrieval
python examples/basic_usage.py

# Batch throughput benchmark + similarity matrix
python examples/batch_encoding.py
```

---

## Publishing to PyPI

### 1. Build the distribution

```bash
pip install build twine
python -m build
# Creates dist/defenx_nlp-0.1.0.tar.gz and dist/defenx_nlp-0.1.0-py3-none-any.whl
```

### 2. Test on TestPyPI first (always)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ defenx-nlp
```

### 3. Publish to real PyPI

```bash
twine upload dist/*
```

### 4. Verify the install

```bash
pip install defenx-nlp
python -c "from defenx_nlp import SemanticEncoder; print(SemanticEncoder())"
```

### Versioning

Update `version` in `pyproject.toml` before each release.
Follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

---

## Project Structure

```
defenx-nlp/
├── defenx_nlp/
│   ├── __init__.py        Public API surface — all exports live here
│   ├── encoder.py         SemanticEncoder — lazy, thread-safe, CUDA-aware
│   ├── device.py          get_device() and device_info() helpers
│   ├── preprocessing.py   clean_text, batch_clean, truncate, deduplicate
│   ├── interfaces.py      BaseEncoder and BaseInferenceEngine ABCs
│   └── utils.py           cosine_similarity, top_k_similar, normalize_*
│
├── tests/
│   └── test_encoder.py    pytest suite — encoder, device, preprocessing, utils
│
├── examples/
│   ├── basic_usage.py     Single-sentence encode, similarity, retrieval
│   └── batch_encoding.py  Throughput benchmark, similarity matrix
│
├── docs/
│   └── api_reference.md   Full API documentation
│
├── README.md              This file
├── pyproject.toml         PEP 621 package metadata + build config
└── LICENSE                MIT
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built on top of:
- [sentence-transformers](https://www.sbert.net/) by UKPLab
- [PyTorch](https://pytorch.org/) by Meta AI
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Microsoft
