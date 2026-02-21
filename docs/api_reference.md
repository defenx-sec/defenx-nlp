# defenx-nlp — API Reference

## Table of Contents
1. [SemanticEncoder](#semanticencoder)
2. [BaseEncoder](#baseencoder)
3. [BaseInferenceEngine](#baseinferenceengine)
4. [Device utilities](#device-utilities)
5. [Preprocessing](#preprocessing)
6. [Similarity & Retrieval](#similarity--retrieval)

---

## SemanticEncoder

```python
from defenx_nlp import SemanticEncoder
```

Thread-safe sentence encoder backed by a SentenceTransformer model.

### Constructor

```python
SemanticEncoder(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "auto",
    lazy: bool = True,
)
```

| Parameter    | Type  | Default              | Description                                          |
|--------------|-------|----------------------|------------------------------------------------------|
| `model_name` | `str` | `"all-MiniLM-L6-v2"`| Any HuggingFace sentence-transformers model name.    |
| `device`     | `str` | `"auto"`             | `"auto"`, `"cuda"`, `"cpu"`, or `"mps"`.            |
| `lazy`       | `bool`| `True`               | Load model on first encode call (False = immediate). |

### Methods

#### `encode(text) → np.ndarray`

Encode a single string.

```python
emb = enc.encode("Hello world")
# shape: (384,), dtype: float32
```

| Parameter | Type  | Description     |
|-----------|-------|-----------------|
| `text`    | `str` | Input sentence. |

**Returns:** `np.ndarray` of shape `(embedding_dim,)`, dtype `float32`.

---

#### `encode_batch(texts, batch_size=32, show_progress=False) → np.ndarray`

Encode a list of strings in one batched forward pass.

```python
embs = enc.encode_batch(["Hello", "World"], batch_size=16)
# shape: (2, 384), dtype: float32
```

| Parameter       | Type         | Default | Description                              |
|-----------------|--------------|---------|------------------------------------------|
| `texts`         | `list[str]`  | —       | Input sentences.                         |
| `batch_size`    | `int`        | `32`    | Sentences per GPU forward pass.          |
| `show_progress` | `bool`       | `False` | Display tqdm progress bar.               |

**Returns:** `np.ndarray` of shape `(N, embedding_dim)`, dtype `float32`.

---

#### `warmup() → None`

Initialise CUDA kernels with a dummy encode. Call once at startup to avoid the first-inference cold-start spike.

---

#### `similarity(a, b) → float`

Cosine similarity between two embeddings. Inherited from `BaseEncoder`.

---

### Properties

| Property        | Type           | Description                            |
|-----------------|----------------|----------------------------------------|
| `embedding_dim` | `int`          | Output vector dimension (e.g. 384).    |
| `device`        | `torch.device` | Hardware the model runs on.            |
| `model_name`    | `str`          | Name of the underlying model.          |

---

## BaseEncoder

```python
from defenx_nlp import BaseEncoder
```

Abstract base class. Subclass to create custom encoder backends.

```python
class MyEncoder(BaseEncoder):
    def encode(self, text: str) -> np.ndarray: ...
    def encode_batch(self, texts: list) -> np.ndarray: ...

    @property
    def embedding_dim(self) -> int: return 768

    @property
    def device(self) -> torch.device: return torch.device("cpu")
```

---

## BaseInferenceEngine

```python
from defenx_nlp import BaseInferenceEngine
```

Abstract base for any model that consumes pre-computed embeddings.

```python
class MyClassifier(BaseInferenceEngine):
    def infer(self, embedding: np.ndarray) -> np.ndarray: ...
    def infer_batch(self, embeddings: np.ndarray) -> np.ndarray: ...
```

---

## Device utilities

### `get_device(preferred="auto") → torch.device`

```python
from defenx_nlp import get_device

device = get_device()          # auto — CUDA if available, else CPU
device = get_device("cuda")    # require CUDA (raises if absent)
device = get_device("cpu")     # force CPU
device = get_device("mps")     # Apple Silicon (fallback to CPU)
```

---

### `device_info() → dict`

```python
from defenx_nlp import device_info

info = device_info()
# {
#   "cuda_available": True,
#   "device_count":   1,
#   "active_device":  "cuda",
#   "device_name":    "NVIDIA GeForce RTX 3050",
#   "vram_gb":        6.0,
#   "torch_version":  "2.3.0+cu128",
#   "cuda_version":   "12.8",
# }
```

---

## Preprocessing

```python
from defenx_nlp import clean_text, batch_clean, truncate
```

### `clean_text(text, **options) → str`

```python
clean_text("  HELLO  WORLD  ", lowercase=True)
# → "hello world"

clean_text("Email me at x@y.com", remove_emails_flag=True)
# → "Email me at "
```

| Parameter            | Default | Description                    |
|----------------------|---------|--------------------------------|
| `lowercase`          | `False` | Convert to lower case.         |
| `remove_urls_flag`   | `False` | Strip HTTP/HTTPS/FTP URLs.     |
| `remove_emails_flag` | `False` | Strip e-mail addresses.        |
| `remove_special`     | `False` | Strip non-alphanumeric chars.  |
| `max_chars`          | `None`  | Hard truncate at N characters. |

---

### `batch_clean(texts, **options) → list[str]`

Apply `clean_text` to every element of a list.

```python
batch_clean(["  A  ", "  B  "])
# → ["A", "B"]
```

---

### `truncate(text, max_chars=512, ellipsis=True) → str`

```python
truncate("hello world", max_chars=5)
# → "hello…"
```

---

## Similarity & Retrieval

```python
from defenx_nlp import (
    cosine_similarity,
    batch_cosine_similarity,
    top_k_similar,
    normalize_embedding,
    normalize_batch,
)
```

### `cosine_similarity(a, b) → float`

Cosine similarity between two 1-D arrays. Returns value in `[-1, 1]`.

```python
sim = cosine_similarity(enc.encode("hello"), enc.encode("hi"))
```

---

### `batch_cosine_similarity(query, matrix) → np.ndarray`

Vectorised cosine similarity: query `(D,)` vs every row of `matrix (N, D)`.

```python
scores = batch_cosine_similarity(qemb, corpus_embs)  # (N,)
```

---

### `top_k_similar(query, corpus, k=5) → list[tuple[int, float]]`

```python
results = top_k_similar(qemb, [e1, e2, e3], k=2)
# → [(2, 0.91), (0, 0.73)]
```

---

### `normalize_embedding(emb) → np.ndarray`

L2-normalise a single embedding to unit length.

---

### `normalize_batch(matrix) → np.ndarray`

Row-wise L2-normalisation of shape `(N, D)` matrix.
