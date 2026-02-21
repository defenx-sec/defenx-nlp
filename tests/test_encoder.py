"""
tests/test_encoder.py — Unit tests for defenx-nlp.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest


# ── SemanticEncoder ───────────────────────────────────────────────────────────

class TestSemanticEncoder:
    """Tests for the primary SemanticEncoder class."""

    @pytest.fixture(scope="class")
    def encoder(self):
        """Single encoder instance shared across all tests in this class."""
        from defenx_nlp import SemanticEncoder
        return SemanticEncoder(lazy=False)

    def test_encode_returns_ndarray(self, encoder):
        emb = encoder.encode("hello world")
        assert isinstance(emb, np.ndarray)

    def test_encode_shape(self, encoder):
        emb = encoder.encode("the quick brown fox")
        assert emb.ndim == 1
        assert emb.shape == (encoder.embedding_dim,)

    def test_encode_dtype(self, encoder):
        emb = encoder.encode("dtype check")
        assert emb.dtype == np.float32

    def test_embedding_dim_property(self, encoder):
        assert encoder.embedding_dim == 384   # all-MiniLM-L6-v2

    def test_encode_batch_shape(self, encoder):
        texts = ["sentence one", "sentence two", "sentence three"]
        embs = encoder.encode_batch(texts)
        assert embs.shape == (3, encoder.embedding_dim)

    def test_encode_batch_dtype(self, encoder):
        embs = encoder.encode_batch(["a", "b"])
        assert embs.dtype == np.float32

    def test_same_text_same_embedding(self, encoder):
        e1 = encoder.encode("deterministic output")
        e2 = encoder.encode("deterministic output")
        np.testing.assert_array_almost_equal(e1, e2, decimal=5)

    def test_different_texts_different_embeddings(self, encoder):
        e1 = encoder.encode("hello there")
        e2 = encoder.encode("completely unrelated content")
        assert not np.allclose(e1, e2)

    def test_similar_texts_high_cosine(self, encoder):
        from defenx_nlp import cosine_similarity
        e1 = encoder.encode("I love machine learning")
        e2 = encoder.encode("I enjoy deep learning very much")
        sim = cosine_similarity(e1, e2)
        assert sim > 0.55, f"Expected high similarity, got {sim:.3f}"

    def test_repr(self, encoder):
        r = repr(encoder)
        assert "SemanticEncoder" in r
        assert "all-MiniLM-L6-v2" in r

    def test_empty_string(self, encoder):
        """Empty string should not raise — returns a valid embedding."""
        emb = encoder.encode("")
        assert emb.shape == (encoder.embedding_dim,)

    def test_unicode_text(self, encoder):
        emb = encoder.encode("Héllo wörld — こんにちは")
        assert emb.shape == (encoder.embedding_dim,)

    def test_long_text_no_crash(self, encoder):
        long_text = "word " * 500   # well beyond token limits
        emb = encoder.encode(long_text)
        assert emb.shape == (encoder.embedding_dim,)


# ── device utilities ──────────────────────────────────────────────────────────

class TestDevice:
    def test_get_device_auto(self):
        import torch
        from defenx_nlp import get_device
        d = get_device("auto")
        assert isinstance(d, torch.device)

    def test_get_device_cpu(self):
        import torch
        from defenx_nlp import get_device
        d = get_device("cpu")
        assert d == torch.device("cpu")

    def test_get_device_invalid_cuda_raises(self):
        import torch
        from defenx_nlp import get_device
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA"):
                get_device("cuda")

    def test_device_info_keys(self):
        from defenx_nlp import device_info
        info = device_info()
        for key in ("cuda_available", "device_count", "active_device",
                    "device_name", "vram_gb", "torch_version", "cuda_version"):
            assert key in info


# ── preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_clean_text_whitespace(self):
        from defenx_nlp import clean_text
        assert clean_text("  hello   world  ") == "hello world"

    def test_clean_text_lowercase(self):
        from defenx_nlp import clean_text
        assert clean_text("HELLO WORLD", lowercase=True) == "hello world"

    def test_truncate(self):
        from defenx_nlp import truncate
        result = truncate("hello world", max_chars=5)
        assert len(result) <= 6   # 5 chars + possible ellipsis

    def test_batch_clean(self):
        from defenx_nlp import batch_clean
        cleaned = batch_clean(["  A  ", "  B  "])
        assert cleaned == ["A", "B"]


# ── similarity utilities ──────────────────────────────────────────────────────

class TestUtils:
    def test_cosine_similarity_identical(self):
        from defenx_nlp import cosine_similarity
        v = np.array([1, 2, 3], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        from defenx_nlp import cosine_similarity
        a = np.array([1, 0], dtype=np.float32)
        b = np.array([0, 1], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_batch_cosine_similarity_shape(self):
        from defenx_nlp import batch_cosine_similarity
        q = np.random.randn(384).astype(np.float32)
        m = np.random.randn(10, 384).astype(np.float32)
        scores = batch_cosine_similarity(q, m)
        assert scores.shape == (10,)

    def test_top_k_similar(self):
        from defenx_nlp import top_k_similar
        corpus = [np.eye(4, dtype=np.float32)[i] for i in range(4)]
        q = corpus[2]
        results = top_k_similar(q, corpus, k=1)
        assert results[0][0] == 2
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_normalize_embedding_unit_length(self):
        from defenx_nlp.utils import normalize_embedding
        v = np.array([3, 4], dtype=np.float32)
        nv = normalize_embedding(v)
        assert abs(np.linalg.norm(nv) - 1.0) < 1e-5
