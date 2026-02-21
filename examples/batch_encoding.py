"""
examples/batch_encoding.py — High-throughput batch embedding with defenx-nlp.

Demonstrates:
  * batch_encode() vs sequential encode()
  * throughput measurement
  * similarity matrix computation
  * custom preprocessing pipeline

Run:
    python examples/batch_encoding.py
"""

import time

import numpy as np

from defenx_nlp import (
    SemanticEncoder,
    batch_clean,
    batch_cosine_similarity,
    normalize_batch,
)


CORPUS = [
    # Greetings
    "Hello, good morning!",
    "Hey there, how are you?",
    "Hi! Nice to see you.",
    # Farewells
    "Goodbye, see you later.",
    "Bye! Take care.",
    "Farewell, until next time.",
    # Questions
    "What is a neural network?",
    "How does backpropagation work?",
    "What is the purpose of dropout?",
    # Commands
    "Start the training process.",
    "Save the model weights now.",
    "Reset the learning rate.",
    # Praise
    "Excellent work, well done!",
    "That result is outstanding.",
    "Brilliant performance today.",
    # Complaints
    "This is not working properly.",
    "I am very frustrated with this.",
    "The output quality is terrible.",
]


def benchmark_sequential_vs_batch(enc: SemanticEncoder, texts: list) -> None:
    print("\n── Throughput benchmark ────────────────────────────────────────────")
    n = len(texts)

    # Sequential
    t0 = time.monotonic()
    for t in texts:
        enc.encode(t)
    seq_ms = (time.monotonic() - t0) * 1000
    print(f"  Sequential  : {n} sentences in {seq_ms:.1f} ms  "
          f"({seq_ms/n:.1f} ms/sent)")

    # Batch
    t0 = time.monotonic()
    enc.encode_batch(texts, batch_size=16)
    bat_ms = (time.monotonic() - t0) * 1000
    print(f"  Batch       : {n} sentences in {bat_ms:.1f} ms  "
          f"({bat_ms/n:.1f} ms/sent)  "
          f"[{seq_ms/bat_ms:.1f}× speedup]")


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity matrix for N embeddings."""
    normed = normalize_batch(embeddings)        # (N, D) unit vectors
    return normed @ normed.T                    # (N, N) — all pairwise sims


def print_similarity_matrix(matrix: np.ndarray, labels: list, top_n: int = 5) -> None:
    print("\n── Top-5 most similar pairs ────────────────────────────────────────")
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((matrix[i, j], i, j))
    pairs.sort(reverse=True)
    for score, i, j in pairs[:top_n]:
        print(f"  {score:.3f}  '{labels[i][:38]}'")
        print(f"         '{labels[j][:38]}'")


def main():
    print("=" * 60)
    print("  defenx-nlp — Batch Encoding Demo")
    print("=" * 60)

    # ── 1. Preprocess corpus ──────────────────────────────────────────────────
    print(f"\n[1/4] Preprocessing {len(CORPUS)} sentences …")
    cleaned = batch_clean(CORPUS, lowercase=False, max_chars=256)
    print(f"      Done — {len(cleaned)} cleaned texts")

    # ── 2. Build encoder & warmup ─────────────────────────────────────────────
    print("\n[2/4] Initialising encoder and warming up CUDA …")
    enc = SemanticEncoder(lazy=False)
    enc.warmup()
    print(f"      {enc}")
    print(f"      Embedding dim : {enc.embedding_dim}")
    print(f"      Device        : {enc.device}")

    # ── 3. Benchmark sequential vs batch ──────────────────────────────────────
    benchmark_sequential_vs_batch(enc, cleaned)

    # ── 4. Compute full pairwise similarity matrix ────────────────────────────
    print("\n[3/4] Computing pairwise similarity matrix …")
    t0 = time.monotonic()
    embeddings = enc.encode_batch(cleaned)          # (N, 384)
    ms = (time.monotonic() - t0) * 1000
    print(f"      Encoded {len(cleaned)} sentences → {embeddings.shape}  ({ms:.1f} ms)")

    sim_matrix = compute_similarity_matrix(embeddings)
    print(f"      Similarity matrix shape: {sim_matrix.shape}")
    print(f"      Diagonal check (all ≈ 1.0): {sim_matrix.diagonal().mean():.6f}")

    print_similarity_matrix(sim_matrix, cleaned, top_n=5)

    # ── 5. Query a new sentence against the corpus ────────────────────────────
    print("\n[4/4] Querying a new sentence against corpus …")
    query = "Could you assist me with this problem?"
    qe    = enc.encode(query)
    scores = batch_cosine_similarity(qe, embeddings)
    top3   = np.argsort(scores)[::-1][:3]

    print(f"\n  Query: '{query}'")
    print("  Top-3 matches:")
    for rank, idx in enumerate(top3, 1):
        print(f"    {rank}. [{scores[idx]:.3f}] '{cleaned[idx]}'")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
