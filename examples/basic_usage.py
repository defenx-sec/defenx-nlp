"""
examples/basic_usage.py — Getting started with defenx-nlp.

Run:
    python examples/basic_usage.py
"""

from defenx_nlp import (
    SemanticEncoder,
    clean_text,
    cosine_similarity,
    device_info,
    top_k_similar,
)


def main():
    # ── 1. Show hardware info ─────────────────────────────────────────────────
    print("=" * 55)
    print("  defenx-nlp — Basic Usage Demo")
    print("=" * 55)

    info = device_info()
    print(f"\n  Device      : {info['active_device']}")
    print(f"  CUDA        : {info['cuda_available']}")
    print(f"  GPU         : {info['device_name']}")
    print(f"  VRAM        : {info['vram_gb']:.1f} GiB")
    print(f"  PyTorch     : {info['torch_version']}\n")

    # ── 2. Build encoder (lazy — model loads on first encode) ─────────────────
    print("[1/4] Creating SemanticEncoder (lazy) …")
    enc = SemanticEncoder()          # model NOT loaded yet
    print(f"      {enc}")

    # ── 3. Encode a single sentence ───────────────────────────────────────────
    print("\n[2/4] Encoding a sentence …")
    sentence = "Neural networks can model complex cognitive processes."
    embedding = enc.encode(sentence)
    print(f"      Text      : '{sentence}'")
    print(f"      Shape     : {embedding.shape}")
    print(f"      Dtype     : {embedding.dtype}")
    print(f"      Norm      : {float((embedding**2).sum()**0.5):.4f}")

    # ── 4. Semantic similarity ────────────────────────────────────────────────
    print("\n[3/4] Semantic similarity …")
    pairs = [
        ("I love machine learning",     "I enjoy deep learning very much"),
        ("The cat sat on the mat",       "A feline rested on the rug"),
        ("Neural networks are powerful", "I had pasta for dinner"),
    ]
    for a, b in pairs:
        e1  = enc.encode(clean_text(a))
        e2  = enc.encode(clean_text(b))
        sim = cosine_similarity(e1, e2)
        bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
        print(f"      [{bar}] {sim:.3f}  '{a[:30]}' vs '{b[:30]}'")

    # ── 5. Top-k retrieval ────────────────────────────────────────────────────
    print("\n[4/4] Top-1 retrieval from a corpus …")
    corpus_texts = [
        "I need help with this task",        # 0 — Help-Request
        "Goodbye, see you tomorrow",         # 1 — Farewell
        "This is terrible and broken",       # 2 — Complaint
        "Yes, that is correct",              # 3 — Confirm
        "How does backpropagation work?",    # 4 — How-Question
    ]
    corpus_embeddings = [enc.encode(t) for t in corpus_texts]

    queries = [
        "Can you assist me please?",
        "I completely disagree",
        "Bye for now!",
    ]
    for q in queries:
        qe = enc.encode(q)
        results = top_k_similar(qe, corpus_embeddings, k=1)
        best_idx, score = results[0]
        print(f"      Query: '{q}'")
        print(f"        → '{corpus_texts[best_idx]}'  (score={score:.3f})\n")

    print("=" * 55)
    print("  Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()
