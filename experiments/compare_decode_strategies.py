"""
Decode 策略统一对比 — 标准 / Speculative / Lookahead / Medusa

用法:
    python -m experiments.compare_decode_strategies
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from modern_llm.speculative_decoding import SimpleLM, SpeculativeDecoder
from modern_llm.lookahead_decoding import LookaheadDecoder
from modern_llm.medusa_heads import MedusaDecoder


def run_baseline(target, prefix, n):
    calls = 0
    out = prefix.copy()
    for _ in range(n):
        tok, _ = target.generate_token(out)
        out = np.append(out, tok)
        calls += 1
    return calls


def main():
    np.random.seed(42)
    vocab = 50
    target = SimpleLM(vocab_size=vocab, d_model=32, seed=42)
    draft = SimpleLM(vocab_size=vocab, d_model=16, seed=99)
    prefix = np.array([5, 12, 3, 8])
    n = 24

    baseline = run_baseline(target, prefix, n)

    spec = SpeculativeDecoder(draft, target, gamma=4)
    spec.generate(prefix, max_new_tokens=n)
    spec_calls = spec.stats["target_calls"]

    look = LookaheadDecoder(target, ngram_size=3, gamma=4)
    look.generate(prefix, max_new_tokens=n)
    look_calls = look.stats["target_calls"]

    medusa = MedusaDecoder(target, num_heads=4, gamma=4)
    medusa.generate(prefix, max_new_tokens=n)
    medusa_calls = medusa.stats["target_calls"]

    print("=" * 62)
    print("Decode 策略对比 (target 前向次数)")
    print("=" * 62)
    print(f"  生成 token 数: {n}\n")
    print(f"{'策略':<22} {'Target前向':>12} {'加速比':>10}")
    print("-" * 46)
    for name, calls in [
        ("标准自回归", baseline),
        ("Speculative Decoding", spec_calls),
        ("Lookahead (n-gram)", look_calls),
        ("Medusa Heads", medusa_calls),
    ]:
        speedup = baseline / max(calls, 1)
        print(f"{name:<22} {calls:>12} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
