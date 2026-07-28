"""
Prefix Cache 基准 — 共享前缀时的 TTFT 对比

用法:
    python -m experiments.compare_prefix_cache
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pytorch.llama_block import GPT
from pytorch.inference_engine import InferenceEngine
from pytorch.prefix_cache import PrefixKVCache


def main():
    torch.manual_seed(0)
    device = "cpu"
    model = GPT(vocab_size=512, d_model=128, num_layers=2, num_heads=4, num_kv_heads=2, d_ff=256).to(device)
    engine = InferenceEngine(model)
    pcache = PrefixKVCache(engine)

    prefix = torch.randint(0, 512, (1, 64), device=device)
    suffix_a = torch.randint(0, 512, (1, 16), device=device)
    suffix_b = torch.randint(0, 512, (1, 16), device=device)

    full_a = torch.cat([prefix, suffix_a], dim=1)
    full_b = torch.cat([prefix, suffix_b], dim=1)

    def time_prefill(ids):
        engine.reset()
        t0 = time.perf_counter()
        engine.prefill(ids)
        return (time.perf_counter() - t0) * 1000

    def time_prefix(ids):
        pcache.reset()
        t0 = time.perf_counter()
        pcache.prefill_with_prefix(ids)
        return (time.perf_counter() - t0) * 1000

    cold_a = time_prefill(full_a)
    cold_b = time_prefill(full_b)
    pcache.reset()
    warm_a = time_prefix(full_a)
    hit_b = time_prefix(full_b)

    print("=" * 58)
    print("Prefix Cache — 共享前缀 TTFT 对比")
    print("=" * 58)
    print(f"  前缀长度: {prefix.shape[1]}  后缀长度: {suffix_a.shape[1]}")
    print(f"  冷启动 A (全量 Prefill):     {cold_a:.2f} ms")
    print(f"  冷启动 B (全量 Prefill):     {cold_b:.2f} ms")
    print(f"  首次 A (建立 Prefix Cache):  {warm_a:.2f} ms")
    print(f"  命中 B (仅 suffix Prefill):  {hit_b:.2f} ms  hit={pcache.cache_hit(full_b)}")
    if hit_b > 0:
        print(f"  Prefix 命中加速比 vs 冷启动 B: {cold_b / hit_b:.2f}x")


if __name__ == "__main__":
    main()
