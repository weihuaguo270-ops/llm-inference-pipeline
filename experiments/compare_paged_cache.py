"""
Paged KV Cache 对照 — block 利用率与连续存储对比

用法:
    python -m experiments.compare_paged_cache
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pytorch.paged_kv_cache import PagedKVCache


def main():
    torch.manual_seed(0)
    H, d_k = 4, 64
    block_size = 16
    paged = PagedKVCache(block_size=block_size, num_heads=H, d_k=d_k)

    total_tokens = 100
    for t in range(total_tokens):
        k = torch.randn(1, H, 1, d_k)
        v = torch.randn(1, H, 1, d_k)
        paged.append(k, v)

    K, V = paged.materialize()
    contiguous_bytes = total_tokens * H * d_k * 4 * 2
    paged_bytes = paged.num_blocks * block_size * H * d_k * 4 * 2

    print("=" * 58)
    print("Paged KV Cache 对照")
    print("=" * 58)
    print(f"  tokens: {total_tokens}  block_size: {block_size}")
    print(f"  blocks  allocated: {paged.num_blocks}")
    print(f"  utilization:       {paged.utilization:.1%}")
    print(f"  materialized K shape: {tuple(K.shape)}")
    print(f"  连续 Cache 体积:   {contiguous_bytes/1024:.1f} KB")
    print(f"  Paged 体积 (含碎片): {paged_bytes/1024:.1f} KB")


if __name__ == "__main__":
    main()
