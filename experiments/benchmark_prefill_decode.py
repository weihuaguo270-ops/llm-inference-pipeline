"""
Prefill / Decode 分离基准 — TTFT 与 TPOT

测量 GPT 推理链路上 Prompt 阶段（Prefill）与逐 token 生成（Decode）的延迟差异，
对比无 Cache 自回归 vs KV Cache Decode。

用法:
    python -m experiments.benchmark_prefill_decode
    python -m experiments.benchmark_prefill_decode --device cuda --prompt_len 512
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pytorch.llama_block import GPT
from pytorch.inference_engine import InferenceEngine


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, warmup, reps, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps * 1000


def main():
    parser = argparse.ArgumentParser(description="Prefill/Decode 分离基准")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--decode_steps", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    device = args.device
    torch.manual_seed(42)

    model = GPT(
        vocab_size=1024,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_model * 2,
        max_seq_len=max(args.prompt_len + args.decode_steps + 8, 256),
    ).to(device)
    engine = InferenceEngine(model)

    prompt = torch.randint(0, 1024, (1, args.prompt_len), device=device)

    def run_prefill():
        engine.prefill(prompt)

    def run_decode_step():
        engine.reset()
        engine.prefill(prompt)
        tok = torch.randint(0, 1024, (1, 1), device=device)
        engine.decode_step(tok)

    def run_naive_one_step():
        seq = prompt
        for _ in range(args.decode_steps):
            cond = seq[:, -model.layers[0].self_attn._cos.shape[0] + 1:]
            _ = model(cond)

    prefill_ms = time_fn(run_prefill, args.warmup, args.reps, device)
    decode_ms = time_fn(run_decode_step, args.warmup, args.reps, device)
    naive_ms = time_fn(run_naive_one_step, max(1, args.warmup // 2), max(3, args.reps // 4), device)

    ttft_ms = prefill_ms + decode_ms
    tpot_cached_ms = decode_ms
    tpot_naive_ms = naive_ms / args.decode_steps

    print("=" * 62)
    print("Prefill / Decode 分离基准")
    print(f"  设备: {device}  prompt={args.prompt_len}  decode_steps={args.decode_steps}")
    print(f"  d_model={args.d_model}  layers={args.num_layers}  GQA kv_heads={args.num_kv_heads}")
    print("=" * 62)
    print(f"\n{'阶段':<22} {'延迟 (ms)':>12} {'说明':>24}")
    print("-" * 62)
    print(f"{'Prefill':<22} {prefill_ms:>10.2f}   批量处理 prompt")
    print(f"{'Decode (KV Cache)':<22} {decode_ms:>10.2f}   单步 + Cache")
    print(f"{'Decode (无 Cache)':<22} {tpot_naive_ms:>10.2f}   每步重算全序列")
    print("-" * 62)
    print(f"{'TTFT (Prefill+首token)':<22} {ttft_ms:>10.2f}")
    print(f"{'TPOT (Cache)':<22} {tpot_cached_ms:>10.2f}   tokens/s ≈ {1000/tpot_cached_ms:.0f}")
    print(f"{'TPOT (无 Cache)':<22} {tpot_naive_ms:>10.2f}   tokens/s ≈ {1000/tpot_naive_ms:.0f}")
    if tpot_naive_ms > 0:
        print(f"\nKV Cache Decode 加速比 vs 无 Cache: {tpot_naive_ms / tpot_cached_ms:.2f}x")


if __name__ == "__main__":
    main()
