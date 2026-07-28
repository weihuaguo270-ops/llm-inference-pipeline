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
from experiments.benchmark_utils import environment_metadata, summarize, write_json


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, warmup, reps, device, setup=None):
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    _sync(device)
    samples = []
    for _ in range(reps):
        if setup is not None:
            setup()
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache_backend",
        choices=["static", "contiguous", "paged"],
        default="static",
    )
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument(
        "--attention_backend", choices=["sdpa", "eager"], default="sdpa"
    )
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--model_dtype", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--matmul_precision", choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--json", help="可选的 JSON 结果输出路径")
    args = parser.parse_args()

    if args.compile_mode:
        from pytorch.windows_toolchain import reexec_with_utf8_for_compile

        return_code = reexec_with_utf8_for_compile()
        if return_code is not None:
            raise SystemExit(return_code)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    device = args.device
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)

    model_dtype = {
        None: torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.model_dtype]
    model = GPT(
        vocab_size=1024,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_model * 2,
        max_seq_len=max(args.prompt_len + args.decode_steps + 8, 256),
        attention_backend=args.attention_backend,
    ).to(device=device, dtype=model_dtype)
    engine = InferenceEngine(
        model,
        cache_backend=args.cache_backend,
        block_size=args.block_size,
        amp_dtype=args.amp_dtype,
        compile_mode=args.compile_mode,
    )

    prompt = torch.randint(0, 1024, (1, args.prompt_len), device=device)

    def run_prefill():
        engine.prefill(prompt)

    decode_token = torch.randint(0, 1024, (1, 1), device=device)

    def prepare_decode_step():
        engine.reset()
        engine.prefill(prompt)

    def run_decode_step():
        engine.decode_step(decode_token)

    def run_naive_one_step():
        _ = model(prompt)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    prefill_stats = summarize(time_fn(run_prefill, args.warmup, args.reps, device))
    decode_stats = summarize(time_fn(
        run_decode_step,
        args.warmup,
        args.reps,
        device,
        setup=prepare_decode_step,
    ))
    naive_stats = summarize(time_fn(
        run_naive_one_step,
        max(1, args.warmup // 2),
        max(3, args.reps // 4),
        device,
    ))

    prefill_ms = prefill_stats["mean_ms"]
    decode_ms = decode_stats["mean_ms"]
    naive_ms = naive_stats["mean_ms"]

    # Prefill already produces the logits used to sample the first output token.
    ttft_ms = prefill_ms
    tpot_cached_ms = decode_ms
    tpot_naive_ms = naive_ms

    print("=" * 62)
    print("Prefill / Decode 分离基准")
    print(f"  设备: {device}  prompt={args.prompt_len}  decode_steps={args.decode_steps}")
    print(f"  d_model={args.d_model}  layers={args.num_layers}  GQA kv_heads={args.num_kv_heads}")
    print("=" * 62)
    print(
        f"  Attention: {args.attention_backend}  Cache: {args.cache_backend}  "
        f"weights: {args.model_dtype or 'float32'}  AMP: {args.amp_dtype or 'off'}  "
        f"compile: {args.compile_mode or 'off'}"
    )
    print(f"\n{'阶段':<22} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10}")
    print("-" * 62)
    print(f"{'Prefill':<22} {prefill_ms:>10.2f} {prefill_stats['p50_ms']:>10.2f} {prefill_stats['p95_ms']:>10.2f}")
    print(f"{'Decode (KV Cache)':<22} {decode_ms:>10.2f} {decode_stats['p50_ms']:>10.2f} {decode_stats['p95_ms']:>10.2f}")
    print(f"{'Decode (无 Cache)':<22} {tpot_naive_ms:>10.2f} {naive_stats['p50_ms']:>10.2f} {naive_stats['p95_ms']:>10.2f}")
    print("-" * 62)
    print(f"{'TTFT (Prefill)':<22} {ttft_ms:>10.2f}   不含采样开销")
    print(f"{'TPOT (Cache)':<22} {tpot_cached_ms:>10.2f}   tokens/s ≈ {1000/tpot_cached_ms:.0f}")
    print(f"{'TPOT (无 Cache)':<22} {tpot_naive_ms:>10.2f}   tokens/s ≈ {1000/tpot_naive_ms:.0f}")
    if tpot_naive_ms > 0:
        print(f"\nKV Cache Decode 加速比 vs 无 Cache: {tpot_naive_ms / tpot_cached_ms:.2f}x")
    print(
        f"KV Cache: used={engine.cache_used_bytes / 1024:.1f} KB  "
        f"allocated={engine.cache_bytes / 1024:.1f} KB"
    )

    payload = {
        "benchmark": "prefill_decode",
        "environment": environment_metadata(device),
        "config": vars(args),
        "metrics": {
            "prefill": prefill_stats,
            "decode_cached": decode_stats,
            "decode_uncached": naive_stats,
            "ttft_ms": ttft_ms,
            "tpot_cached_ms": tpot_cached_ms,
            "cache_speedup": tpot_naive_ms / tpot_cached_ms,
            "cache_bytes": engine.cache_bytes,
            "cache_used_bytes": engine.cache_used_bytes,
            "peak_device_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
        },
    }
    write_json(args.json, payload)


if __name__ == "__main__":
    main()
