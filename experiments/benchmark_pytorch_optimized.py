"""Compare PyTorch 2.x inference execution paths on identical weights."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.benchmark_prefill_decode import time_fn
from experiments.benchmark_utils import environment_metadata, summarize, write_json
from pytorch.inference_engine import InferenceEngine
from pytorch.llama_block import GPT


def build_model(args, attention_backend, state_dict, model_dtype=None):
    dtype = {
        None: torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[model_dtype]
    model = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_model * 2,
        max_seq_len=args.max_seq_len,
        attention_backend=attention_backend,
    ).to(device=args.device, dtype=dtype)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def benchmark_variant(args, name, attention_backend, cache_backend,
                      state_dict, prompt, decode_token, amp_dtype=None,
                      compile_mode=None, model_dtype=None):
    model = build_model(
        args, attention_backend, state_dict, model_dtype=model_dtype
    )
    engine = InferenceEngine(
        model,
        cache_backend=cache_backend,
        block_size=args.block_size,
        amp_dtype=amp_dtype,
        compile_mode=compile_mode,
    )

    def prefill():
        engine.prefill(prompt)

    def prepare_decode():
        engine.prefill(prompt)

    def decode():
        engine.decode_step(decode_token)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prefill_stats = summarize(time_fn(
        prefill, args.warmup, args.reps, args.device
    ))
    decode_stats = summarize(time_fn(
        decode, args.warmup, args.reps, args.device, setup=prepare_decode
    ))
    return {
        "name": name,
        "attention_backend": attention_backend,
        "cache_backend": cache_backend,
        "amp_dtype": amp_dtype,
        "model_dtype": model_dtype,
        "compile_mode": compile_mode,
        "prefill": prefill_stats,
        "decode": decode_stats,
        "cache_used_bytes": engine.cache_used_bytes,
        "cache_allocated_bytes": engine.cache_bytes,
        "peak_device_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch eager/SDPA/Static Cache/AMP/compile 对照"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--decode_steps", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--matmul_precision", choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="额外运行 SDPA + Static Cache 编译变体",
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
    args.max_seq_len = args.prompt_len + args.decode_steps + 8
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)

    reference = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ff=args.d_model * 2,
        max_seq_len=args.max_seq_len,
        attention_backend="eager",
    ).to(args.device)
    state_dict = reference.state_dict()
    prompt = torch.randint(
        0, args.vocab_size, (1, args.prompt_len), device=args.device
    )
    decode_token = torch.randint(
        0, args.vocab_size, (1, 1), device=args.device
    )

    variants = [
        ("eager+contiguous", "eager", "contiguous", None, None, None),
        ("sdpa+contiguous", "sdpa", "contiguous", None, None, None),
        ("sdpa+static", "sdpa", "static", None, None, None),
    ]
    if args.amp_dtype:
        variants.append((
            f"sdpa+static+{args.amp_dtype}",
            "sdpa", "static", args.amp_dtype, None, None,
        ))
        variants.append((
            f"sdpa+static+weights({args.amp_dtype})",
            "sdpa", "static", None, None, args.amp_dtype,
        ))
    if args.compile_mode:
        variants.append((
            f"sdpa+static+compile({args.compile_mode})",
            "sdpa", "static", None, args.compile_mode, args.amp_dtype,
        ))

    results = []
    for (name, attention_backend, cache_backend, amp_dtype,
         compile_mode, model_dtype) in variants:
        try:
            result = benchmark_variant(
                args, name, attention_backend, cache_backend,
                state_dict, prompt, decode_token,
                amp_dtype=amp_dtype, compile_mode=compile_mode,
                model_dtype=model_dtype,
            )
        except RuntimeError as exc:
            if compile_mode is None:
                raise
            result = {
                "name": name,
                "attention_backend": attention_backend,
                "cache_backend": cache_backend,
                "amp_dtype": amp_dtype,
                "model_dtype": model_dtype,
                "compile_mode": compile_mode,
                "error": str(exc),
            }
        results.append(result)

    baseline_decode = results[0]["decode"]["p50_ms"]
    print("=" * 92)
    print("PyTorch 2.x 推理优化矩阵")
    print(
        f"device={args.device} prompt={args.prompt_len} d_model={args.d_model} "
        f"layers={args.num_layers} Q/KV heads={args.num_heads}/{args.num_kv_heads}"
    )
    print("=" * 92)
    print(
        f"{'执行路径':38s} {'Prefill P50':>12s} {'Decode P50':>12s} "
        f"{'Decode vs eager':>16s} {'Cache used/alloc':>18s}"
    )
    print("-" * 100)
    for result in results:
        if "error" in result:
            print(f"{result['name']:38s} unavailable: {result['error']}")
            continue
        speedup = baseline_decode / result["decode"]["p50_ms"]
        result["decode_speedup_vs_eager"] = speedup
        used = result["cache_used_bytes"] / 1024
        allocated = result["cache_allocated_bytes"] / 1024
        print(
            f"{result['name']:38s} {result['prefill']['p50_ms']:>10.2f}ms "
            f"{result['decode']['p50_ms']:>10.2f}ms {speedup:>14.2f}x "
            f"{used:>7.1f}/{allocated:<7.1f}KB"
        )

    sdpa_status = {
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "flash_compiled": torch.backends.cuda.is_flash_attention_available(),
        "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_enabled": torch.backends.cuda.math_sdp_enabled(),
    }
    print(f"\nSDPA backend flags: {sdpa_status}")
    if args.device != "cuda":
        print("当前为 CPU 路径；FlashAttention kernel 需要在 CUDA GPU 上验证。")

    write_json(args.json, {
        "benchmark": "pytorch_optimized",
        "environment": environment_metadata(args.device),
        "config": vars(args),
        "sdpa_backend_flags": sdpa_status,
        "results": results,
    })


if __name__ == "__main__":
    main()
