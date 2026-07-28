"""
PyTorch Spec Decoding 端到端基准 — 真实 GPT 小模型

用法:
    python -m experiments.compare_decoding_pytorch
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch.speculative_decoding import run_speculative_benchmark
from experiments.benchmark_utils import environment_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description="PyTorch Speculative Decoding 基准")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", help="可选的 JSON 结果输出路径")
    args = parser.parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA 不可用，回退到 CPU")
            args.device = "cpu"
    result = run_speculative_benchmark(
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        seed=args.seed,
    )
    print("=" * 58)
    print("PyTorch Speculative Decoding — 真实 GPT 小模型")
    print("=" * 58)
    print(f"  γ={result['gamma']}  生成 {result['max_new_tokens']} tokens")
    print(f"  Target 前向 (baseline): {result['target_calls_baseline']}")
    print(f"  Target 前向 (spec):     {result['target_calls_spec']}")
    print(f"  Target 调用次数比:      {result['target_call_ratio']:.2f}x (理论代理指标)")
    print(f"  Baseline 实测:          {result['baseline_ms']:.2f} ms")
    print(f"  Speculative 实测:       {result['speculative_ms']:.2f} ms")
    print(f"  墙钟加速比:             {result['wall_time_speedup']:.2f}x")
    print(f"  接受率:                 {result['accept_rate']:.0%}")
    write_json(args.json, {
        "benchmark": "speculative_decoding",
        "environment": environment_metadata(args.device),
        "config": vars(args),
        "metrics": result,
        "note": "draft/target 使用训练前随机权重；墙钟结果仅验证实现开销，不代表模型质量",
    })


if __name__ == "__main__":
    main()
