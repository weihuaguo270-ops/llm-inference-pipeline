"""
PyTorch Spec Decoding 端到端基准 — 真实 GPT 小模型

用法:
    python -m experiments.compare_decoding_pytorch
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch.speculative_decoding import run_speculative_benchmark


def main():
    result = run_speculative_benchmark(gamma=4, max_new_tokens=24, device="cpu")
    print("=" * 58)
    print("PyTorch Speculative Decoding — 真实 GPT 小模型")
    print("=" * 58)
    print(f"  γ={result['gamma']}  生成 {result['max_new_tokens']} tokens")
    print(f"  Target 前向 (baseline): {result['target_calls_baseline']}")
    print(f"  Target 前向 (spec):     {result['target_calls_spec']}")
    print(f"  加速比:                 {result['speedup']:.2f}x")
    print(f"  接受率:                 {result['accept_rate']:.0%}")


if __name__ == "__main__":
    main()
