"""
Attention 变体性能基准测试 — 实际推理延迟与吞吐量

测量 MHA、GQA、MLA 在 PyTorch 下的前向推理性能。
支持 CPU（CI 可用）和 GPU（真实结果）两种模式。

用法:
    python -m experiments.benchmark_attention                     # CPU 基准
    python -m experiments.benchmark_attention --device cuda       # GPU 基准
    python -m experiments.benchmark_attention --seq_len 8192      # 长序列
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from experiments.device_policy import resolve_device
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# PyTorch Attention 模块（基准测试用）
# ──────────────────────────────────────────────

class MHALayer(nn.Module):
    """标准 Multi-Head Attention"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return self.Wo(attn.transpose(1, 2).contiguous().view(B, T, self.d_model))


class GQALayer(nn.Module):
    """Grouped Query Attention（n_kv_heads < n_heads）"""
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.head_dim = self.d_k

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.Wv = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        n_repeat = self.num_heads // self.num_kv_heads
        K = self.Wk(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        # GQA: repeat K/V heads
        K = K.repeat_interleave(n_repeat, dim=1)
        V = V.repeat_interleave(n_repeat, dim=1)

        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return self.Wo(attn.transpose(1, 2).contiguous().view(B, T, self.d_model))


class MLALayer(nn.Module):
    """Multi-head Latent Attention (DeepSeek V2) — 带 KV 压缩"""
    def __init__(self, d_model: int, num_heads: int, d_c: int, d_kv_rope: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_c = d_c

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)   # 压缩
        self.W_uk = nn.Linear(d_c, d_model, bias=False)     # 解压 K
        self.W_uv = nn.Linear(d_c, d_model, bias=False)     # 解压 V
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.Wq(x)

        # MLA: 压缩 → 解压
        c = self.W_dkv(x)                     # (B, T, d_c)
        K = self.W_uk(c).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_uv(c).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return self.Wo(attn.transpose(1, 2).contiguous().view(B, T, self.d_model))


# ──────────────────────────────────────────────
# 基准测试
# ──────────────────────────────────────────────

def benchmark_layer(layer, input_tensor, name: str, num_warmup: int = 10,
                    num_iter: int = 100, device: str = "cpu"):
    """测量单层前向推理延迟"""
    layer = layer.to(device)
    x = input_tensor.to(device)

    # Warmup
    for _ in range(num_warmup):
        _ = layer(x)

    # 测量
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iter):
        _ = layer(x)
    torch.cuda.synchronize() if device == "cuda" else None
    end = time.perf_counter()

    avg_ms = (end - start) / num_iter * 1000
    tokens_per_sec = input_tensor.size(1) / (avg_ms / 1000)

    return avg_ms, tokens_per_sec


def count_params(layer) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Attention 变体性能基准测试")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--d_c", type=int, default=128, help="MLA 压缩维度")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()
    args.device = resolve_device(
        args.device,
        cuda_available=torch.cuda.is_available(),
        require_cuda=args.require_cuda,
    )

    B, T, D = args.batch, args.seq_len, args.d_model
    H = args.num_heads
    K = args.num_kv_heads

    print("=" * 60)
    print(f"Attention 变体性能基准测试")
    print(f"  设备: {args.device}  |  Batch: {B}  |  Seq: {T}  |  d_model: {D}")
    print("=" * 60)

    x = torch.randn(B, T, D)
    layers = [
        ("MHA", MHALayer(D, H)),
        ("GQA", GQALayer(D, H, K)),
        ("MLA", MLALayer(D, H, args.d_c)),
    ]

    print(f"\n{'变体':10s} {'参数量':15s} {'延迟(ms)':15s} {'tok/s':15s} {'vs MHA':10s}")
    print("-" * 65)

    mha_latency = None
    results = []

    for name, layer in layers:
        params = count_params(layer)
        latency, tps = benchmark_layer(layer, x, name,
                                       num_warmup=args.warmup,
                                       num_iter=args.iter,
                                       device=args.device)

        if name == "MHA":
            mha_latency = latency
            ratio = "1.00x"
        else:
            ratio = f"{latency / mha_latency:.2f}x" if mha_latency else "-"

        results.append((name, params, latency, tps, ratio))
        print(f"{name:10s} {params / 1e6:>8.2f}M {latency:>10.2f}ms {tps:>10.0f} {ratio:>8s}")

    # 汇总
    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")
    for name, params, latency, tps, ratio in results:
        cache_savings = ""
        if name == "GQA":
            cache_mha = 2 * D * T * 2  # FP16 bytes
            cache_gqa = 2 * (D // H * K) * T * 2
            cache_savings = f"  (KV Cache: {cache_gqa/1024:.0f}KB vs MHA {cache_mha/1024:.0f}KB)"
        elif name == "MLA":
            cache_mha = 2 * D * T * 2
            cache_mla = (args.d_c + args.d_kv_rope if hasattr(args, 'd_kv_rope') else args.d_c) * T * 2
            cache_savings = f"  (KV Cache: ~{cache_mla/1024:.0f}KB vs MHA {cache_mha/1024:.0f}KB)"

        print(f"  {name}: {params/1e6:.1f}M params, {latency:.1f}ms/step ({ratio}){cache_savings}")


if __name__ == "__main__":
    main()
