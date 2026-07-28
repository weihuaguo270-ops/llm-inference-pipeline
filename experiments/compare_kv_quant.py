"""
KV Cache 量化对照 — FP32 vs INT8 存储与反量化开销

测量 Cache 体积节省与 Attention 读取时的反量化误差。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quantize_int8(x):
    scale = np.max(np.abs(x)) / 127.0 + 1e-8
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


def cache_bytes_fp32(seq_len, dim):
    return seq_len * dim * 4 * 2  # K + V


def cache_bytes_int8(seq_len, dim):
    return seq_len * dim * 1 * 2 + seq_len * 4 * 2  # K/V int8 + per-row scale


def attention_output(q, k, v, d_k):
    scores = (q @ k.T) / np.sqrt(d_k)
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v


def main():
    np.random.seed(42)
    seq_len, d_k, d_model = 512, 64, 512
    q = np.random.randn(1, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)

    out_fp32 = attention_output(q, k, v, d_k)
    k_q, k_s = quantize_int8(k)
    v_q, v_s = quantize_int8(v)
    k_dq = dequantize_int8(k_q, k_s)
    v_dq = dequantize_int8(v_q, v_s)
    out_int8 = attention_output(q, k_dq, v_dq, d_k)

    max_err = np.max(np.abs(out_fp32 - out_int8))
    fp32_b = cache_bytes_fp32(seq_len, d_model)
    int8_b = cache_bytes_int8(seq_len, d_model)

    print("=" * 58)
    print("KV Cache 量化对照 (FP32 vs INT8)")
    print("=" * 58)
    print(f"  序列长度: {seq_len}  d_model: {d_model}")
    print(f"  FP32 Cache: {fp32_b/1024:.1f} KB")
    print(f"  INT8 Cache: {int8_b/1024:.1f} KB  (节省 {(1-int8_b/fp32_b)*100:.1f}%)")
    print(f"  Attention 输出 max|Δ|: {max_err:.6f}")


if __name__ == "__main__":
    main()
