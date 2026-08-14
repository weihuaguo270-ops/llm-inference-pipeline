"""
Encoder Block — 纯 NumPy 实现

与 Decoder Block 的区别:
  Decoder: Self-Attention(因果掩码) → Cross-Attention → FFN
  Encoder: Self-Attention(双向，无掩码) → FFN

Encoder 没有 Cross-Attention — 它只负责把输入句子编码成上下文表示。
Cross-Attention 在 Decoder 里，由 Decoder 去"看" Encoder 的输出。

import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
使用场景:
  输入:  "I love you" (3个词)
  Encoder: 每个词看到整个句子 → 上下文感知的向量
  输出:  3个向量，每个融合了整句信息
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np
from .utils import layer_norm
from .multi_head_attention import MultiHeadAttention


class FFN:
    """前馈网络 — 与 transformer_block.py 中的 FFN 完全一致"""
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    def forward(self, x):
        """Apply ReLU expansion and projection to ``(sequence, d_model)`` states."""
        hidden = x @ self.W1 + self.b1
        hidden = np.maximum(0, hidden)
        return hidden @ self.W2 + self.b2
class EncoderBlock:
    """
    单层 Encoder Block
    结构:
      输入 → Self-Attention(双向，无掩码) → +残差 → LayerNorm → FFN → +残差 → LayerNorm → 输出
    """
    def __init__(self, d_model, num_heads, d_ff):
        # 使用双向 Self-Attention（use_mask=False）
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
    def forward(self, x):
        """
        参数:
            x: 输入序列 (seq_len, d_model)
        返回:
            (seq_len, d_model) — 编码后的上下文表示
        """
        # 子层 1: 双向 Self-Attention（无因果掩码）
        attn_out = self.attention.forward(x, use_mask=False)
        x = x + attn_out
        x = layer_norm(x)
        # 子层 2: FFN
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x
# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    d_model, num_heads, d_ff = 8, 2, 16
    seq_len = 4
    X = np.random.randn(seq_len, d_model)
    print(f"配置: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"输入: {seq_len} 个词，每个 {d_model} 维")
    encoder = EncoderBlock(d_model, num_heads, d_ff)
    output = encoder.forward(X)
    print(f"输出: {output.shape[0]} 个词，每个 {output.shape[1]} 维")
    print("\nEncoder 输出已融合整句上下文（无因果掩码，每个词看到全部词）")
