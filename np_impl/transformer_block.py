"""
完整 Transformer Block — 纯 NumPy 实现

这是一个 Decoder layer，使用因果掩码实现自回归生成。

结构:
  输入
    ↓
  [Sinusoidal PE (可选)] ← 仅在 pos_encoding="sinusoidal" 时
    ↓
  Multi-Head Self-Attention（含因果掩码）
    ↓
  + 残差连接
    ↓
  Layer Normalization
    ↓
  Feed-Forward Network (FFN)
    ↓
  + 残差连接
    ↓
  Layer Normalization
    ↓
  输出

import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
位置编码方式通过 pos_encoding 参数控制:
  - "sinusoidal": 标准 Sinusoidal PE，在输入层加到 X 上（调用者做）
  - "rope":      RoPE，在 Attention 内部旋转 Q/K（MHA 自己做）
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np
from .utils import layer_norm
from .multi_head_attention import MultiHeadAttention


# ============================================================
# 1. Feed-Forward Network（前馈神经网络）
# ============================================================
class FFN:
    """
    每个词独立地做"深度思考"——两次线性变换 + 一次非线性激活
    形状变化:
      (seq_len, d_model) → @W1 → (seq_len, d_ff) → ReLU → (seq_len, d_ff) → @W2 → (seq_len, d_model)
    设计原理:
      - 先升维（d_model → d_ff，通常是 4 倍）：给模型更多参数空间去表达复杂模式
      - ReLU 激活：引入非线性，否则两层线性=一层线性
      - 再降维（d_ff → d_model）：恢复原维度，方便堆叠后续层
    Attention 做的是"词与词之间的交互"（横向交流），
    FFN 做的是"每个词独立地深加工"（纵向消化）。
    """
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    def forward(self, x):
        """Apply the educational ReLU feed-forward sublayer."""
        hidden = x @ self.W1 + self.b1
        hidden = np.maximum(0, hidden)
        output = hidden @ self.W2 + self.b2
        return output
# ============================================================
# 2. 完整 Transformer Block
# ============================================================
class TransformerBlock:
    """
    一个完整的 Transformer 层，包含两个子层：
      子层 1: Multi-Head Self-Attention → + 残差连接 → LayerNorm
      子层 2: Feed-Forward Network (FFN) → + 残差连接 → LayerNorm
    每个子层的结构都遵循:
      output = LayerNorm(x + sublayer(x))
                ↑ 残差     ↑ 子层处理
    残差连接（Residual Connection）:
      - 公式: output = input + sublayer(input)
      - 作用: 让梯度可以直接穿过深层网络，防止信息逐层衰减
      - 如果 sublayer 输出全 0，output 至少保留输入值
    层归一化（Layer Normalization）:
      - 公式: LN(x) = (x - mean) / (std + eps)
      - 对每个词的向量独立做标准化，让均值≈0，标准差≈1
      - 让训练更稳定，防止数值过大或过小
    位置编码:
      - pos_encoding="sinusoidal": 标准 Sinusoidal PE，在 forward 入口处加到输入
      - pos_encoding="rope":      RoPE，Attention 内部旋转 Q/K
    """
    def __init__(self, d_model, num_heads, d_ff, pos_encoding="sinusoidal", max_seq_len=128):
        """
        参数:
            d_model: 向量维度（所有子层的输入输出都是这个维度）
            num_heads: 多头注意力中的头数
            d_ff: FFN 中间层维度
            pos_encoding: 位置编码方式 ("sinusoidal" 或 "rope")
            max_seq_len: 最大序列长度（仅 RoPE 需要）
        """
        self.pos_encoding = pos_encoding
        # 子层 1: 多头自注意力
        if pos_encoding == "rope":
            self.attention = MultiHeadAttention(
                d_model, num_heads, use_rope=True, max_seq_len=max_seq_len,
            )
        else:
            self.attention = MultiHeadAttention(d_model, num_heads, use_rope=False)
        # 子层 2: 前馈网络
        self.ffn = FFN(d_model, d_ff)
    def forward(self, x, use_mask=True, positions=None):
        """
        前向传播
        参数:
            x: 输入矩阵，shape (seq_len, d_model)
            use_mask: 是否使用因果掩码（True=GPT风格, False=BERT风格）
            positions: 位置索引，仅 pos_encoding="rope" 时需要
                       默认 None = [0, 1, ..., seq_len-1]
        返回:
            shape (seq_len, d_model) 的输出
        """
        # ═══════════════════════════════════════════════════
        # 子层 0（可选）: Sinusoidal PE
        # ═══════════════════════════════════════════════════
        # pos_encoding="sinusoidal" 时，在进入 Attention 前加到输入
        # pos_encoding="rope" 时，什么也不做（MHA 内部自己旋转 Q/K）
        if self.pos_encoding == "sinusoidal":
            from .positional_encoding import sinusoidal_positional_encoding
            seq_len = x.shape[0]
            d_model = x.shape[1]
            pe = sinusoidal_positional_encoding(seq_len, d_model)
            x = x + pe
        # ═══════════════════════════════════════════════════
        # 子层 1: Multi-Head Self-Attention + 残差 + LayerNorm
        # ═══════════════════════════════════════════════════
        if self.pos_encoding == "rope":
            attn_out = self.attention.forward(x, use_mask, positions=positions)
        else:
            attn_out = self.attention.forward(x, use_mask)
        x = x + attn_out
        x = layer_norm(x)
        # ═══════════════════════════════════════════════════
        # 子层 2: FFN + 残差 + LayerNorm
        # ═══════════════════════════════════════════════════
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x
# ============================================================
# 3. 运行验证
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("完整 Transformer Block 演示")
    print("=" * 60)
    d_model = 8
    num_heads = 2
    d_ff = 16
    seq_len = 4
    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)
    # --- 验证模式 A: Sinusoidal PE ---
    print(f"\n--- 模式 A: Sinusoidal PE ---")
    print(f"配置: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"输入 shape: {X.shape}")
    block_sin = TransformerBlock(d_model, num_heads, d_ff, pos_encoding="sinusoidal")
    output_sin = block_sin.forward(X, use_mask=True)
    print(f"输出 shape: {output_sin.shape}")
    print(f"位置编码方式: Sinusoidal PE（在 Block 入口加到输入）")
    # --- 验证模式 B: RoPE ---
    print(f"\n--- 模式 B: RoPE ---")
    block_rope = TransformerBlock(d_model, num_heads, d_ff, pos_encoding="rope")
    output_rope = block_rope.forward(X, use_mask=True)
    print(f"输出 shape: {output_rope.shape}")
    print(f"位置编码方式: RoPE（在 Attention 内部旋转 Q/K）")
    # --- 堆叠验证 ---
    print(f"\n--- 堆叠 3 层验证（两种模式分别测试）---")
    x_sin = X.copy()
    x_rope = X.copy()
    for i in range(3):
        x_sin = block_sin.forward(x_sin, use_mask=True)
        x_rope = block_rope.forward(x_rope, use_mask=True)
        print(f"  第 {i+1} 层 Sinusoidal: norm={np.linalg.norm(x_sin):.3f}  RoPE: norm={np.linalg.norm(x_rope):.3f}")
    print(f"\nSinusoidal PE: 位置信息在输入层，通过 X+PE 注入")
    print(f"RoPE:          位置信息在 Attention，通过 Q/K 旋转注入")
    print(f"两种方式都可以堆叠多层，核心区别是位置信息的作用位置不同")
