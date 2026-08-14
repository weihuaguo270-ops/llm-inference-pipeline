"""
Llama 风格 Decoder Block — 现代 LLM 架构

与原始 Transformer Decoder Block 的核心差异：

                原始 Transformer (2017)    Llama 系列 (2023-2024)
  ─────────────────────────────────────────────────────────────────
  归一化位置    Post-Norm (子层后)           Pre-Norm (子层前)
  归一化类型    LayerNorm (μ+σ+β)          RMSNorm (仅σ，快30%)
  FFN 激活      ReLU                       SwiGLU (门控机制)
  Attention     MHA                        GQA (KV Cache省80%)
  位置编码      Sinusoidal PE (加法)        RoPE (旋转，可外推)
  Dropout       内置                        极少/无

结构:
  x → RMSNorm → GQA (RoPE + 因果掩码) → +残差 → RMSNorm → SwiGLU → +残差 → 输出
"""
import numpy as np


# ============================================================
# 1. RMSNorm
# ============================================================
class RMSNorm:
    """
    均方根归一化 — 公式: RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

    与 LayerNorm 的区别：
      LayerNorm: (x - mean) / std * γ + β  ← 减均值、除标准差、加偏置
      RMSNorm:    x / rms(x) * γ            ← 只有缩放，无均值偏移

    Llama 弃用 LayerNorm 的原因：
      - 均值偏移在 Transformer 中作用不大（经验结论）
      - 减均值涉及全局通信，对 GPU 不友好
      - 去掉后训练速度提升约 30%，效果持平
    """
    def __init__(self, d_model, eps=1e-6):
        self.weight = np.ones(d_model)
        self.eps = eps

    def forward(self, x):
        """Normalize the last hidden dimension while preserving input shape."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# 2. SwiGLU — 门控 FFN
# ============================================================
class SwiGLU:
    """
    门控 Swish FFN — 公式: SwiGLU(x) = W_down · (Swish(W_gate·x) ⊙ W_up·x)

    比 ReLU FFN 多一个 W_gate 矩阵做「门控」：
      ReLU:    2 个矩阵 (W_up, W_down)
      SwiGLU:  3 个矩阵 (W_gate, W_up, W_down) → 参数多 50%

    参数:
        d_model: 输入输出维度
        d_ff: 中间维度
    """
    def __init__(self, d_model, d_ff):
        self.W_gate = np.random.randn(d_model, d_ff) * 0.01
        self.W_up = np.random.randn(d_model, d_ff) * 0.01
        self.W_down = np.random.randn(d_ff, d_model) * 0.01

    @staticmethod
    def swish(x):
        """x * sigmoid(x) — 平滑的门控函数"""
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward(self, x):
        """Apply gated SiLU expansion and project back to ``d_model``."""
        gate = self.swish(x @ self.W_gate)   # 门控信号 [0~1]
        up = x @ self.W_up                    # 内容
        return (gate * up) @ self.W_down      # 选择性激活后投影回


# ============================================================
# 3. Llama Decoder Block
# ============================================================
class LlamaDecoderBlock:
    """
    完整 Llama 风格 Decoder Block（单层）

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        num_kv_heads: K/V 头数（GQA）
        d_ff: FFN 中间维度
        use_rope: 是否使用 RoPE
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff,
                 use_rope=True, max_seq_len=128):
        from .gqa import GroupedQueryAttention
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, use_rope=use_rope, max_seq_len=max_seq_len,
        )
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x, use_mask=True, positions=None):
        """Apply pre-norm GQA and SwiGLU residual sublayers to one sequence."""
        # Pre-Norm Attention
        residual = x
        x = self.rmsnorm1.forward(x)
        x = self.self_attn.forward(x, use_mask=use_mask, positions=positions)
        x = x + residual

        # Pre-Norm SwiGLU FFN
        residual = x
        x = self.rmsnorm2.forward(x)
        x = self.swiglu.forward(x)
        x = x + residual
        return x


# ============================================================
# 4. 多层堆叠
# ============================================================
class LlamaModel:
    """多层 Llama Decoder 堆叠"""
    def __init__(self, num_layers=4, d_model=8, num_heads=4, num_kv_heads=2,
                 d_ff=32, max_seq_len=128):
        self.layers = [
            LlamaDecoderBlock(d_model, num_heads, num_kv_heads, d_ff,
                              use_rope=True, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ]

    def forward(self, x, use_mask=True, positions=None):
        """Map hidden states through all decoder blocks and final RMSNorm."""
        for layer in self.layers:
            x = layer.forward(x, use_mask=use_mask, positions=positions)
        return x
