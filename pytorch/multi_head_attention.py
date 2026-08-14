"""
PyTorch 版多头注意力 — 与 NumPy 版 multi_head_attention.py 对应

用 nn.Parameter 管理权重，更接近真实框架的使用方式。

位置编码支持两种模式（由 use_rope 参数控制）:
  use_rope=False (默认): Sinusoidal PE 由调用者在外部加
  use_rope=True:          在 Q/K 拆头后旋转每个 head
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn as nn
from pytorch.utils import softmax, split_heads, combine_heads


def precompute_rotary_frequencies(d_k, max_seq_len=128, base=10000.0):
    """预计算 RoPE 的 cos/sin 表（PyTorch 版）"""
    theta = base ** (-2 * torch.arange(0, d_k, 2).float() / d_k)
    pos = torch.arange(max_seq_len).float()
    angles = pos[:, None] * theta[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x, cos_table, sin_table, positions=None):
    """对 Q 或 K 应用 RoPE 旋转（PyTorch 版）"""
    seq_len = x.shape[0]
    if positions is None:
        positions = torch.arange(seq_len, device=x.device)
    cos_val = cos_table[positions].to(x.device)
    sin_val = sin_table[positions].to(x.device)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    x_even_rotated = x_even * cos_val - x_odd * sin_val
    x_odd_rotated = x_even * sin_val + x_odd * cos_val

    result = torch.empty_like(x)
    result[:, 0::2] = x_even_rotated
    result[:, 1::2] = x_odd_rotated
    return result


class MultiHeadAttention(nn.Module):
    """
    PyTorch 版多头自注意力

    与 NumPy 版 MultiHeadAttention 逻辑完全一致，但：
      1. 用 nn.Module 封装 — 方便管理参数和 forward
      2. 权重用 nn.Parameter — 可自动微分
      3. forward 接受 (seq_len, d_model) 输入
      4. 支持 use_rope 参数 — 选择是否用 RoPE
    """
    def __init__(self, d_model, num_heads, use_rope=False, max_seq_len=128):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.Wq = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wk = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wv = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wo = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        if use_rope:
            cos_t, sin_t = precompute_rotary_frequencies(self.d_k, max_seq_len)
            self.register_buffer("_cos_table", cos_t)
            self.register_buffer("_sin_table", sin_t)

    def forward(self, x, use_mask=True, positions=None):
        """Run educational self-attention on ``(sequence, d_model)`` states.

        This loop-oriented implementation exposes head splitting and causal mask
        mechanics; production paths should use the batched SDPA module.
        """
        seq_len = x.shape[0]

        # Step 1: QKV 投影 + 拆头
        Q = split_heads(x @ self.Wq, self.num_heads)
        K = split_heads(x @ self.Wk, self.num_heads)
        V = split_heads(x @ self.Wv, self.num_heads)

        # Step 2: 可选 RoPE
        if self.use_rope:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
            for h in range(self.num_heads):
                Q[h] = apply_rotary(Q[h], self._cos_table, self._sin_table, positions)
                K[h] = apply_rotary(K[h], self._cos_table, self._sin_table, positions)

        # Step 3: 所有头并行 Attention
        scores = (Q @ K.transpose(1, 2)) / (self.d_k ** 0.5)

        # Step 4: 因果掩码
        if use_mask:
            mask = torch.triu(torch.full((seq_len, seq_len), -1e9), diagonal=1)
            scores = scores + mask

        # Step 5: Softmax
        attn_weights = softmax(scores)

        # Step 6: 加权求和
        head_outputs = attn_weights @ V

        # Step 7: 合并 + 输出投影
        combined = combine_heads(head_outputs, self.num_heads)
        return combined @ self.Wo


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    d_model, num_heads = 4, 2
    X = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])

    mha = MultiHeadAttention(d_model, num_heads, use_rope=False)
    output = mha(X, use_mask=False)
    print("多头 Attention 输出 (无 RoPE):")
    print(output)

    mha_r = MultiHeadAttention(d_model, num_heads, use_rope=True)
    output_r = mha_r(X, use_mask=False)
    print("\n多头 Attention 输出 (RoPE):")
    print(output_r)
    print(f"\n配置: {num_heads} 个头, d_model={d_model}, 每头维度={d_model // num_heads}")
