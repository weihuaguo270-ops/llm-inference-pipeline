"""
PyTorch 版 Encoder Block — 与 NumPy 版对应

双向 Self-Attention（无因果掩码）+ FFN。
"""
import torch
import torch.nn as nn
from utils import layer_norm
from multi_head_attention import MultiHeadAttention


class FFN(nn.Module):
    """Position-wise feed-forward sublayer for the encoder example."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True)
        self.W2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        """Preserve batch and sequence axes while projecting hidden states."""
        return self.W2(torch.relu(self.W1(x)))


class EncoderBlock(nn.Module):
    """
    单层 Encoder Block — 双向 Attention，无因果掩码
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        """Apply bidirectional self-attention and FFN residual sublayers."""
        # 子层 1: 双向 Self-Attention（use_mask=False）
        attn_out = self.attention(x, use_mask=False)
        x = x + attn_out
        x = layer_norm(x)

        # 子层 2: FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)

        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model, num_heads, d_ff = 8, 2, 16
    X = torch.randn(4, d_model)

    encoder = EncoderBlock(d_model, num_heads, d_ff)
    output = encoder(X)
    print(f"Encoder 输出形状: {tuple(output.shape)}")
