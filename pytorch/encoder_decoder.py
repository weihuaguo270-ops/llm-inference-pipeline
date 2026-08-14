"""
PyTorch 版 Encoder-Decoder 完整架构 — 与 NumPy 版对应
"""
import torch
import torch.nn as nn
from utils import layer_norm
from multi_head_attention import MultiHeadAttention
from cross_attention import MultiHeadCrossAttention


class FFN(nn.Module):
    """Position-wise feed-forward sublayer shared by encoder and decoder."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True)
        self.W2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        """Project the final hidden dimension and preserve leading axes."""
        return self.W2(torch.relu(self.W1(x)))


class EncoderLayer(nn.Module):
    """单层 Encoder — 双向 Self-Attention + FFN"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        """Encode source states with unmasked self-attention."""
        attn_out = self.self_attn(x, use_mask=False)
        x = x + attn_out
        x = layer_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    """单层 Decoder — Self-Attention(因果掩码) + Cross-Attention + FFN"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, encoder_output):
        """Apply causal self-attention, cross-attention, then the FFN."""
        # Self-Attention（因果掩码）
        attn_out = self.self_attn(x, use_mask=True)
        x = x + attn_out
        x = layer_norm(x)

        # Cross-Attention
        cross_out = self.cross_attn(x, encoder_output)
        x = x + cross_out
        x = layer_norm(x)

        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x


class EncoderDecoder(nn.Module):
    """完整 Encoder-Decoder"""
    def __init__(self, d_model, num_heads, d_ff, num_layers=2):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, source, target):
        """
        参数:
            source: (seq_enc, d_model) — 原句子
            target: (seq_dec, d_model) — 译文（逐步生成）
        """
        # Encoder
        x = source
        for layer in self.encoder_layers:
            x = layer(x)
        encoder_output = x

        # Decoder
        x = target
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model, num_heads, d_ff = 8, 2, 16

    source = torch.randn(3, d_model)  # 原句子 3 个词
    target = torch.randn(2, d_model)  # 已生成 2 个词

    model = EncoderDecoder(d_model, num_heads, d_ff, num_layers=2)
    output = model(source, target)

    print(f"Encoder-Decoder 输出形状: {tuple(output.shape)}")
    print("Encoder 编码原句子 → Decoder 通过 Cross-Attention 逐词生成")
