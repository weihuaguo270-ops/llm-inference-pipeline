"""
Llama Block (RMSNorm + SwiGLU + GQA + RoPE + Pre-Norm) — PyTorch 版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gqa import GroupedQueryAttention


class RMSNorm(nn.Module):
    """均方根归一化 — 公式: x / sqrt(mean(x²) + eps) * weight"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        """Normalize the final hidden dimension without centering activations."""
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    """门控 Swish FFN"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """Apply the gated SiLU feed-forward projection to batched states."""
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class LlamaDecoderBlock(nn.Module):
    """完整 Llama Decoder Block（单层）"""
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff,
                 use_rope=True, max_seq_len=128, attention_backend="sdpa"):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads,
            use_rope=use_rope, max_seq_len=max_seq_len,
            attention_backend=attention_backend,
        )
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None, positions=None, is_causal=False):
        """Apply pre-norm GQA and SwiGLU residual sublayers.

        ``x`` keeps shape ``(batch, sequence, d_model)``; mask semantics are
        delegated to the attention backend.
        """
        # Pre-Norm Attention
        residual = x
        x = self.rmsnorm1(x)
        x = self.self_attn(
            x, mask=mask, positions=positions, is_causal=is_causal
        )
        x = x + residual

        # Pre-Norm SwiGLU FFN
        residual = x
        x = self.rmsnorm2(x)
        x = self.swiglu(x)
        x = x + residual
        return x


class GPT(nn.Module):
    """
    完整 GPT 语言模型

    架构:
        Token Embedding → LlamaBlock × num_layers → RMSNorm → LM Head

    参数:
        vocab_size: 词表大小
        d_model: 模型维度
        num_layers: 层数
        num_heads: Q 头数
        num_kv_heads: K/V 头数
        d_ff: FFN 中间维度
        max_seq_len: 最大序列长度
        use_rope: 是否使用 RoPE
    """
    def __init__(self, vocab_size=1000, d_model=64, num_layers=4,
                 num_heads=4, num_kv_heads=2, d_ff=128,
                 max_seq_len=128, use_rope=True, attention_backend="sdpa"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.attention_backend = attention_backend
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LlamaDecoderBlock(d_model, num_heads, num_kv_heads, d_ff,
                              use_rope=use_rope, max_seq_len=max_seq_len,
                              attention_backend=attention_backend)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定（embedding 和 lm_head 共享权重）
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) — token ids
        返回: (batch, seq_len, vocab_size) — logits
        """
        seq_len = x.shape[1]
        x = self.token_embedding(x)

        # 因果掩码
        is_causal = mask is None

        positions = torch.arange(seq_len, device=x.device)

        for layer in self.layers:
            x = layer(
                x, mask=mask, positions=positions, is_causal=is_causal
            )

        x = self.norm(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens=20, temperature=1.0):
        """自回归生成"""
        self.eval()
        max_len = self.layers[0].self_attn._cos.shape[0]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len+1:] if idx.shape[1] >= max_len else idx
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
