"""
Grouped Query Attention (GQA) — PyTorch 版

与 np_impl/gqa.py 逻辑一致，使用 nn.Module 封装。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 — PyTorch 版

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        num_kv_heads: K/V 头数
        use_rope: 是否使用 RoPE
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model, num_heads, num_kv_heads, use_rope=False, max_seq_len=128):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, self.d_k * num_kv_heads, bias=False)
        self.Wv = nn.Linear(d_model, self.d_k * num_kv_heads, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        if use_rope:
            from .positional_encoding import precompute_rope
            cos, sin = precompute_rope(self.d_k, max_seq_len)
            self.register_buffer("_cos", cos)
            self.register_buffer("_sin", sin)

    def _apply_rope(self, x, positions=None):
        """对 x 应用 RoPE 旋转"""
        seq_len = x.shape[2]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        cos = self._cos[positions].unsqueeze(0).unsqueeze(0)  # (1,1,S,d/2)
        sin = self._sin[positions].unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def forward(self, x, mask=None, positions=None, return_cache=False):
        """
        前向传播
        x: (batch, seq, d_model)
        mask: (1, 1, seq, seq) 或 None
        """
        B, S, _ = x.shape

        Q = self.Wq(x)  # (B, S, d_model)
        K = self.Wk(x)  # (B, S, d_kv)
        V = self.Wv(x)  # (B, S, d_kv)

        # 拆头
        Q = Q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)

        # RoPE
        if self.use_rope:
            Q = self._apply_rope(Q, positions)
            K = self._apply_rope(K, positions)

        # K/V 重复
        n_repeat = self.num_heads // self.num_kv_heads
        if n_repeat > 1:
            K = K.repeat_interleave(n_repeat, dim=1)
            V = V.repeat_interleave(n_repeat, dim=1)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, H, S, d_k)

        # 合并
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        result = self.Wo(out)
        if return_cache:
            return result, (K, V)
        return result

    def forward_with_cache(self, x, positions, kv_cache=None):
        """单步 Decode：x (B,1,d)，positions 为当前位置索引，返回 (out, new_cache)。"""
        B, S, _ = x.shape
        assert S == 1

        Q = self.Wq(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q, positions)
            K = self._apply_rope(K, positions)

        n_repeat = self.num_heads // self.num_kv_heads
        if n_repeat > 1:
            K = K.repeat_interleave(n_repeat, dim=1)
            V = V.repeat_interleave(n_repeat, dim=1)

        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.Wo(out), (K, V)
