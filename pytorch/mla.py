"""
Multi-head Latent Attention (MLA) — PyTorch 版

与 modern_llm/mla.py 逻辑对齐：解压路径 + 吸收矩阵推理路径。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .positional_encoding import precompute_rope
except ImportError:
    from positional_encoding import precompute_rope


def _apply_rope(x, cos, sin, positions):
    """x: (B, H, S, d), cos/sin: (max_seq, d//2), positions: (S,)"""
    cos_p = cos[positions].unsqueeze(0).unsqueeze(0)
    sin_p = sin[positions].unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos_p - x_odd * sin_p
    out[..., 1::2] = x_even * sin_p + x_odd * cos_p
    return out


class MultiHeadLatentAttention(nn.Module):
    """Compress K/V state into a shared latent cache with a separate RoPE key."""

    def __init__(self, d_model, num_heads, d_c, d_kv_rope=32, max_seq_len=128):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_c = d_c
        self.d_kv_rope = d_kv_rope
        self.d_total = self.d_k + d_kv_rope

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.W_qr = nn.Linear(d_model, d_kv_rope, bias=False)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, d_model, bias=False)
        self.W_kr = nn.Linear(d_model, d_kv_rope, bias=False)
        self.W_uv = nn.Linear(d_c, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope(d_kv_rope, max_seq_len)
        self.register_buffer("_cos", cos)
        self.register_buffer("_sin", sin)

        self._absorbed_q = None
        self._absorbed_v = None

    def absorb_weights(self):
        """Precompute inference-only query/key and value latent projections."""
        Wq = self.Wq.weight.T.reshape(self.d_model, self.num_heads, self.d_k)
        Wuk = self.W_uk.weight.T.reshape(self.d_c, self.num_heads, self.d_k)
        Wuv = self.W_uv.weight.T.reshape(self.d_c, self.num_heads, self.d_k)

        absorbed_q, absorbed_v = [], []
        for h in range(self.num_heads):
            absorbed_q.append(Wq[:, h, :] @ Wuk[:, h, :].T)
            absorbed_v.append(Wuv[:, h, :])
        self._absorbed_q = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in absorbed_q]
        )
        self._absorbed_v = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in absorbed_v]
        )
        return self

    def _ensure_absorbed(self):
        if self._absorbed_q is None:
            self.absorb_weights()

    def forward(self, x, use_mask=True):
        """全序列前向（解压路径）。"""
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)

        q_c = self.Wq(x)
        q_r = _apply_rope(
            self.W_qr(x).unsqueeze(1), self._cos, self._sin, positions
        ).squeeze(1)

        c = self.W_dkv(x)
        k_c = self.W_uk(c).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_uv(c).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        q_heads = q_c.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k_r = _apply_rope(
            self.W_kr(x).unsqueeze(1), self._cos, self._sin, positions
        ).squeeze(1)

        scores = torch.zeros(B, self.num_heads, T, T, device=x.device)
        for h in range(self.num_heads):
            q_h = torch.cat([q_heads[:, h], q_r], dim=-1)
            k_h = torch.cat([k_c[:, h], k_r], dim=-1)
            scores[:, h] = (q_h @ k_h.transpose(-2, -1)) / (self.d_total ** 0.5)

        if use_mask:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        head_out = attn @ v
        combined = head_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(combined)

    def forward_with_cache(
        self, x_step, c_kv_cache=None, k_r_cache=None, positions=None, use_absorb=False
    ):
        """Decode one step and return output plus updated latent and RoPE caches."""
        if use_absorb:
            return self._forward_cache_absorb(x_step, c_kv_cache, k_r_cache, positions)
        return self._forward_cache_decompress(x_step, c_kv_cache, k_r_cache, positions)

    def _append_cache(self, c_kv, k_r, c_kv_cache, k_r_cache):
        if c_kv_cache is None:
            return c_kv, k_r
        return (
            torch.cat([c_kv_cache, c_kv], dim=1),
            torch.cat([k_r_cache, k_r], dim=1),
        )

    def _forward_cache_decompress(self, x_step, c_kv_cache, k_r_cache, positions):
        B = x_step.shape[0]
        if positions is None:
            pos = 0 if c_kv_cache is None else c_kv_cache.shape[1]
            positions = torch.tensor([pos], device=x_step.device)

        c_kv = self.W_dkv(x_step)
        k_r = self.W_kr(x_step)
        c_kv_cache, k_r_cache = self._append_cache(c_kv, k_r, c_kv_cache, k_r_cache)

        cache_len = c_kv_cache.shape[1]
        pos_all = torch.arange(cache_len, device=x_step.device)
        k_full = self.W_uk(c_kv_cache)
        v_full = self.W_uv(c_kv_cache)
        k_r_rot = _apply_rope(
            k_r_cache.unsqueeze(1), self._cos, self._sin, pos_all
        ).squeeze(1)

        q_c = self.Wq(x_step)
        q_r = _apply_rope(
            self.W_qr(x_step).unsqueeze(1), self._cos, self._sin, positions
        ).squeeze(1)

        q_heads = q_c.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = k_full.view(B, cache_len, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = v_full.view(B, cache_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.zeros(B, self.num_heads, 1, cache_len, device=x_step.device)
        for h in range(self.num_heads):
            q_h = torch.cat([q_heads[:, h], q_r], dim=-1)
            k_h = torch.cat([k_heads[:, h], k_r_rot], dim=-1)
            scores[:, h] = (q_h @ k_h.transpose(-2, -1)) / (self.d_total ** 0.5)

        attn = F.softmax(scores, dim=-1)
        head_out = attn @ v_heads
        combined = head_out.transpose(1, 2).contiguous().view(B, 1, -1)
        return self.Wo(combined), c_kv_cache, k_r_cache

    def _forward_cache_absorb(self, x_step, c_kv_cache, k_r_cache, positions):
        self._ensure_absorbed()
        B = x_step.shape[0]
        if positions is None:
            pos = 0 if c_kv_cache is None else c_kv_cache.shape[1]
            positions = torch.tensor([pos], device=x_step.device)

        c_kv = self.W_dkv(x_step)
        k_r = self.W_kr(x_step)
        c_kv_cache, k_r_cache = self._append_cache(c_kv, k_r, c_kv_cache, k_r_cache)

        cache_len = c_kv_cache.shape[1]
        pos_all = torch.arange(cache_len, device=x_step.device)
        k_r_rot = _apply_rope(
            k_r_cache.unsqueeze(1), self._cos, self._sin, pos_all
        ).squeeze(1)
        q_r = _apply_rope(
            self.W_qr(x_step).unsqueeze(1), self._cos, self._sin, positions
        ).squeeze(1)

        head_outputs = []
        xs = x_step.squeeze(1)
        for h in range(self.num_heads):
            q_abs = xs @ self._absorbed_q[h]
            score_c = q_abs.unsqueeze(1) @ c_kv_cache.transpose(-2, -1)
            score_r = q_r @ k_r_rot.transpose(-2, -1)
            scores = (score_c + score_r) / (self.d_total ** 0.5)
            attn = F.softmax(scores, dim=-1)
            latent = attn @ c_kv_cache
            head_outputs.append(latent @ self._absorbed_v[h])

        combined = torch.cat(head_outputs, dim=-1)
        return self.Wo(combined), c_kv_cache, k_r_cache
