"""
Grouped Query Attention (GQA) — PyTorch 版

与 np_impl/gqa.py 逻辑一致，使用 nn.Module 封装。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


_SDPA_HAS_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


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
    def __init__(self, d_model, num_heads, num_kv_heads, use_rope=False,
                 max_seq_len=128, attention_backend="sdpa"):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        if attention_backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be 'sdpa' or 'eager'")
        self.attention_backend = attention_backend

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

    def _attention(self, q, k, v, mask=None, is_causal=False):
        """Dispatch to fused SDPA kernels or the readable eager reference."""
        if self.attention_backend == "sdpa":
            if q.shape[1] == k.shape[1]:
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0,
                    is_causal=is_causal,
                )
            # On current Windows CUDA wheels, native enable_gqa may fall back
            # to the math kernel. Expanding only at read time preserves the
            # compressed cache while unlocking the fused efficient kernel.
            if q.device.type == "cuda":
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0,
                    is_causal=is_causal,
                )
            if _SDPA_HAS_GQA:
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0,
                    is_causal=is_causal, enable_gqa=True,
                )

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if is_causal:
            query_len, key_len = q.shape[-2], k.shape[-2]
            causal = torch.ones(
                query_len, key_len, dtype=torch.bool, device=q.device
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
        if mask is not None:
            scores = scores + mask
        return F.softmax(scores, dim=-1) @ v

    def forward(self, x, mask=None, positions=None, return_cache=False,
                is_causal=False, kv_cache=None):
        """
        前向传播
        x: (batch, seq, d_model)
        mask: (1, 1, seq, seq) 或 None
        """
        if kv_cache is not None:
            return self.forward_with_cache(x, positions, kv_cache)

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

        cache_k, cache_v = K, V

        out = self._attention(Q, K, V, mask=mask, is_causal=is_causal)

        # 合并
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        result = self.Wo(out)
        if return_cache:
            return result, (cache_k, cache_v)
        return result

    def forward_with_cache(self, x, positions, kv_cache=None):
        """Cached attention for one decode token or a suffix chunk."""
        B, S, _ = x.shape

        Q = self.Wq(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q, positions)
            K = self._apply_rope(K, positions)

        n_repeat = self.num_heads // self.num_kv_heads

        def expand_kv(k, v):
            if n_repeat > 1:
                k = k.repeat_interleave(n_repeat, dim=1)
                v = v.repeat_interleave(n_repeat, dim=1)
            return k, v

        cache_backend = kv_cache if hasattr(kv_cache, "append") else None
        if cache_backend is not None:
            past_len = cache_backend.length
        elif kv_cache is not None:
            past_len = kv_cache[0].shape[2]
        else:
            past_len = 0

        total_len = past_len + S
        query_positions = past_len + torch.arange(S, device=x.device)
        key_positions = torch.arange(total_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if cache_backend is not None:
            cache_backend.append(K, V)
            if getattr(cache_backend, "supports_sdpa", False):
                cached_k, cached_v = cache_backend.materialize()
                mask = None if S == 1 else torch.zeros(
                    S, total_len, device=x.device, dtype=Q.dtype
                ).masked_fill(causal_mask[0, 0], float("-inf"))
                out = self._attention(Q, cached_k, cached_v, mask=mask)
                out = out.transpose(1, 2).contiguous().view(B, S, -1)
                return self.Wo(out), cache_backend

            blocks = list(cache_backend.iter_blocks())
            score_blocks = [
                (Q @ expand_kv(k_block, v_block)[0].transpose(-2, -1))
                / (self.d_k ** 0.5)
                for k_block, v_block in blocks
            ]
            scores = torch.cat(score_blocks, dim=-1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.zeros_like(Q)
            for weights, (k_block, v_block) in zip(
                torch.split(attn, [score.shape[-1] for score in score_blocks], dim=-1),
                blocks,
            ):
                _, expanded_v = expand_kv(k_block, v_block)
                out = out + weights @ expanded_v
            out = out.transpose(1, 2).contiguous().view(B, S, -1)
            return self.Wo(out), cache_backend

        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)

        cache_result = (K, V)
        K, V = expand_kv(K, V)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.Wo(out), cache_result
