"""
GPT 推理引擎 — Prefill / Decode 分离 + KV Cache

提供 TTFT / TPOT 基准所需的端到端推理循环接口。
"""
import torch
import torch.nn.functional as F


class InferenceEngine:
    """在 GPT 模型上实现 Prefill + KV Cache Decode。"""

    def __init__(self, model):
        self.model = model
        self.kv_caches = None
        self.seq_len = 0

    def reset(self):
        self.kv_caches = [None] * len(self.model.layers)
        self.seq_len = 0

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor) -> torch.Tensor:
        """Prompt 阶段：批量处理 prompt，建立 KV Cache。"""
        self.reset()
        self.model.eval()
        B, T = idx.shape
        x = self.model.token_embedding(idx)
        positions = torch.arange(T, device=idx.device)

        new_caches = []
        for layer in self.model.layers:
            x, cache = self._layer_forward(
                layer, x, positions, kv_cache=None, prefill=True
            )
            new_caches.append(cache)

        self.kv_caches = new_caches
        self.seq_len = T
        x = self.model.norm(x)
        return self.model.lm_head(x)

    @torch.no_grad()
    def decode_step(self, idx: torch.Tensor) -> torch.Tensor:
        """Decode 阶段：单 token 前向，复用 KV Cache。"""
        self.model.eval()
        B, _ = idx.shape
        assert B == 1
        x = self.model.token_embedding(idx)
        pos = torch.tensor([self.seq_len], device=idx.device)

        new_caches = []
        for i, layer in enumerate(self.model.layers):
            x, cache = self._layer_forward(
                layer, x, pos, kv_cache=self.kv_caches[i], prefill=False
            )
            new_caches.append(cache)

        self.kv_caches = new_caches
        self.seq_len += 1
        x = self.model.norm(x)
        return self.model.lm_head(x)

    @torch.no_grad()
    def generate_naive(self, idx, max_new_tokens):
        """无 Cache 自回归（对照组）。"""
        self.model.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.model.layers[0].self_attn._cos.shape[0] + 1:]
            logits = self.model(cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens):
        """带 Cache 自回归。"""
        logits = self.prefill(idx)
        next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
        out = torch.cat([idx, next_id], dim=1)

        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_id)
            next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            out = torch.cat([out, next_id], dim=1)
        return out

    def _layer_forward(self, layer, x, positions, kv_cache, prefill=False):
        residual = x
        x = layer.rmsnorm1(x)
        if prefill:
            seq_len = x.shape[1]
            if kv_cache is None and seq_len > 1:
                mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
            else:
                mask = None
            x, cache = layer.self_attn.forward(
                x, mask=mask, positions=positions, return_cache=True
            )
        else:
            x, cache = layer.self_attn.forward_with_cache(x, positions, kv_cache)
        x = x + residual

        residual = x
        x = layer.rmsnorm2(x)
        x = layer.swiglu(x)
        x = x + residual
        return x, cache
