"""
Prefix KV Cache — 共享 system prompt 的 Cache 复用

Agent / RAG 场景中，多条请求共享相同前缀时，跳过重复 Prefill。
"""
import torch


class PrefixKVCache:
    """缓存前缀 token 的 KV，后缀变化时只 Prefill 增量部分。"""

    def __init__(self, engine):
        self.engine = engine
        self.prefix_ids = None

    def reset(self):
        self.prefix_ids = None
        self.engine.reset()

    @torch.no_grad()
    def prefill_with_prefix(self, full_ids: torch.Tensor) -> torch.Tensor:
        """
        full_ids: (1, total_len)
        若前缀匹配则只处理 suffix；否则全量 Prefill 并更新缓存前缀。
        """
        if self.prefix_ids is None:
            logits = self.engine.prefill(full_ids)
            self.prefix_ids = full_ids.clone()
            return logits

        plen = self.prefix_ids.shape[1]
        if plen <= full_ids.shape[1] and torch.equal(full_ids[:, :plen], self.prefix_ids):
            suffix = full_ids[:, plen:]
            if suffix.shape[1] == 0:
                return self.engine.model(self.prefix_ids)
            for i in range(suffix.shape[1]):
                pos = torch.tensor([self.engine.seq_len], device=full_ids.device)
                x = self.engine.model.token_embedding(suffix[:, i:i + 1])
                new_caches = []
                for li, layer in enumerate(self.engine.model.layers):
                    x, cache = self.engine._layer_forward(
                        layer, x, pos, self.engine.kv_caches[li], prefill=False
                    )
                    new_caches.append(cache)
                self.engine.kv_caches = new_caches
                self.engine.seq_len += 1
                x = self.engine.model.norm(x)
            return self.engine.model.lm_head(x)

        logits = self.engine.prefill(full_ids)
        self.prefix_ids = full_ids.clone()
        return logits

    def cache_hit(self, full_ids: torch.Tensor) -> bool:
        if self.prefix_ids is None:
            return False
        plen = self.prefix_ids.shape[1]
        return plen <= full_ids.shape[1] and torch.equal(full_ids[:, :plen], self.prefix_ids)
