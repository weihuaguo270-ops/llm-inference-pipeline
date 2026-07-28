"""Continuous batching with shared weights and per-request KV cache state."""

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch

from .cache_backends import build_cache_backend
from .inference_engine import InferenceEngine


@dataclass
class Request:
    req_id: int
    prompt: torch.Tensor
    max_new: int
    generated: int = 0
    done: bool = False
    stage: str = "prefill"


class ContinuousBatcher:
    """Batch requests with matching stage and sequence length.

    All request engines share one model. Cache state remains isolated per request
    and is merged only for a batched model forward.
    """

    def __init__(self, engine_factory):
        prototype = engine_factory() if callable(engine_factory) else engine_factory
        if not isinstance(prototype, InferenceEngine):
            raise TypeError("engine_factory must produce an InferenceEngine")
        self.model = prototype.model
        self.cache_backend = prototype.cache_backend
        self.block_size = prototype.block_size
        self.amp_dtype = prototype.amp_dtype
        self.compile_mode = prototype.compile_mode
        self.queue: List[Request] = []
        self.engines = {}
        self.stats = {
            "prefill_batches": 0,
            "decode_batches": 0,
            "model_forwards": 0,
            "tokens_out": 0,
            "max_batch_size": 0,
        }

    def _new_engine(self):
        return InferenceEngine(
            self.model,
            cache_backend=self.cache_backend,
            block_size=self.block_size,
            amp_dtype=self.amp_dtype,
            compile_mode=None,
        )

    def add_request(self, req: Request):
        if req.prompt.ndim != 2 or req.prompt.shape[0] != 1:
            raise ValueError("each request prompt must have shape (1, seq_len)")
        if req.max_new <= 0:
            req.done = True
            return
        if req.req_id in self.engines or any(r.req_id == req.req_id for r in self.queue):
            raise ValueError(f"duplicate request id: {req.req_id}")
        self.queue.append(req)

    def _group_active(self, active):
        groups = defaultdict(list)
        for req in active:
            if req.stage == "prefill":
                key = ("prefill", req.prompt.shape[1])
            else:
                key = ("decode", self.engines[req.req_id].seq_len)
            groups[key].append(req)
        return groups

    def _split_cache(self, batch_engine, requests):
        for batch_index, req in enumerate(requests):
            engine = self.engines.setdefault(req.req_id, self._new_engine())
            engine.kv_caches = []
            for cache in batch_engine.kv_caches:
                k, v = cache.materialize()
                engine.kv_caches.append(build_cache_backend(
                    engine.cache_backend,
                    k[batch_index:batch_index + 1],
                    v[batch_index:batch_index + 1],
                    block_size=engine.block_size,
                    max_seq_len=engine.max_seq_len,
                ))
            engine.seq_len = batch_engine.seq_len

    def _merge_decode_cache(self, requests):
        batch_engine = self._new_engine()
        batch_engine.seq_len = self.engines[requests[0].req_id].seq_len
        batch_engine.kv_caches = []
        for layer_index in range(len(self.model.layers)):
            layer_k, layer_v = [], []
            for req in requests:
                cache = self.engines[req.req_id].kv_caches[layer_index]
                k, v = cache.materialize()
                layer_k.append(k)
                layer_v.append(v)
            batch_engine.kv_caches.append(build_cache_backend(
                self.cache_backend,
                torch.cat(layer_k, dim=0),
                torch.cat(layer_v, dim=0),
                block_size=self.block_size,
                max_seq_len=batch_engine.max_seq_len,
            ))
        return batch_engine

    def _record_tokens(self, requests, tokens):
        for req, token in zip(requests, tokens):
            req.prompt = torch.cat([req.prompt, token.unsqueeze(0)], dim=1)
            req.generated += 1
            req.done = req.generated >= req.max_new
            req.stage = "decode"
        count = len(requests)
        self.stats["tokens_out"] += count
        self.stats["max_batch_size"] = max(self.stats["max_batch_size"], count)
        return count

    def _prefill_group(self, requests):
        batch_engine = self._new_engine()
        prompts = torch.cat([req.prompt for req in requests], dim=0)
        logits = batch_engine.prefill(prompts)
        tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        self._split_cache(batch_engine, requests)
        self.stats["prefill_batches"] += 1
        self.stats["model_forwards"] += 1
        return self._record_tokens(requests, tokens)

    def _decode_group(self, requests):
        batch_engine = self._merge_decode_cache(requests)
        input_tokens = torch.cat([req.prompt[:, -1:] for req in requests], dim=0)
        logits = batch_engine.decode_step(input_tokens)
        tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        self._split_cache(batch_engine, requests)
        self.stats["decode_batches"] += 1
        self.stats["model_forwards"] += 1
        return self._record_tokens(requests, tokens)

    def step(self, max_batch=4) -> int:
        active = [req for req in self.queue if not req.done][:max_batch]
        if not active:
            return 0
        produced = 0
        for (stage, _), requests in self._group_active(active).items():
            if stage == "prefill":
                produced += self._prefill_group(requests)
            else:
                produced += self._decode_group(requests)
        self.queue = [req for req in self.queue if not req.done]
        return produced

    def run_until_done(self, max_batch=4, max_steps=1000):
        steps = 0
        while self.queue and steps < max_steps:
            self.step(max_batch)
            steps += 1
        if self.queue:
            raise RuntimeError(f"scheduler exceeded max_steps={max_steps}")
        return dict(self.stats)
