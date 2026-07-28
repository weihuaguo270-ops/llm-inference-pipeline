"""
Continuous Batching — 多请求交错 Prefill/Decode 模拟

模拟服务化推理中不同请求处于 Prefill 或 Decode 阶段的 batch 调度。
"""
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    req_id: int
    prompt: torch.Tensor
    max_new: int
    generated: int = 0
    done: bool = False
    stage: str = "prefill"  # prefill | decode


class ContinuousBatcher:
    """简化版 continuous batching 调度器。"""

    def __init__(self, engine_factory):
        self.engine_factory = engine_factory
        self.queue: List[Request] = []
        self.engines = {}
        self.stats = {"prefill_batches": 0, "decode_batches": 0, "tokens_out": 0}

    def add_request(self, req: Request):
        self.queue.append(req)

    def step(self, max_batch=4) -> int:
        """执行一步调度，返回本步产出 token 数。"""
        active = [r for r in self.queue if not r.done][:max_batch]
        if not active:
            return 0

        tokens = 0
        for r in active:
            eng = self.engines.setdefault(r.req_id, self.engine_factory())
            if r.stage == "prefill":
                logits = eng.prefill(r.prompt)
                r.stage = "decode"
                self.stats["prefill_batches"] += 1
                tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                r.prompt = torch.cat([r.prompt, tok], dim=1)
                r.generated += 1
                tokens += 1
                self.stats["tokens_out"] += 1
            else:
                logits = eng.decode_step(r.prompt[:, -1:])
                tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                r.prompt = torch.cat([r.prompt, tok], dim=1)
                r.generated += 1
                tokens += 1
                self.stats["tokens_out"] += 1
                self.stats["decode_batches"] += 1
                if r.generated >= r.max_new:
                    r.done = True
        self.queue = [r for r in self.queue if not r.done]
        return tokens

    def run_until_done(self, max_batch=4, max_steps=1000):
        steps = 0
        while self.queue and steps < max_steps:
            self.step(max_batch)
            steps += 1
        return self.stats
