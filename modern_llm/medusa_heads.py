"""
Medusa Heads — 多 token 并行预测（简化版）

在主模型 hidden state 上挂多个 lm head，每步并行预测 t+1, t+2, ... 候选，
再用目标分布验证。本实现为教学级 NumPy 演示。
"""
import numpy as np

from .speculative_decoding import SimpleLM, softmax


class MedusaHeads:
    """K 个额外 lm head，共享 embedding 路径。"""

    def __init__(self, base_lm: SimpleLM, num_heads=4, seed=7):
        self.base = base_lm
        self.num_heads = num_heads
        rng = np.random.RandomState(seed)
        d, v = base_lm.d_model, base_lm.vocab_size
        self.heads = [rng.randn(d, v) * 0.05 for _ in range(num_heads)]

    def propose(self, token_ids, gamma):
        _ = self.base.forward(token_ids)
        hidden = self.base.embedding[token_ids]
        drafts = []
        for h in range(min(gamma, self.num_heads)):
            lg = hidden[-1] @ self.heads[h]
            probs = softmax(lg)
            drafts.append(np.random.choice(len(probs), p=probs))
        return np.array(drafts), None


class MedusaDecoder:
    def __init__(self, base_lm, num_heads=4, gamma=4):
        self.medusa = MedusaHeads(base_lm, num_heads)
        self.base = base_lm
        self.gamma = gamma
        self.stats = {"target_calls": 0, "accepted": 0, "rejected": 0}

    def generate(self, prefix, max_new_tokens=20):
        output = prefix.copy()
        while len(output) - len(prefix) < max_new_tokens:
            n = min(self.gamma, max_new_tokens - (len(output) - len(prefix)))
            drafts, _ = self.medusa.propose(output, n)
            draft_seq = np.concatenate([output, drafts])
            target_logits = self.base.forward(draft_seq)
            self.stats["target_calls"] += 1

            for i, dt in enumerate(drafts):
                t_logits = target_logits[len(output) + i - 1]
                probs = softmax(t_logits)
                if np.random.random() < probs[dt]:
                    output = np.append(output, dt)
                    self.stats["accepted"] += 1
                else:
                    output = np.append(output, np.random.choice(len(probs), p=probs))
                    self.stats["rejected"] += 1
                    break
        return output
