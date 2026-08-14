"""
Lookahead Decoding — 无 draft 模型的 n-gram 并行验证

用 prompt 内 n-gram 作为候选 token，目标模型批量验证，适合重复模式较多的场景。
"""
import numpy as np

from .speculative_decoding import SimpleLM, softmax


class LookaheadDecoder:
    """基于 n-gram 查表的轻量 Decode 加速（NumPy 演示）。"""

    def __init__(self, target_model, ngram_size=3, gamma=4):
        self.target = target_model
        self.ngram_size = ngram_size
        self.gamma = gamma
        self.stats = {"target_calls": 0, "accepted": 0, "rejected": 0}

    def _draft_from_ngrams(self, context, n):
        """从已有上下文中找后续 n-gram 作为候选。"""
        if len(context) < self.ngram_size:
            tok, _ = self.target.generate_token(context)
            return np.array([tok])

        key = tuple(context[-(self.ngram_size - 1):])
        drafts = []
        seq = context.copy()
        for i in range(len(seq) - self.ngram_size + 1):
            gram = tuple(seq[i:i + self.ngram_size])
            if gram[:-1] == key:
                drafts.append(gram[-1])
        if not drafts:
            tok, _ = self.target.generate_token(context)
            return np.array([tok])
        return np.array(drafts[:n])

    def generate(self, prefix, max_new_tokens=20):
        """Generate greedily while reusing n-gram continuations as lookahead guesses."""
        output = prefix.copy()
        while len(output) - len(prefix) < max_new_tokens:
            n = min(self.gamma, max_new_tokens - (len(output) - len(prefix)))
            drafts = self._draft_from_ngrams(output, n)
            draft_seq = np.concatenate([output, drafts])
            target_logits = self.target.forward(draft_seq)
            self.stats["target_calls"] += 1

            accepted = 0
            for i, dt in enumerate(drafts):
                t_logits = target_logits[len(output) + i - 1]
                probs = softmax(t_logits)
                if np.random.random() < probs[dt]:
                    output = np.append(output, dt)
                    self.stats["accepted"] += 1
                    accepted += 1
                else:
                    sampled = np.random.choice(len(probs), p=probs)
                    output = np.append(output, sampled)
                    self.stats["rejected"] += 1
                    break
            if accepted == len(drafts) and len(output) - len(prefix) < max_new_tokens:
                last_probs = softmax(target_logits[-1])
                output = np.append(output, np.random.choice(len(last_probs), p=last_probs))
                self.stats["accepted"] += 1
        return output
