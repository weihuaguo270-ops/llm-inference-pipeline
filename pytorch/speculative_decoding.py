"""
PyTorch Speculative Decoding — 真实 GPT 小模型适配

将 pytorch.llama_block.GPT 接入 modern_llm.speculative_decoding.SpeculativeDecoder，
用不同规模的训练前随机权重模型验证执行链路、墙钟耗时与接受率。
"""
import numpy as np
import torch
import torch.nn.functional as F
import time

try:
    from .llama_block import GPT
except ImportError:
    from llama_block import GPT


class TorchLM:
    """SpeculativeDecoder 所需的 LM 接口（NumPy token ids ↔ PyTorch GPT）。"""

    def __init__(self, model: GPT, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            idx = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model(idx)[0].cpu().numpy()
        return logits

    def generate_token(self, token_ids):
        logits = self.forward(token_ids)
        last = logits[-1]
        probs = self._softmax(last)
        tok = np.random.choice(len(probs), p=probs)
        return tok, last

    def generate_n_tokens(self, token_ids, n):
        tokens, logits_list = [], []
        current = token_ids.copy()
        for _ in range(n):
            tok, lg = self.generate_token(current)
            tokens.append(tok)
            logits_list.append(lg)
            current = np.append(current, tok)
        return np.array(tokens), logits_list

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()


def build_draft_target_pair(
    vocab_size=512,
    d_model_draft=64,
    d_model_target=128,
    num_layers=2,
    device="cpu",
):
    """构建 draft / target 小模型对（同词表、不同容量）。"""
    common = dict(vocab_size=vocab_size, num_heads=4, num_kv_heads=2, d_ff=128, max_seq_len=128)
    draft = GPT(d_model=d_model_draft, num_layers=num_layers, **common)
    target = GPT(d_model=d_model_target, num_layers=num_layers + 1, **common)
    return TorchLM(draft, device), TorchLM(target, device)


def run_speculative_benchmark(
    gamma=4,
    max_new_tokens=24,
    prefix_len=8,
    device="cpu",
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    draft, target = build_draft_target_pair(device=device)
    from modern_llm.speculative_decoding import SpeculativeDecoder

    prefix = np.random.randint(0, 512, size=prefix_len)

    baseline_tokens = []
    current = prefix.copy()
    if device == "cuda":
        torch.cuda.synchronize()
    baseline_start = time.perf_counter()
    for _ in range(max_new_tokens):
        tok, _ = target.generate_token(current)
        baseline_tokens.append(tok)
        current = np.append(current, tok)
    if device == "cuda":
        torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - baseline_start) * 1000

    decoder = SpeculativeDecoder(draft, target, gamma=gamma)
    if device == "cuda":
        torch.cuda.synchronize()
    speculative_start = time.perf_counter()
    output = decoder.generate(prefix, max_new_tokens=max_new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    speculative_ms = (time.perf_counter() - speculative_start) * 1000

    target_call_ratio = max_new_tokens / max(decoder.stats["target_calls"], 1)
    wall_time_speedup = baseline_ms / speculative_ms
    total = decoder.stats["accepted"] + decoder.stats["rejected"]
    accept_rate = decoder.stats["accepted"] / total if total else 1.0

    return {
        "gamma": gamma,
        "max_new_tokens": max_new_tokens,
        "target_calls_baseline": max_new_tokens,
        "target_calls_spec": decoder.stats["target_calls"],
        "speedup": wall_time_speedup,
        "wall_time_speedup": wall_time_speedup,
        "target_call_ratio": target_call_ratio,
        "baseline_ms": baseline_ms,
        "speculative_ms": speculative_ms,
        "accept_rate": accept_rate,
        "output_len": len(output) - len(prefix),
        "stats": decoder.stats,
    }
