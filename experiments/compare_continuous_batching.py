"""
Continuous Batching 模拟 — 多请求吞吐对照

用法:
    python -m experiments.compare_continuous_batching
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pytorch.llama_block import GPT
from pytorch.inference_engine import InferenceEngine
from pytorch.continuous_batching import ContinuousBatcher, Request


def main():
    torch.manual_seed(0)
    model = GPT(vocab_size=256, d_model=64, num_layers=2, num_heads=4, num_kv_heads=2, d_ff=128)

    def factory():
        return InferenceEngine(model)

    batcher = ContinuousBatcher(factory)
    for i in range(3):
        prompt = torch.randint(0, 256, (1, 8))
        batcher.add_request(Request(req_id=i, prompt=prompt, max_new=6))

    stats = batcher.run_until_done(max_batch=3)
    print("=" * 58)
    print("Continuous Batching 模拟")
    print("=" * 58)
    print(f"  请求数: 3  max_batch: 3  每请求 max_new: 6")
    print(f"  prefill_batches: {stats['prefill_batches']}")
    print(f"  decode_batches:  {stats['decode_batches']}")
    print(f"  model_forwards:  {stats['model_forwards']}")
    print(f"  max_batch_size:  {stats['max_batch_size']}")
    print(f"  tokens_out:      {stats['tokens_out']}")


if __name__ == "__main__":
    main()
