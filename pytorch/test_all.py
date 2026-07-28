"""
PyTorch 版 — 数值验证测试

与 test_all.py 的测试逻辑完全一致，但使用 PyTorch。
验证 PyTorch 版与 NumPy 版的数学等价性。

用法:
    python test_all.py
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from console_io import FAIL, PASS, configure_stdio, safe_print

configure_stdio()

errors = []


def check(name, cond, detail=""):
    if cond:
        safe_print(f"  {PASS} {name}")
    else:
        msg = f"  {FAIL} {name}" + (f" — {detail}" if detail else "")
        safe_print(msg)
        errors.append(name)


# ============================================================
# 1. utils
# ============================================================
print("\n【utils 工具函数】")
from utils import softmax, split_heads, combine_heads, layer_norm

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
s = softmax(x)
check("softmax 形状", s.shape == (2, 3))
check("softmax 行和为1", torch.allclose(s.sum(dim=-1), torch.tensor([1.0, 1.0])))
check("softmax 单调性", s[0, 0] < s[0, 1] < s[0, 2])

# split_heads
x2d = torch.randn(4, 8)
sh = split_heads(x2d, num_heads=2)
check("split_heads 形状", sh.shape == (2, 4, 4))

# combine_heads
ch = combine_heads(sh, num_heads=2)
check("combine_heads 还原", ch.shape == (4, 8))
check("combine_heads 值不变", torch.allclose(ch, x2d))

# layer_norm
x_ln = torch.randn(2, 4, 8) * 2 + 1
ln_out = layer_norm(x_ln, eps=1e-5)
check("layer_norm 形状", ln_out.shape == (2, 4, 8))
check("layer_norm 均值≈0", torch.allclose(ln_out.mean(dim=-1), torch.zeros(2, 4), atol=1e-5))
check("layer_norm 方差≈1", torch.allclose(ln_out.std(dim=-1, unbiased=False), torch.ones(2, 4), atol=1e-4))


# ============================================================
# 2. attention
# ============================================================
print("\n【attention 单头 Self-Attention】")
torch.manual_seed(42)
d_model, d_k = 4, 3
W_q = torch.randn(d_model, d_k)
W_k = torch.randn(d_model, d_k)
W_v = torch.randn(d_model, d_k)
X = torch.randn(3, d_model)

Q = X @ W_q
K = X @ W_k
V = X @ W_v
scores = Q @ K.T / (d_k ** 0.5)
weights = softmax(scores)
output = weights @ V

check("无掩码输出形状", output.shape == (3, d_k))
check("权重行和为1", torch.allclose(weights.sum(dim=-1), torch.tensor([1.0, 1.0, 1.0])))

# 因果掩码
causal_mask = torch.triu(torch.full((3, 3), -1e9), diagonal=1)
causal_scores = Q @ K.T / (d_k ** 0.5) + causal_mask
causal_weights = softmax(causal_scores)

check("词0只看自己", causal_weights[0, 1].item() == 0.0 and causal_weights[0, 2].item() == 0.0)
check("词1只看前2", causal_weights[1, 2].item() == 0.0)


# ============================================================
# 3. multi_head_attention
# ============================================================
print("\n【multi_head_attention 多头注意力】")
from multi_head_attention import MultiHeadAttention

torch.manual_seed(42)
mha = MultiHeadAttention(d_model=8, num_heads=2)
X_mha = torch.randn(4, 8)
out_mha = mha(X_mha, use_mask=False)
check("多头输出形状", out_mha.shape == (4, 8))
check("多头输出非零", torch.norm(out_mha).item() > 0)

out_masked = mha(X_mha, use_mask=True)
check("多头+掩码输出形状", out_masked.shape == (4, 8))


# ============================================================
# 4. kv_cache
# ============================================================
print("\n【kv_cache KV Cache】")
torch.manual_seed(42)
d_k = 4

q1 = torch.randn(1, d_k)
k1 = torch.randn(1, d_k)
v1 = torch.randn(1, d_k)
scores1 = (q1 @ k1.T) / (d_k ** 0.5)
out1 = softmax(scores1) @ v1

q2 = torch.randn(1, d_k)
k2 = torch.randn(1, d_k)
v2 = torch.randn(1, d_k)
K_cache = k1
V_cache = v1
K_full = torch.cat([K_cache, k2])
V_full = torch.cat([V_cache, v2])
scores2 = (q2 @ K_full.T) / (d_k ** 0.5)
out2_cached = softmax(scores2) @ V_full

scores2_direct = (q2 @ torch.cat([k1, k2]).T) / (d_k ** 0.5)
out2_direct = softmax(scores2_direct) @ torch.cat([v1, v2])

diff = (out2_cached - out2_direct).abs().max().item()
check("KV Cache 输出一致", diff < 1e-6)


# ============================================================
# 5. positional_encoding
# ============================================================
print("\n【positional_encoding 位置编码】")
from positional_encoding import sinusoidal_positional_encoding

pe = sinusoidal_positional_encoding(seq_len=10, d_model=8)
check("位置编码形状", pe.shape == (10, 8))
check("位置编码非零", torch.norm(pe).item() > 0)
check("相邻位置编码不同", torch.norm(pe[0] - pe[1]).item() > 0)
check("值域在[-1,1]", (pe.abs() <= 1.0 + 1e-6).all().item())


# ============================================================
# 6. transformer_block
# ============================================================
print("\n【transformer_block 完整 Block】")
from transformer_block import TransformerBlock

torch.manual_seed(42)
block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
X_tb = torch.randn(4, 8)
out_tb = block(X_tb, use_mask=True)
check("Block输出形状", out_tb.shape == (4, 8))
check("Block输出稳定", torch.isfinite(out_tb).all().item())

x = X_tb
for i in range(3):
    x = block(x, use_mask=True)
check("3层堆叠稳定", torch.isfinite(x).all().item())


# ============================================================
# 7. MLA (PyTorch)
# ============================================================
print("\n【MLA PyTorch — 解压 vs 吸收】")
from pytorch.mla import MultiHeadLatentAttention

torch.manual_seed(0)
mla = MultiHeadLatentAttention(d_model=8, num_heads=2, d_c=3, d_kv_rope=4, max_seq_len=32)
mla.absorb_weights()
x_step = torch.randn(1, 1, 8)
out_d, c_d, k_d = mla.forward_with_cache(x_step, use_absorb=False)
out_a, c_a, k_a = mla.forward_with_cache(x_step, use_absorb=True)
check("MLA absorb 输出形状", out_a.shape == (1, 1, 8))
max_diff = (out_d - out_a).abs().max().item()
check("MLA absorb≈解压", max_diff < 1e-5, f"max_diff={max_diff}")


# ============================================================
# 8. InferenceEngine (Prefill + Cache Decode)
# ============================================================
print("\n【InferenceEngine Prefill/Decode】")
from pytorch.llama_block import GPT
from pytorch.inference_engine import InferenceEngine

gpt = GPT(vocab_size=128, d_model=32, num_layers=2, num_heads=4, num_kv_heads=2, d_ff=64, max_seq_len=64)
eng = InferenceEngine(gpt)
prompt = torch.randint(0, 128, (1, 10))
logits_p = eng.prefill(prompt)
tok = torch.randint(0, 128, (1, 1))
logits_d = eng.decode_step(tok)
check("Prefill logits 形状", logits_p.shape == (1, 10, 128))
check("Decode logits 形状", logits_d.shape == (1, 1, 128))

# Cache 路径必须与每步完整重算的最后一个位置严格对齐。
gpt.eval()
sequence = torch.randint(0, 128, (2, 8))
eng = InferenceEngine(gpt)
cached_prefill = eng.prefill(sequence[:, :5])
full_prefill = gpt(sequence[:, :5])
check(
    "Prefill 与完整前向一致",
    torch.allclose(cached_prefill, full_prefill, atol=1e-5, rtol=1e-5),
)
for pos in range(5, sequence.shape[1]):
    cached_step = eng.decode_step(sequence[:, pos:pos + 1])
    full_step = gpt(sequence[:, :pos + 1])[:, -1:, :]
    check(
        f"Batch Decode 第 {pos} 步与完整前向一致",
        torch.allclose(cached_step, full_step, atol=1e-5, rtol=1e-5),
        f"max_diff={(cached_step - full_step).abs().max().item()}",
    )

fresh_eng = InferenceEngine(gpt)
try:
    fresh_eng.decode_step(torch.randint(0, 128, (1, 1)))
except RuntimeError:
    decode_requires_prefill = True
else:
    decode_requires_prefill = False
check("Decode 前必须 Prefill", decode_requires_prefill)

paged_eng = InferenceEngine(gpt, cache_backend="paged", block_size=3)
paged_prefill = paged_eng.prefill(sequence[:, :5])
check(
    "Paged Cache Prefill 与完整前向一致",
    torch.allclose(paged_prefill, full_prefill, atol=1e-5, rtol=1e-5),
)
for pos in range(5, sequence.shape[1]):
    paged_step = paged_eng.decode_step(sequence[:, pos:pos + 1])
    full_step = gpt(sequence[:, :pos + 1])[:, -1:, :]
    check(
        f"Paged Decode 第 {pos} 步与完整前向一致",
        torch.allclose(paged_step, full_step, atol=1e-5, rtol=1e-5),
        f"max_diff={(paged_step - full_step).abs().max().item()}",
    )
check("Paged Cache 跨越多个 block", paged_eng.kv_caches[0].num_blocks == 3)

# Static Cache 必须原地更新，避免 Decode 阶段重复分配 K/V。
static_eng = InferenceEngine(gpt, cache_backend="static")
static_eng.prefill(sequence[:, :5])
static_ptr = static_eng.kv_caches[0].k.data_ptr()
static_eng.decode_step(sequence[:, 5:6])
check("Static Cache Decode 不重新分配", static_eng.kv_caches[0].k.data_ptr() == static_ptr)
check("Static Cache 逻辑长度更新", static_eng.kv_caches[0].length == 6)
check(
    "Static Cache 使用预分配容量",
    static_eng.kv_caches[0].k.shape[2] == gpt.max_seq_len,
)
from pytorch.cache_backends import StaticKVCache

capacity_k = torch.randn(1, 1, 1, 4)
capacity_cache = StaticKVCache(2, capacity_k, capacity_k)
try:
    capacity_cache.append(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
except RuntimeError:
    static_capacity_guard = True
else:
    static_capacity_guard = False
check("Static Cache 容量越界会失败", static_capacity_guard)

# SDPA/原生 GQA kernel 与 eager 参考实现保持一致。
eager_gpt = GPT(
    vocab_size=128, d_model=32, num_layers=2, num_heads=4,
    num_kv_heads=2, d_ff=64, max_seq_len=64,
    attention_backend="eager",
)
eager_gpt.load_state_dict(gpt.state_dict())
sdpa_logits = gpt(sequence)
eager_logits = eager_gpt(sequence)
check(
    "SDPA GQA 与 eager 完整前向一致",
    torch.allclose(sdpa_logits, eager_logits, atol=1e-5, rtol=1e-5),
    f"max_diff={(sdpa_logits - eager_logits).abs().max().item()}",
)

amp_eng = InferenceEngine(gpt, cache_backend="static", amp_dtype="bfloat16")
amp_logits = amp_eng.prefill(sequence[:, :5])
check("BF16 autocast 推理输出有限", torch.isfinite(amp_logits).all().item())


# ============================================================
# 9. Continuous Batching
# ============================================================
print("\n【Continuous Batching】")
from pytorch.continuous_batching import ContinuousBatcher, Request

batch_model = GPT(
    vocab_size=64, d_model=16, num_layers=1, num_heads=2,
    num_kv_heads=1, d_ff=32, max_seq_len=32,
)
batcher = ContinuousBatcher(lambda: InferenceEngine(batch_model))
requests = [
    Request(i, torch.randint(0, 64, (1, 4)), max_new=3)
    for i in range(3)
]
for request in requests:
    batcher.add_request(request)
batch_stats = batcher.run_until_done(max_batch=3)
check("相同长度请求使用一次 Batched Prefill", batch_stats["prefill_batches"] == 1)
check("Decode 按 batch 执行", batch_stats["decode_batches"] == 2)
check("批量大小达到 3", batch_stats["max_batch_size"] == 3)
check("每个请求严格生成 max_new", all(r.generated == 3 for r in requests))

single = Request(99, torch.randint(0, 64, (1, 3)), max_new=1)
batcher = ContinuousBatcher(lambda: InferenceEngine(batch_model))
batcher.add_request(single)
single_stats = batcher.run_until_done()
check("max_new=1 不会多生成", single.generated == 1)
check("max_new=1 无额外 Decode", single_stats["decode_batches"] == 0)


# ============================================================
# 10. Prefix Cache
# ============================================================
print("\n【Prefix Cache】")
from pytorch.prefix_cache import PrefixKVCache

prefix_engine = InferenceEngine(batch_model, cache_backend="paged", block_size=3)
prefix_cache = PrefixKVCache(prefix_engine)
shared = torch.randint(0, 64, (1, 5))
suffix_a = torch.randint(0, 64, (1, 2))
suffix_b = torch.randint(0, 64, (1, 3))
full_a = torch.cat([shared, suffix_a], dim=1)
full_b = torch.cat([shared, suffix_b], dim=1)
cached_a = prefix_cache.prefill_with_prefix(full_a, prefix_len=shared.shape[1])
cached_b = prefix_cache.prefill_with_prefix(full_b)
check("共享前缀 B 命中", prefix_cache.cache_hit(full_b))
check(
    "Prefix Cache 请求 A 数值一致",
    torch.allclose(cached_a, batch_model(full_a)[:, -1:, :], atol=1e-5, rtol=1e-5),
)
check(
    "Prefix Cache 恢复快照后请求 B 数值一致",
    torch.allclose(cached_b, batch_model(full_b)[:, -1:, :], atol=1e-5, rtol=1e-5),
)


# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*50}")
if errors:
    safe_print(f"{FAIL} {len(errors)} 项失败:")
    for e in errors:
        safe_print(f"   - {e}")
else:
    safe_print(f"{PASS} 全部测试通过!")
print(f"{'='*50}")

raise SystemExit(1 if errors else 0)
