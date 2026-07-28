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
