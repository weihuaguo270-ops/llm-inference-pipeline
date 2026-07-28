# 推理优化实现（2023-2024）

`modern_llm/` — 纯 NumPy 实现，覆盖当前主流 LLM 推理链路上的 Attention 变体与 Decode 加速手段。

包含 Llama 路线（GQA + Pre-Norm Block）和 DeepSeek 路线（MLA 潜空间压缩）两大类 Cache 优化，以及 Speculative Decoding、StreamingLLM 等 Decode 层加速。

独立包，不依赖 `np_impl/` 目录。

## 文件说明

| 文件 | 推理链路角色 |
|------|-------------|
| `gqa.py` | **Cache 压缩**：分组 KV 头、广播、与 RoPE 集成 |
| `mla.py` | **Cache 压缩 + 计算优化**：低维 KV 压缩、解压/吸收双路径 |
| `attention_sinks.py` | **Cache 管理**：StreamingLLM 长文本淘汰策略 |
| `speculative_decoding.py` | **Decode 加速**：Draft Model 并行验证 |
| `lookahead_decoding.py` | **Decode 加速**：n-gram 候选验证（无 draft 模型） |
| `medusa_heads.py` | **Decode 加速**：多 lm head 并行预测 |
| `llama_block.py` | 推理常用 Block 结构（Pre-Norm + RMSNorm + SwiGLU + GQA） |
| `rotary.py` | RoPE 旋转位置编码（长度外推） |
| `utils.py` | 工具函数 |
| `test.py` | 冒烟测试（含 MLA 吸收≈解压） |

MLA 吸收路径用法：

```python
from modern_llm.mla import MultiHeadLatentAttention
mla = MultiHeadLatentAttention(d_model=8, num_heads=2, d_k=4, d_c=3, d_kv_rope=2)
mla.absorb_weights()
out, c, kr = mla.forward_with_cache(x_step, use_absorb=True)
```

## 对比实验

参见 [`experiments/`](../experiments/README.md) 目录。

## 运行测试

```bash
python -m modern_llm.test
```
