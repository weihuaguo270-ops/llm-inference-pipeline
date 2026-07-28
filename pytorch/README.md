# PyTorch 版 — 推理引擎与基准

`pytorch/` 用 PyTorch 框架 API 重写各模块，提供 **Prefill/Decode 推理引擎**、GPU 级基准与 Attention 变体训练验证。

## 推理链路模块

| 文件 | 角色 |
|------|------|
| `inference_engine.py` | Prefill + KV Cache Decode 端到端循环 |
| `gqa.py` | GQA + RoPE + `forward_with_cache` |
| `mla.py` | MLA 解压/吸收双路径（PyTorch） |
| `prefix_cache.py` | 共享前缀 KV 复用 |
| `paged_kv_cache.py` | Block 化 KV 存储 |
| `speculative_decoding.py` | GPT 小模型 Spec Decoding 适配 |
| `continuous_batching.py` | 多请求交错调度模拟 |

## 与 NumPy 版的对应关系

| 模块 | NumPy 版 | PyTorch 版 | 差异 |
|------|---------|-----------|------|
| Self-Attention | `np_impl/attention.py` | `attention.py` | nn.Linear + F.softmax |
| MHA | `np_impl/multi_head_attention.py` | `attention.py` | 同上 |
| RoPE | `np_impl/rotary.py` | `attention.py` | 集成到 attention 中 |
| Cross Attention | `np_impl/cross_attention.py` | `cross_attention.py` | nn.Linear + F.softmax |
| GQA | `modern_llm/gqa.py` | `gqa.py` | nn.Linear + F.scaled_dot_product_attention |
| Llama Block | `modern_llm/llama_block.py` | `llama_block.py` | RMSNorm + SwiGLU + 完整 GPT 模型 |

## 训练（Attention 变体验证）

```bash
python -m pytorch.train_gpt --epochs 3 --d_model 64 --num_heads 4
python -m pytorch.train_gpt --epochs 5 --num_kv_heads 2  # GQA
```

- 数据集：TinyStories
- 支持命令行调参和交互式输入
- 自动记录实验配置和结果

## 推理基准

Attention 变体延迟对比见 [`experiments/benchmark_attention.py`](../experiments/benchmark_attention.py)。

## 实验记录

参见 [`experiments/runs/`](../experiments/runs/)。
