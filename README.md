# Transformer 推理链路优化

[![CI](https://github.com/weihuaguo270-ops/transformer-attention/actions/workflows/test.yml/badge.svg)](https://github.com/weihuaguo270-ops/transformer-attention/actions/workflows/test.yml) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**LLM 自回归推理链路的实现与基准** — NumPy 提供可读的数学参考；PyTorch 2.x 路径面向真实性能工程，默认使用原生 SDPA/GQA kernel、预分配 Static KV Cache、`inference_mode`，并支持 CUDA AMP 与 `torch.compile`。仓库提供可复现的延迟、缓存和解码实验，不包含大规模预训练设施。

## 推理链路概览

自回归 LLM 推理可分为 **Prefill**（处理 prompt）与 **Decode**（逐 token 生成）两阶段。Decode 循环是延迟与显存的主要瓶颈，本仓库聚焦该链路上的优化层：

```
Prompt
  │
  ▼
Prefill ──► KV Cache 写入
  │
  ▼
Decode Loop（每步）
  ├── Attention（Q·K^T·V，读/写 KV Cache）  ◄── GQA / MLA / StreamingLLM
  ├── FFN
  └── 采样下一 token                          ◄── Speculative Decoding
  │
  ▼
输出 token 流
```

| 瓶颈 | 优化手段 | 本仓库实现 |
|------|----------|------------|
| Decode 重复计算 K/V | KV Cache | [`np_impl/kv_cache.py`](np_impl/kv_cache.py) |
| KV Cache 显存占用 | GQA / MQA | [`modern_llm/gqa.py`](modern_llm/gqa.py) |
| KV Cache 进一步压缩 | MLA + 吸收矩阵 | [`modern_llm/mla.py`](modern_llm/mla.py) |
| 长上下文 Cache 淘汰 | Attention Sinks | [`modern_llm/attention_sinks.py`](modern_llm/attention_sinks.py) |
| Decode 串行延迟 | Speculative / Lookahead / Medusa | [`modern_llm/speculative_decoding.py`](modern_llm/speculative_decoding.py) 等 |
| 共享前缀重复 Prefill | Prefix Cache | [`pytorch/prefix_cache.py`](pytorch/prefix_cache.py) |
| Cache 显存碎片 | Paged KV Cache（block 读取路径） | [`pytorch/paged_kv_cache.py`](pytorch/paged_kv_cache.py) |
| 服务化吞吐 | Continuous Batching | [`pytorch/continuous_batching.py`](pytorch/continuous_batching.py) |
| 优化效果量化 | 基准与对比实验 | [`experiments/`](experiments/README.md) |

## 快速开始

```bash
pip install -e .
# 可选：训练 / GPU 基准依赖
pip install -e ".[pytorch]"

# 运行全部测试
python test_all.py
```

## 项目结构

```
transformer-attention/
│
├── np_impl/                    # 推理基线：Attention + KV Cache（NumPy）
│   ├── attention.py            单头 Self-Attention + 因果掩码
│   ├── multi_head_attention.py 多头注意力（MHA，KV Cache 开销基线）
│   ├── kv_cache.py             KV Cache：Decode 阶段 O(n·d) 加速
│   └── ...
│
├── modern_llm/                 # 推理优化核心实现（2023-2024）
│   ├── gqa.py                  GQA：KV 头分组，压缩 Cache
│   ├── mla.py                  MLA：潜空间压缩 + 吸收矩阵加速
│   ├── speculative_decoding.py Speculative Decoding：并行验证候选 token
│   ├── lookahead_decoding.py   Lookahead：n-gram 候选验证
│   ├── medusa_heads.py         Medusa：多 head 并行预测
│   ├── attention_sinks.py      StreamingLLM：长文本 Cache 淘汰策略
│   ├── llama_block.py          Llama Block（Pre-Norm + SwiGLU，推理常用结构）
│   └── test.py                 ~33 项冒烟测试
│
├── experiments/                # 推理链路对比实验与基准
│   ├── compare_attention.py    MHA vs GQA vs MLA：Cache 大小 / 参数量
│   ├── compare_cache.py        完整 Cache vs StreamingLLM：质量 / 节省
│   ├── compare_decoding.py     标准 Decode vs Spec Decoding：加速比
│   ├── benchmark_attention.py  前向延迟 / 吞吐量（CPU/CUDA）
│   ├── benchmark_prefill_decode.py  TTFT / TPOT（Prefill vs Decode）
│   ├── compare_decoding_pytorch.py  PyTorch GPT Spec Decoding 端到端
│   ├── compare_decode_strategies.py 标准 / Spec / Lookahead / Medusa
│   ├── compare_prefix_cache.py      Prefix Cache TTFT 对比
│   ├── compare_paged_cache.py       Paged KV block 利用率
│   ├── compare_kv_quant.py          FP32 vs INT8 Cache 体积/误差
│   ├── compare_continuous_batching.py 多请求真实批量前向对比
│   └── runs/                   实验记录 + mla_absorb_*.csv
│
├── pytorch/                    # PyTorch 实现 + 推理引擎
│   ├── gqa.py                  GQA + RoPE + forward_with_cache
│   ├── mla.py                  MLA 解压/吸收双路径（PyTorch）
│   ├── inference_engine.py     Prefill + KV Cache Decode 引擎
│   ├── prefix_cache.py         共享前缀 Cache 复用
│   ├── paged_kv_cache.py       Block 化 KV 存储
│   ├── speculative_decoding.py PyTorch GPT Spec Decoding 适配
│   ├── cache_backends.py       连续 / Paged KV Cache 统一接口
│   ├── continuous_batching.py  共享模型的批量 Prefill / Decode 调度
│   ├── llama_block.py          完整 GPT 模型
│   ├── train_gpt.py            训练脚本（TinyStories 级）
│   └── test_all.py             PyTorch 侧冒烟测试
│
├── test_all.py                 统一测试入口（np_impl + modern_llm，约 74 项）
└── pyproject.toml
```

| 目录 | 在推理链路中的角色 |
|------|-------------------|
| [`np_impl/`](np_impl/README.md) | 建立 MHA + KV Cache 基线，理解 Decode 瓶颈来源 |
| [`modern_llm/`](modern_llm/README.md) | 主流推理优化技术的可读实现 |
| [`experiments/`](experiments/README.md) | 量化 Cache 节省、延迟、解码加速比 |
| [`pytorch/`](pytorch/README.md) | GPU 级基准与 Attention 变体训练验证 |

## 核心优化实现

### KV Cache — Decode 阶段基础加速

自回归解码时缓存 K/V 张量，避免每步重算历史 token 的投影，复杂度从 O(n²·d) 降至 O(n·d)。

### GQA — 压缩 KV Cache（2023）

减少 K/V 头数、保留 Q 头数，在 MHA 质量与 MQA 效率间取平衡。Llama 2/3、Mistral、Qwen 等主流模型的默认方案。

| 变体 | KV 头数 | KV Cache（32h, 4096seq, FP16） | 代表模型 |
|------|---------|-------------------------------|---------|
| MHA | 32 | 64.0 MB | 原始 Transformer |
| GQA | 8 | 16.0 MB | Llama 3 70B |
| GQA | 4 | 8.0 MB | Mistral 7B |
| MQA | 1 | 2.0 MB | Falcon |

### MLA — 潜空间 Cache 压缩 + 吸收矩阵（2024）

DeepSeek V2/V3 思路：K/V 压入低维潜空间，Cache 维度远小于 MHA。

```
MHA:   K = h · W_K,       缓存 K-V（d_model 维）
MLA:   c = h · W_DKV,     缓存 c（d_c 维, d_c << d_model）
       K = c · W_UK,       V = c · W_UV（从压缩缓存解压）
```

**吸收矩阵技巧（推理可用）：**
```
Q_h · (C · W_UK_h) = (Q_h · W_UK_h^T) · C
attn · (C · W_UV_h) = (attn · C) · W_UV_h
```
`forward_with_cache(..., use_absorb=True)` 走吸收路径；`use_absorb=False` 为逐步解压对照。单测验证两者数值对齐（max|Δ| < 1e-6）。

实际参数（DeepSeek V2, d_model=5120）：
- MHA 每步缓存：2 × 5120 = 10,240 维
- MLA 每步缓存：512 + 64 = 576 维
- **压缩比：约 18x**

**本机微基准（NumPy CPU，教学规模；完整 CSV 见 `experiments/runs/mla_absorb_20260714.csv`）：**

| 配置 | 解压 ms | 吸收 ms | 加速比 | 对齐 max\|Δ\| |
|------|--------:|--------:|-------:|-------------:|
| d=256, h=8, d_c=64, seq=64 | 11.1 | 9.3 | 1.20× | ~1e-18 |
| d=512, h=8, d_c=128, seq=256 | 425.6 | 76.5 | 5.56× | ~1e-17 |

```bash
python -m experiments.benchmark_mla_absorb
python -m experiments.benchmark_mla_absorb --seq_len 256 --d_model 512 --d_c 128 --csv experiments/runs/mla_absorb.csv
```

### Speculative Decoding — Decode 并行加速

小模型（Draft Model）先生成 K 个候选 token，目标模型批量验证并用 rejection sampling 保持目标分布。PyTorch 基准同时报告 Target 调用次数比与墙钟加速比；当前 draft/target 仍是训练前小模型，因此只验证执行链路，不代表训练后模型的质量或加速收益。

### Attention Sinks / StreamingLLM — 长上下文 Cache 管理

保留最近 tokens + 开头若干 tokens（attention sink），在有限 Cache 窗口下处理远超训练长度的序列，无需完整 KV 重算。适用 Agent 长对话、连续推理场景。

## 基准与对比实验

```bash
# Prefill / Decode 分离：TTFT、TPOT、KV Cache vs 无 Cache
python -m experiments.benchmark_prefill_decode
python -m experiments.benchmark_prefill_decode --device cuda --prompt_len 512
python -m experiments.benchmark_prefill_decode --cache_backend paged --json experiments/runs/prefill_decode.json

# PyTorch 2.x 优化矩阵：eager / SDPA / Static Cache / AMP / compile
python -m experiments.benchmark_pytorch_optimized --device cuda --prompt_len 1024 \
  --amp_dtype bfloat16 --compile_mode reduce-overhead \
  --json experiments/runs/pytorch_optimized.json

# Attention 变体：Cache 大小 / 参数量 / 60 层推估
python -m experiments.compare_attention

# KV Cache 策略：完整 vs StreamingLLM
python -m experiments.compare_cache

# 解码策略：标准 / Spec / Lookahead / Medusa
python -m experiments.compare_decode_strategies
python -m experiments.compare_decoding_pytorch --json experiments/runs/speculative.json

# Cache 层优化：Prefix / Paged / 量化
python -m experiments.compare_prefix_cache
python -m experiments.compare_paged_cache
python -m experiments.compare_kv_quant

# 服务化：Continuous Batching
python -m experiments.compare_continuous_batching  # 真实 batched forward 次数

# 前向延迟 / 吞吐量（CI 短序列；本地可换 GPU / 长序列）
python -m experiments.benchmark_attention
python -m experiments.benchmark_attention --device cuda --seq_len 2048

# MLA 解压 vs 吸收路径
python -m experiments.benchmark_mla_absorb --seq_len 256 --d_model 512 --d_c 128
```

PyTorch 训练 pipeline 用于验证 Attention 变体（TinyStories 级，非大规模预训练）：

```bash
python -m pytorch.train_gpt --epochs 3 --d_model 64 --num_heads 4
python -m pytorch.train_gpt --epochs 5 --num_kv_heads 2   # GQA 对比
```

## 测试

```bash
python test_all.py                              # np_impl + modern_llm（约 74 项）
python -m np_impl.test                          # 基线：~41 项
python -m modern_llm.test                       # 优化实现：~33 项
python -m pytorch.test_all                      # PyTorch 数值与调度测试
```

测试入口在任意检查失败时返回非零退出码。PyTorch 套件额外验证完整前向与连续/Paged Cache、chunked prefix prefill 的逐步数值一致性，以及 Continuous Batching 的生成边界。

基准 JSON 包含 Python/PyTorch/设备环境、完整参数、Mean/P50/P95、Cache 逻辑占用与预分配容量等字段。CPU 或共享 CI runner 的耗时只适合冒烟验证，FlashAttention、AMP 和编译收益应在固定 CUDA GPU 与固定软件栈下重复采集。

## 环境要求

- Python 3.10+
- NumPy（所有模块）
- PyTorch 2.0+（训练 pipeline / GPU 基准需要，其他模块可选）

## 相关项目

- [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) — LLM 评估实验框架
- [react-agent](https://github.com/weihuaguo270-ops/react-agent) — Agent 运行时（另一条线）

## License

MIT

## 贡献与安全

见 [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md)。
