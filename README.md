# Transformer Attention

[![CI](https://github.com/weihuaguo270-ops/transformer-attention/actions/workflows/test.yml/badge.svg)](https://github.com/weihuaguo270-ops/transformer-attention/actions/workflows/test.yml) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**NumPy/PyTorch 手写 Attention** — 从 2017 Transformer 到 GQA、Llama Block、DeepSeek MLA（含 absorb 路径）、Speculative Decoding、Attention Sinks 的对照实现。教学规模，不是预训练工程。

## 快速开始

```bash
pip install -e .
# 可选：训练相关依赖
pip install -e ".[pytorch]"

# 运行全部测试
python test_all.py
```

## 架构

项目追踪原始 Transformer 之后的两条独立演进路线：

```
原始 Transformer（2017）
       /                     Llama 路线               DeepSeek 路线
(GQA + Pre-Norm +       (MLA 潜空间压缩)
 SwiGLU + RoPE)
    2023-2024               2024
```

| 目录 | 覆盖内容 |
|------|----------|
| [`np_impl/`](np_impl/README.md) | 原始 Transformer（2017）NumPy 实现 |
| [`modern_llm/`](modern_llm/README.md) | GQA、Llama Block、MLA、Spec Decoding、StreamingLLM |
| [`pytorch/`](pytorch/README.md) | PyTorch GQA + Llama Block + GPT 训练 pipeline + 实验记录 |
| [`experiments/`](experiments/README.md) | Attention 变体 / KV 策略 / 解码 / 超参数对比实验 |

## 项目结构

```
transformer-attention/
│
├── np_impl/                    # 原始 Transformer（NumPy）
│   ├── attention.py            单头 Self-Attention + 因果掩码
│   ├── multi_head_attention.py 多头注意力（MHA）
│   ├── kv_cache.py             KV Cache 推理加速
│   ├── positional_encoding.py  Sinusoidal 位置编码
│   ├── rotary.py               RoPE 旋转位置编码
│   ├── transformer_block.py    原始 Decoder Block（Post-Norm + ReLU）
│   ├── cross_attention.py      编码器-解码器交叉注意力
│   ├── encoder_block.py        Encoder Block
│   ├── encoder_decoder.py      Encoder-Decoder 完整架构
│   ├── utils.py                softmax / LayerNorm / head 拆合等
│   └── test.py                 ~41 项冒烟测试（含 RoPE）
│
├── modern_llm/                 # 现代 LLM 架构（2023-2024）
│   ├── gqa.py                  Grouped Query Attention
│   ├── llama_block.py          Llama Decoder Block（Pre-Norm + RMSNorm + SwiGLU）
│   ├── mla.py                  Multi-head Latent Attention（DeepSeek V2/V3）
│   ├── speculative_decoding.py Speculative Decoding 推理加速
│   ├── attention_sinks.py      StreamingLLM / Attention Sinks
│   ├── rotary.py               RoPE（独立模块）
│   ├── utils.py                共享工具
│   └── test.py                 ~33 项冒烟测试
│
├── experiments/                # 对比实验
│   ├── compare_attention.py    MHA vs GQA vs MLA：缓存/参数量对比
│   ├── compare_cache.py        完整缓存 vs StreamingLLM 质量/节省
│   ├── compare_decoding.py     标准 vs Spec Decoding 加速比
│   ├── compare_training.py     超参数训练效果对比
│   ├── benchmark_attention.py  前向耗时微基准（CPU/CUDA）
│   ├── benchmark_mla_absorb.py 解压 vs 吸收路径微基准（导出 CSV）
│   └── runs/                   实验记录 + mla_absorb_*.csv
│
├── pytorch/                    # PyTorch 训练 pipeline
│   ├── gqa.py                  GQA + RoPE（PyTorch nn.Module）
│   ├── llama_block.py          RMSNorm + SwiGLU + 完整 GPT 模型
│   ├── train_gpt.py            训练脚本（交互式/命令行，自动记录实验）
│   ├── train_transformer.py    原始 Transformer 对照训练
│   ├── data.py                 TinyStories 数据加载 + 词表构建
│   ├── test_all.py             PyTorch 侧冒烟测试（~25 项，独立于根目录入口）
│   └── ...                     各模块的 PyTorch 实现
│
├── test_all.py                 统一测试入口（np_impl + modern_llm，约 74 项）
└── pyproject.toml
```

## 核心实现

### MHA — Multi-Head Attention（2017）

Transformer 的基础。Q、K、V 投影到 d_model，拆分为 n_heads，缩放点积注意力计算。

**KV Cache**（`np_impl/kv_cache.py`）：自回归解码时缓存 K/V 张量，避免重复计算，每步复杂度从 O(n²·d) 降至 O(n·d)。

### GQA — Grouped Query Attention（2023）

Llama 2/3、Mistral、Qwen 等主流模型采用的方案。减少 K/V 头数同时保留 Q 头数，在 MHA 质量和 MQA 效率间取得平衡。

| 变体 | KV 头数 | KV Cache（32h, 4096seq, FP16） | 代表模型 |
|------|---------|-------------------------------|---------|
| MHA | 32 | 64.0 MB | 原始 Transformer |
| GQA | 8 | 16.0 MB | Llama 3 70B |
| GQA | 4 | 8.0 MB | Mistral 7B |
| MQA | 1 | 2.0 MB | Falcon |

### Llama Decoder Block

与原始 Transformer 的 5 项关键差异：

| 维度 | 原始（2017） | Llama（2023） | 改进原因 |
|------|-------------|---------------|----------|
| 归一化位置 | Post-Norm（子层后） | Pre-Norm（子层前） | 梯度直通残差路径，深层训练稳定 |
| 归一化类型 | LayerNorm（μ+σ+γ+β） | RMSNorm（仅σ） | 计算快 30%，效果持平 |
| FFN 激活 | ReLU | SwiGLU（门控） | 可学习的选择性激活 |
| Attention | MHA（32 KV heads） | GQA（4-8 KV heads） | 节省 75-87% KV Cache |
| 位置编码 | Sinusoidal PE（加法） | RoPE（旋转） | 支持长度外推 |

```
# 原始 Transformer Block（Post-Norm）
x → MHA → +残差 → LayerNorm → FFN(ReLU) → +残差 → LayerNorm → 输出

# Llama Decoder Block（Pre-Norm）
x → RMSNorm → GQA(RoPE) → +残差 → RMSNorm → SwiGLU → +残差 → 输出
```

### MLA — Multi-head Latent Attention（2024）

DeepSeek V2/V3 用的思路：把 K/V 压到低维潜空间，KV Cache 明显小于 MHA。

```
MHA:   K = h · W_K,       缓存 K-V（d_model 维）
MLA:   c = h · W_DKV,     缓存 c（d_c 维, d_c << d_model）
       K = c · W_UK,       V = c · W_UV（从压缩缓存解压）
```

**吸收矩阵技巧（已实现，推理可用）：**
```
Q_h · (C · W_UK_h) = (Q_h · W_UK_h^T) · C
attn · (C · W_UV_h) = (attn · C) · W_UV_h
```
`forward_with_cache(..., use_absorb=True)` 走吸收路径；`use_absorb=False` 为逐步解压对照。单测验证两者数值对齐（max|Δ| < 1e-6）。

实际参数（DeepSeek V2, d_model=5120，来自论文量级说明）：
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

> 说明：加速比随序列与压缩比变化；仓库训练脚本仍为 TinyStories 级小模型，非大规模预训练结果。

### Speculative Decoding

小模型（Draft Model）先生成 K 个候选 token，目标模型批量验证并用 rejection sampling 保持目标分布。当前仓库是 NumPy 算法演示；尚未用真实 draft/target 模型验证端到端质量等价或 2-3x 加速。

### Attention Sinks / StreamingLLM

保留最近 tokens + 开头几个 tokens（attention sink），处理远超训练长度的序列，无需完整 KV 重算。

## 训练与实验

PyTorch pipeline 提供完整的 GPT 训练流程：

```bash
# 训练 baseline
python -m pytorch.train_gpt --epochs 3 --d_model 64 --num_heads 4

# GQA 对比训练
python -m pytorch.train_gpt --epochs 5 --num_kv_heads 2

# 实验对比
python experiments/runs/compare.py
```

实验自动记录配置快照、loss 曲线和模型 checkpoint。

## 测试与基准

```bash
# 运行 np_impl + modern_llm（约 74 项）
python test_all.py

# 分别运行
python -m np_impl.test         # 原始 Transformer：~41 项（含 RoPE）
python -m modern_llm.test      # 现代 LLM：~33 项
python -m pytorch.test_all     # PyTorch 侧独立冒烟：~25 项

# 实际前向耗时微基准（CI 使用短序列；本地可加长 / 换 GPU）
python -m experiments.benchmark_attention
python -m experiments.benchmark_attention --device cuda --seq_len 2048

# MLA 解压 vs 吸收（NumPy；写出 CSV）
python -m experiments.benchmark_mla_absorb --seq_len 256 --d_model 512 --d_c 128
```

## 环境要求

- Python 3.10+
- NumPy（所有模块）
- PyTorch 2.0+（训练 pipeline / benchmark 需要，其他模块可选）

## 相关项目

- [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) — LLM 评估实验框架
- [react-agent](https://github.com/weihuaguo270-ops/react-agent) — Agent 运行时（另一条线）

## License

MIT

## 贡献与安全

见 [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md)。
