# 推理链路对比实验

对 LLM 推理链路上各优化手段的横向对比与量化基准。覆盖 Cache 压缩、Decode 加速、Prefill/Decode 分离、服务化调度等维度。

## 实验列表

| 阶段 | 实验 | 文件 | 对比内容 |
|------|------|------|---------|
| P1 | Prefill/Decode 基准 | `benchmark_prefill_decode.py` | TTFT、TPOT、KV Cache vs 无 Cache |
| P1 | PyTorch Spec Decoding | `compare_decoding_pytorch.py` | 未训练 GPT 小模型的调用次数与墙钟耗时 |
| P1 | Attention 变体对比 | `compare_attention.py` | MHA vs GQA vs MLA：Cache / 参数量 |
| P1 | 性能基准 | `benchmark_attention.py` | MHA vs GQA vs MLA 延迟 / 吞吐 |
| P1 | PyTorch 优化矩阵 | `benchmark_pytorch_optimized.py` | eager / SDPA / Static Cache / AMP / compile |
| P1 | MLA 吸收微基准 | `benchmark_mla_absorb.py` | 解压 vs 吸收路径 + CSV |
| P2 | Prefix Cache | `compare_prefix_cache.py` | 共享前缀 TTFT 节省 |
| P2 | Paged KV Cache | `compare_paged_cache.py` | Block 利用率 / 体积 |
| P2 | KV 量化 | `compare_kv_quant.py` | FP32 vs INT8 体积 / 误差 |
| P3 | Decode 策略统一对比 | `compare_decode_strategies.py` | 标准 / Spec / Lookahead / Medusa |
| P3 | Continuous Batching | `compare_continuous_batching.py` | 多请求交错调度 |
| — | KV Cache 策略 | `compare_cache.py` | 完整 vs StreamingLLM |
| — | 解码策略 (NumPy) | `compare_decoding.py` | Spec Decoding gamma / 接受率 |
| — | 超参数对比 | `compare_training.py` | 训练曲线 |

## Phase 1 — 链路基准与核心优化验证

```bash
# TTFT / TPOT
python -m experiments.benchmark_prefill_decode
python -m experiments.benchmark_prefill_decode --device cuda --prompt_len 512
python -m experiments.benchmark_prefill_decode --cache_backend paged \
  --json experiments/runs/prefill_decode.json

# PyTorch GPT Spec Decoding
python -m experiments.compare_decoding_pytorch \
  --json experiments/runs/speculative.json

# Attention 变体延迟
python -m experiments.benchmark_attention --device cuda --seq_len 4096

# 同权重 PyTorch 2.x 执行路径对照
python -m experiments.benchmark_pytorch_optimized --device cuda \
  --prompt_len 1024 --amp_dtype bfloat16 --compile_mode reduce-overhead \
  --json experiments/runs/pytorch_optimized.json
```

## Phase 2 — Cache 管理层

```bash
python -m experiments.compare_prefix_cache
python -m experiments.compare_paged_cache
python -m experiments.compare_kv_quant
```

## Phase 3 — Decode 加速与服务化

```bash
python -m experiments.compare_decode_strategies
python -m experiments.compare_continuous_batching
```

## 指标与结果格式

`benchmark_prefill_decode` 将 Cache 状态准备移出 Decode 计时区间，TTFT 对应产生首 token logits 的 Prefill，TPOT 对应已有 Cache 后的单 token 前向。输出包含 Mean、P50、P95、Cache 实际分配和可选的峰值设备内存。

`compare_decoding_pytorch` 区分 Target 调用次数比（代理指标）与真实墙钟加速比。当前使用训练前随机权重的小模型，只用于验证执行链路与 rejection sampling，不能作为模型质量结论。

使用 `--json` 时会记录运行参数、Python/PyTorch/设备环境及结构化指标，便于固定环境下重复对照。

## MLA 吸收路径微基准

```bash
python -m experiments.benchmark_mla_absorb
python -m experiments.benchmark_mla_absorb --d_model 512 --d_c 128 --seq_len 256 \
  --csv experiments/runs/mla_absorb_20260714.csv
```

输出含：解压/吸收全序列 decode 耗时、加速比、前缀步数值误差、相对 MHA 的 cache 压缩比。

## 实验记录

每次训练自动存档至 `runs/` 目录：

```bash
python experiments/runs/compare.py
```
