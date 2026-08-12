# Changelog

## Unreleased

### Fixed

- Run the focused CI regression through `python -m pytest` so repository modules resolve consistently on clean runners

### Documentation and evidence

- Replaced machine-specific CUDA reproduction paths with repository-relative commands
- Added the normalized RTX 4060 source result and `agent-release-evidence/v1` budget report

## 0.2.0 (2026-08-12)

### Release additions

- `agent-release-evidence/v1` conversion for latency, cache, and memory budgets
- Strict `--require-cuda` mode to prevent silent CPU fallback
- Reproducible CPU and CUDA 13.0 constraints
- RTX 4060 benchmark evidence with environment metadata
- Linux strict-device and performance-evidence CI checks

### Verified

- Portability and release-evidence regression: 7 passed
- PyTorch 2.13.0+cu130 on RTX 4060 Laptop GPU
- TTFT 2.68 ms and cached TPOT 2.71 ms for the recorded workload

### Changed
- 仓库与包重命名: `transformer-attention` → `llm-inference-pipeline`，与「LLM 推理链路优化」定位对齐
- PyTorch 性能路径默认切换为原生 SDPA/GQA + 预分配 Static KV Cache + `torch.inference_mode`
- 新增 CUDA FP16/BF16 autocast、matmul precision 与可选 `torch.compile`；Windows 缺少 MSVC 时提供前置诊断
- 新增 `benchmark_pytorch_optimized`，在相同权重上对照 eager、SDPA、动态/静态 Cache、AMP 与 compile
- Cache 指标拆分为逻辑已用字节和实际预分配字节；新增静态存储地址、容量保护、SDPA/eager 对齐测试
- Windows `torch.compile` 自动通过 `vswhere` 发现 Build Tools 并加载 `vcvars64.bat`，不再要求手动打开 Developer Shell
- CUDA GQA 根据平台 kernel 能力分派：避免 Windows wheel 的 native GQA math fallback，读取时广播 KV heads 以启用 memory-efficient SDPA
- Llama RMSNorm 切换为 PyTorch 原生 `F.rms_norm`，允许框架选择优化 kernel
- 新增 `python -m pytorch.check_environment`，区分 CUDA wheel、GPU、Flash 编译能力、Triton 与 MSVC 状态
- 修正 TTFT/TPOT 基准口径：Decode 计时不再包含 Prefill；新增 Mean/P50/P95、Cache 分配、环境元数据与 JSON 输出
- 测试失败现在返回非零退出码；新增完整前向与 Contiguous/Paged Cache、Prefix Cache、批量调度数值与边界验证
- 推理引擎新增可切换 Contiguous/Paged KV Cache 后端及 chunked suffix prefill；Paged 路径逐 block 读取 K/V
- Continuous Batching 改为共享模型权重、独立请求 Cache 的真实 batched Prefill/Decode
- Prefix Cache 改为不可变前缀快照，每次命中恢复快照并批量处理 suffix
- Speculative Decoding 区分 Target 调用次数代理指标和墙钟实测加速比
- GitHub 仓库描述与 topics 调整为「推理链路优化」定位；`pyproject.toml` description 同步
- **三阶段推理链路优化落地**：
  - Phase 1：`benchmark_prefill_decode`、`pytorch/inference_engine`、`pytorch/mla`（absorb）、`compare_decoding_pytorch`
  - Phase 2：`prefix_cache`、`paged_kv_cache`、`compare_kv_quant` / `compare_prefix_cache` / `compare_paged_cache`
  - Phase 3：`lookahead_decoding`、`medusa_heads`、`continuous_batching`、`compare_decode_strategies`
- **应用方向调整为推理链路优化**：README / 子目录文档 / pyproject 描述以 Prefill-Decode 链路与 Cache/Decode 优化为主线重组
- 测试输出改用 `[PASS]` / `[FAIL]` + `console_io.safe_print`；CI 增加 Windows × 3.10/3.11

## 0.1.0 (2026-07-12)
### Changed
- 仓库更名: attention-from-scratch → transformer-attention
- README 重写为中文专业版
- 包名更新: attention-from-scratch → transformer-attention

### Added
- GitHub Actions CI 工作流
- CI Badge 到 README
