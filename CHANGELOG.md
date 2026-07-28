# Changelog

## Unreleased

### Changed
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
