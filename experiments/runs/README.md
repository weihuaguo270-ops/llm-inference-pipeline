# 实验记录系统

每次训练的结果记录在 `runs/` 下。

## 目录命名规则

| 来源 | 命名格式 | 示例 |
|------|---------|------|
| 手动记录 | `legacy_{序号}_{描述}/` | `legacy_001_baseline/` |
| 自动记录 | `{时间戳}_{tag}_{参数}_auto/` | `20260712_140332_lr-test_lr0.001_auto/` |

## 对比工具

```bash
python experiments/runs/compare.py
```

支持按 tag、参数过滤，横向对比不同实验的 loss 曲线和最终指标。

## 发布性能证据

以下文件保存已经用于 Release 的、带环境元数据且使用仓库相对路径的结果：

- `pytorch_cuda_4060_20260812_prefill_decode.json`：固定 RTX 4060 负载的原始 TTFT/TPOT 结果；
- `agent_release_performance_20260812.json`：`agent-release-evidence/v1` 预算判断。

证据仅适用于文件记录的硬件、软件版本和负载，不代表其他 GPU 或线上流量收益。
