# 贡献指南（Contributing）

**推理链路优化**对照实现仓库。欢迎补充优化手段、修正实现错误、扩展基准实验。

```bash
pip install -e .
python test_all.py
```

贡献方向示例：
- 新的 Cache 策略或 Decode 加速算法（含单测与对比实验）
- 现有优化路径（如 MLA absorb、StreamingLLM）的数值对齐或性能基准
- 文档与实验记录完善

请勿提交大型 checkpoint / 数据集文件。
