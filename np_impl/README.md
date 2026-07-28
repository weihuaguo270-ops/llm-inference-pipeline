# 推理基线：Attention + KV Cache

`np_impl/` — 纯 NumPy 实现。提供 MHA 与 KV Cache 基线，用于理解 Decode 阶段的计算与显存瓶颈，再对照 `modern_llm/` 中的优化手段。

## 文件说明

| 文件 | 推理链路角色 |
|------|-------------|
| `attention.py` | Attention 计算细节：QKV 投影、缩放点积、因果掩码 |
| `multi_head_attention.py` | MHA 基线：拆分/合并、可切换 RoPE（KV Cache 开销参照） |
| `kv_cache.py` | **Decode 基础加速**：有/无 Cache 的计算量对比 |
| `positional_encoding.py` | Sinusoidal 位置编码 |
| `rotary.py` | RoPE：长度外推（推理常用） |
| `transformer_block.py` | 原始 Decoder Block（Post-Norm + ReLU FFN） |
| `cross_attention.py` | 编码器-解码器交叉注意力 |
| `encoder_block.py` | Encoder Block |
| `encoder_decoder.py` | Encoder-Decoder 完整串联 |
| `utils.py` | 公共工具函数 |
| `test.py` | 36+ 项测试 |

## 运行测试

```bash
python -m np_impl.test
```

## 阅读顺序

```
想理解什么 → 看哪个文件
Attention 计算瓶颈 → attention.py
MHA + KV Cache 基线 → multi_head_attention.py → kv_cache.py
位置编码（RoPE 外推） → rotary.py
完整 Block 结构 → transformer_block.py
```
