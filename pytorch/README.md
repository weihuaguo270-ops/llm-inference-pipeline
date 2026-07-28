# PyTorch 2.x 性能后端 — 推理引擎与基准

`pytorch/` 是项目的性能工程主线，提供 **Prefill/Decode 推理引擎**、PyTorch 原生融合 Attention、GPU 基准与 Attention 变体训练验证。NumPy 目录负责数学可读性，PyTorch 路径优先使用框架的优化 kernel 和稳定内存布局。

默认推理配置：

- `F.scaled_dot_product_attention`：CPU 使用原生 GQA；CUDA 上保持压缩 Cache，并在读取时广播 KV heads，使 PyTorch 选择可用的 FlashAttention 或 memory-efficient kernel，避免原生 GQA 落到 math backend。
- `StaticKVCache`：按模型最大上下文预分配 K/V，decode 原地写入，不执行逐 token `torch.cat`。
- `torch.inference_mode()`：关闭 autograd 及额外版本计数开销。
- 可选 CUDA FP16/BF16 autocast、TF32 matmul precision 和 `torch.compile`。

## 推理链路模块

| 文件 | 角色 |
|------|------|
| `inference_engine.py` | Prefill、chunked suffix prefill 与 KV Cache Decode |
| `cache_backends.py` | Static / Contiguous / Paged Cache 可切换后端 |
| `gqa.py` | GQA + RoPE + `forward_with_cache` |
| `mla.py` | MLA 解压/吸收双路径（PyTorch） |
| `prefix_cache.py` | 共享前缀 KV 复用 |
| `paged_kv_cache.py` | Block 化 KV 存储与逐 block Attention 读取 |
| `speculative_decoding.py` | GPT 小模型 Spec Decoding 适配 |
| `continuous_batching.py` | 共享权重、独立 Cache 的分组批量调度 |

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

推理引擎可直接切换 Cache 后端：

```python
engine = InferenceEngine(
    model,
    cache_backend="static",       # 默认：预分配、原地更新
    amp_dtype="bfloat16",         # CUDA Ampere+ 推荐评测项
    compile_mode="reduce-overhead",
)
```

Static/Contiguous 后端直接进入 PyTorch SDPA；单 token decode 无需 attention mask，可使用框架选择的融合 kernel。Paged 后端逐 block 计算 attention score，仍属于可移植参考实现，不包含 vLLM 等系统的自定义分页 CUDA kernel。

Windows 官方 CUDA wheel 可能设置 `flash_sdp_enabled=True`，但仍未编译 FlashAttention；应以 `torch.backends.cuda.is_flash_attention_available()` 和 profiler 中的实际 kernel 为准。本仓库会在这种环境使用 memory-efficient/CUTLASS attention，而不是退回 math GQA。

Continuous Batching 按阶段与当前序列长度分组，因此每组会执行真实 batched forward；当前可移植调度器会在组批/拆批时复制 Cache。对极致服务吞吐的部署，应使用 block table 或融合 kernel 消除这部分搬运。

## 优化矩阵基准

```bash
python -m experiments.benchmark_pytorch_optimized --device cuda \
  --prompt_len 1024 --decode_steps 128 --d_model 1024 --num_layers 16 \
  --amp_dtype bfloat16 --compile_mode reduce-overhead \
  --json experiments/runs/pytorch_optimized.json
```

该基准在相同权重和输入上对照 eager+动态 Cache、SDPA+动态 Cache、SDPA+Static Cache、autocast、常驻 FP16/BF16 权重与 compile，报告 Prefill/Decode P50、相对加速、逻辑 Cache 占用、实际预分配和峰值设备内存。首次编译发生在 warmup，不计入稳态延迟。低延迟 Decode 通常应优先评估常驻低精度权重，逐 token 进入 autocast 可能产生明显上下文开销。

Windows 基准请求 `torch.compile` 时会通过 `vswhere` 自动发现 Visual Studio Build Tools、加载 `vcvars64.bat`，并在需要时以 Python UTF-8 模式自动重启，兼容本地化 MSVC 输出。如果自动发现失败，需要安装“使用 C++ 的桌面开发”工作负载（MSVC x64/x86 与 Windows SDK）。在其他 Python 入口中直接创建编译引擎时，应使用 `python -X utf8` 或设置 `PYTHONUTF8=1`。Linux/CUDA 需要与 PyTorch 版本匹配的 Triton/编译工具链。

环境诊断：

```bash
python -m pytorch.check_environment
```

当前官方 Windows CUDA wheel 不包含 Triton，因此原生 Windows 可以运行 CUDA eager/SDPA/Static Cache/AMP，但 CUDA `torch.compile` 需要经过兼容性验证的 `triton-windows`，或改在 WSL2/Linux 中运行。项目不会自动安装第三方 Triton 移植包。

NVIDIA Windows 环境应显式安装 CUDA wheel；普通 `pip install torch` 可能得到 CPU 构建：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.0+cu130 torchvision==0.27.0+cu130
```

## 实验记录

参见 [`experiments/runs/`](../experiments/runs/)。
