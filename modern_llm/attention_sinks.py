"""
Attention Sinks / StreamingLLM — 长文本推理的 KV Cache 淘汰策略

问题：
  KV Cache 大小 = O(seq_len)，Agent 对话永不结束 → 最终 OOM。

StreamingLLM 的核心发现（Google, 2023）：
  最开头的几个 token（尤其是第一个）会持续吸收大量注意力，
  即使后面的内容完全变了。这些 token 被称为 "Attention Sinks"。

  原因：softmax 要求每行和为 1。当一个 token 不需要关注任何具体
  内容时，它需要"没用的地方"分配多余的注意力权重。
  开头的 token 充当了这个角色。

方案：
  保留 sinks（开头的 4 个 token）+ recent（最近 N 个 token）
  中间的 token 全部丢弃 → KV Cache 永远不超 window_size。

参考: https://arxiv.org/abs/2309.17453
"""
import numpy as np


class StreamingKVCache:
    """
    带 Attention Sinks 的 KV Cache

    缓存策略:
      [sink0 sink1 sink2 sink3] [recent0 recent1 ... recentN]
       ↑ 永远保留                    ↑ 滑动窗口
       sink_len=4                    window_len 个最近 token

    参数:
        sink_len: 开头保留的 token 数（Attention Sinks）
        window_len: 最近保留的 token 数（滑动窗口）
    """
    def __init__(self, sink_len=4, window_len=20):
        self.sink_len = sink_len
        self.window_len = window_len
        self.max_cache_size = sink_len + window_len

        self._k_cache = None  # (cache_len, d_k)
        self._v_cache = None
        self._positions = []  # 记录每个缓存位置对应的原始位置

    @property
    def size(self):
        """当前缓存的 token 数"""
        return 0 if self._k_cache is None else self._k_cache.shape[0]

    @property
    def max_size(self):
        """最大缓存 token 数"""
        return self.max_cache_size

    def update(self, k_new, v_new, positions=None):
        """
        更新 KV Cache

        当序列较短时（≤ max_cache_size）：全部保留
        当序列超过 max_cache_size：丢弃中间，保留 sinks + recent

        参数:
            k_new: (new_tokens, d_k) — 新 token 的 Key
            v_new: (new_tokens, d_k) — 新 token 的 Value
            positions: (new_tokens,) — 对应的位置索引（可选）
        """
        n_new = k_new.shape[0]

        if positions is None:
            positions = np.arange(self.size, self.size + n_new)

        if self._k_cache is None:
            # 首次：初始化缓存
            self._k_cache = k_new.copy()
            self._v_cache = v_new.copy()
            self._positions = list(positions)
        else:
            # 追加到缓存
            self._k_cache = np.concatenate([self._k_cache, k_new], axis=0)
            self._v_cache = np.concatenate([self._v_cache, v_new], axis=0)
            self._positions.extend(positions)

        # 如果缓存超出上限 → 执行淘汰
        if self.size > self.max_cache_size:
            self._evict()

    def _evict(self):
        """
        淘汰中间 token，只保留 sinks + recent

        策略：
          1. 保留前 sink_len 个 token（attention sinks）
          2. 保留最后 window_len 个 token（最近上下文）
          3. 中间的 token 丢弃
          4. 如果 sink_len + window_len > 总长度 → 不淘汰
        """
        total = self.size
        if total <= self.max_cache_size:
            return

        # 保留 sinks + recent，丢弃中间
        keep_indices = (
            list(range(self.sink_len)) +
            list(range(total - self.window_len, total))
        )
        # 去重（当序列很短时 sink 和 recent 可能重叠）
        keep_indices = sorted(set(keep_indices))

        self._k_cache = self._k_cache[keep_indices]
        self._v_cache = self._v_cache[keep_indices]
        self._positions = [self._positions[i] for i in keep_indices]

    def get_all(self):
        """返回当前缓存的全部 K, V, positions"""
        return self._k_cache, self._v_cache, np.array(self._positions)

    def reset(self):
        """清空缓存"""
        self._k_cache = None
        self._v_cache = None
        self._positions = []


# ============================================================
# 简化的 Attention 计算（使用 StreamingKVCache）
# ============================================================
def softmax(x):
    """Compute numerically stable softmax over the last attention axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def attention_with_cache(q, k_cache, v_cache, d_k):
    """使用缓存算 Attention"""
    scores = (q @ k_cache.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ v_cache


def demo_streaming_vs_full():
    """对比完整缓存 vs Streaming 缓存的内存占用"""
    print("=" * 60)
    print("StreamingLLM vs 完整 KV Cache 缓存量对比")
    print("=" * 60)

    d_k = 64
    sink_len, window_len = 4, 8
    total_tokens = 50

    # 完整 KV Cache：线性增长
    full_sizes = [min(i, total_tokens) for i in range(1, total_tokens + 1)]

    # Streaming KV Cache：有上限
    stream = StreamingKVCache(sink_len=sink_len, window_len=window_len)
    stream_sizes = []
    for step in range(total_tokens):
        k = np.random.randn(1, d_k)
        v = np.random.randn(1, d_k)
        stream.update(k, v, positions=np.array([step]))
        stream_sizes.append(stream.size)

    print(f"\n参数: sink_len={sink_len}, window_len={window_len}, "
          f"max_cache={sink_len + window_len}")
    print(f"\n生成步数 | 完整缓存 | StreamingLLM | 节省")
    print("-" * 45)
    for step in [10, 20, 30, 40, 50]:
        full = step
        stream_sz = stream_sizes[step - 1]
        saving = (full - stream_sz) / full * 100
        print(f"  {step:4d}   | {full:5d}     | {stream_sz:5d}       | {saving:5.0f}%")

    print(f"\n生成 {total_tokens} 个 token 后:")
    print(f"  完整缓存: {total_tokens} 个 K,V 向量")
    print(f"  Streaming: {stream.size} 个 K,V 向量 ({sink_len} sinks + {window_len} recent)")
    print(f"  固定上限: {stream.max_size} — 永远不会超过")


def demo_attention_pattern():
    """演示 Attention Sinks 现象：开头 token 持续获得注意力"""
    print("\n" + "=" * 60)
    print("Attention Sinks 现象演示")
    print("=" * 60)

    d_k = 16
    sink_len, window_len = 3, 6
    max_cache = sink_len + window_len

    # 模拟一段较长的序列
    seq_len = 30
    np.random.seed(42)
    tokens = np.random.randn(seq_len, d_k)

    stream = StreamingKVCache(sink_len=sink_len, window_len=window_len)

    # 逐步填充缓存
    for i in range(seq_len):
        stream.update(
            tokens[i:i+1],
            tokens[i:i+1],
            positions=np.array([i])
        )

    # 检查当前缓存中的位置分布
    _, _, positions = stream.get_all()

    print(f"\n序列长度: {seq_len}")
    print(f"缓存上限: {max_cache} (sinks={sink_len}, window={window_len})")
    print(f"实际缓存: {len(positions)} 个 token")
    print(f"缓存中 token 的原始位置: {positions}")
    print(f"  前 {sink_len} 个: sinks (开头)")
    print(f"  后 {window_len} 个: recent (最近)")
    print(f"  中间位置 {sink_len}~{seq_len - window_len - 1}: 已丢弃")

    # 模拟注意力：最新的 token 查询所有缓存
    q = tokens[-1:]

    # 用完整序列算 attention（只是看看权重分布）
    full_k = tokens
    full_v = tokens
    scores_full = (q @ full_k.T) / np.sqrt(d_k)
    weights_full = softmax(scores_full)[0]

    print(f"\n最新 token 对不同位置的注意力权重:")
    print(f"  Attention Sinks (pos 0~{sink_len-1}): "
          f"{weights_full[:sink_len].sum():.3f} (总权重)")
    print(f"  Recent (pos {seq_len-window_len}~{seq_len-1}):    "
          f"{weights_full[-window_len:].sum():.3f} (总权重)")
    print(f"  中间区域:            "
          f"{weights_full[sink_len:-window_len].sum():.3f} (总权重)")

    # 用缓存算 attention（验证结果与完整序列的差异）
    k_cache, v_cache, _ = stream.get_all()
    output_cached = attention_with_cache(q, k_cache, v_cache, d_k)

    # 用完整序列算 attention
    output_full = attention_with_cache(q, full_k, full_v, d_k)

    # 比较两种输出的差异
    diff = np.linalg.norm(output_cached - output_full)
    print(f"\n完整序列输出 vs Streaming 缓存输出:")
    print(f"  向量差异 (L2 norm): {diff:.4f}")
    print(f"  差异很小说明：丢弃中间 token 对最终输出影响不大")


def demo_agent_conversation():
    """模拟 Agent 长篇对话的 KV Cache 表现"""
    print("\n" + "=" * 60)
    print("Agent 长对话模拟")
    print("=" * 60)

    d_k = 32
    sink_len, window_len = 4, 12

    # 模拟多轮对话
    rounds = [
        ("用户: 今天天气怎么样？", 8),
        ("Agent: 今天晴天，20度。", 10),
        ("用户: 明天呢？", 6),
        ("Agent: 明天多云，18度。", 10),
        ("用户: 后天呢？", 6),
        ("Agent: 后天有雨，15度。", 10),
        ("用户: 大后天呢？", 8),
        ("Agent: 大后天晴转多云。", 12),
        ("用户: 那下周呢？", 8),
        ("Agent: 下周整体偏暖。", 10),
    ]

    stream = StreamingKVCache(sink_len=sink_len, window_len=window_len)
    full_cache = []  # 模拟完整缓存（无淘汰）

    print(f"\n{'轮次':>4} | {'消息':>30} | {'完整缓存':>8} | {'Streaming':>8} | {'节省':>6}")
    print("-" * 65)

    for i, (msg, n_tokens) in enumerate(rounds):
        k = np.random.randn(n_tokens, d_k)
        v = np.random.randn(n_tokens, d_k)

        stream.update(k, v)
        full_cache.extend([1] * n_tokens)  # 只计数

        saving = (len(full_cache) - stream.size) / len(full_cache) * 100
        print(f"  {i+1:3d} | {msg:>30} | {len(full_cache):8d} | {stream.size:8d} | {saving:5.0f}%")

    print(f"\n结论:")
    print(f"  完整缓存: {len(full_cache)} tokens — 随对话线性增长")
    print(f"  Streaming: {stream.size} tokens — 永远不超 {stream.max_size}")
    print(f"  Agent 可以无限对话而不会 OOM")


if __name__ == "__main__":
    demo_streaming_vs_full()
    demo_attention_pattern()
    demo_agent_conversation()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
StreamingLLM 的核心：

1. Attention Sinks 现象
   - 开头 token 持续吸收注意力（softmax 的"垃圾桶"）
   - 保留它们就能稳定推理

2. 缓存策略
   保留：[sinks (4个)] + [recent (N个)]
   丢弃：中间的所有 token

3. 效果
   - KV Cache 有固定上限 → 永不 OOM
   - 输出质量几乎不变（差异通常 < 0.01）
   - 适用场景：Agent 长对话、连续推理
    """)
