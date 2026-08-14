"""
Grouped Query Attention (GQA) — 分组查询注意力

GQA 是 MHA 和 MQA 的折中方案，Llama 2/3、Mistral、Qwen 等主流模型全在用。

KV Cache 对比（num_heads=32, d_k=128, seq_len=4096, FP16）：
  MHA (32 KV heads): 64.0 MB
  GQA (8 KV heads):  16.0 MB  → Llama 3 70B
  GQA (4 KV heads):   8.0 MB  → Mistral 7B
  MQA (1 KV head):    2.0 MB  → Falcon

工作原理：
  1. Q 投影到 num_heads 个头，K/V 投影到 num_kv_heads 个头（更少）
  2. K/V 头通过 np.repeat 广播以匹配 Q 头数
  3. 每个 Q 头与自己组内的 K/V 头做 Attention
"""
import numpy as np
from .utils import softmax


class GroupedQueryAttention:
    """
    分组查询注意力

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        num_kv_heads: K/V 头数（必须能整除 num_heads）
        use_rope: 是否对 Q/K 应用 RoPE
        max_seq_len: 最大序列长度（RoPE 需要）
    """
    def __init__(self, d_model, num_heads, num_kv_heads, use_rope=False, max_seq_len=128):
        assert num_heads % num_kv_heads == 0, \
            f"num_heads({num_heads}) 必须能被 num_kv_heads({num_kv_heads}) 整除"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.d_kv = self.d_k * num_kv_heads
        self.Wk = np.random.randn(d_model, self.d_kv) * 0.01
        self.Wv = np.random.randn(d_model, self.d_kv) * 0.01
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        if use_rope:
            from .rotary import precompute_rotary_frequencies
            self._cos_table, self._sin_table = precompute_rotary_frequencies(
                self.d_k, max_seq_len=max_seq_len
            )

    def forward(self, x, use_mask=True, positions=None):
        """Run GQA on ``(sequence, d_model)`` states.

        Query heads share each K/V head within a group; an upper-triangular mask
        prevents future-token attention when ``use_mask`` is true.
        """
        seq_len = x.shape[0]

        # 1. 投影
        Q = x @ self.Wq   # (seq_len, d_model)
        K = x @ self.Wk   # (seq_len, d_kv)
        V = x @ self.Wv   # (seq_len, d_kv)

        # 2. 拆头
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_kv_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_kv_heads, self.d_k).transpose(1, 0, 2)

        # 3. RoPE
        if self.use_rope:
            from .rotary import apply_rotary
            if positions is None:
                positions = np.arange(seq_len)
            for h in range(self.num_kv_heads):
                K[h] = apply_rotary(K[h], self._cos_table, self._sin_table, positions)
            for h in range(self.num_heads):
                Q[h] = apply_rotary(Q[h], self._cos_table, self._sin_table, positions)

        # 4. 重复 K/V 以匹配 Q 头数
        num_repeats = self.num_heads // self.num_kv_heads
        K = np.repeat(K, num_repeats, axis=0)
        V = np.repeat(V, num_repeats, axis=0)

        # 5. Attention
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        if use_mask:
            scores += np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        attn_weights = softmax(scores)
        head_outputs = attn_weights @ V

        # 6. 合并 + 输出投影
        combined = head_outputs.transpose(1, 0, 2).reshape(seq_len, -1)
        return combined @ self.Wo
