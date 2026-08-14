"""
Encoder-Decoder 完整架构 — 纯 NumPy 实现

整合:
  Encoder:   N 层 EncoderBlock（双向 Self-Attention）
  Decoder:   N 层 DecoderBlock（Self-Attention + Cross-Attention + FFN）

完整流程:
  输入句子 "I love you"
    ↓
  Positional Encoding
    ↓
  Encoder × N 层
    ↓
  encoder_output (原句子的上下文表示)
    ↓
  Decoder × N 层:
    每层: Self-Attention(因果掩码) → Cross-Attention(Q=decoder, KV=encoder) → FFN
    ↓
  输出投影 → 预测下一个词

import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
AutoRegressive 生成:
  Step 1: 输入 <BOS> → Decoder 用 Cross-Attention 看 encoder_output → 预测 "我"
  Step 2: 输入 <BOS> 我 → Decoder 用 Cross-Attention 看 encoder_output → 预测 "爱"
  Step 3: 输入 <BOS> 我爱 → ... → 预测 "你"
  Step 4: 输入 <BOS> 我爱你 → ... → 预测 <EOS>
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np
from .utils import layer_norm
from .multi_head_attention import MultiHeadAttention
from .cross_attention import MultiHeadCrossAttention
from .positional_encoding import sinusoidal_positional_encoding


class FFN:
    """Two-layer ReLU feed-forward network used by the reference decoder."""

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    def forward(self, x):
        """Project the final dimension through the reference FFN."""
        hidden = x @ self.W1 + self.b1
        hidden = np.maximum(0, hidden)
        return hidden @ self.W2 + self.b2
class DecoderBlock:
    """
    Decoder Block — 含 Cross-Attention
    与 transformer_block.py 中的 TransformerBlock 区别:
      多了 Cross-Attention 层：Q=decoder, K,V=encoder_output
    结构:
      输入 → Self-Attention(因果掩码) → +残差 → LayerNorm
           → Cross-Attention → +残差 → LayerNorm
           → FFN → +残差 → LayerNorm → 输出
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
    def forward(self, decoder_input, encoder_output):
        """
        参数:
            decoder_input: (seq_len_dec, d_model) — Decoder 当前层的输入
            encoder_output: (seq_len_enc, d_model) — Encoder 最终输出
        返回:
            (seq_len_dec, d_model)
        """
        # 子层 1: Self-Attention（因果掩码）
        attn_out = self.self_attn.forward(decoder_input, use_mask=True)
        x = decoder_input + attn_out
        x = layer_norm(x)
        # 子层 2: Cross-Attention（Q=decoder, K,V=encoder）
        cross_out = self.cross_attn.forward(x, encoder_output)
        x = x + cross_out
        x = layer_norm(x)
        # 子层 3: FFN
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x
class EncoderDecoder:
    """
    完整 Encoder-Decoder 架构
    用法:
        model = EncoderDecoder(d_model=8, num_heads=2, d_ff=16, num_layers=2)
        encoder_out = model.encode(source_sequence)
        output = model.decode(target_sequence, encoder_out)
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers=2):
        self.d_model = d_model
        self.encoders = []
        self.decoders = []
        for _ in range(num_layers):
            from .encoder_block import EncoderBlock
            # 不 import 在开头是为了避免循环依赖
            # 这里实际执行的是同一个 EncoderBlock

        # 为了简单，直接在 __init__ 里创建
        class _EncoderBlock:
            def __init__(self, d_model, num_heads, d_ff):
                self.attention = MultiHeadAttention(d_model, num_heads)
                self.ffn = FFN(d_model, d_ff)
            def forward(self, x):
                attn_out = self.attention.forward(x, use_mask=False)
                x = x + attn_out
                x = layer_norm(x)
                ffn_out = self.ffn.forward(x)
                x = x + ffn_out
                x = layer_norm(x)
                return x
        self.encoder_layers = [_EncoderBlock(d_model, num_heads, d_ff)
                               for _ in range(num_layers)]
        self.decoder_layers = [DecoderBlock(d_model, num_heads, d_ff)
                               for _ in range(num_layers)]
    def encode(self, x):
        """Encoder 前向 — 双向 Attention"""
        for layer in self.encoder_layers:
            x = layer.forward(x)
        return x
    def decode(self, decoder_input, encoder_output):
        """Decoder 前向 — 含 Cross-Attention"""
        x = decoder_input
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output)
        return x
# ============================================================
# 演示 — 完整 Encoder-Decoder 流程
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    d_model, num_heads, d_ff = 8, 2, 16
    num_layers = 2
    # 原句子: 3 个词（"I love you"）
    source = np.random.randn(3, d_model)
    # 已生成的译文: 2 个词（"我 爱"）
    target = np.random.randn(2, d_model)
    print("=" * 50)
    print("Encoder-Decoder 完整架构演示")
    print("=" * 50)
    print(f"\n配置: d_model={d_model}, heads={num_heads}, layers={num_layers}")
    print(f"原句子: {source.shape[0]} 个词")
    print(f"已生成: {target.shape[0]} 个词")
    model = EncoderDecoder(d_model, num_heads, d_ff, num_layers)
    # Step 1: Encoder 编码原句子
    encoder_output = model.encode(source)
    print(f"\nEncoder 输出: {encoder_output.shape[0]} 个词 × {encoder_output.shape[1]} 维")
    # Step 2: Decoder 用 Cross-Attention 生成译文
    decoder_output = model.decode(target, encoder_output)
    print(f"Decoder 输出: {decoder_output.shape[0]} 个词 × {decoder_output.shape[1]} 维")
    print("\n流程完成: Encoder 理解原句子 → Decoder 通过 Cross-Attention 逐词生成译文")
