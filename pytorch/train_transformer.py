"""
PyTorch 版完整 Transformer — 含训练流程

把现有组件组装成一个可训练的翻译模型：
  Encoder → Decoder → LM Head → Softmax → 预测下一个词

训练一个极简的"序列复制"任务：
  输入: [0, 1, 2, 3]  →  输出: [1, 2, 3, 4]
  让模型学习"每个元素+1"的映射
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import layer_norm
from multi_head_attention import MultiHeadAttention
from cross_attention import MultiHeadCrossAttention
from positional_encoding import sinusoidal_positional_encoding


# ============================================================
# 1. 完整 Transformer 模型（含 LM Head）
# ============================================================

class FFN(nn.Module):
    """Training example's position-wise feed-forward sublayer."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Apply the training model's feed-forward transform."""
        return self.W2(F.relu(self.W1(x)))


class EncoderLayer(nn.Module):
    """Training example encoder layer with bidirectional self-attention."""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        """Update source states through attention and FFN residual blocks."""
        attn_out = self.self_attn(x, use_mask=False)
        x = x + attn_out
        x = layer_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    """Training example decoder layer with causal and cross-attention blocks."""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, encoder_output):
        """Update target states using causal self-attention and source context."""
        attn_out = self.self_attn(x, use_mask=True)
        x = x + attn_out
        x = layer_norm(x)

        cross_out = self.cross_attn(x, encoder_output)
        x = x + cross_out
        x = layer_norm(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x


class Transformer(nn.Module):
    """
    可训练的 Encoder-Decoder Transformer

    完整流程: Embedding → Positional Encoding
             → Encoder × N
             → Decoder × N（含 Cross-Attention）
             → LM Head → vocab 概率
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers=2):
        super().__init__()

        # Token Embedding: 把词 ID 映射为向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Encoder / Decoder 层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        # LM Head: 把向量映射回词表大小
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        """
        参数:
            src_ids: (seq_src,) — 原句子的 token ID 序列
            tgt_ids: (seq_tgt,) — 目标句子的 token ID 序列（已生成部分）

        返回:
            logits: (seq_tgt, vocab_size) — 每一步对词表每个词的分数
        """
        d_model = self.token_embedding.embedding_dim

        # --- Encoder ---
        src_emb = self.token_embedding(src_ids)  # (seq_src, d_model)
        # 加位置编码
        pe = sinusoidal_positional_encoding(src_ids.shape[0], d_model)
        src_emb = src_emb + pe

        x = src_emb
        for layer in self.encoder_layers:
            x = layer(x)
        encoder_output = x

        # --- Decoder ---
        tgt_emb = self.token_embedding(tgt_ids)  # (seq_tgt, d_model)
        pe = sinusoidal_positional_encoding(tgt_ids.shape[0], d_model)
        tgt_emb = tgt_emb + pe

        x = tgt_emb
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)

        # --- LM Head ---
        logits = self.lm_head(x)  # (seq_tgt, vocab_size)
        return logits


# ============================================================
# 2. 训练演示
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 造一个极简词表: 数字 0~9 共 10 个词
    vocab_size = 10
    d_model = 16
    num_heads = 2
    d_ff = 32

    model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers=2)

    # 训练数据: 直接预测下一个元素（Teacher Forcing）
    # Decoder 看到的是"从第2个开始的正确输出"
    # Loss 算的是"每一步预测下一个词的准确率"
    train_data = [
        (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 1, 2, 3])),  # dec_in=[1,0,1,2,3] → 预测 [0,1,2,3]?
        (torch.tensor([1, 2, 3, 4]), torch.tensor([2, 1, 2, 3, 4])),
        (torch.tensor([2, 3, 4, 5]), torch.tensor([3, 2, 3, 4, 5])),
        (torch.tensor([3, 4, 5, 6]), torch.tensor([4, 3, 4, 5, 6])),
        (torch.tensor([4, 5, 6, 7]), torch.tensor([5, 4, 5, 6, 7])),
        (torch.tensor([5, 6, 7, 8]), torch.tensor([6, 5, 6, 7, 8])),
        (torch.tensor([0, 1]), torch.tensor([1, 0, 1])),
        (torch.tensor([7, 8]), torch.tensor([8, 7, 8])),
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 50)
    print("训练 Encoder-Decoder Transformer")
    print("任务: 输入[idx1, idx2, ...] → 预测 [idx1+1, idx2+1, ...]")
    print("注意: 这是 Teacher Forcing 训练，不是自回归生成")
    print("      LM Head 将 (seq, d_model) 映射到 (seq, vocab_size)")
    print("=" * 50)
    print()

    n_epochs = 200
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_acc = 0.0
        n_total = 0
        for src, tgt_full in train_data:
            # tgt_full[0] 是 Decoder 的起始标记
            # tgt_full[1:] 是目标输出
            dec_in = tgt_full[:-1]  # Decoder 输入
            dec_out = tgt_full[1:]  # 要预测的目标

            model.train()
            logits = model(src, dec_in)

            loss = loss_fn(logits, dec_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 准确率
            preds = logits.argmax(dim=-1)
            total_acc += (preds == dec_out).sum().item()
            n_total += dec_out.shape[0]

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(train_data)
            acc = total_acc / n_total * 100
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")

    print(f"\n训练完成!")

    # ===== 测试 Teacher Forcing =====
    print("\n" + "=" * 50)
    print("测试（Teacher Forcing — 给出正确 Decoder 输入）")
    print("=" * 50)

    test_cases = [
        (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 1, 2])),  # dec_in=[1,0,1] → 预测 [0,1,2]?
        (torch.tensor([4, 5]), torch.tensor([5, 4, 5])),
        (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 1, 2, 3])),
    ]

    with torch.no_grad():
        model.eval()
        for src, tgt_full in test_cases:
            dec_in = tgt_full[:-1]
            dec_out = tgt_full[1:]

            logits = model(src, dec_in)
            preds = logits.argmax(dim=-1)

            correct = (preds == dec_out).all().item()
            mark = "✅" if correct else "❌"
            print(f"  {mark}  输入 {src.tolist()} → 预测 {preds.tolist()} (期望 {dec_out.tolist()})")
