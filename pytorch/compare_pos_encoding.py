"""
位置编码对比实验 — Sinusoidal PE vs RoPE

在同一个序列复制任务上，分别用两种位置编码训练 Transformer，
对比 loss 下降速度和最终准确率。

运行方式:
  python pytorch/compare_pos_encoding.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.utils import layer_norm
from pytorch.multi_head_attention import MultiHeadAttention
from pytorch.cross_attention import MultiHeadCrossAttention


# ============================================================
# 可切换位置编码的 Encoder / Decoder
# ============================================================

class FFN(nn.Module):
    """Feed-forward sublayer used only by the positional-encoding comparison."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        """Apply the comparison model's position-wise feed-forward transform."""
        return self.W2(F.relu(self.W1(x)))


class EncoderLayer(nn.Module):
    """Minimal encoder layer for comparing positional-encoding variants."""

    def __init__(self, d_model, num_heads, d_ff, use_rope=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_rope=use_rope)
        self.ffn = FFN(d_model, d_ff)
    def forward(self, x):
        """Apply unmasked self-attention and FFN residual updates."""
        attn_out = self.self_attn(x, use_mask=False)
        x = x + attn_out
        x = layer_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    """Minimal decoder layer for the positional-encoding comparison."""

    def __init__(self, d_model, num_heads, d_ff, use_rope=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_rope=use_rope)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
    def forward(self, x, encoder_output):
        """Apply causal self-attention followed by encoder cross-attention."""
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
    """可切换位置编码的 Transformer"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers=2,
                 pos_encoding="sinusoidal"):
        super().__init__()
        self.pos_encoding = pos_encoding
        use_rope = (pos_encoding == "rope")

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, use_rope=use_rope)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, use_rope=use_rope)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        """Return target-token logits for one source/target comparison batch."""
        d_model = self.token_embedding.embedding_dim
        device = src_ids.device

        # --- Encoder ---
        src_emb = self.token_embedding(src_ids)
        if self.pos_encoding == "sinusoidal":
            pe = self._sinusoidal_pe(src_ids.shape[0], d_model, device)
            src_emb = src_emb + pe
        x = src_emb
        for layer in self.encoder_layers:
            x = layer(x)
        encoder_output = x

        # --- Decoder ---
        tgt_emb = self.token_embedding(tgt_ids)
        if self.pos_encoding == "sinusoidal":
            pe = self._sinusoidal_pe(tgt_ids.shape[0], d_model, device)
            tgt_emb = tgt_emb + pe
        x = tgt_emb
        for layer in self.decoder_layers:
            x = layer(x, encoder_output)

        logits = self.lm_head(x)
        return logits

    @staticmethod
    def _sinusoidal_pe(seq_len, d_model, device):
        pe = torch.zeros(seq_len, d_model, device=device)
        pos = torch.arange(seq_len, device=device).float().view(-1, 1)
        i = torch.arange(d_model, device=device).float()
        div_term = torch.exp(i * -math.log(10000.0) / d_model)
        angles = pos * div_term
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe


# ============================================================
# 训练函数
# ============================================================

import math


def train_model(pos_encoding: str, n_epochs: int = 200, seed: int = 42) -> dict:
    """用指定位置编码训练一个 Transformer，返回训练指标"""
    torch.manual_seed(seed)

    vocab_size = 10
    d_model = 16
    num_heads = 2
    d_ff = 32

    model = Transformer(vocab_size, d_model, num_heads, d_ff,
                        num_layers=2, pos_encoding=pos_encoding)

    train_data = [
        (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 1, 2, 3])),
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

    history = {"loss": [], "acc": []}

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_acc = 0.0
        n_total = 0
        for src, tgt_full in train_data:
            dec_in = tgt_full[:-1]
            dec_out = tgt_full[1:]

            model.train()
            logits = model(src, dec_in)
            loss = loss_fn(logits, dec_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_acc += (preds == dec_out).sum().item()
            n_total += dec_out.shape[0]

        avg_loss = total_loss / len(train_data)
        acc = total_acc / n_total * 100

        if (epoch + 1) % 20 == 0 or epoch == 0:
            history["loss"].append((epoch + 1, avg_loss))
            history["acc"].append((epoch + 1, acc))

    # 测试
    test_cases = [
        (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 1, 2])),
        (torch.tensor([4, 5]), torch.tensor([5, 4, 5])),
        (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 1, 2, 3])),
    ]
    with torch.no_grad():
        model.eval()
        test_correct = 0
        test_total = 0
        for src, tgt_full in test_cases:
            dec_in = tgt_full[:-1]
            dec_out = tgt_full[1:]
            logits = model(src, dec_in)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == dec_out).all().item()
            test_total += 1

    return {
        "pos_encoding": pos_encoding,
        "history": history,
        "final_loss": history["loss"][-1][1],
        "final_acc": history["acc"][-1][1],
        "test_pass_rate": test_correct / test_total * 100,
    }


# ============================================================
# 运行对比
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("位置编码对比实验")
    print("任务: 序列复制（元素+1）")
    print("=" * 60)

    for pe in ["sinusoidal", "rope"]:
        print(f"\n▶ 训练 {pe} ...")
        result = train_model(pe, n_epochs=200)

        print(f"\n  {pe} 训练结果:")
        for epoch, loss in result["history"]["loss"]:
            acc = [a for e, a in result["history"]["acc"] if e == epoch][0]
            mark = "← 较好" if loss < 0.15 else ""
            print(f"    Epoch {epoch:4d}  Loss: {loss:.4f}  Acc: {acc:.1f}%  {mark}")

        print(f"\n  最终: Loss={result['final_loss']:.4f}  Acc={result['final_acc']:.1f}%")
        print(f"  测试通过率: {result['test_pass_rate']:.0f}%")

    # 再跑一次 Sinusoidal 看看能不能收敛
    print("\n" + "=" * 60)
    print("两种方案最终对比")
    print("=" * 60)

    results = {}
    for pe in ["sinusoidal", "rope"]:
        results[pe] = train_model(pe, n_epochs=300)

    r_sin = results["sinusoidal"]
    r_rope = results["rope"]

    print(f"\n{'指标':<20} {'Sinusoidal PE':<18} {'RoPE':<18}")
    print("-" * 56)
    print(f"{'最终 Loss':<20} {r_sin['final_loss']:<18.4f} {r_rope['final_loss']:<18.4f}")
    print(f"{'最终 Acc':<20} {r_sin['final_acc']:<18.1f}% {r_rope['final_acc']:<18.1f}%")
    print(f"{'测试通过率':<20} {r_sin['test_pass_rate']:<18.0f}% {r_rope['test_pass_rate']:<18.0f}%")

    # 对比收敛速度（到达 90% Acc 的 epoch）
    for pe in ["sinusoidal", "rope"]:
        r = results[pe]
        for epoch, acc in r["history"]["acc"]:
            if acc >= 90:
                print(f"\n  {pe} 到达 90% Acc: epoch {epoch}")
                break
