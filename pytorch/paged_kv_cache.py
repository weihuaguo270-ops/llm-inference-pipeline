"""
Paged KV Cache — 按 block 管理的 KV 存储（简化版）

模拟 vLLM PagedAttention 的核心思想：固定大小 block 分配，减少显存碎片。
"""
import torch


class PagedKVCache:
    """按 block_size 分块的 K/V 存储。"""

    def __init__(self, block_size=16, num_heads=4, d_k=64, device="cpu", dtype=torch.float32):
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_k = d_k
        self.device = device
        self.dtype = dtype
        self.k_blocks = []
        self.v_blocks = []
        self.length = 0

    def _new_block(self):
        k = torch.zeros(1, self.num_heads, self.block_size, self.d_k, device=self.device, dtype=self.dtype)
        v = torch.zeros(1, self.num_heads, self.block_size, self.d_k, device=self.device, dtype=self.dtype)
        self.k_blocks.append(k)
        self.v_blocks.append(v)
        return len(self.k_blocks) - 1

    def append(self, k_new, v_new):
        """k_new, v_new: (1, H, S, d_k)"""
        S = k_new.shape[2]
        offset = 0
        while offset < S:
            if self.length % self.block_size == 0:
                self._new_block()
            bi = self.length // self.block_size
            pos = self.length % self.block_size
            space = self.block_size - pos
            take = min(space, S - offset)
            self.k_blocks[bi][0, :, pos:pos + take] = k_new[0, :, offset:offset + take]
            self.v_blocks[bi][0, :, pos:pos + take] = v_new[0, :, offset:offset + take]
            self.length += take
            offset += take

    def materialize(self):
        """拼成连续 (1, H, length, d_k) 供 Attention 读取。"""
        if self.length == 0:
            return None, None
        parts_k, parts_v = [], []
        for bi in range(len(self.k_blocks)):
            end = min(self.block_size, self.length - bi * self.block_size)
            if end <= 0:
                break
            parts_k.append(self.k_blocks[bi][:, :, :end])
            parts_v.append(self.v_blocks[bi][:, :, :end])
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    @property
    def num_blocks(self):
        return len(self.k_blocks)

    @property
    def utilization(self):
        if not self.k_blocks:
            return 0.0
        used = self.length
        cap = len(self.k_blocks) * self.block_size
        return used / cap if cap else 0.0
