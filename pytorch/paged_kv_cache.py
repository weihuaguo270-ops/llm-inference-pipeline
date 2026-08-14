"""
Paged KV Cache — 按 block 管理的 KV 存储（简化版）

模拟 vLLM PagedAttention 的核心思想：固定大小 block 分配，减少显存碎片。
"""
import torch


class PagedKVCache:
    """按 block_size 分块的 K/V 存储。"""

    def __init__(self, block_size=16, num_heads=None, d_k=None, device="cpu", dtype=torch.float32):
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_k = d_k
        self.device = device
        self.dtype = dtype
        self.k_blocks = []
        self.v_blocks = []
        self.length = 0
        self.batch_size = None

    def _new_block(self):
        k = torch.empty(
            self.batch_size, self.num_heads, self.block_size, self.d_k,
            device=self.device, dtype=self.dtype,
        )
        v = torch.empty_like(k)
        self.k_blocks.append(k)
        self.v_blocks.append(v)
        return len(self.k_blocks) - 1

    def append(self, k_new, v_new):
        """Append tensors shaped ``(batch, heads, seq, head_dim)``."""
        if k_new.shape != v_new.shape or k_new.ndim != 4:
            raise ValueError("K and V must have the same 4-D shape")
        if self.batch_size is None:
            self.batch_size = k_new.shape[0]
            self.num_heads = k_new.shape[1]
            self.d_k = k_new.shape[3]
            self.device = k_new.device
            self.dtype = k_new.dtype
        expected = (self.batch_size, self.num_heads, self.d_k)
        actual = (k_new.shape[0], k_new.shape[1], k_new.shape[3])
        if actual != expected:
            raise ValueError(f"cache shape mismatch: expected {expected}, got {actual}")
        S = k_new.shape[2]
        offset = 0
        while offset < S:
            if self.length % self.block_size == 0:
                self._new_block()
            bi = self.length // self.block_size
            pos = self.length % self.block_size
            space = self.block_size - pos
            take = min(space, S - offset)
            self.k_blocks[bi][:, :, pos:pos + take] = k_new[:, :, offset:offset + take]
            self.v_blocks[bi][:, :, pos:pos + take] = v_new[:, :, offset:offset + take]
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

    def iter_blocks(self):
        """Yield only the populated slice of each physical block."""
        for bi, (k, v) in enumerate(zip(self.k_blocks, self.v_blocks)):
            end = min(self.block_size, self.length - bi * self.block_size)
            if end > 0:
                yield k[:, :, :end], v[:, :, :end]

    @property
    def num_blocks(self):
        """Return the number of allocated physical K/V blocks."""
        return len(self.k_blocks)

    @property
    def utilization(self):
        """Return initialized positions divided by allocated block capacity."""
        if not self.k_blocks:
            return 0.0
        used = self.length
        cap = len(self.k_blocks) * self.block_size
        return used / cap if cap else 0.0

    @property
    def allocated_bytes(self):
        """Return bytes reserved by every allocated K/V block."""
        return sum(
            k.numel() * k.element_size() + v.numel() * v.element_size()
            for k, v in zip(self.k_blocks, self.v_blocks)
        )

    @property
    def used_bytes(self):
        """Return bytes corresponding to initialized sequence positions."""
        if self.batch_size is None:
            return 0
        elements = 2 * self.batch_size * self.num_heads * self.length * self.d_k
        return elements * torch.empty((), dtype=self.dtype).element_size()
