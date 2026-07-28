"""Pluggable KV cache storage backends used by the inference engine."""

import torch


class ContiguousKVCache:
    """K/V tensors stored contiguously along the sequence dimension."""

    supports_sdpa = True

    def __init__(self, k=None, v=None):
        self.k = k
        self.v = v

    def append(self, k_new, v_new):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self

    def materialize(self):
        return self.k, self.v

    def iter_blocks(self):
        if self.k is not None:
            yield self.k, self.v

    @property
    def length(self):
        return 0 if self.k is None else self.k.shape[2]

    @property
    def allocated_bytes(self):
        if self.k is None:
            return 0
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()

    @property
    def used_bytes(self):
        return self.allocated_bytes


class StaticKVCache:
    """Preallocated K/V tensors with in-place token appends."""

    supports_sdpa = True

    def __init__(self, max_seq_len, k=None, v=None):
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        self.k = None
        self.v = None
        self._length = 0
        if k is not None:
            self._allocate(k, v)
            self.append(k, v)

    def _allocate(self, k, v):
        if k.shape != v.shape or k.ndim != 4:
            raise ValueError("K and V must have the same 4-D shape")
        shape = (k.shape[0], k.shape[1], self.max_seq_len, k.shape[3])
        self.k = torch.empty(shape, dtype=k.dtype, device=k.device)
        self.v = torch.empty_like(self.k)

    def append(self, k_new, v_new):
        if self.k is None:
            self._allocate(k_new, v_new)
        if k_new.shape != v_new.shape:
            raise ValueError("K and V shapes must match")
        if (
            k_new.shape[0] != self.k.shape[0]
            or k_new.shape[1] != self.k.shape[1]
            or k_new.shape[3] != self.k.shape[3]
        ):
            raise ValueError("cache shape mismatch")
        end = self._length + k_new.shape[2]
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache capacity exceeded: {end} > {self.max_seq_len}"
            )
        self.k[:, :, self._length:end].copy_(k_new)
        self.v[:, :, self._length:end].copy_(v_new)
        self._length = end
        return self

    def materialize(self):
        return self.k[:, :, :self._length], self.v[:, :, :self._length]

    def iter_blocks(self):
        if self._length:
            yield self.materialize()

    @property
    def length(self):
        return self._length

    @property
    def allocated_bytes(self):
        if self.k is None:
            return 0
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()

    @property
    def used_bytes(self):
        if self.k is None:
            return 0
        elements = 2 * self.k.shape[0] * self.k.shape[1] * self._length * self.k.shape[3]
        return elements * self.k.element_size()


def build_cache_backend(kind, k, v, block_size=16, max_seq_len=None):
    """Create a cache backend initialized with a prefill K/V pair."""
    if kind == "contiguous":
        return ContiguousKVCache(k, v)
    if kind == "static":
        if max_seq_len is None:
            raise ValueError("static cache requires max_seq_len")
        return StaticKVCache(max_seq_len=max_seq_len, k=k, v=v)
    if kind == "paged":
        from .paged_kv_cache import PagedKVCache

        cache = PagedKVCache(block_size=block_size, device=k.device, dtype=k.dtype)
        cache.append(k, v)
        return cache
    raise ValueError(f"unknown cache backend: {kind}")
