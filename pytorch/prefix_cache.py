"""Reusable KV snapshots for requests that share a token prefix."""

import torch

from .cache_backends import build_cache_backend


class PrefixKVCache:
    """Cache an immutable prefix snapshot and replay only each request suffix."""

    def __init__(self, engine):
        self.engine = engine
        self.prefix_ids = None
        self._prefix_caches = None

    def reset(self):
        """Discard the cached prefix and associated K/V backend state."""
        self.prefix_ids = None
        self._prefix_caches = None
        self.engine.reset()

    def _clone_caches(self, caches):
        cloned = []
        for cache in caches:
            k, v = cache.materialize()
            cloned.append(build_cache_backend(
                self.engine.cache_backend,
                k.clone(),
                v.clone(),
                block_size=self.engine.block_size,
                max_seq_len=self.engine.max_seq_len,
            ))
        return cloned

    @torch.no_grad()
    def prime(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Build and retain an immutable KV snapshot for ``prefix_ids``."""
        logits = self.engine.prefill(prefix_ids)
        self.prefix_ids = prefix_ids.clone()
        self._prefix_caches = self._clone_caches(self.engine.kv_caches)
        return logits

    def _restore_prefix(self):
        self.engine.kv_caches = self._clone_caches(self._prefix_caches)
        self.engine.seq_len = self.prefix_ids.shape[1]

    @torch.no_grad()
    def prefill_with_prefix(self, full_ids: torch.Tensor, prefix_len=None) -> torch.Tensor:
        """Prefill a request, reusing a previously primed prefix when possible.

        On the first call, ``prefix_len`` selects the reusable part. Omitting it
        preserves the old exact-prefix behavior by caching the full input.
        """
        if self.prefix_ids is None:
            reusable_len = full_ids.shape[1] if prefix_len is None else prefix_len
            if reusable_len <= 0 or reusable_len > full_ids.shape[1]:
                raise ValueError("prefix_len must be within the input sequence")
            self.prime(full_ids[:, :reusable_len])

        if not self.cache_hit(full_ids):
            return self.engine.prefill(full_ids)

        self._restore_prefix()
        suffix = full_ids[:, self.prefix_ids.shape[1]:]
        if suffix.shape[1] == 0:
            return self.engine.model(self.prefix_ids)[:, -1:, :]

        return self.engine.prefill_suffix(suffix)[:, -1:, :]

    def cache_hit(self, full_ids: torch.Tensor) -> bool:
        """Return whether ``full_ids`` begins with the exact cached token prefix."""
        if self.prefix_ids is None:
            return False
        prefix_len = self.prefix_ids.shape[1]
        return (
            prefix_len <= full_ids.shape[1]
            and torch.equal(full_ids[:, :prefix_len], self.prefix_ids)
        )
