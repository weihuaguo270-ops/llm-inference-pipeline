"""
GPT 推理引擎 — Prefill / Decode 分离 + KV Cache

提供 TTFT / TPOT 基准所需的端到端推理循环接口。
"""
from contextlib import nullcontext
import importlib.util
import platform
import sys

import torch
import torch.nn.functional as F

from .cache_backends import build_cache_backend


class InferenceEngine:
    """在 GPT 模型上实现 Prefill + KV Cache Decode。"""

    def __init__(self, model, cache_backend="static", block_size=16,
                 amp_dtype=None, compile_mode=None):
        self.model = model
        self.cache_backend = cache_backend
        self.block_size = block_size
        self.max_seq_len = model.max_seq_len
        dtype_names = {
            None: None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            torch.float16: torch.float16,
            torch.bfloat16: torch.bfloat16,
        }
        if amp_dtype not in dtype_names:
            raise ValueError("amp_dtype must be None, 'float16', or 'bfloat16'")
        self.amp_dtype = dtype_names[amp_dtype]
        self.compile_mode = compile_mode
        if compile_mode is not None:
            if not hasattr(torch.nn.Module, "compile"):
                raise RuntimeError("torch.compile requires PyTorch 2.x")
            if platform.system() == "Windows":
                from .windows_toolchain import ensure_msvc_environment

                if ensure_msvc_environment() is None:
                    raise RuntimeError(
                        "torch.compile requires the Visual Studio C++ Build "
                        "Tools workload (MSVC x64/x86 and a Windows SDK)"
                    )
                if not sys.flags.utf8_mode:
                    raise RuntimeError(
                        "torch.compile with localized MSVC output requires "
                        "Python UTF-8 mode; restart Python with PYTHONUTF8=1 "
                        "or -X utf8"
                    )
            model_device = next(self.model.parameters()).device
            if (
                model_device.type == "cuda"
                and importlib.util.find_spec("triton") is None
            ):
                raise RuntimeError(
                    "CUDA torch.compile requires a compatible Triton build. "
                    "The official Windows PyTorch wheel does not include one; "
                    "use WSL2/Linux or validate a matching triton-windows package"
                )
            for layer in self.model.layers:
                layer.self_attn.compile(mode=compile_mode, dynamic=True)
                layer.swiglu.compile(mode=compile_mode, dynamic=True)
        self.kv_caches = None
        self.seq_len = 0

    def reset(self):
        self.kv_caches = [None] * len(self.model.layers)
        self.seq_len = 0

    def _autocast(self, idx):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=idx.device.type, dtype=self.amp_dtype, enabled=True
        )

    @torch.inference_mode()
    def prefill(self, idx: torch.Tensor) -> torch.Tensor:
        """Prompt 阶段：批量处理 prompt，建立 KV Cache。"""
        with self._autocast(idx):
            return self._prefill_impl(idx)

    def _prefill_impl(self, idx: torch.Tensor) -> torch.Tensor:
        # Prefill starts a new sequence; stale cache pages must not leak across requests.
        self.reset()
        self.model.eval()
        B, T = idx.shape
        x = self.model.token_embedding(idx)
        positions = torch.arange(T, device=idx.device)

        # Each layer returns its updated backend so contiguous, static and paged
        # caches share one engine-level update path.
        new_caches = []
        for layer in self.model.layers:
            x, cache = self._layer_forward(
                layer, x, positions, kv_cache=None, prefill=True
            )
            k, v = cache
            new_caches.append(build_cache_backend(
                self.cache_backend, k, v, block_size=self.block_size,
                max_seq_len=self.max_seq_len,
            ))

        self.kv_caches = new_caches
        self.seq_len = T
        x = self.model.norm(x)
        return self.model.lm_head(x)

    @property
    def cache_bytes(self):
        if not self.kv_caches:
            return 0
        return sum(
            cache.allocated_bytes for cache in self.kv_caches if cache is not None
        )

    @property
    def cache_used_bytes(self):
        if not self.kv_caches:
            return 0
        return sum(
            cache.used_bytes for cache in self.kv_caches if cache is not None
        )

    @torch.inference_mode()
    def decode_step(self, idx: torch.Tensor) -> torch.Tensor:
        """Decode 阶段：单 token 前向，复用 KV Cache。"""
        if idx.shape[1] != 1:
            raise ValueError("decode_step expects exactly one token")
        with self._autocast(idx):
            return self._cached_forward(idx)

    @torch.inference_mode()
    def prefill_suffix(self, idx: torch.Tensor) -> torch.Tensor:
        """Process a suffix chunk against an existing prefix cache."""
        if idx.shape[1] == 0:
            raise ValueError("suffix must contain at least one token")
        with self._autocast(idx):
            return self._cached_forward(idx)

    def _cached_forward(self, idx: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if self.kv_caches is None:
            raise RuntimeError("cached forward requires prefill() first")
        token_count = idx.shape[1]
        x = self.model.token_embedding(idx)
        positions = torch.arange(
            self.seq_len, self.seq_len + token_count, device=idx.device
        )

        new_caches = []
        for i, layer in enumerate(self.model.layers):
            x, cache = self._layer_forward(
                layer, x, positions, kv_cache=self.kv_caches[i], prefill=False
            )
            new_caches.append(cache)

        self.kv_caches = new_caches
        self.seq_len += token_count
        x = self.model.norm(x)
        return self.model.lm_head(x)

    @torch.inference_mode()
    def generate_naive(self, idx, max_new_tokens):
        """无 Cache 自回归（对照组）。"""
        self.model.eval()
        # This deliberately recomputes the full context as the cache benchmark control.
        for _ in range(max_new_tokens):
            cond = idx[:, -self.model.layers[0].self_attn._cos.shape[0] + 1:]
            logits = self.model(cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    @torch.inference_mode()
    def generate_cached(self, idx, max_new_tokens):
        """带 Cache 自回归。"""
        logits = self.prefill(idx)
        next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
        out = torch.cat([idx, next_id], dim=1)

        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_id)
            next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            out = torch.cat([out, next_id], dim=1)
        return out

    def _layer_forward(self, layer, x, positions, kv_cache, prefill=False):
        residual = x
        x = layer.rmsnorm1(x)
        if prefill:
            x, cache = layer.self_attn(
                x, positions=positions, return_cache=True, is_causal=True
            )
        else:
            x, cache = layer.self_attn(
                x, positions=positions, kv_cache=kv_cache
            )
        x = x + residual

        residual = x
        x = layer.rmsnorm2(x)
        x = layer.swiglu(x)
        x = x + residual
        return x, cache
