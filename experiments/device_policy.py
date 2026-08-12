"""Device selection policy shared by reproducible benchmarks."""

from __future__ import annotations


def resolve_device(requested: str, *, cuda_available: bool, require_cuda: bool) -> str:
    """Resolve a benchmark device and fail closed when CUDA evidence is required."""
    if requested != "cuda" or cuda_available:
        return requested
    if require_cuda:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
    print("CUDA unavailable; falling back to CPU")
    return "cpu"
