"""Shared helpers for reproducible benchmark output."""

import json
import platform
from pathlib import Path

import torch


def summarize(samples_ms):
    values = sorted(float(value) for value in samples_ms)
    if not values:
        raise ValueError("at least one timing sample is required")

    def percentile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "samples": len(values),
    }


def environment_metadata(device):
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "cuda": torch.version.cuda,
    }
    if device == "cuda" and torch.cuda.is_available():
        metadata["accelerator"] = torch.cuda.get_device_name()
    else:
        metadata["accelerator"] = platform.processor() or "cpu"
    return metadata


def write_json(path, payload):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
