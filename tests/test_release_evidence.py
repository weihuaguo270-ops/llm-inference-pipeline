import pytest

from experiments.release_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    PerformanceBudget,
    build_performance_evidence,
)


def _benchmark():
    return {
        "benchmark": "prefill_decode",
        "environment": {"device": "cpu", "torch": "2.x"},
        "metrics": {
            "ttft_ms": 80.0,
            "tpot_cached_ms": 12.0,
            "cache_bytes": 4096,
            "peak_device_bytes": None,
        },
    }


def test_performance_evidence_passes_recorded_budget():
    evidence = build_performance_evidence(
        _benchmark(),
        PerformanceBudget(max_ttft_ms=100, max_tpot_ms=15, max_cache_bytes=8192),
    )
    payload = evidence.to_dict()
    assert payload["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert payload["passed"] is True
    assert payload["environment"]["device"] == "cpu"


def test_performance_evidence_fails_latency_regression():
    evidence = build_performance_evidence(
        _benchmark(),
        PerformanceBudget(max_tpot_ms=10),
    )
    assert evidence.passed is False
    assert evidence.checks[0]["metric"] == "tpot_cached_ms"


def test_performance_evidence_requires_budget():
    with pytest.raises(ValueError, match="at least one"):
        build_performance_evidence(_benchmark(), PerformanceBudget())


def test_multi_result_benchmark_requires_explicit_variant():
    benchmark = {
        "benchmark": "pytorch_optimized",
        "environment": {"device": "cuda"},
        "results": [
            {
                "name": "sdpa+static",
                "prefill": {"mean_ms": 20.0},
                "decode": {"mean_ms": 4.0},
                "cache_allocated_bytes": 4096,
                "peak_device_bytes": 8192,
            },
            {
                "name": "eager+contiguous",
                "prefill": {"mean_ms": 30.0},
                "decode": {"mean_ms": 6.0},
                "cache_allocated_bytes": 8192,
                "peak_device_bytes": 16384,
            },
        ],
    }
    with pytest.raises(ValueError, match="variant is required"):
        build_performance_evidence(
            benchmark,
            PerformanceBudget(max_ttft_ms=100),
        )
    evidence = build_performance_evidence(
        benchmark,
        PerformanceBudget(max_ttft_ms=25, max_tpot_ms=5),
        variant="sdpa+static",
    )
    assert evidence.passed is True
    assert evidence.source == "pytorch_optimized:sdpa+static"
