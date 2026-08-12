"""Normalize inference benchmark output into a release-gate evidence record."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


EVIDENCE_SCHEMA_VERSION = "agent-release-evidence/v1"


@dataclass(frozen=True)
class PerformanceBudget:
    max_ttft_ms: float | None = None
    max_tpot_ms: float | None = None
    max_cache_bytes: int | None = None
    max_peak_device_bytes: int | None = None


@dataclass
class PerformanceEvidence:
    source: str
    environment: dict[str, Any]
    metrics: dict[str, Any]
    budget: dict[str, Any]
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check["passed"] for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "evidence_type": "inference_performance",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": self.source,
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
            "budget": dict(self.budget),
            "checks": list(self.checks),
            "passed": self.passed,
            "claim_boundary": (
                "Benchmark evidence is valid only for the recorded environment and workload."
            ),
        }


def build_performance_evidence(
    benchmark: Mapping[str, Any],
    budget: PerformanceBudget,
    *,
    variant: str | None = None,
) -> PerformanceEvidence:
    """Build hard checks without treating proxy speedups as Agent quality scores."""
    metrics, selected_variant = _select_metrics(benchmark, variant=variant)
    checks = []
    _append_check(checks, "ttft_ms", metrics, budget.max_ttft_ms)
    _append_check(checks, "tpot_cached_ms", metrics, budget.max_tpot_ms)
    _append_check(checks, "cache_bytes", metrics, budget.max_cache_bytes)
    _append_check(
        checks,
        "peak_device_bytes",
        metrics,
        budget.max_peak_device_bytes,
        allow_none=True,
    )
    if not checks:
        raise ValueError("at least one performance budget is required")
    return PerformanceEvidence(
        source=(
            f"{benchmark.get('benchmark') or 'unknown'}:{selected_variant}"
            if selected_variant
            else str(benchmark.get("benchmark") or "unknown")
        ),
        environment=dict(benchmark.get("environment") or {}),
        metrics=metrics,
        budget={
            key: value
            for key, value in {
                "max_ttft_ms": budget.max_ttft_ms,
                "max_tpot_ms": budget.max_tpot_ms,
                "max_cache_bytes": budget.max_cache_bytes,
                "max_peak_device_bytes": budget.max_peak_device_bytes,
            }.items()
            if value is not None
        },
        checks=checks,
    )


def _select_metrics(
    benchmark: Mapping[str, Any],
    *,
    variant: str | None,
) -> tuple[dict[str, Any], str | None]:
    direct = benchmark.get("metrics")
    if isinstance(direct, Mapping) and direct:
        if variant:
            raise ValueError("variant is not valid for a single-result benchmark")
        return dict(direct), None

    results = benchmark.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("benchmark.metrics or benchmark.results is required")
    if not variant:
        raise ValueError("variant is required for a multi-result benchmark")
    selected = next(
        (
            result
            for result in results
            if isinstance(result, Mapping) and result.get("name") == variant
        ),
        None,
    )
    if selected is None:
        names = [
            str(result.get("name"))
            for result in results
            if isinstance(result, Mapping)
        ]
        raise ValueError(f"unknown variant {variant!r}; available: {names}")
    prefill = selected.get("prefill") or {}
    decode = selected.get("decode") or {}
    return {
        "ttft_ms": prefill.get("mean_ms"),
        "tpot_cached_ms": decode.get("mean_ms"),
        "cache_bytes": selected.get(
            "cache_allocated_bytes", selected.get("cache_bytes")
        ),
        "cache_used_bytes": selected.get("cache_used_bytes"),
        "peak_device_bytes": selected.get("peak_device_bytes"),
    }, variant


def _append_check(
    checks: list[dict[str, Any]],
    metric: str,
    metrics: Mapping[str, Any],
    maximum: float | int | None,
    *,
    allow_none: bool = False,
) -> None:
    if maximum is None:
        return
    actual = metrics.get(metric)
    passed = actual is not None and float(actual) <= float(maximum)
    if allow_none and actual is None:
        passed = False
    checks.append(
        {
            "metric": metric,
            "operator": "<=",
            "limit": maximum,
            "actual": actual,
            "passed": passed,
        }
    )
