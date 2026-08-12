"""CLI for converting a benchmark JSON file into release-gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.release_evidence import PerformanceBudget, build_performance_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--max-ttft-ms", type=float)
    parser.add_argument("--max-tpot-ms", type=float)
    parser.add_argument("--max-cache-bytes", type=int)
    parser.add_argument("--max-peak-device-bytes", type=int)
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    evidence = build_performance_evidence(
        benchmark,
        PerformanceBudget(
            max_ttft_ms=args.max_ttft_ms,
            max_tpot_ms=args.max_tpot_ms,
            max_cache_bytes=args.max_cache_bytes,
            max_peak_device_bytes=args.max_peak_device_bytes,
        ),
        variant=args.variant,
    )
    payload = evidence.to_dict()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if evidence.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
