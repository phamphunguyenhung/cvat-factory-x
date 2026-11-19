"""Standalone CLI script for running the V2 vs V3 Factory Benchmark."""
from __future__ import annotations

import argparse
import json

from cvat_tools.factory.benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CVAT Autolabel Factory V2 vs V3")
    parser.add_argument("--job", type=int, default=1623, help="Job ID (default: 1623)")
    parser.add_argument("--limit", type=int, default=20, help="Number of representative frames")
    parser.add_argument("--no-v2", action="store_true", help="Skip V2 comparison")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    res = run_benchmark(
        job_id=args.job,
        limit=args.limit,
        compare_v2=not args.no_v2,
        output_path=args.output,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
