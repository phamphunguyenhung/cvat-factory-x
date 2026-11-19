"""Lightweight reproducible benchmark for the CVAT Antigravity harness."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_stats(paths: list[Path]) -> tuple[int, int, int]:
    files = 0
    total_bytes = 0
    total_lines = 0
    for p in paths:
        if p.exists() and p.is_file():
            raw = p.read_bytes()
            files += 1
            total_bytes += len(raw)
            total_lines += raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
    return files, total_bytes, total_lines


def bench_discovery() -> dict:
    # Legacy always-on + discovery footprint (what agents were instructed to read)
    legacy_paths = [
        ROOT / "REUSE_MAP.md",
        ROOT / "MODEL_INVENTORY.md",
        ROOT / "AGENT_ORCHESTRATION.md",
        ROOT / "memory-bank" / "architecture.md",
        ROOT / "memory-bank" / "lessons-learned.md",
        ROOT / "memory-bank" / "workflows.md",
        ROOT / "week02" / "plan.md",
    ]
    t0 = time.perf_counter()
    leg_files, leg_bytes, leg_lines = _read_stats(legacy_paths)
    leg_ms = (time.perf_counter() - t0) * 1000.0

    # Optimized FAST path: AGENTS.md + docs/AGENT_MAP.md + targeted rg for validate_face50_frame
    t1 = time.perf_counter()
    opt_paths = [ROOT / "AGENTS.md", ROOT / "docs" / "AGENT_MAP.md"]
    opt_files, opt_bytes, opt_lines = _read_stats(opt_paths)
    rg_res = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-n", "def validate_face50_frame", "--", "src/cvat_tools/pose"],
        capture_output=True,
        text=True,
        check=False,
    )
    opt_ms = (time.perf_counter() - t1) * 1000.0

    return {
        "task": "1_locate_symbol_validate_face50_frame",
        "legacy_discovery_files": leg_files,
        "legacy_discovery_bytes": leg_bytes,
        "legacy_discovery_lines": leg_lines,
        "legacy_read_ms": round(leg_ms, 2),
        "optimized_files_read": opt_files,
        "optimized_bytes_read": opt_bytes + len(rg_res.stdout.encode("utf-8")),
        "optimized_lines_read": opt_lines + len(rg_res.stdout.splitlines()),
        "optimized_ms": round(opt_ms, 2),
        "rg_hits": len(rg_res.stdout.splitlines()),
    }


def bench_targeted_test() -> dict:
    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_face50_engine.py", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "task": "2_targeted_pytest_face50_engine",
        "returncode": res.returncode,
        "elapsed_ms": round(elapsed_ms, 2),
        "output_bytes": len(res.stdout.encode("utf-8")) + len(res.stderr.encode("utf-8")),
        "output_lines": len(res.stdout.splitlines()) + len(res.stderr.splitlines()),
        "summary": res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "",
    }


def bench_diff_and_checkpoint() -> dict:
    t0 = time.perf_counter()
    diff_res = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--stat", "--", "AGENTS.md", "rules.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "task": "3_git_diff_stat_review",
        "elapsed_ms": round(elapsed_ms, 2),
        "output_bytes": len(diff_res.stdout.encode("utf-8")),
        "output_lines": len(diff_res.stdout.splitlines()),
    }


def main() -> int:
    results = {
        "discovery": bench_discovery(),
        "targeted_test": bench_targeted_test(),
        "diff_review": bench_diff_and_checkpoint(),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
