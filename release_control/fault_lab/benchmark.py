"""Fault benchmarking and metamorphic testing for CVAT Factory X ValidationEngine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Sequence

from release_control.fault_lab.inject import inject_faults
from release_control.models import FaultType
from release_control.validators.gate import ValidationEngine

CRITICAL_FAULT_TYPES: tuple[FaultType, ...] = (
    FaultType.T1_MISSING_IMAGE,
    FaultType.T2_WRONG_HASH,
    FaultType.T3_INVALID_CLASS,
    FaultType.T4_BBOX_OUT_OF_BOUNDS,
    FaultType.T7_SCENE_LEAK,
    FaultType.T8_GUIDELINE_MISMATCH,
)

ALL_FAULT_TYPES: tuple[FaultType, ...] = tuple(FaultType)


def _short_code(ft: FaultType) -> str:
    return ft.value.split("_", 1)[0]


@dataclass(frozen=True)
class BenchmarkReport:
    """Structured benchmark & metamorphic evaluation summary."""

    per_type_recall: dict[str, float]
    overall_critical_fault_recall: float
    overall_fault_recall: float
    false_hold_rate: float
    false_release_rate: float
    avg_latency_ms: float
    dev_seeds: list[int]
    held_out_seeds: list[int]
    total_fault_runs: int
    total_clean_runs: int
    metamorphic_tests: dict[str, Any]
    runs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FaultBenchmarker:
    """Runs dev/held-out fault injection benchmarks and metamorphic tests."""

    def __init__(
        self,
        dev_seeds: Sequence[int] = (42, 43),
        held_out_seeds: Sequence[int] = (101, 102, 103),
    ) -> None:
        self.dev_seeds = list(dev_seeds)
        self.held_out_seeds = list(held_out_seeds)

    def run(
        self,
        source_dir: str | Path,
        contract: dict[str, Any] | None = None,
        output_path: str | Path | None = Path("reports/FAULT_BENCHMARK.json"),
    ) -> BenchmarkReport:
        """Execute full fault benchmark across dev/held-out seeds + metamorphic tests."""
        src_root = Path(source_dir).resolve()
        contract = contract or {"expected_guideline_version": "v2.0", "num_classes": 80}

        runs: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        detected_by_type: dict[FaultType, list[bool]] = {ft: [] for ft in ALL_FAULT_TYPES}
        false_releases = 0

        with tempfile.TemporaryDirectory(prefix="fault_bench_") as tmp:
            tmp_root = Path(tmp)

            # 1. Clean baseline run
            clean_dir = tmp_root / "clean_baseline"
            shutil.copytree(src_root, clean_dir)
            t0 = time.perf_counter()
            clean_report = ValidationEngine.validate_dataset_report(
                clean_dir,
                contract=contract,
                reference_dir=src_root,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            clean_false_hold = clean_report.recommended_status != "VALIDATED"

            # 2. Fault injection runs across dev and held-out seeds
            seed_groups: list[tuple[str, list[int]]] = [
                ("dev", self.dev_seeds),
                ("held_out", self.held_out_seeds),
            ]
            for split_name, seeds in seed_groups:
                for seed in seeds:
                    for ft in ALL_FAULT_TYPES:
                        work_dir = tmp_root / f"{split_name}_{seed}_{ft.value}"
                        inject_faults(src_root, work_dir, [ft], seed=seed)

                        start = time.perf_counter()
                        report = ValidationEngine.validate_dataset_report(
                            work_dir,
                            contract=contract,
                            reference_dir=src_root,
                        )
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        latencies_ms.append(elapsed_ms)

                        short = _short_code(ft)
                        detected = report.counts_by_fault_type.get(short, 0) > 0
                        held = report.recommended_status == "HOLD"
                        if not held:
                            false_releases += 1

                        detected_by_type[ft].append(detected)
                        runs.append(
                            {
                                "split": split_name,
                                "seed": seed,
                                "fault_type": ft.value,
                                "detected": detected,
                                "recommended_status": report.recommended_status,
                                "counts_by_fault_type": report.counts_by_fault_type,
                                "latency_ms": round(elapsed_ms, 3),
                            }
                        )

            # 3. Metamorphic tests
            metamorphic = self._run_metamorphic_tests(
                src_root=src_root,
                tmp_root=tmp_root,
                clean_report=clean_report.to_dict(),
                contract=contract,
            )

        per_type_recall: dict[str, float] = {}
        for ft in ALL_FAULT_TYPES:
            outcomes = detected_by_type[ft]
            per_type_recall[ft.value] = (
                sum(1 for ok in outcomes if ok) / len(outcomes) if outcomes else 0.0
            )

        critical_outcomes = [
            ok for ft in CRITICAL_FAULT_TYPES for ok in detected_by_type[ft]
        ]
        all_outcomes = [ok for ft in ALL_FAULT_TYPES for ok in detected_by_type[ft]]

        overall_critical_recall = (
            sum(1 for ok in critical_outcomes if ok) / len(critical_outcomes)
            if critical_outcomes
            else 0.0
        )
        overall_recall = (
            sum(1 for ok in all_outcomes if ok) / len(all_outcomes)
            if all_outcomes
            else 0.0
        )
        total_fault_runs = len(runs)
        false_release_rate = (
            false_releases / total_fault_runs if total_fault_runs > 0 else 0.0
        )
        false_hold_rate = 1.0 if clean_false_hold else 0.0
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        bench_report = BenchmarkReport(
            per_type_recall=per_type_recall,
            overall_critical_fault_recall=round(overall_critical_recall, 6),
            overall_fault_recall=round(overall_recall, 6),
            false_hold_rate=round(false_hold_rate, 6),
            false_release_rate=round(false_release_rate, 6),
            avg_latency_ms=round(avg_latency_ms, 3),
            dev_seeds=self.dev_seeds,
            held_out_seeds=self.held_out_seeds,
            total_fault_runs=total_fault_runs,
            total_clean_runs=1,
            metamorphic_tests=metamorphic,
            runs=runs,
        )

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(
                json.dumps(bench_report.to_dict(), indent=2),
                encoding="utf-8",
            )

        return bench_report

    def _run_metamorphic_tests(
        self,
        src_root: Path,
        tmp_root: Path,
        clean_report: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        # Test A: Shuffle manifest image order -> validation counts/status must be invariant
        shuffled_dir = tmp_root / "meta_shuffled"
        shutil.copytree(src_root, shuffled_dir)
        manifest_path = shuffled_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        images = list(manifest.get("images", []))
        rng = random.Random(2026)
        rng.shuffle(images)
        manifest["images"] = images
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        shuffled_report = ValidationEngine.validate_dataset(
            shuffled_dir,
            contract=contract,
            reference_dir=src_root,
        )
        order_invariant = (
            shuffled_report["passed"] == clean_report["passed"]
            and shuffled_report["recommended_status"] == clean_report["recommended_status"]
            and shuffled_report["counts_by_fault_type"] == clean_report["counts_by_fault_type"]
            and len(shuffled_report["issues"]) == len(clean_report["issues"])
        )

        # Test B: Clean copy/re-materialization -> 0 false positives
        scaled_dir = tmp_root / "meta_clean_copy"
        shutil.copytree(src_root, scaled_dir)
        copy_report = ValidationEngine.validate_dataset(
            scaled_dir,
            contract=contract,
            reference_dir=src_root,
        )
        zero_false_positives = (
            copy_report["passed"] is True
            and copy_report["recommended_status"] == "VALIDATED"
            and len(copy_report["issues"]) == 0
        )

        return {
            "manifest_order_invariance": {
                "passed": order_invariant,
                "baseline_issue_count": len(clean_report["issues"]),
                "shuffled_issue_count": len(shuffled_report["issues"]),
            },
            "clean_copy_zero_false_positives": {
                "passed": zero_false_positives,
                "recommended_status": copy_report["recommended_status"],
                "issue_count": len(copy_report["issues"]),
            },
            "all_passed": order_invariant and zero_false_positives,
        }
