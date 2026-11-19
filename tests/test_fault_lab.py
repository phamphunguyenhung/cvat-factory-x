"""Tests for release_control.fault_lab (inject.py and benchmark.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from release_control.fault_lab import FaultBenchmarker, FaultInjector, inject_faults
from release_control.models import FaultManifest, FaultType
from release_control.validators import ValidationEngine


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_clean_dataset(root: Path) -> tuple[Path, dict]:
    ds_dir = root / "clean_dataset"
    (ds_dir / "images").mkdir(parents=True)
    (ds_dir / "labels").mkdir(parents=True)

    samples = [
        ("img1", "train", "scene_01", "0 0.50 0.50 0.20 0.20\n"),
        ("img2", "train", "scene_02", "2 0.30 0.40 0.15 0.15\n"),
        ("img3", "val", "scene_03", "7 0.60 0.45 0.20 0.20\n"),
        ("img4", "val", "scene_04", "2 0.40 0.60 0.12 0.14\n"),
    ]

    image_entries = []
    for img_id, split, scene_id, label_text in samples:
        img_bytes = f"jpeg-bytes-{img_id}".encode("utf-8")
        img_rel = f"images/{img_id}.jpg"
        lbl_rel = f"labels/{img_id}.txt"
        (ds_dir / img_rel).write_bytes(img_bytes)
        (ds_dir / lbl_rel).write_text(label_text, encoding="utf-8")
        image_entries.append(
            {
                "image_id": img_id,
                "image_path": img_rel,
                "label_path": lbl_rel,
                "sha256": _sha256_bytes(img_bytes),
                "split": split,
                "scene_id": scene_id,
                "guideline_version": "v2.0",
            }
        )

    manifest = {
        "dataset_id": "ds-fault-lab-001",
        "guideline_version": "v2.0",
        "classes": [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
        ],
        "images": image_entries,
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    contract = {"expected_guideline_version": "v2.0", "num_classes": 80}
    return ds_dir, contract


def test_source_is_never_mutated_and_seed_is_deterministic(tmp_path: Path) -> None:
    src_dir, contract = _create_clean_dataset(tmp_path)
    before_manifest = (src_dir / "manifest.json").read_text(encoding="utf-8")

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    fm_a = inject_faults(src_dir, out_a, [FaultType.T1_MISSING_IMAGE, FaultType.T6_CLASS_SWAP], seed=42)
    fm_b = inject_faults(src_dir, out_b, [FaultType.T1_MISSING_IMAGE, FaultType.T6_CLASS_SWAP], seed=42)

    # Source untouched and still passes validation
    assert (src_dir / "manifest.json").read_text(encoding="utf-8") == before_manifest
    assert ValidationEngine.validate_dataset(src_dir, contract)["recommended_status"] == "VALIDATED"

    # Deterministic outputs
    assert fm_a.run_id == fm_b.run_id
    assert [f.model_dump(exclude={"fault_id"}) for f in fm_a.faults] == [
        f.model_dump(exclude={"fault_id"}) for f in fm_b.faults
    ]

    # fault_manifest.json validates as FaultManifest and includes provenance/seed
    raw_json = json.loads((out_a / "fault_manifest.json").read_text(encoding="utf-8"))
    parsed = FaultManifest.model_validate(raw_json)
    assert len(parsed.faults) == 2
    assert raw_json["seed"] == 42
    assert raw_json["provenance"]["source_dir"] == str(src_dir.resolve())
    assert all("target" in f and "ground_truth" in f for f in raw_json["faults"])


def test_all_eight_fault_types_detected_by_validation_engine(tmp_path: Path) -> None:
    src_dir, contract = _create_clean_dataset(tmp_path)
    injector = FaultInjector(seed=77)

    for ft in FaultType:
        work_dir = tmp_path / f"work_{ft.value}"
        injector.inject(src_dir, work_dir, [ft], seed=77)
        report = ValidationEngine.validate_dataset_report(
            work_dir,
            contract=contract,
            reference_dir=src_dir,
        )
        short = ft.value.split("_", 1)[0]
        assert report.counts_by_fault_type[short] >= 1, f"{ft.value} not detected"
        assert report.recommended_status == "HOLD", f"{ft.value} did not trigger HOLD"


def test_fault_benchmarker_full_run_and_report_output(tmp_path: Path) -> None:
    src_dir, contract = _create_clean_dataset(tmp_path)
    report_path = tmp_path / "reports" / "FAULT_BENCHMARK.json"

    bench = FaultBenchmarker(dev_seeds=(42, 43), held_out_seeds=(101, 102, 103))
    result = bench.run(src_dir, contract=contract, output_path=report_path)

    assert report_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["overall_critical_fault_recall"] == 1.0
    assert saved["overall_fault_recall"] == 1.0
    assert saved["false_hold_rate"] == 0.0
    assert saved["false_release_rate"] == 0.0
    assert saved["metamorphic_tests"]["all_passed"] is True
    assert result.total_fault_runs == 5 * 8
    assert all(recall == 1.0 for recall in result.per_type_recall.values())
