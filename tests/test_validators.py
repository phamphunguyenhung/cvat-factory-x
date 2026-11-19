"""Tests for release_control.validators (T1..T8, schema, duplicates, privacy, gate)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from release_control.validators import ValidationEngine


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_valid_dataset(tmp_path: Path) -> tuple[Path, dict]:
    ds_dir = tmp_path / "dataset"
    (ds_dir / "images").mkdir(parents=True)
    (ds_dir / "labels").mkdir(parents=True)

    img1_bytes = b"fake-jpeg-1"
    img2_bytes = b"fake-jpeg-2"
    (ds_dir / "images" / "img1.jpg").write_bytes(img1_bytes)
    (ds_dir / "images" / "img2.jpg").write_bytes(img2_bytes)

    (ds_dir / "labels" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds_dir / "labels" / "img2.txt").write_text("2 0.3 0.4 0.1 0.15\n", encoding="utf-8")

    manifest = {
        "dataset_id": "ds-001",
        "guideline_version": "v2.0",
        "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"],
        "images": [
            {
                "image_id": "img1",
                "image_path": "images/img1.jpg",
                "label_path": "labels/img1.txt",
                "sha256": _sha256_bytes(img1_bytes),
                "split": "train",
                "scene_id": "scene_a",
                "guideline_version": "v2.0",
            },
            {
                "image_id": "img2",
                "image_path": "images/img2.jpg",
                "label_path": "labels/img2.txt",
                "sha256": _sha256_bytes(img2_bytes),
                "split": "val",
                "scene_id": "scene_b",
                "guideline_version": "v2.0",
            },
        ],
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    contract = {
        "expected_guideline_version": "v2.0",
        "num_classes": 80,
    }
    return ds_dir, contract


def test_valid_dataset_passes_with_validated_status(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    report = ValidationEngine.validate_dataset(ds_dir, contract)

    assert report["passed"] is True
    assert report["requires_human_review"] is False
    assert report["recommended_status"] == "VALIDATED"
    assert report["issues"] == []
    assert all(report["counts_by_fault_type"][f"T{i}"] == 0 for i in range(1, 9))


def test_schema_missing_required_fields(tmp_path: Path) -> None:
    ds_dir = tmp_path / "bad_schema"
    ds_dir.mkdir()
    (ds_dir / "manifest.json").write_text(json.dumps({"dataset_id": "x"}), encoding="utf-8")

    report = ValidationEngine.validate_dataset(ds_dir, {})
    assert report["passed"] is False
    assert report["recommended_status"] == "HOLD"
    assert report["counts_by_fault_type"]["SCHEMA"] >= 3


def test_t1_missing_image_and_label(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    (ds_dir / "images" / "img1.jpg").unlink()
    (ds_dir / "labels" / "img2.txt").unlink()

    report = ValidationEngine.validate_dataset(ds_dir, contract)
    assert report["passed"] is False
    assert report["recommended_status"] == "HOLD"
    assert report["counts_by_fault_type"]["T1"] == 2


def test_t2_wrong_hash(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    (ds_dir / "images" / "img1.jpg").write_bytes(b"corrupted-bytes")

    report = ValidationEngine.validate_dataset(ds_dir, contract)
    assert report["passed"] is False
    assert report["counts_by_fault_type"]["T2"] == 1
    assert any(i["code"] == "T2_WRONG_HASH" for i in report["issues"])


def test_t3_invalid_class_and_malformed_row(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    (ds_dir / "labels" / "img1.txt").write_text(
        "99 0.5 0.5 0.2 0.2\n1 0.5 0.5\n",
        encoding="utf-8",
    )

    report = ValidationEngine.validate_dataset(ds_dir, contract)
    assert report["passed"] is False
    assert report["counts_by_fault_type"]["T3"] == 2


def test_t4_bbox_out_of_bounds_and_duplicate(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    (ds_dir / "labels" / "img1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "0 0.502 0.501 0.201 0.199\n"  # duplicate within 0.005
        "0 1.2 0.5 0.2 0.2\n"          # out of bounds
        "0 0.5 0.5 0.0 0.2\n",         # non-positive w
        encoding="utf-8",
    )

    report = ValidationEngine.validate_dataset(ds_dir, contract)
    assert report["passed"] is False
    assert report["counts_by_fault_type"]["T4"] == 3
    codes = {i["code"] for i in report["issues"]}
    assert "T4_BBOX_OUT_OF_BOUNDS" in codes
    assert "T4_DUPLICATE_BBOX" in codes


def test_t5_bbox_drift_and_t6_class_swap_against_reference(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()

    # img1 has class 0 drifted far; img2 has car(2) swapped to truck(7) at exact same box
    (ds_dir / "labels" / "img2.txt").write_text("7 0.3 0.4 0.1 0.15\n", encoding="utf-8")
    ref_boxes = {
        "img1": [{"class_id": 0, "cx": 0.2, "cy": 0.2, "w": 0.2, "h": 0.2}],
        "img2": [{"class_id": 2, "cx": 0.3, "cy": 0.4, "w": 0.1, "h": 0.15}],
    }
    (ref_dir / "reference_boxes.json").write_text(json.dumps(ref_boxes), encoding="utf-8")

    report = ValidationEngine.validate_dataset(ds_dir, contract, reference_dir=ref_dir)
    assert report["passed"] is True  # warnings only, not critical
    assert report["requires_human_review"] is True
    assert report["recommended_status"] == "HOLD"
    assert report["counts_by_fault_type"]["T5"] >= 1
    assert report["counts_by_fault_type"]["T6"] >= 1

    t5_issues = [i for i in report["issues"] if i["code"] == "T5_BBOX_DRIFT"]
    assert t5_issues[0]["severity"] == "WARNING"
    assert "NOT ground-truth proof" in t5_issues[0]["message"]


def test_t7_scene_leak_and_t8_guideline_mismatch(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    manifest_path = ds_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["images"][1]["scene_id"] = "scene_a"  # leaks across train & val
    manifest["images"][1]["guideline_version"] = "v1.0"  # mismatch vs v2.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = ValidationEngine.validate_dataset(ds_dir, contract)
    assert report["passed"] is False
    assert report["recommended_status"] == "HOLD"
    assert report["counts_by_fault_type"]["T7"] == 1
    assert report["counts_by_fault_type"]["T8"] == 1


def test_privacy_policy_blocks_unapproved_faces_or_pii(tmp_path: Path) -> None:
    ds_dir, contract = _create_valid_dataset(tmp_path)
    manifest_path = ds_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contains_faces"] = True
    manifest["privacy_approved"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = ValidationEngine.validate_dataset(ds_dir, contract)
    assert report["passed"] is False
    assert report["recommended_status"] == "HOLD"
    assert report["counts_by_fault_type"]["PRIVACY"] == 1
