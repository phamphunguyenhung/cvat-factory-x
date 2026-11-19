"""Structural validator: manifest schema, T1 missing files, T2 SHA-256, T3 class/label integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .models import (
    ValidationIssue,
    parse_yolo_lines,
    resolve_image_id,
    resolve_image_relpath,
    resolve_label_relpath,
)

REQUIRED_MANIFEST_FIELDS = ("dataset_id", "guideline_version", "images", "classes")


def compute_sha256(file_path: Path) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_allowed_class_ids(
    manifest: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> set[int]:
    """Determine allowed class IDs from contract taxonomy or default 0..79."""
    contract = contract or {}
    if "valid_class_ids" in contract and contract["valid_class_ids"] is not None:
        return {int(cid) for cid in contract["valid_class_ids"]}

    if "num_classes" in contract and isinstance(contract["num_classes"], int):
        return set(range(contract["num_classes"]))

    contract_classes = contract.get("classes")
    if isinstance(contract_classes, dict) and contract_classes:
        parsed: set[int] = set()
        for k, v in contract_classes.items():
            if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                parsed.add(int(k))
            elif isinstance(v, int):
                parsed.add(v)
        if parsed:
            return parsed
    elif isinstance(contract_classes, list) and contract_classes:
        return set(range(len(contract_classes)))

    # Default COCO 80-class range 0..79
    return set(range(80))


def validate_manifest_schema(manifest: Any) -> tuple[list[ValidationIssue], bool]:
    """Validate manifest top-level schema and required fields."""
    issues: list[ValidationIssue] = []
    if not isinstance(manifest, dict):
        issues.append(
            ValidationIssue(
                code="SCHEMA_ERROR",
                fault_type="SCHEMA",
                severity="CRITICAL",
                message="Manifest must be a JSON object.",
            )
        )
        return issues, False

    for field_name in REQUIRED_MANIFEST_FIELDS:
        if field_name not in manifest:
            issues.append(
                ValidationIssue(
                    code="SCHEMA_ERROR",
                    fault_type="SCHEMA",
                    severity="CRITICAL",
                    message=f"Missing required manifest field '{field_name}'.",
                    details={"missing_field": field_name},
                )
            )

    if "dataset_id" in manifest and (
        not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"].strip()
    ):
        issues.append(
            ValidationIssue(
                code="SCHEMA_ERROR",
                fault_type="SCHEMA",
                severity="CRITICAL",
                message="Manifest field 'dataset_id' must be a non-empty string.",
            )
        )

    if "guideline_version" in manifest and (
        not isinstance(manifest["guideline_version"], str) or not manifest["guideline_version"].strip()
    ):
        issues.append(
            ValidationIssue(
                code="SCHEMA_ERROR",
                fault_type="SCHEMA",
                severity="CRITICAL",
                message="Manifest field 'guideline_version' must be a non-empty string.",
            )
        )

    if "images" in manifest and not isinstance(manifest["images"], list):
        issues.append(
            ValidationIssue(
                code="SCHEMA_ERROR",
                fault_type="SCHEMA",
                severity="CRITICAL",
                message="Manifest field 'images' must be a list.",
            )
        )

    if "classes" in manifest and not isinstance(manifest["classes"], (list, dict)):
        issues.append(
            ValidationIssue(
                code="SCHEMA_ERROR",
                fault_type="SCHEMA",
                severity="CRITICAL",
                message="Manifest field 'classes' must be a list or mapping.",
            )
        )

    return issues, len(issues) == 0


def validate_structural(
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> list[ValidationIssue]:
    """Run structural checks (schema, T1 missing file, T2 hash mismatch, T3 invalid class/row)."""
    schema_issues, schema_ok = validate_manifest_schema(manifest)
    if not schema_ok:
        return schema_issues

    issues: list[ValidationIssue] = list(schema_issues)
    dataset_root = Path(dataset_dir)
    allowed_classes = resolve_allowed_class_ids(manifest, contract)

    for idx, entry in enumerate(manifest["images"]):
        if not isinstance(entry, dict):
            issues.append(
                ValidationIssue(
                    code="SCHEMA_ERROR",
                    fault_type="SCHEMA",
                    severity="CRITICAL",
                    message=f"Image entry at index {idx} must be an object.",
                    details={"index": idx},
                )
            )
            continue

        image_id = resolve_image_id(entry, idx)
        img_rel = resolve_image_relpath(entry)

        # T1: Missing image file
        if img_rel is None:
            issues.append(
                ValidationIssue(
                    code="T1_MISSING_IMAGE",
                    fault_type="T1",
                    severity="CRITICAL",
                    message=f"Image entry '{image_id}' is missing an image path.",
                    image_id=image_id,
                )
            )
        else:
            img_path = dataset_root / img_rel
            if not img_path.is_file():
                issues.append(
                    ValidationIssue(
                        code="T1_MISSING_IMAGE",
                        fault_type="T1",
                        severity="CRITICAL",
                        message=f"Referenced image file does not exist on disk: '{img_rel}'.",
                        file_path=img_rel,
                        image_id=image_id,
                    )
                )
            else:
                # T2: SHA-256 verification for image
                expected_sha = entry.get("sha256") or entry.get("image_sha256")
                if isinstance(expected_sha, str) and expected_sha.strip():
                    actual_sha = compute_sha256(img_path)
                    if actual_sha.lower() != expected_sha.strip().lower():
                        issues.append(
                            ValidationIssue(
                                code="T2_WRONG_HASH",
                                fault_type="T2",
                                severity="CRITICAL",
                                message=f"SHA-256 mismatch for '{img_rel}'.",
                                file_path=img_rel,
                                image_id=image_id,
                                details={
                                    "expected_sha256": expected_sha.strip().lower(),
                                    "actual_sha256": actual_sha,
                                },
                            )
                        )

        # T1 / T2 / T3 for label file
        label_rel = resolve_label_relpath(entry)
        if label_rel is not None:
            label_path = dataset_root / label_rel
            if not label_path.is_file():
                issues.append(
                    ValidationIssue(
                        code="T1_MISSING_IMAGE",
                        fault_type="T1",
                        severity="CRITICAL",
                        message=f"Referenced label file does not exist on disk: '{label_rel}'.",
                        file_path=label_rel,
                        image_id=image_id,
                    )
                )
            else:
                expected_label_sha = entry.get("label_sha256")
                if isinstance(expected_label_sha, str) and expected_label_sha.strip():
                    actual_label_sha = compute_sha256(label_path)
                    if actual_label_sha.lower() != expected_label_sha.strip().lower():
                        issues.append(
                            ValidationIssue(
                                code="T2_WRONG_HASH",
                                fault_type="T2",
                                severity="CRITICAL",
                                message=f"SHA-256 mismatch for label file '{label_rel}'.",
                                file_path=label_rel,
                                image_id=image_id,
                                details={
                                    "expected_sha256": expected_label_sha.strip().lower(),
                                    "actual_sha256": actual_label_sha,
                                },
                            )
                        )

                boxes, row_errors = parse_yolo_lines(label_path.read_text(encoding="utf-8"))
                for line_no, raw_line, reason in row_errors:
                    issues.append(
                        ValidationIssue(
                            code="T3_INVALID_CLASS",
                            fault_type="T3",
                            severity="CRITICAL",
                            message=f"Malformed label row in '{label_rel}' line {line_no}: {reason}.",
                            file_path=label_rel,
                            image_id=image_id,
                            line_number=line_no,
                            details={"raw_line": raw_line, "reason": reason},
                        )
                    )
                for box in boxes:
                    if box.class_id not in allowed_classes:
                        issues.append(
                            ValidationIssue(
                                code="T3_INVALID_CLASS",
                                fault_type="T3",
                                severity="CRITICAL",
                                message=(
                                    f"Invalid class_id {box.class_id} in '{label_rel}' "
                                    f"line {box.line_number}; not in contract taxonomy."
                                ),
                                file_path=label_rel,
                                image_id=image_id,
                                line_number=box.line_number,
                                details={"class_id": box.class_id},
                            )
                        )

        # T3 for inline annotations if present
        inline = entry.get("annotations") or entry.get("boxes")
        if isinstance(inline, list):
            for ann_idx, item in enumerate(inline, start=1):
                if not isinstance(item, dict):
                    issues.append(
                        ValidationIssue(
                            code="T3_INVALID_CLASS",
                            fault_type="T3",
                            severity="CRITICAL",
                            message=f"Malformed inline annotation at index {ann_idx} for '{image_id}'.",
                            image_id=image_id,
                            line_number=ann_idx,
                        )
                    )
                    continue
                cls_id = item.get("class_id")
                if not isinstance(cls_id, int) or isinstance(cls_id, bool) or cls_id not in allowed_classes:
                    issues.append(
                        ValidationIssue(
                            code="T3_INVALID_CLASS",
                            fault_type="T3",
                            severity="CRITICAL",
                            message=(
                                f"Invalid inline class_id {cls_id!r} for '{image_id}' "
                                f"at annotation {ann_idx}."
                            ),
                            image_id=image_id,
                            line_number=ann_idx,
                            details={"class_id": cls_id},
                        )
                    )

    return issues
