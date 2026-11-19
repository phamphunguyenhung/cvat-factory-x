"""Geometry validator: T4 out-of-bounds / non-positive boxes and duplicate box detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    NormalizedBox,
    ValidationIssue,
    load_boxes_for_entry,
    resolve_image_id,
    resolve_label_relpath,
)

DEFAULT_BBOX_TOLERANCE = 1e-4
DEFAULT_DUPLICATE_DELTA = 0.005


def _check_normalized_box(
    box: NormalizedBox,
    tol: float,
    image_id: str,
    file_path: str | None,
) -> ValidationIssue | None:
    """Return a T4_BBOX_OUT_OF_BOUNDS issue if a normalized box is invalid or out of [0, 1]."""
    if box.w <= 0.0 or box.h <= 0.0:
        return ValidationIssue(
            code="T4_BBOX_OUT_OF_BOUNDS",
            fault_type="T4",
            severity="CRITICAL",
            message=(
                f"Non-positive bbox dimensions (w={box.w}, h={box.h}) in '{image_id}' "
                f"at line {box.line_number}."
            ),
            file_path=file_path,
            image_id=image_id,
            line_number=box.line_number,
            details={"cx": box.cx, "cy": box.cy, "w": box.w, "h": box.h},
        )

    out_of_bounds = (
        box.cx < -tol
        or box.cx > 1.0 + tol
        or box.cy < -tol
        or box.cy > 1.0 + tol
        or box.w > 1.0 + tol
        or box.h > 1.0 + tol
        or box.x1 < -tol
        or box.y1 < -tol
        or box.x2 > 1.0 + tol
        or box.y2 > 1.0 + tol
    )
    if out_of_bounds:
        return ValidationIssue(
            code="T4_BBOX_OUT_OF_BOUNDS",
            fault_type="T4",
            severity="CRITICAL",
            message=(
                f"Normalized bbox out of [0, 1] bounds (tol={tol}) in '{image_id}' "
                f"at line {box.line_number}."
            ),
            file_path=file_path,
            image_id=image_id,
            line_number=box.line_number,
            details={
                "cx": box.cx,
                "cy": box.cy,
                "w": box.w,
                "h": box.h,
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
            },
        )
    return None


def _check_pixel_boxes(
    entry: dict[str, Any],
    tol: float,
    image_id: str,
) -> list[ValidationIssue]:
    """Validate inline pixel-space bboxes against [0, width] x [0, height]."""
    issues: list[ValidationIssue] = []
    img_w = entry.get("width")
    img_h = entry.get("height")
    inline = entry.get("annotations") or entry.get("boxes")
    if not isinstance(inline, list):
        return issues

    has_dims = isinstance(img_w, (int, float)) and img_w > 0 and isinstance(img_h, (int, float)) and img_h > 0
    for idx, item in enumerate(inline, start=1):
        if not isinstance(item, dict) or "bbox" not in item:
            continue
        raw_bbox = item["bbox"]
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in raw_bbox)
        pw = x2 - x1
        ph = y2 - y1
        if pw <= 0.0 or ph <= 0.0:
            issues.append(
                ValidationIssue(
                    code="T4_BBOX_OUT_OF_BOUNDS",
                    fault_type="T4",
                    severity="CRITICAL",
                    message=f"Non-positive pixel bbox dimensions (w={pw}, h={ph}) in '{image_id}' at {idx}.",
                    image_id=image_id,
                    line_number=idx,
                    details={"bbox": [x1, y1, x2, y2]},
                )
            )
            continue

        if has_dims and (
            x1 < -tol or y1 < -tol or x2 > float(img_w) + tol or y2 > float(img_h) + tol
        ):
            issues.append(
                ValidationIssue(
                    code="T4_BBOX_OUT_OF_BOUNDS",
                    fault_type="T4",
                    severity="CRITICAL",
                    message=(
                        f"Pixel bbox {[x1, y1, x2, y2]} exceeds image bounds "
                        f"[0, {img_w}] x [0, {img_h}] (tol={tol}) in '{image_id}'."
                    ),
                    image_id=image_id,
                    line_number=idx,
                    details={"bbox": [x1, y1, x2, y2], "width": img_w, "height": img_h},
                )
            )
    return issues


def validate_geometry(
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> list[ValidationIssue]:
    """Run geometry checks (T4 out-of-bounds, non-positive dims, duplicate boxes)."""
    contract = contract or {}
    tol = float(contract.get("bbox_tolerance", DEFAULT_BBOX_TOLERANCE))
    center_delta = float(contract.get("duplicate_center_delta", DEFAULT_DUPLICATE_DELTA))
    size_delta = float(contract.get("duplicate_size_delta", DEFAULT_DUPLICATE_DELTA))

    dataset_root = Path(dataset_dir)
    issues: list[ValidationIssue] = []

    images = manifest.get("images", [])
    if not isinstance(images, list):
        return issues

    for idx, entry in enumerate(images):
        if not isinstance(entry, dict):
            continue
        image_id = resolve_image_id(entry, idx)
        label_rel = resolve_label_relpath(entry)

        # Direct pixel-box bounds check if inline pixel boxes exist
        pixel_issues = _check_pixel_boxes(entry, tol, image_id)
        issues.extend(pixel_issues)

        boxes = load_boxes_for_entry(dataset_root, entry)
        # Avoid double-reporting if pixel boxes already reported for inline entries
        if not pixel_issues:
            for box in boxes:
                oob = _check_normalized_box(box, tol, image_id, label_rel)
                if oob is not None:
                    issues.append(oob)

        # Duplicate box detection (same class, center delta <= 0.005, size delta <= 0.005)
        for i in range(len(boxes)):
            b1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                b2 = boxes[j]
                if b1.class_id != b2.class_id:
                    continue
                if (
                    abs(b1.cx - b2.cx) <= center_delta
                    and abs(b1.cy - b2.cy) <= center_delta
                    and abs(b1.w - b2.w) <= size_delta
                    and abs(b1.h - b2.h) <= size_delta
                ):
                    issues.append(
                        ValidationIssue(
                            code="T4_DUPLICATE_BBOX",
                            fault_type="T4",
                            severity="WARNING",
                            message=(
                                f"Duplicate bbox for class_id {b1.class_id} in '{image_id}' "
                                f"(lines {b1.line_number} and {b2.line_number})."
                            ),
                            file_path=label_rel,
                            image_id=image_id,
                            line_number=b2.line_number,
                            requires_human_review=True,
                            details={
                                "class_id": b1.class_id,
                                "first_line": b1.line_number,
                                "second_line": b2.line_number,
                                "center_delta": [abs(b1.cx - b2.cx), abs(b1.cy - b2.cy)],
                                "size_delta": [abs(b1.w - b2.w), abs(b1.h - b2.h)],
                            },
                        )
                    )

    return issues
