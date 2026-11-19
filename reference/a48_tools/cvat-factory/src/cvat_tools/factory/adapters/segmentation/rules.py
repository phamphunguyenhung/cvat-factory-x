"""Segmentation rules: mask<->polygon conversion, mask read-back, layer ordering, and geometry validation.

Zero heavyweight imports (shapely, cv2, numpy) at module import time; imported lazily inside functions.
"""
from __future__ import annotations

from typing import Any, Sequence

from cvat_tools.factory.core.models import (
    Proposal,
    QualityVerdict,
    TaskSchema,
    ValidationIssue,
    ValidationReport,
)

# Canonical layer ordering matching cvat_tools.segmentation.layers.LAYER_ORDER
LAYER_ORDER: dict[str, int] = {
    "sky": -9,
    "building": -8,
    "vegetation": -8,
    "terrain": -8,
    "wall": -8,
    "fence": -8,
    "sidewalk": -7,
    "pole": -2,
    "road": -1,
    "area/alternative": 0,
    "area/direct": 0,
    "lane/crosswalk": 0,
    "lane/road curb": 0,
    "lane/single white": 0,
    "lane/single yellow": 0,
    "lane/double white": 0,
    "lane/double yellow": 0,
    "lane/single other": 0,
    "lane/double other": 0,
    "traffic sign": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "train": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "pedestrian": 0,
    "rider": 0,
    "traffic light": 0,
}

LABEL_ALIASES: dict[str, str] = {
    "person": "pedestrian",
    "traffic_light": "traffic light",
    "traffic_sign": "traffic sign",
}


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label.strip(), label.strip())


def layer_z_order(label: str) -> int:
    return LAYER_ORDER.get(canonical_label(label), 0)


def mask_to_flat_polygon(
    mask: Any,
    *,
    max_points: int = 40,
    eps: float = 1.0,
) -> list[float] | None:
    """Convert a 2D binary mask into a flat [x1, y1, ...] polygon using sam_refine.mask_to_polygon."""
    from cvat_tools.segmentation.sam_refine import mask_to_polygon

    return mask_to_polygon(mask, max_points=max_points, eps=eps)


def compute_mask_readback_iou(
    polygon_flat: Sequence[float],
    source_mask: Any,
    width: int | None = None,
    height: int | None = None,
) -> float:
    """Rasterize a flat polygon back to a binary mask and compute IoU against source_mask."""
    import cv2
    import numpy as np

    src = np.asarray(source_mask) > 0
    h, w = src.shape[:2] if src.ndim >= 2 else (int(height or 0), int(width or 0))
    if h <= 0 or w <= 0 or len(polygon_flat) < 6:
        return 0.0
    pts = np.round(np.asarray(polygon_flat, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
    raster = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(raster, [pts], 1)
    pred = raster > 0
    inter = int(np.logical_and(pred, src).sum())
    union = int(np.logical_or(pred, src).sum())
    return round(float(inter / union), 4) if union > 0 else 0.0


def check_polygon_validity(
    polygon_flat: Sequence[float],
    *,
    min_area: float = 4.0,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[bool, bool, float, list[ValidationIssue]]:
    """Validate polygon geometry: >=3 vertices, finite, no self-intersection, area >= min_area, in-bounds."""
    from shapely.geometry import Polygon

    issues: list[ValidationIssue] = []
    if len(polygon_flat) < 6 or len(polygon_flat) % 2 != 0:
        issues.append(
            ValidationIssue(
                code="SEG_INVALID_VERTEX_COUNT",
                severity="error",
                message=f"Polygon needs >=6 even coordinates, got {len(polygon_flat)}",
                region="geometry",
            )
        )
        return False, False, 0.0, issues

    coords = [(float(polygon_flat[i]), float(polygon_flat[i + 1])) for i in range(0, len(polygon_flat), 2)]
    if not all(abs(x) < 1e9 and abs(y) < 1e9 and x == x and y == y for x, y in coords):
        issues.append(
            ValidationIssue(
                code="SEG_NON_FINITE_COORDS",
                severity="error",
                message="Polygon contains NaN or Inf coordinates",
                region="geometry",
            )
        )
        return False, False, 0.0, issues

    poly = Polygon(coords)
    geometry_valid = True
    alignment_good = True

    if not poly.is_valid:
        geometry_valid = False
        issues.append(
            ValidationIssue(
                code="SEG_SELF_INTERSECTION",
                severity="error",
                message="Polygon has self-intersection or invalid ring topology",
                region="topology",
            )
        )

    area = float(poly.area)
    if area < min_area:
        geometry_valid = False
        issues.append(
            ValidationIssue(
                code="SEG_DEGENERATE_AREA",
                severity="warning",
                message=f"Polygon area {area:.2f} < min_area={min_area}",
                region="geometry",
            )
        )

    if image_width is not None and image_height is not None:
        minx, miny, maxx, maxy = poly.bounds
        if minx < -1.0 or miny < -1.0 or maxx > image_width + 1.0 or maxy > image_height + 1.0:
            alignment_good = False
            issues.append(
                ValidationIssue(
                    code="SEG_OUT_OF_BOUNDS",
                    severity="warning",
                    message=f"Polygon bounds exceed frame ({image_width}x{image_height})",
                    region="boundary",
                )
            )

    return geometry_valid, alignment_good, area, issues


def validate_segmentation_proposal(
    proposal: Proposal,
    *,
    task_schema: TaskSchema | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    min_area: float = 4.0,
    min_conf: float = 0.25,
    min_readback_iou: float = 0.70,
) -> ValidationReport:
    """Validate a single polygon/mask segmentation Proposal."""
    issues: list[ValidationIssue] = []
    schema_valid = True
    confidence_sufficient = True

    if proposal.shape_type not in ("polygon", "mask"):
        issues.append(
            ValidationIssue(
                code="SEG_INVALID_SHAPE_TYPE",
                severity="error",
                message=f"Expected polygon/mask, got {proposal.shape_type!r}",
                region="geometry",
            )
        )
        return ValidationReport(
            proposal_id=proposal.proposal_id,
            verdict=QualityVerdict.REVIEW,
            geometry_valid=False,
            image_alignment_good=False,
            schema_valid=False,
            confidence_sufficient=False,
            no_regression=True,
            issues=issues,
        )

    pts = proposal.geometry.get("points") or []
    geometry_valid, alignment_good, area, geom_issues = check_polygon_validity(
        pts,
        min_area=min_area,
        image_width=image_width,
        image_height=image_height,
    )
    issues.extend(geom_issues)

    expected_z = layer_z_order(proposal.label)
    if proposal.z_order != expected_z and canonical_label(proposal.label) in LAYER_ORDER:
        issues.append(
            ValidationIssue(
                code="SEG_LAYER_ORDER_MISMATCH",
                severity="warning",
                message=f"z_order={proposal.z_order} != expected {expected_z} for {proposal.label}",
                region="layer_order",
            )
        )

    readback_iou = proposal.metadata.get("mask_readback_iou")
    if readback_iou is not None and float(readback_iou) < min_readback_iou:
        alignment_good = False
        issues.append(
            ValidationIssue(
                code="SEG_MASK_READBACK_MISMATCH",
                severity="warning",
                message=f"Mask read-back IoU {float(readback_iou):.3f} < {min_readback_iou:.2f}",
                region="mask_readback",
            )
        )

    if task_schema is not None:
        lbl = task_schema.label_by_name(proposal.label)
        if lbl is None:
            schema_valid = False
            issues.append(
                ValidationIssue(
                    code="SEG_UNKNOWN_LABEL",
                    severity="error",
                    message=f"Label {proposal.label!r} not in task schema",
                    region="schema",
                )
            )

    if proposal.confidence is not None and proposal.confidence < min_conf:
        confidence_sufficient = False
        issues.append(
            ValidationIssue(
                code="SEG_LOW_CONFIDENCE",
                severity="warning",
                message=f"Confidence {proposal.confidence:.3f} < {min_conf:.2f}",
                region="confidence",
            )
        )

    if not schema_valid:
        verdict = QualityVerdict.REVIEW
    elif not geometry_valid:
        verdict = QualityVerdict.REFINE if any(i.code == "SEG_SELF_INTERSECTION" for i in issues) else QualityVerdict.REVIEW
    elif not alignment_good or not confidence_sufficient or any(i.code == "SEG_LAYER_ORDER_MISMATCH" for i in issues):
        verdict = QualityVerdict.REFINE if alignment_good or any(i.code in ("SEG_OUT_OF_BOUNDS", "SEG_LAYER_ORDER_MISMATCH") for i in issues) else QualityVerdict.REVIEW
    else:
        verdict = QualityVerdict.PASS

    return ValidationReport(
        proposal_id=proposal.proposal_id,
        verdict=verdict,
        geometry_valid=geometry_valid,
        image_alignment_good=alignment_good,
        schema_valid=schema_valid,
        confidence_sufficient=confidence_sufficient,
        no_regression=True,
        issues=issues,
        metrics={"area": round(area, 2), "mask_readback_iou": readback_iou},
    )


def validate_frame_shared_boundaries(
    proposals: Sequence[Proposal],
    reports: dict[str, ValidationReport],
    *,
    overlap_ratio_threshold: float = 0.05,
) -> None:
    """Check stuff-over-thing overlaps and shared-boundary conflicts per frame."""
    from cvat_tools.segmentation.stuff_things import STUFF_LABELS, THING_LABELS, ThingRegion, overlap_report

    by_frame: dict[int, list[Proposal]] = {}
    for p in proposals:
        by_frame.setdefault(p.frame_id, []).append(p)

    for frame_id, frame_props in by_frame.items():
        things = [
            ThingRegion(label=canonical_label(p.label), polygon=list(p.geometry.get("points") or []), source="mask")
            for p in frame_props
            if canonical_label(p.label) in THING_LABELS and len(p.geometry.get("points") or []) >= 6
        ]
        if not things:
            continue
        for p in frame_props:
            if canonical_label(p.label) not in STUFF_LABELS:
                continue
            pts = p.geometry.get("points") or []
            if len(pts) < 6:
                continue
            overlaps = [r for r in overlap_report(pts, things) if r["ratio_of_thing"] >= overlap_ratio_threshold]
            if overlaps:
                rep = reports[p.proposal_id]
                rep.issues.append(
                    ValidationIssue(
                        code="SEG_STUFF_OVER_THING_BLEED",
                        severity="warning",
                        message=f"Stuff {p.label} overlaps {len(overlaps)} thing(s) (max ratio={max(o['ratio_of_thing'] for o in overlaps):.2f})",
                        region="shared_boundary",
                        details={"overlaps": overlaps},
                    )
                )
                if rep.verdict == QualityVerdict.PASS:
                    rep.verdict = QualityVerdict.REFINE
