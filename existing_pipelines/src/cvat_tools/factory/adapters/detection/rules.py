"""Detection taxonomy mapping, bbox geometry validation, duplicate/overlap, and calibration rules.

Zero heavyweight imports at module import time; numpy/cv2 are imported lazily only when needed.
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

# Canonical COCO / alias -> CVAT label mapping (7-class & 10-class traffic + common aliases)
COCO_TO_CVAT_LABELS: dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "train": "train",
    "traffic light": "traffic light",
    "traffic_light": "traffic light",
    "traffic sign": "traffic sign",
    "traffic_sign": "traffic sign",
    "stop sign": "traffic sign",
}

SEVEN_CLASS_TRAFFIC: frozenset[str] = frozenset(
    {"car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "rider"}
)
VEHICLE_4W_GROUP: frozenset[str] = frozenset({"car", "truck", "bus", "train"})
TWO_WHEELER_GROUP: frozenset[str] = frozenset({"motorcycle", "bicycle"})


def normalize_label_key(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def map_raw_label(raw_label: str, task_schema: TaskSchema | None = None) -> tuple[str, int | None]:
    """Map a raw model label to the canonical CVAT label name and label_id (if in task_schema)."""
    key = normalize_label_key(raw_label)
    canon = COCO_TO_CVAT_LABELS.get(key, raw_label.strip())

    if task_schema is None:
        return canon, None

    # Exact match first
    lbl = task_schema.label_by_name(canon)
    if lbl is not None:
        return lbl.name, lbl.id

    # Normalized match across task_schema.labels
    for candidate in task_schema.labels:
        if normalize_label_key(candidate.name) in (normalize_label_key(canon), key):
            return candidate.name, candidate.id

    return canon, None


def compute_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def calibrate_confidence(
    raw_conf: float,
    label: str,
    *,
    is_night: bool = False,
    is_small: bool = False,
    from_tile: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    """Apply deterministic confidence calibration and return (calibrated_conf, rule_decisions)."""
    conf = max(0.0, min(1.0, float(raw_conf)))
    decisions: list[dict[str, Any]] = []

    if is_night:
        # Night frames have lower raw detector logits; apply bounded calibration uplift for ranking
        adj = min(1.0, conf * 1.15)
        decisions.append({"rule": "night_conf_calibration", "raw": round(conf, 4), "calibrated": round(adj, 4)})
        conf = adj

    if from_tile and is_small:
        adj = min(1.0, conf * 1.05)
        decisions.append({"rule": "small_tile_calibration", "raw": round(conf, 4), "calibrated": round(adj, 4)})
        conf = adj

    if label in ("traffic light", "traffic sign") and conf < 0.25:
        decisions.append({"rule": "small_traffic_cue_floor_flag", "calibrated": round(conf, 4)})

    return round(conf, 4), decisions


def validate_bbox_proposal(
    proposal: Proposal,
    *,
    task_schema: TaskSchema | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    min_box_px: float = 6.0,
    min_conf: float = 0.20,
) -> ValidationReport:
    """Validate a single rectangle Proposal on geometry, alignment, schema, and confidence."""
    issues: list[ValidationIssue] = []
    pts = proposal.geometry.get("points") or []
    geometry_valid = True
    image_alignment_good = True
    schema_valid = True
    confidence_sufficient = True

    if proposal.shape_type != "rectangle":
        geometry_valid = False
        issues.append(
            ValidationIssue(
                code="BBOX_INVALID_SHAPE_TYPE",
                severity="error",
                message=f"Expected rectangle, got {proposal.shape_type!r}",
                region="geometry",
            )
        )

    if len(pts) != 4:
        geometry_valid = False
        issues.append(
            ValidationIssue(
                code="BBOX_INVALID_POINT_COUNT",
                severity="error",
                message=f"Expected 4 coordinates [x1,y1,x2,y2], got {len(pts)}",
                region="geometry",
            )
        )
        x1 = y1 = x2 = y2 = 0.0
    else:
        x1, y1, x2, y2 = (float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3]))
        if not all(abs(v) < 1e9 and v == v for v in (x1, y1, x2, y2)):
            geometry_valid = False
            issues.append(
                ValidationIssue(
                    code="BBOX_NON_FINITE_COORDS",
                    severity="error",
                    message="Bounding box contains NaN or Inf coordinates",
                    region="geometry",
                )
            )
        elif x2 <= x1 or y2 <= y1:
            geometry_valid = False
            issues.append(
                ValidationIssue(
                    code="BBOX_INVERTED_OR_ZERO_AREA",
                    severity="error",
                    message=f"Invalid box ordering: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]",
                    region="geometry",
                )
            )
        else:
            bw, bh = x2 - x1, y2 - y1
            if bw < min_box_px or bh < min_box_px:
                geometry_valid = False
                issues.append(
                    ValidationIssue(
                        code="BBOX_TOO_SMALL",
                        severity="warning",
                        message=f"Box {bw:.1f}x{bh:.1f}px is below min_box_px={min_box_px}",
                        region="geometry",
                    )
                )

            if image_width is not None and image_height is not None:
                if x1 < -1.0 or y1 < -1.0 or x2 > image_width + 1.0 or y2 > image_height + 1.0:
                    image_alignment_good = False
                    issues.append(
                        ValidationIssue(
                            code="BBOX_OUT_OF_BOUNDS",
                            severity="warning",
                            message=f"Box exceeds frame bounds ({image_width}x{image_height})",
                            region="boundary",
                        )
                    )
                cy = (y1 + y2) / 2.0
                if cy > 0.88 * image_height and proposal.label not in ("traffic light", "traffic sign"):
                    image_alignment_good = False
                    issues.append(
                        ValidationIssue(
                            code="BBOX_HOOD_REFLECTION_RISK",
                            severity="warning",
                            message=f"Box center y={cy:.1f} is in lower hood band (>0.88H)",
                            region="lower_frame",
                        )
                    )
                if (bw * bh) > 0.30 * image_width * image_height and (proposal.confidence or 0.0) < 0.35:
                    image_alignment_good = False
                    issues.append(
                        ValidationIssue(
                            code="BBOX_OVERSIZED_LOW_CONF",
                            severity="warning",
                            message="Oversized box (>30% of frame) with low confidence",
                            region="size",
                        )
                    )

    if task_schema is not None:
        lbl = task_schema.label_by_name(proposal.label)
        if lbl is None:
            schema_valid = False
            issues.append(
                ValidationIssue(
                    code="BBOX_UNKNOWN_LABEL",
                    severity="error",
                    message=f"Label {proposal.label!r} not found in task schema",
                    region="schema",
                )
            )
        elif proposal.geometry.get("label_id") is not None and int(proposal.geometry["label_id"]) != lbl.id:
            schema_valid = False
            issues.append(
                ValidationIssue(
                    code="BBOX_LABEL_ID_MISMATCH",
                    severity="error",
                    message=f"label_id {proposal.geometry['label_id']} != schema {lbl.id}",
                    region="schema",
                )
            )

    if proposal.confidence is not None and proposal.confidence < min_conf:
        confidence_sufficient = False
        issues.append(
            ValidationIssue(
                code="BBOX_LOW_CONFIDENCE",
                severity="warning",
                message=f"Confidence {proposal.confidence:.3f} < {min_conf:.2f}",
                region="confidence",
            )
        )

    if not geometry_valid or not schema_valid:
        verdict = QualityVerdict.REVIEW if any(i.code in ("BBOX_UNKNOWN_LABEL", "BBOX_NON_FINITE_COORDS", "BBOX_INVALID_POINT_COUNT", "BBOX_INVALID_SHAPE_TYPE") for i in issues) else QualityVerdict.REFINE
    elif not image_alignment_good or not confidence_sufficient or issues:
        verdict = QualityVerdict.REFINE if any(i.code == "BBOX_OUT_OF_BOUNDS" for i in issues) and confidence_sufficient else QualityVerdict.REVIEW
    else:
        verdict = QualityVerdict.PASS

    return ValidationReport(
        proposal_id=proposal.proposal_id,
        verdict=verdict,
        geometry_valid=geometry_valid,
        image_alignment_good=image_alignment_good,
        schema_valid=schema_valid,
        confidence_sufficient=confidence_sufficient,
        no_regression=True,
        issues=issues,
        metrics={"confidence": proposal.confidence},
    )


def detect_duplicates_and_missing(
    proposals: Sequence[Proposal],
    reports: dict[str, ValidationReport],
    *,
    dup_iou: float = 0.70,
    reference_boxes: Sequence[dict[str, Any]] | None = None,
    missing_iou: float = 0.30,
) -> None:
    """Mutate reports in-place with duplicate/overlap and missing-object candidate issues per frame."""
    by_frame: dict[int, list[Proposal]] = {}
    for p in proposals:
        by_frame.setdefault(p.frame_id, []).append(p)

    for frame_id, frame_props in by_frame.items():
        n = len(frame_props)
        for i in range(n):
            pi = frame_props[i]
            bi = pi.geometry.get("points") or []
            if len(bi) != 4:
                continue
            for j in range(i + 1, n):
                pj = frame_props[j]
                bj = pj.geometry.get("points") or []
                if len(bj) != 4:
                    continue
                iou = compute_iou(bi, bj)
                same_cls = pi.label == pj.label
                same_4w = pi.label in VEHICLE_4W_GROUP and pj.label in VEHICLE_4W_GROUP
                if (same_cls and iou >= dup_iou) or (same_4w and iou >= 0.75):
                    # Flag lower-confidence one for REFINE/REVIEW
                    loser = pj if (pi.confidence or 0.0) >= (pj.confidence or 0.0) else pi
                    rep = reports[loser.proposal_id]
                    rep.issues.append(
                        ValidationIssue(
                            code="BBOX_DUPLICATE_OVERLAP",
                            severity="warning",
                            message=f"Overlaps {pi.label}/{pj.label} with IoU={iou:.2f}",
                            region="overlap",
                            details={"iou": round(iou, 3), "peer_id": (pi.proposal_id if loser is pj else pj.proposal_id)},
                        )
                    )
                    if rep.verdict == QualityVerdict.PASS:
                        rep.verdict = QualityVerdict.REFINE

        if reference_boxes:
            refs_for_frame = [r for r in reference_boxes if int(r.get("frame_id", frame_id)) == frame_id]
            for ref in refs_for_frame:
                rbox = ref.get("points") or ref.get("box") or []
                if len(rbox) != 4:
                    continue
                matched = any(
                    len(p.geometry.get("points") or []) == 4 and compute_iou(p.geometry["points"], rbox) >= missing_iou
                    for p in frame_props
                )
                if not matched and frame_props:
                    rep = reports[frame_props[0].proposal_id]
                    rep.issues.append(
                        ValidationIssue(
                            code="BBOX_MISSING_OBJECT_CANDIDATE",
                            severity="info",
                            message=f"Unmatched reference candidate {ref.get('label', 'object')} at {[round(float(x), 1) for x in rbox]}",
                            region="missing_candidate",
                            details={"reference": ref},
                        )
                    )
                    if rep.verdict == QualityVerdict.PASS:
                        rep.verdict = QualityVerdict.REVIEW
