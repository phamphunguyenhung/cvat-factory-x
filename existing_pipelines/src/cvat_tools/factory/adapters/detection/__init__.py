"""Detection (BBox) adapter package."""
from __future__ import annotations

from cvat_tools.factory.adapters.detection.adapter import DetectionAdapter
from cvat_tools.factory.adapters.detection.rules import (
    COCO_TO_CVAT_LABELS,
    SEVEN_CLASS_TRAFFIC,
    calibrate_confidence,
    compute_iou,
    map_raw_label,
    validate_bbox_proposal,
)

__all__ = [
    "COCO_TO_CVAT_LABELS",
    "DetectionAdapter",
    "SEVEN_CLASS_TRAFFIC",
    "calibrate_confidence",
    "compute_iou",
    "map_raw_label",
    "validate_bbox_proposal",
]
