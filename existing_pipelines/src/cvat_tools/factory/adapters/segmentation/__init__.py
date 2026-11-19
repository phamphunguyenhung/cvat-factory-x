"""Segmentation adapter package."""
from __future__ import annotations

from cvat_tools.factory.adapters.segmentation.adapter import SegmentationAdapter
from cvat_tools.factory.adapters.segmentation.rules import (
    LAYER_ORDER,
    canonical_label,
    check_polygon_validity,
    compute_mask_readback_iou,
    layer_z_order,
    mask_to_flat_polygon,
    validate_segmentation_proposal,
)

__all__ = [
    "LAYER_ORDER",
    "SegmentationAdapter",
    "canonical_label",
    "check_polygon_validity",
    "compute_mask_readback_iou",
    "layer_z_order",
    "mask_to_flat_polygon",
    "validate_segmentation_proposal",
]
