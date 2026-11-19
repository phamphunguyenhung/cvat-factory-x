"""Skeleton AnnotationAdapter package for Pose17, Face50, and custom skeletons."""
from __future__ import annotations

from cvat_tools.factory.adapters.skeleton.adapter import SkeletonAdapter
from cvat_tools.factory.adapters.skeleton.schemas import (
    FACE50_GROUP_NAMES,
    POSE17_COCO_NAMES,
    SkeletonProfile,
    build_skeleton_geometry,
    detect_skeleton_profile,
    extract_skeleton_schemas,
    is_skeleton_task,
    normalize_keypoint_record,
    sublabel_lookup_from_task,
)

__all__ = [
    "FACE50_GROUP_NAMES",
    "POSE17_COCO_NAMES",
    "SkeletonAdapter",
    "SkeletonProfile",
    "build_skeleton_geometry",
    "detect_skeleton_profile",
    "extract_skeleton_schemas",
    "is_skeleton_task",
    "normalize_keypoint_record",
    "sublabel_lookup_from_task",
]
