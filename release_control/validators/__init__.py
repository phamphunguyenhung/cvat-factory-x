"""Validation Engine for CVAT Factory X."""

from .gate import FAULT_TYPES, ValidationEngine, ValidationReport
from .geometry import validate_geometry
from .models import NormalizedBox, ValidationIssue, compute_iou
from .privacy import validate_privacy
from .semantic import validate_semantic
from .structural import (
    compute_sha256,
    resolve_allowed_class_ids,
    validate_manifest_schema,
    validate_structural,
)

__all__ = [
    "FAULT_TYPES",
    "NormalizedBox",
    "ValidationEngine",
    "ValidationIssue",
    "ValidationReport",
    "compute_iou",
    "compute_sha256",
    "resolve_allowed_class_ids",
    "validate_geometry",
    "validate_manifest_schema",
    "validate_privacy",
    "validate_semantic",
    "validate_structural",
]
