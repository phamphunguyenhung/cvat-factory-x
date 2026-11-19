"""Tracking adapter package."""
from __future__ import annotations

from cvat_tools.factory.adapters.tracking.adapter import TrackingAdapter
from cvat_tools.factory.adapters.tracking.rules import (
    detect_fragmentation_and_id_switches,
    validate_track_proposal,
)

__all__ = [
    "TrackingAdapter",
    "detect_fragmentation_and_id_switches",
    "validate_track_proposal",
]
