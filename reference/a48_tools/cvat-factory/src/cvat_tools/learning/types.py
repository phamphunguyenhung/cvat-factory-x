"""Shared data types for CVAT-learning all-object detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    label: str
    box: list[float]  # [x1, y1, x2, y2] in pixel coordinates
    conf: float
    source_model: str = "unknown"
    occluded: bool = False
    truncated: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def to_cvat_shape(self, frame: int, label_id: int, z_order: int = 0) -> dict[str, Any]:
        attrs = []
        for k, v in self.attributes.items():
            attrs.append({"name": k, "value": str(v).lower() if isinstance(v, bool) else str(v)})
        return {
            "type": "rectangle",
            "frame": frame,
            "label_id": label_id,
            "points": [round(float(x), 2) for x in self.box],
            "occluded": bool(self.occluded),
            "z_order": z_order,
            "source": "manual",
            "attributes": attrs,
        }
