"""Shared data models and deterministic parsing helpers for CVAT Factory X validators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ValidationIssue:
    """Structured validation issue emitted by a validator."""

    code: str
    fault_type: str
    severity: str
    message: str
    file_path: Optional[str] = None
    image_id: Optional[str] = None
    line_number: Optional[int] = None
    requires_human_review: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize issue to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class NormalizedBox:
    """Normalized YOLO bounding box representation."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    line_number: Optional[int] = None

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2.0


def resolve_image_id(entry: dict[str, Any], index: int) -> str:
    """Return a deterministic identifier for a manifest image entry."""
    for key in ("image_id", "id", "file_name", "image_path", "path"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return f"image_{index}"


def resolve_image_relpath(entry: dict[str, Any]) -> Optional[str]:
    """Return the relative image path from a manifest image entry."""
    for key in ("image_path", "file_name", "path"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def resolve_label_relpath(entry: dict[str, Any]) -> Optional[str]:
    """Return the relative label path from a manifest image entry if specified."""
    for key in ("label_path", "label_file"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def parse_yolo_lines(text: str) -> tuple[list[NormalizedBox], list[tuple[int, str, str]]]:
    """Parse YOLO label text into boxes and malformed row tuples (line_no, raw_line, reason)."""
    boxes: list[NormalizedBox] = []
    errors: list[tuple[int, str, str]] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            errors.append((line_no, raw_line, f"expected 5 tokens, got {len(parts)}"))
            continue

        cls_token, cx_tok, cy_tok, w_tok, h_tok = parts
        try:
            # Reject floats like '1.0' or '1.5' in class_id column
            if "." in cls_token or "e" in cls_token.lower():
                raise ValueError("class_id must be an integer")
            class_id = int(cls_token)
        except ValueError:
            errors.append((line_no, raw_line, f"invalid integer class_id '{cls_token}'"))
            continue

        try:
            cx, cy, w, h = (float(cx_tok), float(cy_tok), float(w_tok), float(h_tok))
        except ValueError:
            errors.append((line_no, raw_line, "non-numeric bbox coordinates"))
            continue

        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            errors.append((line_no, raw_line, "non-finite bbox coordinates"))
            continue

        boxes.append(NormalizedBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h, line_number=line_no))

    return boxes, errors


def load_boxes_for_entry(
    dataset_root: Path,
    entry: dict[str, Any],
) -> list[NormalizedBox]:
    """Load normalized boxes from an entry's YOLO label file or inline annotations."""
    label_rel = resolve_label_relpath(entry)
    if label_rel:
        label_file = dataset_root / label_rel
        if label_file.is_file():
            boxes, _ = parse_yolo_lines(label_file.read_text(encoding="utf-8"))
            return boxes

    inline = entry.get("annotations") or entry.get("boxes") or []
    boxes: list[NormalizedBox] = []
    if isinstance(inline, list):
        img_w = entry.get("width")
        img_h = entry.get("height")
        for idx, item in enumerate(inline, start=1):
            if not isinstance(item, dict):
                continue
            cls_id = item.get("class_id")
            if not isinstance(cls_id, int) or isinstance(cls_id, bool):
                continue
            if all(k in item for k in ("cx", "cy", "w", "h")):
                boxes.append(
                    NormalizedBox(
                        class_id=cls_id,
                        cx=float(item["cx"]),
                        cy=float(item["cy"]),
                        w=float(item["w"]),
                        h=float(item["h"]),
                        line_number=idx,
                    )
                )
            elif "bbox" in item and isinstance(item["bbox"], (list, tuple)) and len(item["bbox"]) == 4:
                x1, y1, x2, y2 = (float(v) for v in item["bbox"])
                if isinstance(img_w, (int, float)) and img_w > 0 and isinstance(img_h, (int, float)) and img_h > 0:
                    cx = ((x1 + x2) / 2.0) / float(img_w)
                    cy = ((y1 + y2) / 2.0) / float(img_h)
                    w = (x2 - x1) / float(img_w)
                    h = (y2 - y1) / float(img_h)
                    boxes.append(
                        NormalizedBox(class_id=cls_id, cx=cx, cy=cy, w=w, h=h, line_number=idx)
                    )
    return boxes


def compute_iou(box_a: NormalizedBox, box_b: NormalizedBox) -> float:
    """Compute Intersection over Union (IoU) for two normalized boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a.w) * max(0.0, box_a.h)
    area_b = max(0.0, box_b.w) * max(0.0, box_b.h)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
