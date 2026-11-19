"""Convention & Attribute Engine for CVAT rectangle object detections (Job 285 / Chung)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from cvat_tools.learning.types import Detection

IGNORED_DUPLICATE_LABELS: set[str] = {"person", "traffic_light", "traffic_sign"}

ALL_OBJECT_LABELS: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
)

LABEL_ALIASES: dict[str, str] = {
    "person": "pedestrian",
    "traffic_light": "traffic light",
    "traffic_sign": "traffic sign",
}

VEHICLE_LABELS: frozenset[str] = frozenset(
    {"car", "truck", "bus", "train", "motorcycle", "bicycle"}
)


def _normalize_box(box: list[float] | tuple[float, ...]) -> list[float]:
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _box_iou(a: list[float], b: list[float]) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _is_hood_or_dashboard_reflection(
    label: str, box: list[float], img_h: int
) -> bool:
    """Return True if box is likely a hood/dashboard reflection at the bottom of frame."""
    y1, y2 = box[1], box[3]
    cy = (y1 + y2) / 2.0
    if cy <= 0.88 * img_h:
        return False
    # Large foreground vehicles extending above 0.75 * img_h are real vehicles, not reflections.
    if label in VEHICLE_LABELS and y1 < 0.75 * img_h:
        return False
    return True


def _touches_image_edge(
    box: list[float], img_w: int, img_h: int, edge_margin_px: float
) -> bool:
    x1, y1, x2, y2 = box
    return (
        x1 <= edge_margin_px
        or y1 <= edge_margin_px
        or x2 >= img_w - edge_margin_px
        or y2 >= img_h - edge_margin_px
    )


def compute_z_orders(
    detections: list[Detection], by_bottom_y: bool = False
) -> list[int]:
    """Compute z_order values for rectangle detections.

    Per Job 285 spec, all rectangle objects default to z_order=0.
    When ``by_bottom_y=True``, assigns non-decreasing integer z_orders
    ordered by bottom-y (y2) ascending (closer objects get higher z_order).
    """
    if not detections:
        return []
    if not by_bottom_y:
        return [0] * len(detections)

    order = sorted(range(len(detections)), key=lambda idx: detections[idx].box[3])
    z_orders = [0] * len(detections)
    for rank, idx in enumerate(order):
        z_orders[idx] = rank
    return z_orders


def apply_conventions(
    detections: list[Detection],
    img_w: int = 1280,
    img_h: int = 720,
    edge_margin_px: float = 4.0,
    occluded_iou_thresh: float = 0.30,
) -> list[Detection]:
    """Normalize labels, filter reflections, and predict truncated/occluded attributes."""
    filtered: list[Detection] = []

    for det in detections:
        norm_label = LABEL_ALIASES.get(det.label, det.label)
        box = _normalize_box(det.box)
        if _box_area(box) <= 0.0:
            continue
        if box[2] <= 0.0 or box[3] <= 0.0 or box[0] >= img_w or box[1] >= img_h:
            continue
        if _is_hood_or_dashboard_reflection(norm_label, box, img_h):
            continue

        truncated = bool(det.truncated) or _touches_image_edge(
            box, img_w, img_h, edge_margin_px
        )
        filtered.append(
            replace(
                det,
                label=norm_label,
                box=box,
                truncated=truncated,
                occluded=bool(det.occluded),
                attributes=dict(det.attributes),
            )
        )

    # Predict occlusion: a box is occluded if >= occluded_iou_thresh of its area
    # is covered by a foreground box (higher y2 = closer to camera; tie-break by larger area).
    areas = [_box_area(d.box) for d in filtered]
    for i, det_i in enumerate(filtered):
        if det_i.occluded:
            continue
        area_i = areas[i]
        if area_i <= 0.0:
            continue
        y2_i = det_i.box[3]
        for j, det_j in enumerate(filtered):
            if i == j:
                continue
            y2_j = det_j.box[3]
            is_foreground = (y2_j > y2_i) or (y2_j == y2_i and areas[j] > area_i)
            if not is_foreground:
                continue
            inter = _intersection_area(det_i.box, det_j.box)
            if inter / area_i >= occluded_iou_thresh:
                det_i.occluded = True
                break

    return filtered


def _extract_human_box(shape: dict[str, Any]) -> list[float] | None:
    shape_type = shape.get("type", "rectangle")
    if shape_type != "rectangle":
        return None
    pts = shape.get("points")
    if isinstance(pts, (list, tuple)) and len(pts) == 4:
        return _normalize_box(pts)
    box = shape.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return _normalize_box(box)
    return None


def dedupe_against_human(
    detections: list[Detection],
    human_shapes: list[dict[str, Any]],
    iou_thresh: float = 0.45,
) -> list[Detection]:
    """Filter out detections that overlap existing human-drawn rectangle shapes of the same class/group."""
    vehicle_group = {"car", "truck", "bus", "train"}
    human_entries: list[tuple[str, list[float]]] = []
    for shape in human_shapes:
        if shape.get("outside", False):
            continue
        box = _extract_human_box(shape)
        if box is not None and _box_area(box) > 0.0:
            raw_lbl = str(shape.get("label", "")).strip().lower()
            lbl = LABEL_ALIASES.get(raw_lbl, raw_lbl)
            human_entries.append((lbl, box))

    if not human_entries:
        return list(detections)

    kept: list[Detection] = []
    for det in detections:
        det_box = _normalize_box(det.box)
        det_raw = det.label.strip().lower()
        det_lbl = LABEL_ALIASES.get(det_raw, det_raw)
        dup = False
        for h_lbl, hbox in human_entries:
            same_or_group = (not h_lbl) or (h_lbl == det_lbl) or (h_lbl in vehicle_group and det_lbl in vehicle_group)
            if same_or_group and _box_iou(det_box, hbox) >= iou_thresh:
                dup = True
                break
        if dup:
            continue
        kept.append(det)
    return kept
