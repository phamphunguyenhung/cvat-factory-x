"""CVAT Active-Learner & Preference Calibrator.

Learns per-label geometry, confidence, edge-truncation, and class-confusion
calibration from human CVAT rectangle annotations, and applies the learned
profile to filter and calibrate model detections.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import median
from typing import Any

from cvat_tools.learning.types import Detection

DEFAULT_CONF_DAY = 0.30
DEFAULT_CONF_NIGHT = 0.25
DEFAULT_NIGHT_LUMA_THRESHOLD = 70.0
EDGE_MARGIN_PX = 10.0


@dataclass
class LearnedProfile:
    """Per-label calibration learned from human CVAT annotations."""

    min_w: dict[str, float] = field(default_factory=dict)
    min_h: dict[str, float] = field(default_factory=dict)
    min_area: dict[str, float] = field(default_factory=dict)
    max_area_ratio: dict[str, float] = field(default_factory=dict)
    conf_day: dict[str, float] = field(default_factory=dict)
    conf_night: dict[str, float] = field(default_factory=dict)
    edge_conf_night: dict[str, float] = field(default_factory=dict)
    y_min_ratio: dict[str, float] = field(default_factory=dict)
    y_max_ratio: dict[str, float] = field(default_factory=dict)
    box_expand_ratio: dict[str, float] = field(default_factory=dict)
    class_remaps: dict[str, dict[str, Any]] = field(default_factory=dict)
    ignore_labels: list[str] = field(default_factory=list)
    frame_stats: dict[str, Any] = field(default_factory=dict)


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _is_edge_box(box: list[float], img_w: int, img_h: int, margin: float = EDGE_MARGIN_PX) -> bool:
    return (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= img_w - margin
        or box[3] >= img_h - margin
    )


def _sweep_best_conf(
    gt_by_frame: dict[int, list[list[float]]],
    preds_by_frame: dict[int, list[Detection]],
    frames: set[int],
    default_conf: float,
    iou_thresh: float = 0.50,
) -> float:
    """Sweep confidence thresholds in [0.10, 0.90] to maximize F1 against GT boxes."""
    total_gt = sum(len(gt_by_frame.get(f, [])) for f in frames)
    total_preds = sum(len(preds_by_frame.get(f, [])) for f in frames)
    if total_gt == 0 or total_preds == 0:
        return default_conf

    best_f1 = -1.0
    best_conf = default_conf
    candidates = [round(0.10 + 0.05 * i, 2) for i in range(17)]

    for thresh in candidates:
        tp = 0
        fp = 0
        fn = 0
        for f in frames:
            gts = gt_by_frame.get(f, [])
            preds = sorted(
                (p for p in preds_by_frame.get(f, []) if p.conf >= thresh),
                key=lambda p: p.conf,
                reverse=True,
            )
            matched_gt: set[int] = set()
            for p in preds:
                best_iou = 0.0
                best_idx = -1
                for idx, g in enumerate(gts):
                    if idx in matched_gt:
                        continue
                    iou = _box_iou(p.box, g)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_thresh:
                    matched_gt.add(best_idx)
                    tp += 1
                else:
                    fp += 1
            fn += len(gts) - len(matched_gt)

        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1 or (math.isclose(f1, best_f1) and abs(thresh - default_conf) < abs(best_conf - default_conf)):
            best_f1 = f1
            best_conf = thresh

    return best_conf


def _extract_rect_boxes(
    shapes: list[dict],
    id_to_label: dict[int, str],
) -> tuple[dict[str, dict[int, list[list[float]]]], int, set[int]]:
    """Extract normalized [x1, y1, x2, y2] boxes per label and frame from CVAT shapes."""
    str_to_label = {str(k): v for k, v in id_to_label.items()}
    gt_boxes: dict[str, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    ignored_non_rect = 0
    seen_frames: set[int] = set()

    for s in shapes:
        if s.get("type") != "rectangle":
            ignored_non_rect += 1
            continue
        if s.get("outside", False):
            continue
        pts = s.get("points") or []
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        lid = s.get("label_id")
        label = id_to_label.get(lid) if lid in id_to_label else str_to_label.get(str(lid))
        if not label:
            label = s.get("label")
        if not label:
            continue

        frame = int(s.get("frame", 0))
        seen_frames.add(frame)
        gt_boxes[label][frame].append([x1, y1, x2, y2])

    return gt_boxes, ignored_non_rect, seen_frames


def learn_from_cvat_shapes(
    shapes: list[dict],
    id_to_label: dict[int, str],
    frame_luma: dict[int, float] | None = None,
    model_preds_by_frame: dict[int, list[Detection]] | None = None,
    img_w: int = 1280,
    img_h: int = 720,
    night_luma_threshold: float = DEFAULT_NIGHT_LUMA_THRESHOLD,
) -> LearnedProfile:
    """Learn per-label calibration profile from human CVAT rectangle annotations."""
    gt_boxes, ignored_non_rect, shape_frames = _extract_rect_boxes(shapes, id_to_label)

    all_frames = set(shape_frames)
    if frame_luma:
        all_frames.update(frame_luma.keys())
    if model_preds_by_frame:
        all_frames.update(model_preds_by_frame.keys())

    night_frames: set[int] = set()
    day_frames: set[int] = set()
    unknown_luma_frames: set[int] = set()
    for f in all_frames:
        if frame_luma is None or f not in frame_luma:
            unknown_luma_frames.add(f)
        elif frame_luma[f] < night_luma_threshold:
            night_frames.add(f)
        else:
            day_frames.add(f)
    if not day_frames and unknown_luma_frames:
        day_frames = set(unknown_luma_frames)

    # Index model predictions by label -> frame -> list[Detection]
    preds_by_label: dict[str, dict[int, list[Detection]]] = defaultdict(lambda: defaultdict(list))
    if model_preds_by_frame:
        for f, dets in model_preds_by_frame.items():
            for d in dets:
                preds_by_label[d.label][f].append(d)

    profile = LearnedProfile()
    img_area = float(img_w * img_h)
    label_counts: dict[str, int] = {}

    for label, frames_dict in sorted(gt_boxes.items()):
        boxes = [b for bs in frames_dict.values() for b in bs]
        label_counts[label] = len(boxes)

        widths = [b[2] - b[0] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        area_ratios = [a / img_area for a in areas]
        y1_ratios = [b[1] / img_h for b in boxes]
        y2_ratios = [b[3] / img_h for b in boxes]

        # Size bounds with safety margin so we don't over-prune
        profile.min_w[label] = round(max(4.0, min(widths) * 0.75), 2)
        profile.min_h[label] = round(max(4.0, min(heights) * 0.75), 2)
        profile.min_area[label] = round(max(16.0, min(areas) * 0.65), 2)
        profile.max_area_ratio[label] = round(min(1.0, max(area_ratios) * 1.35), 4)
        profile.y_min_ratio[label] = round(max(0.0, min(y1_ratios) - 0.10), 4)
        profile.y_max_ratio[label] = round(min(1.0, max(y2_ratios) + 0.10), 4)

        # Day vs night confidence calibration
        lbl_preds = preds_by_label.get(label, {})
        c_day = _sweep_best_conf(frames_dict, lbl_preds, day_frames, DEFAULT_CONF_DAY)

        interior_gt = {
            f: [b for b in bs if not _is_edge_box(b, img_w, img_h)]
            for f, bs in frames_dict.items()
        }
        interior_preds = {
            f: [p for p in ps if not p.truncated and not _is_edge_box(p.box, img_w, img_h)]
            for f, ps in lbl_preds.items()
        }
        has_night_int_gt = any(interior_gt.get(f) for f in night_frames)
        has_night_int_preds = any(interior_preds.get(f) for f in night_frames)
        if has_night_int_gt and has_night_int_preds:
            c_night = _sweep_best_conf(interior_gt, interior_preds, night_frames, DEFAULT_CONF_NIGHT)
        else:
            c_night = _sweep_best_conf(frames_dict, lbl_preds, night_frames, DEFAULT_CONF_NIGHT)

        profile.conf_day[label] = c_day
        profile.conf_night[label] = c_night

        # Edge-truncation sensitivity at night
        edge_gt = {
            f: [b for b in bs if _is_edge_box(b, img_w, img_h)]
            for f, bs in frames_dict.items()
        }
        edge_preds = {
            f: [p for p in ps if p.truncated or _is_edge_box(p.box, img_w, img_h)]
            for f, ps in lbl_preds.items()
        }
        has_night_edge_gt = any(edge_gt.get(f) for f in night_frames)
        has_night_edge_preds = any(edge_preds.get(f) for f in night_frames)
        if has_night_edge_gt and has_night_edge_preds:
            profile.edge_conf_night[label] = _sweep_best_conf(
                edge_gt, edge_preds, night_frames, max(0.15, round(c_night - 0.08, 2))
            )
        else:
            edge_frac = sum(1 for b in boxes if _is_edge_box(b, img_w, img_h)) / len(boxes)
            margin = 0.08 if edge_frac >= 0.10 else 0.05
            profile.edge_conf_night[label] = round(max(0.15, c_night - margin), 2)

        # Box expansion ratio from matched TP pairs
        size_ratios: list[float] = []
        if model_preds_by_frame:
            for f, gts in frames_dict.items():
                for g in gts:
                    gw, gh = g[2] - g[0], g[3] - g[1]
                    for p in lbl_preds.get(f, []):
                        if _box_iou(p.box, g) >= 0.50 and p.width > 0 and p.height > 0:
                            size_ratios.append(math.sqrt((gw * gh) / (p.width * p.height)))
                            break
        if size_ratios:
            med_ratio = median(size_ratios)
            if 0.97 <= med_ratio <= 1.03:
                profile.box_expand_ratio[label] = 1.0
            else:
                profile.box_expand_ratio[label] = round(min(1.25, max(0.90, med_ratio)), 3)
        else:
            profile.box_expand_ratio[label] = 1.0

    # Class confusion remapping & ignore_labels from model predictions
    if model_preds_by_frame:
        confusion_counts: Counter[tuple[str, str]] = Counter()
        confusion_confs: dict[tuple[str, str], list[float]] = defaultdict(list)
        pred_label_counts: Counter[str] = Counter()
        matched_pred_counts: Counter[str] = Counter()

        for f, dets in model_preds_by_frame.items():
            # Collect all GT boxes on frame f
            frame_gts: list[tuple[str, list[float]]] = []
            for lbl, fdict in gt_boxes.items():
                for g in fdict.get(f, []):
                    frame_gts.append((lbl, g))

            for d in dets:
                pred_label_counts[d.label] += 1
                best_iou = 0.0
                best_gt_lbl: str | None = None
                for g_lbl, g_box in frame_gts:
                    iou = _box_iou(d.box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_lbl = g_lbl

                if best_iou >= 0.50 and best_gt_lbl is not None:
                    if best_gt_lbl == d.label:
                        matched_pred_counts[d.label] += 1
                    else:
                        pair = (d.label, best_gt_lbl)
                        confusion_counts[pair] += 1
                        confusion_confs[pair].append(d.conf)

        # Group confusions by predicted label, pick dominant target
        by_pred: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (p_lbl, g_lbl), cnt in confusion_counts.items():
            by_pred[p_lbl].append((g_lbl, cnt))

        for p_lbl, targets in by_pred.items():
            targets.sort(key=lambda x: x[1], reverse=True)
            g_lbl, cnt = targets[0]
            confs = confusion_confs[(p_lbl, g_lbl)]
            # Learn remap if repeated confusion or unannotated predicted class with low conf
            if cnt >= 2 or (p_lbl not in gt_boxes and max(confs) < 0.60):
                profile.class_remaps[p_lbl] = {
                    "target": g_lbl,
                    "max_conf": round(min(0.95, max(confs) + 0.05), 2),
                    "count": cnt,
                }

        # Ignore predicted labels that appear frequently (>= 5) with 0 GT and 0 matches and no remap
        for p_lbl, cnt in pred_label_counts.items():
            if (
                cnt >= 5
                and p_lbl not in gt_boxes
                and matched_pred_counts[p_lbl] == 0
                and p_lbl not in profile.class_remaps
            ):
                profile.ignore_labels.append(p_lbl)
        profile.ignore_labels.sort()

    profile.frame_stats = {
        "total_frames": len(all_frames),
        "day_frames": len(day_frames),
        "night_frames": len(night_frames),
        "total_rectangles": sum(label_counts.values()),
        "ignored_non_rectangles": ignored_non_rect,
        "label_counts": label_counts,
        "night_luma_threshold": night_luma_threshold,
    }
    return profile


def save_profile(profile: LearnedProfile, path: Path) -> None:
    """Serialize a LearnedProfile to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(profile), indent=2, ensure_ascii=False), encoding="utf-8")


def load_profile(path: Path) -> LearnedProfile:
    """Load a LearnedProfile from JSON."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    valid_fields = {f.name for f in fields(LearnedProfile)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return LearnedProfile(**filtered)


def _expand_box(box: list[float], ratio: float, img_w: int, img_h: int) -> list[float]:
    if math.isclose(ratio, 1.0, abs_tol=1e-4):
        return list(box)
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    hw = ((box[2] - box[0]) / 2.0) * ratio
    hh = ((box[3] - box[1]) / 2.0) * ratio
    return [
        max(0.0, cx - hw),
        max(0.0, cy - hh),
        min(float(img_w), cx + hw),
        min(float(img_h), cy + hh),
    ]


def apply_profile_filter(
    detections: list[Detection],
    profile: LearnedProfile,
    is_night: bool,
    img_w: int = 1280,
    img_h: int = 720,
) -> list[Detection]:
    """Filter and calibrate model detections using a LearnedProfile."""
    img_area = float(img_w * img_h)
    ignore_set = set(profile.ignore_labels)
    kept: list[Detection] = []

    for det in detections:
        label = det.label

        # 1. Class confusion remap (only above noise floor)
        remap = profile.class_remaps.get(label)
        if det.conf >= 0.10:
            if isinstance(remap, dict):
                if det.conf < float(remap.get("max_conf", 1.0)):
                    label = str(remap["target"])
            elif isinstance(remap, str):
                label = remap

        # 2. Ignore labels
        if label in ignore_set:
            continue

        # 3. Confidence threshold (day / night / night-edge)
        is_edge = det.truncated or _is_edge_box(det.box, img_w, img_h)
        if is_night:
            if is_edge and label in profile.edge_conf_night:
                thresh = profile.edge_conf_night[label]
            else:
                thresh = profile.conf_night.get(label, DEFAULT_CONF_NIGHT)
        else:
            thresh = profile.conf_day.get(label, DEFAULT_CONF_DAY)

        if det.conf < thresh:
            continue

        # 4. Box expansion
        expand_ratio = profile.box_expand_ratio.get(label, 1.0)
        box = _expand_box(det.box, expand_ratio, img_w, img_h)

        # 5. Geometry bounds
        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        area = w * h
        area_ratio = area / img_area if img_area > 0 else 0.0
        cy_ratio = ((box[1] + box[3]) / 2.0) / img_h if img_h > 0 else 0.5

        if label in profile.min_w and w < profile.min_w[label]:
            continue
        if label in profile.min_h and h < profile.min_h[label]:
            continue
        if label in profile.min_area and area < profile.min_area[label]:
            continue
        if label in profile.max_area_ratio and area_ratio > profile.max_area_ratio[label]:
            continue
        if label in profile.y_min_ratio and cy_ratio < profile.y_min_ratio[label]:
            continue
        if label in profile.y_max_ratio and cy_ratio > profile.y_max_ratio[label]:
            continue

        kept.append(
            Detection(
                label=label,
                box=box,
                conf=det.conf,
                source_model=det.source_model,
                occluded=det.occluded,
                truncated=det.truncated,
                attributes=dict(det.attributes),
            )
        )

    return kept
