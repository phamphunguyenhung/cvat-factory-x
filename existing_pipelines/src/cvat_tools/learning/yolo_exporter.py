"""Export CVAT rectangle annotations to Ultralytics YOLO dataset format.

Supports:
- Canonical 10-class traffic object mapping with alias normalization (e.g. person -> pedestrian).
- Skipping non-rectangle shapes (polygons, polylines, etc.) and stuff/background labels.
- Deduplicating exact and near-identical human annotations per (frame, class).
- Day/night luma-stratified train/val splitting.
"""
from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

CVAT_TO_YOLO_CLASSES: list[str] = [
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
]

CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CVAT_TO_YOLO_CLASSES)}

LABEL_ALIASES: dict[str, str] = {
    "person": "pedestrian",
    "motorbike": "motorcycle",
    "bike": "bicycle",
    "traffic_light": "traffic light",
    "traffic-light": "traffic light",
    "traffic_sign": "traffic sign",
    "traffic-sign": "traffic sign",
}


def normalize_label(name: str) -> str:
    """Normalize a CVAT label name and resolve known aliases."""
    key = name.strip().lower()
    return LABEL_ALIASES.get(key, key)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def _is_near_duplicate(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    iou_thresh: float = 0.85,
    px_thresh: float = 2.0,
) -> bool:
    if max(abs(a[i] - b[i]) for i in range(4)) <= px_thresh:
        return True
    return _box_iou(a, b) >= iou_thresh


def _read_image_info(img_path: Path) -> tuple[int, int, float]:
    """Return (width, height, mean_gray_luma) for an image file."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")
    h, w = img.shape[:2]
    luma = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    return w, h, luma


def _split_stratum(fids: list[int], val_ratio: float, rng: random.Random) -> tuple[list[int], list[int]]:
    items = list(fids)
    rng.shuffle(items)
    n = len(items)
    if n == 0 or val_ratio <= 0.0:
        return items, []
    if val_ratio >= 1.0:
        return [], items
    if n == 1:
        return items, []
    n_val = max(1, min(n - 1, round(n * val_ratio)))
    return items[n_val:], items[:n_val]


def _stratified_split(
    frame_luma: dict[int, float],
    val_ratio: float,
    seed: int,
    night_luma_thresh: float = 70.0,
) -> tuple[list[int], list[int], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    day_fids = sorted(fid for fid, luma in frame_luma.items() if luma >= night_luma_thresh)
    night_fids = sorted(fid for fid, luma in frame_luma.items() if luma < night_luma_thresh)

    day_train, day_val = _split_stratum(day_fids, val_ratio, rng)
    night_train, night_val = _split_stratum(night_fids, val_ratio, rng)

    train_fids = sorted(day_train + night_train)
    val_fids = sorted(day_val + night_val)

    # Edge case: 1 day + 1 night frame with 0 < val_ratio < 1 would leave val empty.
    if not val_fids and len(train_fids) >= 2 and 0.0 < val_ratio < 1.0:
        moved = train_fids.pop()
        val_fids.append(moved)
        if moved in day_train:
            day_train.remove(moved)
            day_val.append(moved)
        else:
            night_train.remove(moved)
            night_val.append(moved)

    stats = {
        "day": {"train": len(day_train), "val": len(day_val)},
        "night": {"train": len(night_train), "val": len(night_val)},
    }
    return train_fids, val_fids, stats


def _extract_box(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if shape.get("type") != "rectangle":
        return None
    pts = shape.get("points") or []
    if len(pts) < 4:
        return None
    x1, y1, x2, y2 = float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def export_cvat_to_yolo(
    shapes: list[dict],
    id_to_label: dict[int, str],
    frame_images: dict[int, Path],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    iou_dedupe_thresh: float = 0.85,
    night_luma_thresh: float = 70.0,
) -> dict[str, Any]:
    """Export CVAT rectangle shapes to an Ultralytics YOLO dataset directory.

    Args:
        shapes: Raw CVAT shape dicts (type, frame, label_id or label, points, ...).
        id_to_label: Mapping from CVAT label_id to label name.
        frame_images: Mapping from frame index to local image file Path.
        out_dir: Output dataset root directory.
        val_ratio: Fraction of frames assigned to validation split.
        seed: Random seed for deterministic stratified splitting.
        iou_dedupe_thresh: IoU threshold for deduplicating near-identical same-class boxes.
        night_luma_thresh: Mean grayscale luma below which a frame is classified as night.

    Returns:
        Summary dictionary with paths, split counts, class counts, and filtering stats.
    """
    out_dir = Path(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    valid_frames: dict[int, Path] = {}
    frame_dims: dict[int, tuple[int, int]] = {}
    frame_luma: dict[int, float] = {}
    for fid, img_path in sorted(frame_images.items()):
        p = Path(img_path)
        if not p.exists():
            continue
        w, h, luma = _read_image_info(p)
        valid_frames[fid] = p
        frame_dims[fid] = (w, h)
        frame_luma[fid] = luma

    # Collect and filter shapes per frame, preferring manual annotations when deduplicating.
    ordered_shapes = sorted(
        shapes,
        key=lambda s: (0 if s.get("source") == "manual" else 1, int(s.get("frame", -1))),
    )

    frame_boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    skipped_shapes = 0
    deduped_boxes = 0

    for shape in ordered_shapes:
        fid = shape.get("frame")
        if fid not in valid_frames:
            skipped_shapes += 1
            continue

        box = _extract_box(shape)
        if box is None:
            skipped_shapes += 1
            continue

        raw_label = shape.get("label")
        if raw_label is None:
            lid = shape.get("label_id")
            raw_label = id_to_label.get(lid) if lid is not None else None
        if not raw_label:
            skipped_shapes += 1
            continue

        norm_label = normalize_label(str(raw_label))
        cls_id = CLASS_TO_ID.get(norm_label)
        if cls_id is None:
            skipped_shapes += 1
            continue

        existing = frame_boxes[fid]
        if any(
            ex_cls == cls_id and _is_near_duplicate(ex_box, box, iou_thresh=iou_dedupe_thresh)
            for ex_cls, ex_box in existing
        ):
            deduped_boxes += 1
            continue

        existing.append((cls_id, box))

    train_fids, val_fids, luma_stats = _stratified_split(
        frame_luma, val_ratio=val_ratio, seed=seed, night_luma_thresh=night_luma_thresh
    )

    class_counts: Counter[str] = Counter()
    total_boxes = 0

    for split, fids in (("train", train_fids), ("val", val_fids)):
        for fid in fids:
            src_img = valid_frames[fid]
            stem = f"frame_{fid:06d}"
            ext = src_img.suffix.lower() or ".jpg"
            dst_img = out_dir / "images" / split / f"{stem}{ext}"
            shutil.copy2(src_img, dst_img)

            img_w, img_h = frame_dims[fid]
            lines: list[str] = []
            for cls_id, (x1, y1, x2, y2) in frame_boxes.get(fid, []):
                xc = max(0.0, min(1.0, ((x1 + x2) / 2.0) / img_w))
                yc = max(0.0, min(1.0, ((y1 + y2) / 2.0) / img_h))
                bw = max(0.0, min(1.0, (x2 - x1) / img_w))
                bh = max(0.0, min(1.0, (y2 - y1) / img_h))
                if bw <= 0.0 or bh <= 0.0:
                    continue
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                class_counts[CVAT_TO_YOLO_CLASSES[cls_id]] += 1
                total_boxes += 1

            label_file = out_dir / "labels" / split / f"{stem}.txt"
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml_path = out_dir / "data.yaml"
    data_cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CVAT_TO_YOLO_CLASSES),
        "names": {i: name for i, name in enumerate(CVAT_TO_YOLO_CLASSES)},
    }
    data_yaml_path.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml_path),
        "total_frames": len(valid_frames),
        "train_frames": len(train_fids),
        "val_frames": len(val_fids),
        "day_frames": luma_stats["day"],
        "night_frames": luma_stats["night"],
        "total_boxes": total_boxes,
        "deduped_boxes": deduped_boxes,
        "skipped_shapes": skipped_shapes,
        "class_counts": dict(class_counts),
    }
