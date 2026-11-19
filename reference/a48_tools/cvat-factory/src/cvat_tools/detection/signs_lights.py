"""Specialist detector for `traffic sign` and `traffic light` (job 285 blind spots).

Combines:
1. `traffic light`: high-resolution upper-ROI tiling (top 65% of image, overlapping
   horizontal tiles + full upper ROI at imgsz=960/1280 with low conf) + bright-bulb
   HSV/color cue filter to reject dark false positives.
2. `traffic sign`: SegFormer-B2 ONNX (Cityscapes class 7) connected components ->
   nearby fragment merging -> tight bbox + aspect/size/position filtering (upper 75%)
   + red/blue/yellow HSV sign-color confidence boost.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import cv2
import numpy as np

from cvat_tools.core import paths
from cvat_tools.learning.types import Detection
from cvat_tools.segmentation.onnx_seg import CITYSCAPES, OnnxSeg

_SEGFORMER_CACHE: Optional[OnnxSeg] = None
_YOLO_CACHE: Any = None

TRAFFIC_LIGHT_COCO_ID = 9
UPPER_LIGHT_ROI_RATIO = 0.65
UPPER_SIGN_ROI_RATIO = 0.75
TILE_EDGE_MARGIN = 4
NMS_IOU_THRESH = 0.45


def _get_segformer() -> OnnxSeg:
    """Lazy-load and cache SegFormer-B2 ONNX once per process."""
    global _SEGFORMER_CACHE
    if _SEGFORMER_CACHE is None:
        _SEGFORMER_CACHE = OnnxSeg("b2")
    return _SEGFORMER_CACHE


def _get_yolo():
    """Lazy-load and cache YOLO26l checkpoint once per process."""
    global _YOLO_CACHE
    if _YOLO_CACHE is None:
        from ultralytics import YOLO

        ckpt = paths.CHECKPOINTS / "yolo26l.pt"
        _YOLO_CACHE = YOLO(str(ckpt))
    return _YOLO_CACHE


def _box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0.0 else 0.0


def _nms(dets: list[Detection], iou_thresh: float = NMS_IOU_THRESH) -> list[Detection]:
    """Per-label greedy NMS sorted by descending confidence."""
    ordered = sorted(dets, key=lambda d: -d.conf)
    keep: list[Detection] = []
    for d in ordered:
        if any(k.label == d.label and _box_iou(k.box, d.box) > iou_thresh for k in keep):
            continue
        keep.append(d)
    return keep


def _has_bright_bulb_cue(img_bgr: np.ndarray, box: list[float], is_night: bool) -> bool:
    """Check if a traffic light bbox contains a dark housing + lit red/amber/green signal."""
    H, W = img_bgr.shape[:2]
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(W, int(np.ceil(box[2])))
    y2 = min(H, int(np.ceil(box[3])))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False

    crop = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch = hsv[..., 0]
    s_ch = hsv[..., 1]
    v_ch = hsv[..., 2]

    # Hue-restricted traffic-signal colors: red (<=12 or >=165), amber (12..35), green (40..90)
    sig_hue = (h_ch <= 12) | (h_ch >= 165) | ((h_ch > 12) & (h_ch <= 35)) | ((h_ch >= 40) & (h_ch <= 90))
    sig_px = sig_hue & (s_ch >= 80) & (v_ch >= 100)
    min_px = max(2, int(0.02 * crop.shape[0] * crop.shape[1]))
    if int(sig_px.sum()) >= min_px:
        return True

    # Dark housing + bright bulb contrast (rejects uniform bright sky/windows)
    p40 = float(np.percentile(v_ch, 40))
    p95 = float(np.percentile(v_ch, 95))
    housing_max = 75.0 if is_night else 110.0
    bulb_min = 120.0 if is_night else 160.0
    if p40 < housing_max and p95 >= bulb_min and (p95 - p40) >= 55.0:
        return True

    return False


def _predict_yolo_lights(
    yolo_model: Any,
    crop_bgr: np.ndarray,
    imgsz: int,
    conf: float,
) -> list[tuple[list[float], float]]:
    """Run YOLO on a BGR crop and return [(xyxy, conf)] for COCO traffic light (id=9)."""
    if callable(yolo_model) and not hasattr(yolo_model, "predict"):
        return list(yolo_model(crop_bgr, imgsz=imgsz, conf=conf))

    results = yolo_model.predict(
        crop_bgr,
        imgsz=imgsz,
        conf=conf,
        classes=[TRAFFIC_LIGHT_COCO_ID],
        device="cpu",
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    return [([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(c)) for b, c in zip(boxes, confs)]


def _detect_traffic_lights(
    img_bgr: np.ndarray,
    is_night: bool,
    yolo_model: Any,
) -> list[Detection]:
    """Detect traffic lights via upper-ROI tiling (top 65%) + bright-bulb HSV filter."""
    H, W = img_bgr.shape[:2]
    roi_h = max(32, int(round(H * UPPER_LIGHT_ROI_RATIO)))
    roi = np.ascontiguousarray(img_bgr[0:roi_h, :])
    conf_thresh = 0.12 if is_night else 0.15

    raw: list[Detection] = []

    # 1. Full upper ROI at high resolution (imgsz=1280)
    for box, c in _predict_yolo_lights(yolo_model, roi, imgsz=1280, conf=conf_thresh):
        raw.append(Detection(label="traffic light", box=box, conf=c, source_model="yolo_upper_roi"))

    # 2. Overlapping 3x1 horizontal tiles across upper ROI at imgsz=960
    tile_w = max(64, int(round(W * 0.45)))
    if tile_w < W:
        xs = sorted({0, (W - tile_w) // 2, W - tile_w})
        for ox in xs:
            crop = np.ascontiguousarray(roi[:, ox : ox + tile_w])
            cw = crop.shape[1]
            for box, c in _predict_yolo_lights(yolo_model, crop, imgsz=960, conf=conf_thresh):
                x1, y1, x2, y2 = box
                if (ox > 0 and x1 < TILE_EDGE_MARGIN) or (ox + cw < W and x2 > cw - TILE_EDGE_MARGIN):
                    continue
                raw.append(
                    Detection(
                        label="traffic light",
                        box=[x1 + ox, y1, x2 + ox, y2],
                        conf=c,
                        source_model="yolo_upper_tile",
                    )
                )

    merged = _nms(raw, iou_thresh=NMS_IOU_THRESH)

    filtered: list[Detection] = []
    for d in merged:
        bw, bh = d.width, d.height
        if bw < 6 or bh < 6 or bw * bh > 0.08 * W * H:
            continue
        if bw / max(bh, 1e-6) > 3.5 or bh / max(bw, 1e-6) > 5.0:
            continue
        cy = (d.box[1] + d.box[3]) / 2.0
        if cy > 0.50 * H and d.conf < 0.35:
            continue
        # Reject low/mid-confidence dark false positives without a bright bulb / color cue
        if d.conf < 0.50 and not _has_bright_bulb_cue(img_bgr, d.box, is_night=is_night):
            continue
        filtered.append(d)

    return filtered


def _merge_nearby_boxes(boxes: list[list[float]], max_gap: float = 12.0) -> list[list[float]]:
    """Iteratively merge nearby bounding boxes belonging to fragmented sign masks."""
    if len(boxes) <= 1:
        return [list(b) for b in boxes]

    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_boxes: list[list[float]] = []
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                gap_x = max(0.0, max(cur[0], other[0]) - min(cur[2], other[2]))
                gap_y = max(0.0, max(cur[1], other[1]) - min(cur[3], other[3]))
                if gap_x <= max_gap and gap_y <= max_gap:
                    cand = [
                        min(cur[0], other[0]),
                        min(cur[1], other[1]),
                        max(cur[2], other[2]),
                        max(cur[3], other[3]),
                    ]
                    cw = cand[2] - cand[0]
                    ch = cand[3] - cand[1]
                    if 0.25 <= cw / max(ch, 1e-6) <= 4.0:
                        cur = cand
                        used[j] = True
                        changed = True
            next_boxes.append(cur)
        merged = next_boxes
    return merged


def _sign_color_ratio(img_bgr: np.ndarray, box: list[float]) -> float:
    """Fraction of pixels in bbox matching red, blue, or yellow traffic-sign HSV ranges."""
    H, W = img_bgr.shape[:2]
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(W, int(np.ceil(box[2])))
    y2 = min(H, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    hsv = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sv_ok = (s >= 60) & (v >= 45)
    red = ((h <= 12) | (h >= 165)) & sv_ok
    blue = (h >= 95) & (h <= 130) & sv_ok
    yellow = (h >= 15) & (h <= 35) & sv_ok
    matched = red | blue | yellow
    return float(matched.sum()) / float(matched.size)


def _detect_traffic_signs(
    img_bgr: np.ndarray,
    segformer: OnnxSeg,
) -> list[Detection]:
    """Detect traffic signs from SegFormer-B2 Cityscapes mask + fragment merging + HSV boost."""
    H, W = img_bgr.shape[:2]
    class_map, _ = segformer.class_map(img_bgr, policy={"enhance": "none"})
    sign_mask = (class_map == CITYSCAPES["traffic sign"]).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(sign_mask, connectivity=8)
    raw_boxes: list[list[float]] = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < 40 or w < 6 or h < 6:
            continue
        cy = y + h / 2.0
        if cy > UPPER_SIGN_ROI_RATIO * H:
            continue
        raw_boxes.append([float(x), float(y), float(x + w), float(y + h)])

    merged_boxes = _merge_nearby_boxes(raw_boxes, max_gap=6.0)

    dets: list[Detection] = []
    for box in merged_boxes:
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        cy = (box[1] + box[3]) / 2.0
        if bw < 10 or bh < 10 or bw * bh < 100:
            continue
        if bw > 0.35 * W or bh > 0.45 * H:
            continue
        if cy > UPPER_SIGN_ROI_RATIO * H or box[3] > 0.85 * H:
            continue
        aspect = bw / max(bh, 1e-6)
        if not (0.25 <= aspect <= 4.0):
            continue

        x1, y1, x2, y2 = map(int, [np.floor(box[0]), np.floor(box[1]), np.ceil(box[2]), np.ceil(box[3])])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        fill = float(sign_mask[y1:y2, x1:x2].sum()) / max(1.0, float((x2 - x1) * (y2 - y1)))
        if fill < 0.18:
            continue

        color_ratio = _sign_color_ratio(img_bgr, box)
        base_conf = min(0.85, 0.45 + 0.35 * fill)
        if color_ratio >= 0.05:
            base_conf = min(0.95, base_conf + 0.12)

        dets.append(
            Detection(
                label="traffic sign",
                box=box,
                conf=round(base_conf, 4),
                source_model="segformer_b2",
                attributes={"color_cue": round(color_ratio, 3)} if color_ratio >= 0.05 else {},
            )
        )

    return _nms(dets, iou_thresh=NMS_IOU_THRESH)


def detect_signs_and_lights(
    img_bgr: np.ndarray,
    is_night: bool = False,
    use_segformer: bool = True,
    yolo_model: Optional[Any] = None,
) -> list[Detection]:
    """Detect `traffic light` and `traffic sign` on a BGR image.

    Args:
        img_bgr: Input BGR uint8 image (H, W, 3).
        is_night: True for night frames (uses lower YOLO confidence & night bulb cue).
        use_segformer: If True, run lazy-loaded SegFormer-B2 ONNX for `traffic sign`.
        yolo_model: Optional pre-loaded Ultralytics YOLO instance or callable for `traffic light`.

    Returns:
        List of `Detection` objects with labels `"traffic light"` and `"traffic sign"`.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.size == 0:
        return []

    detections: list[Detection] = []

    ym = yolo_model if yolo_model is not None else _get_yolo()
    detections.extend(_detect_traffic_lights(img_bgr, is_night=is_night, yolo_model=ym))

    if use_segformer:
        seg = _get_segformer()
        detections.extend(_detect_traffic_signs(img_bgr, segformer=seg))

    return detections
