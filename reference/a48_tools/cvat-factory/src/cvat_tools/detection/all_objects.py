"""Multi-scale vehicle, pedestrian, and rider detector for CVAT job 285.

Combines full-frame inference, horizon-band tiling (for distant small objects),
and left/right edge crops (for border-truncated vehicles), with class-aware
NMS and spatial pedestrian->rider relabeling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from cvat_tools.learning.types import Detection

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENVINO_DIR = REPO_ROOT / "models" / "openvino"
CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints"

COCO_TO_CVAT: dict[str, str] = {
    "person": "pedestrian",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "train": "train",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
}

TWO_WHEELER_LABELS = frozenset({"motorcycle", "bicycle"})
HORIZON_Y_RANGE = (0.30, 0.72)
EDGE_CROP_FRAC = 0.35
TILE_OVERLAP = 0.30
INTERIOR_EDGE_MARGIN = 6.0
NMS_IOU_THRESHOLD = 0.50
RIDER_IOU_THRESHOLD = 0.15


def compute_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _parse_imgsz_from_name(path: Path, default: int = 640) -> int:
    """Extract exported imgsz from OpenVINO folder name or metadata.yaml."""
    for part in path.name.split("_"):
        if part.isdigit():
            val = int(part)
            if val in (320, 480, 640, 800, 960, 1280, 1920):
                return val
    return default


def _find_openvino_model(model_name: str) -> Path | None:
    """Find best matching OpenVINO model directory for `model_name`."""
    if not OPENVINO_DIR.exists():
        return None
    candidates = sorted(
        p for p in OPENVINO_DIR.iterdir()
        if p.is_dir() and p.name.startswith(f"{model_name}_") and p.name.endswith("_openvino_model")
    )
    if not candidates:
        return None
    non_fp16 = [p for p in candidates if "_fp16_" not in p.name]
    pool = non_fp16 or candidates
    pool.sort(key=lambda p: _parse_imgsz_from_name(p, 0), reverse=True)
    return pool[0]


@dataclass
class LoadedDetector:
    """Lazy-loading Ultralytics detector wrapper for OpenVINO or PyTorch."""

    model_name: str
    backend: str
    weights_path: Path
    imgsz: int
    _model: Any = field(default=None, repr=False, compare=False)

    def load(self) -> LoadedDetector:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights_path), task="detect")
        return self

    def predict(
        self,
        img_bgr: np.ndarray,
        conf: float = 0.22,
        imgsz: int | None = None,
    ) -> list[tuple[str, list[float], float]]:
        """Run detection on a BGR image and return [(coco_label, [x1,y1,x2,y2], conf)]."""
        self.load()
        run_imgsz = self.imgsz if self.backend == "openvino" else (imgsz or self.imgsz)
        result = self._model.predict(
            img_bgr,
            conf=conf,
            imgsz=run_imgsz,
            device="cpu",
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        cls_ids = result.boxes.cls.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        return [
            (str(names[int(c)]), [float(v) for v in box], float(score))
            for c, box, score in zip(cls_ids, xyxy, confs)
        ]


def load_detector(backend: str = "auto", model_name: str = "yolo26l") -> LoadedDetector:
    """Lazy-load an OpenVINO or PyTorch detector via Ultralytics.

    Args:
        backend: "auto", "openvino", or "pytorch".
        model_name: e.g. "yolo26l", "yolo26s", "yolo26n".
    """
    if backend not in ("auto", "openvino", "pytorch"):
        raise ValueError(f"Unsupported backend: {backend!r}")

    ov_path = _find_openvino_model(model_name)
    pt_path = CHECKPOINTS_DIR / f"{model_name}.pt"

    if backend == "openvino":
        if ov_path is None:
            raise FileNotFoundError(f"No OpenVINO model found for {model_name!r} in {OPENVINO_DIR}")
        return LoadedDetector(
            model_name=model_name,
            backend="openvino",
            weights_path=ov_path,
            imgsz=_parse_imgsz_from_name(ov_path, 1280 if model_name == "yolo26l" else 960),
        )

    if backend == "pytorch":
        if not pt_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {pt_path}")
        return LoadedDetector(
            model_name=model_name,
            backend="pytorch",
            weights_path=pt_path,
            imgsz=1280 if model_name == "yolo26l" else 960,
        )

    # auto: prefer OpenVINO on CPU when available, otherwise PyTorch checkpoint
    if ov_path is not None:
        return LoadedDetector(
            model_name=model_name,
            backend="openvino",
            weights_path=ov_path,
            imgsz=_parse_imgsz_from_name(ov_path, 1280 if model_name == "yolo26l" else 960),
        )
    if pt_path.exists():
        return LoadedDetector(
            model_name=model_name,
            backend="pytorch",
            weights_path=pt_path,
            imgsz=1280 if model_name == "yolo26l" else 960,
        )
    raise FileNotFoundError(
        f"No weights found for {model_name!r} in {OPENVINO_DIR} or {CHECKPOINTS_DIR}"
    )


def _map_label(raw_label: str) -> str | None:
    """Map COCO/YOLO class name to CVAT job 285 label, or None if ignored."""
    key = raw_label.strip().lower().replace("_", " ")
    return COCO_TO_CVAT.get(key)


def _run_detector(
    detector: Any,
    img_bgr: np.ndarray,
    conf: float,
) -> list[tuple[str, list[float], float]]:
    """Invoke either a LoadedDetector or a callable mock detector."""
    if hasattr(detector, "predict") and callable(detector.predict):
        return detector.predict(img_bgr, conf=conf)
    if callable(detector):
        return detector(img_bgr, conf)
    raise TypeError(f"detector must have .predict(img, conf) or be callable, got {type(detector)}")


def _horizon_tile_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Generate overlapping (x1, y1, x2, y2) tiles across the primary horizon band + lower road band."""
    bands = [HORIZON_Y_RANGE, (0.56, 0.88)]
    windows: list[tuple[int, int, int, int]] = []
    for y_lo, y_hi in bands:
        y1 = max(0, int(round(y_lo * height)))
        y2 = min(height, int(round(y_hi * height)))
        band_h = max(1, y2 - y1)
        tile_w = min(width, max(band_h, min(640, max(320, width // 2))))
        if tile_w >= width:
            windows.append((0, y1, width, y2))
            continue
        step = max(1, int(round(tile_w * (1.0 - TILE_OVERLAP))))
        xs: list[int] = []
        x = 0
        while x + tile_w < width:
            xs.append(x)
            x += step
        xs.append(width - tile_w)
        windows.extend((ox, y1, ox + tile_w, y2) for ox in sorted(set(xs)))
    return windows


def _edge_crop_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Generate left and right edge crop windows [0, 0.35W] and [0.65W, W]."""
    crop_w = max(1, int(round(EDGE_CROP_FRAC * width)))
    return [
        (0, 0, crop_w, height),
        (max(0, width - crop_w), 0, width, height),
    ]


def _is_clipped_by_interior_border(
    box: Sequence[float],
    ox: int,
    oy: int,
    cw: int,
    ch: int,
    full_w: int,
    full_h: int,
) -> bool:
    """Return True if box touches an interior crop boundary (not the image border)."""
    x1, y1, x2, y2 = box
    if ox > 0 and x1 < INTERIOR_EDGE_MARGIN:
        return True
    if oy > 0 and y1 < INTERIOR_EDGE_MARGIN:
        return True
    if ox + cw < full_w and x2 > cw - INTERIOR_EDGE_MARGIN:
        return True
    if oy + ch < full_h and y2 > ch - INTERIOR_EDGE_MARGIN:
        return True
    return False


def _valid_detection(label: str, box: Sequence[float], conf: float, width: int, height: int) -> bool:
    """Filter out hood reflections and tiny/degenerate noise boxes."""
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    if bw < 6.0 or bh < 6.0:
        return False
    cy = (box[1] + box[3]) / 2.0
    if cy > 0.88 * height and label not in ("traffic light", "traffic sign"):
        return False
    if (bw * bh) > 0.30 * width * height and conf < 0.35:
        return False
    return True


def _class_aware_nms(
    detections: list[Detection],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[Detection]:
    """Apply class-aware NMS plus cross-class vehicle suppression (car vs truck/bus/train)."""
    vehicle_group = {"car", "truck", "bus", "train"}
    ordered = sorted(
        detections,
        key=lambda d: (d.conf + (0.15 if d.label == "car" else 0.0)),
        reverse=True,
    )
    kept: list[Detection] = []
    for cand in ordered:
        conflict = False
        for k in kept:
            iou = compute_iou(k.box, cand.box)
            if k.label == cand.label and iou >= iou_threshold:
                conflict = True
                break
            if k.label in vehicle_group and cand.label in vehicle_group and iou >= 0.70:
                conflict = True
                break
        if conflict:
            continue
        kept.append(cand)
    return kept


def _apply_rider_rule(detections: list[Detection]) -> list[Detection]:
    """Relabel pedestrian -> rider when overlapping a motorcycle/bicycle."""
    bikes = [d for d in detections if d.label in TWO_WHEELER_LABELS]
    if not bikes:
        return detections
    for det in detections:
        if det.label != "pedestrian":
            continue
        cx = (det.box[0] + det.box[2]) / 2.0
        foot_y = det.box[3]
        for bike in bikes:
            bx1, by1, bx2, by2 = bike.box
            foot_in_bike = (bx1 <= cx <= bx2) and (by1 <= foot_y <= by2 + 10.0)
            if foot_in_bike or compute_iou(det.box, bike.box) >= RIDER_IOU_THRESHOLD:
                det.label = "rider"
                break
    return detections


def _source_tag(detector: Any) -> str:
    if hasattr(detector, "model_name") and hasattr(detector, "backend"):
        return f"{detector.model_name}:{detector.backend}"
    return "custom"


def detect_objects_multiscale(
    img_bgr: np.ndarray,
    detector: Any,
    is_night: bool = False,
    conf_day: float = 0.22,
    conf_night: float = 0.09,
    use_horizon_tiles: bool = True,
    use_edge_crops: bool = True,
) -> list[Detection]:
    """Run multi-scale detection (full-frame + horizon tiles + edge crops).

    Args:
        img_bgr: Input BGR image (H, W, 3).
        detector: LoadedDetector or callable(img_bgr, conf) returning
            [(coco_label, [x1, y1, x2, y2], conf), ...].
        is_night: Use lower confidence threshold (`conf_night`) when True.
        conf_day: Confidence threshold for daytime frames.
        conf_night: Confidence threshold for nighttime frames.
        use_horizon_tiles: Enable overlapping tiles along y in [0.30H, 0.72H].
        use_edge_crops: Enable left/right border crops [0, 0.35W] and [0.65W, W].
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    height, width = img_bgr.shape[:2]
    conf = conf_night if is_night else conf_day
    src_model = _source_tag(detector)
    raw: list[Detection] = []

    def _collect(crop: np.ndarray, ox: int, oy: int, filter_interior: bool) -> None:
        ch, cw = crop.shape[:2]
        for raw_label, box, score in _run_detector(detector, crop, conf):
            mapped = _map_label(raw_label)
            if mapped is None:
                continue
            if filter_interior and _is_clipped_by_interior_border(box, ox, oy, cw, ch, width, height):
                continue
            gx1 = max(0.0, min(float(width), float(box[0]) + ox))
            gy1 = max(0.0, min(float(height), float(box[1]) + oy))
            gx2 = max(0.0, min(float(width), float(box[2]) + ox))
            gy2 = max(0.0, min(float(height), float(box[3]) + oy))
            gbox = [gx1, gy1, gx2, gy2]
            if not _valid_detection(mapped, gbox, float(score), width, height):
                continue
            truncated = (gx1 <= 2.0) or (gy1 <= 2.0) or (gx2 >= width - 2.0) or (gy2 >= height - 2.0)
            raw.append(
                Detection(
                    label=mapped,
                    box=gbox,
                    conf=float(score),
                    source_model=src_model,
                    truncated=truncated,
                )
            )

    # 1. Full-frame pass
    _collect(img_bgr, 0, 0, filter_interior=False)

    # 2. Horizon-band overlapping tiles
    if use_horizon_tiles:
        for x1, y1, x2, y2 in _horizon_tile_windows(width, height):
            tile = np.ascontiguousarray(img_bgr[y1:y2, x1:x2])
            _collect(tile, x1, y1, filter_interior=True)

    # 3. Left and right edge crops for border-truncated vehicles
    if use_edge_crops:
        for x1, y1, x2, y2 in _edge_crop_windows(width, height):
            crop = np.ascontiguousarray(img_bgr[y1:y2, x1:x2])
            _collect(crop, x1, y1, filter_interior=True)

    deduped = _class_aware_nms(raw, iou_threshold=NMS_IOU_THRESHOLD)
    return _apply_rider_rule(deduped)
