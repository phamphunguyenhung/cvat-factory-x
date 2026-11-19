"""Fast unit tests for cvat_tools.detection.signs_lights using synthetic images and mocks."""
from __future__ import annotations

import numpy as np

from cvat_tools.detection import signs_lights
from cvat_tools.detection.signs_lights import detect_signs_and_lights
from cvat_tools.learning.types import Detection
from cvat_tools.segmentation.onnx_seg import CITYSCAPES


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)

    def __len__(self) -> int:
        return len(self.xyxy._arr)


class _FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeResult:
    def __init__(self, xyxy: list[list[float]], conf: list[float]):
        if xyxy:
            self.boxes = _FakeBoxes(np.asarray(xyxy, dtype=np.float32), np.asarray(conf, dtype=np.float32))
        else:
            self.boxes = None


class _FakeYolo:
    """Mock Ultralytics YOLO returning detections only when full ROI is passed."""

    def __init__(self, detections_by_min_w: dict[int, tuple[list[list[float]], list[float]]]):
        self._dets = detections_by_min_w

    def predict(self, img, imgsz=640, conf=0.25, classes=None, device="cpu", verbose=False):
        w = img.shape[1]
        if w >= 1200 and "full" in self._dets:
            xyxy, cf = self._dets["full"]
            kept = [(b, c) for b, c in zip(xyxy, cf) if c >= conf]
            return [_FakeResult([b for b, _ in kept], [c for _, c in kept])]
        return [_FakeResult([], [])]


class _FakeSegFormer:
    def __init__(self, class_map: np.ndarray):
        self._cm = class_map

    def class_map(self, img_bgr: np.ndarray, policy=None):
        return self._cm, {"s": 0.01}


def test_empty_image_returns_empty():
    assert detect_signs_and_lights(np.zeros((0, 0, 3), dtype=np.uint8), use_segformer=False, yolo_model=_FakeYolo({})) == []


def test_traffic_light_bright_bulb_filter_and_night_conf():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Lit green bulb inside box [100, 100, 124, 148]
    img[115:130, 106:118] = (0, 255, 0)
    # Dark box [300, 100, 324, 148] remains all zeros

    yolo = _FakeYolo(
        {
            "full": (
                [[100, 100, 124, 148], [300, 100, 324, 148], [500, 100, 524, 148]],
                [0.13, 0.18, 0.55],  # 0.13 needs is_night=True + bulb cue; 0.18 is dark -> rejected; 0.55 >= 0.35 -> kept
            )
        }
    )

    dets = detect_signs_and_lights(img, is_night=True, use_segformer=False, yolo_model=yolo)
    assert len(dets) == 2
    assert all(isinstance(d, Detection) and d.label == "traffic light" for d in dets)
    xs = sorted(round(d.box[0]) for d in dets)
    assert xs == [100, 500]


def test_traffic_sign_segformer_fragment_merge_roi_and_color_boost(monkeypatch):
    H, W = 720, 1280
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cm = np.zeros((H, W), dtype=np.uint8)
    sign_id = CITYSCAPES["traffic sign"]

    # Sign 1: fragmented into two nearby blobs (gap=4px) with red color -> should merge + color boost
    cm[200:230, 400:440] = sign_id
    cm[234:254, 405:438] = sign_id
    img[200:254, 400:440] = (0, 0, 220)  # red BGR

    # Sign 2: neutral gray sign in upper ROI -> kept, lower confidence than red sign
    cm[200:240, 700:740] = sign_id
    img[200:240, 700:740] = (128, 128, 128)

    # Noise 3: in bottom 25% (hood/road) -> must be filtered out
    cm[600:640, 500:540] = sign_id

    # Noise 4: tiny 4x4 speck -> must be filtered out
    cm[100:104, 100:104] = sign_id

    monkeypatch.setattr(signs_lights, "_get_segformer", lambda: _FakeSegFormer(cm))

    dets = detect_signs_and_lights(img, use_segformer=True, yolo_model=_FakeYolo({}))
    signs = [d for d in dets if d.label == "traffic sign"]
    assert len(signs) == 2

    red_sign = min(signs, key=lambda d: d.box[0])
    gray_sign = max(signs, key=lambda d: d.box[0])

    assert red_sign.box == [400.0, 200.0, 440.0, 254.0]
    assert "color_cue" in red_sign.attributes
    assert red_sign.conf > gray_sign.conf
    assert gray_sign.box == [700.0, 200.0, 740.0, 240.0]
