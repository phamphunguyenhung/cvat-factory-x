"""Unit tests for cvat_tools.detection.all_objects."""
from __future__ import annotations

import numpy as np
import pytest

from cvat_tools.detection.all_objects import (
    LoadedDetector,
    compute_iou,
    detect_objects_multiscale,
    load_detector,
)
from cvat_tools.learning.types import Detection


def test_load_detector_lazy_and_backend_resolution():
    det_ov = load_detector(backend="auto", model_name="yolo26l")
    assert isinstance(det_ov, LoadedDetector)
    assert det_ov.backend == "openvino"
    assert det_ov._model is None  # lazy-loaded
    assert det_ov.imgsz == 1280

    det_s = load_detector(backend="openvino", model_name="yolo26s")
    assert det_s.backend == "openvino"
    assert det_s.imgsz == 960

    det_pt = load_detector(backend="pytorch", model_name="yolo26l")
    assert det_pt.backend == "pytorch"
    assert det_pt._model is None

    with pytest.raises(ValueError):
        load_detector(backend="invalid", model_name="yolo26l")


def test_label_mapping_and_night_threshold():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    seen_confs: list[float] = []

    def fake_detector(crop: np.ndarray, conf: float):
        seen_confs.append(conf)
        if crop.shape[:2] == (720, 1280):
            return [
                ("person", [100.0, 200.0, 150.0, 320.0], 0.80),
                ("motorbike", [400.0, 350.0, 480.0, 440.0], 0.75),
                ("dog", [200.0, 200.0, 240.0, 240.0], 0.90),  # ignored class
            ]
        return []

    dets = detect_objects_multiscale(
        img,
        fake_detector,
        is_night=True,
        conf_day=0.22,
        conf_night=0.14,
        use_horizon_tiles=False,
        use_edge_crops=False,
    )
    assert seen_confs == [0.14]
    assert all(isinstance(d, Detection) for d in dets)
    labels = {d.label for d in dets}
    assert labels == {"pedestrian", "motorcycle"}


def test_horizon_tiles_and_nms_dedup():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_detector(crop: np.ndarray, conf: float):
        h, w = crop.shape[:2]
        if (h, w) == (720, 1280):
            # Full-frame detects one car
            return [("car", [500.0, 340.0, 560.0, 390.0], 0.70)]
        # Horizon tile (height ~ 302) detects the same car with higher conf + a tiny distant car
        if h < 400 and w <= 640:
            return [
                ("car", [500.0, 124.0, 560.0, 174.0], 0.85),  # if ox=0, oy=216 -> y=[340, 390]
                ("car", [200.0, 130.0, 224.0, 150.0], 0.45),  # tiny distant car missed by full-frame
            ]
        return []

    dets = detect_objects_multiscale(
        img,
        fake_detector,
        use_horizon_tiles=True,
        use_edge_crops=False,
    )
    cars = [d for d in dets if d.label == "car"]
    # Should have the deduplicated [500, 340, 560, 390] (with conf 0.85) and the tiny distant car
    assert any(abs(d.box[0] - 200.0) < 1e-3 and abs(d.box[1] - 346.0) < 1e-3 for d in cars)
    assert sum(1 for d in cars if compute_iou(d.box, [500.0, 340.0, 560.0, 390.0]) > 0.9) == 1


def test_edge_crops_catch_truncated_vehicles_and_drop_interior_clips():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :10, 0] = 1  # mark left border

    def fake_detector(crop: np.ndarray, conf: float):
        h, w = crop.shape[:2]
        if (h, w) == (720, 1280):
            return []
        # Left edge crop (marked by channel 0 == 1 at x=0)
        if h == 720 and w == 448 and crop[0, 0, 0] == 1:
            return [
                # Left border truncated car (x1=0 -> kept, truncated=True)
                ("car", [0.0, 320.0, 240.0, 510.0], 0.60),
                # Interior-clipped box touching right edge of left crop (x2=447 -> dropped)
                ("car", [380.0, 330.0, 447.0, 410.0], 0.70),
            ]
        return []

    dets = detect_objects_multiscale(
        img,
        fake_detector,
        use_horizon_tiles=False,
        use_edge_crops=True,
    )
    assert len(dets) == 1
    assert dets[0].label == "car"
    assert dets[0].truncated is True
    assert dets[0].box == [0.0, 320.0, 240.0, 510.0]


def test_rider_spatial_rule_iou_and_foot_inside_bike():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_detector(crop: np.ndarray, conf: float):
        return [
            # Motorcycle 1 + rider via foot-inside-bike
            ("motorcycle", [300.0, 380.0, 380.0, 460.0], 0.80),
            ("person", [320.0, 320.0, 360.0, 420.0], 0.85),
            # Bicycle 2 + rider via IoU >= 0.15
            ("bicycle", [600.0, 380.0, 680.0, 450.0], 0.78),
            ("person", [610.0, 360.0, 660.0, 430.0], 0.82),
            # Standalone pedestrian far away
            ("person", [900.0, 330.0, 940.0, 440.0], 0.88),
        ]

    dets = detect_objects_multiscale(
        img,
        fake_detector,
        use_horizon_tiles=False,
        use_edge_crops=False,
    )
    riders = [d for d in dets if d.label == "rider"]
    peds = [d for d in dets if d.label == "pedestrian"]
    assert len(riders) == 2
    assert len(peds) == 1
    assert peds[0].box == [900.0, 330.0, 940.0, 440.0]


def test_hood_reflection_filtered_except_traffic_light():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_detector(crop: np.ndarray, conf: float):
        return [
            ("car", [500.0, 640.0, 620.0, 710.0], 0.90),  # cy=675 > 0.88*720=633.6 -> filtered
            ("traffic light", [500.0, 640.0, 530.0, 700.0], 0.70),  # kept
        ]

    dets = detect_objects_multiscale(
        img,
        fake_detector,
        use_horizon_tiles=False,
        use_edge_crops=False,
    )
    assert [d.label for d in dets] == ["traffic light"]
