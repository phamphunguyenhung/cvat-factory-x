"""Unit tests for cvat_tools.learning.convention_engine."""
from __future__ import annotations

from cvat_tools.learning.convention_engine import (
    ALL_OBJECT_LABELS,
    IGNORED_DUPLICATE_LABELS,
    apply_conventions,
    compute_z_orders,
    dedupe_against_human,
)
from cvat_tools.learning.types import Detection


def test_constants() -> None:
    assert IGNORED_DUPLICATE_LABELS == {"person", "traffic_light", "traffic_sign"}
    assert ALL_OBJECT_LABELS == (
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


def test_alias_normalization_and_immutability() -> None:
    dets = [
        Detection(label="person", box=[100.0, 200.0, 140.0, 300.0], conf=0.9),
        Detection(label="traffic_light", box=[300.0, 100.0, 330.0, 160.0], conf=0.85),
        Detection(label="traffic_sign", box=[500.0, 120.0, 540.0, 160.0], conf=0.8),
        Detection(label="car", box=[600.0, 300.0, 750.0, 420.0], conf=0.95),
    ]
    out = apply_conventions(dets)
    assert [d.label for d in out] == [
        "pedestrian",
        "traffic light",
        "traffic sign",
        "car",
    ]
    # Original objects are not mutated
    assert dets[0].label == "person"
    assert dets[1].label == "traffic_light"
    assert dets[2].label == "traffic_sign"


def test_hood_reflection_filtering() -> None:
    # img_h = 720 -> 0.88 * 720 = 633.6, 0.75 * 720 = 540.0
    reflection_car = Detection(
        label="car", box=[400.0, 620.0, 600.0, 700.0], conf=0.6
    )  # cy = 660 > 633.6, y1 = 620 >= 540 -> filtered
    foreground_truck = Detection(
        label="truck", box=[400.0, 500.0, 800.0, 780.0], conf=0.9
    )  # cy = 640 > 633.6, y1 = 500 < 540 -> kept
    bottom_sign_reflection = Detection(
        label="traffic sign", box=[200.0, 630.0, 260.0, 710.0], conf=0.5
    )  # cy = 670 > 633.6, not a vehicle -> filtered
    normal_car = Detection(
        label="car", box=[100.0, 400.0, 250.0, 500.0], conf=0.92
    )  # cy = 450 <= 633.6 -> kept

    out = apply_conventions(
        [reflection_car, foreground_truck, bottom_sign_reflection, normal_car],
        img_w=1280,
        img_h=720,
    )
    assert len(out) == 2
    assert out[0].label == "truck"
    assert out[1].label == "car"
    assert out[1].box == [100.0, 400.0, 250.0, 500.0]


def test_truncated_edge_detection() -> None:
    dets = [
        Detection(label="car", box=[2.0, 200.0, 120.0, 300.0], conf=0.9),       # left
        Detection(label="traffic light", box=[300.0, 3.0, 340.0, 60.0], conf=0.9),  # top
        Detection(label="bus", box=[1150.0, 200.0, 1277.0, 350.0], conf=0.9),   # right
        Detection(label="car", box=[500.0, 500.0, 700.0, 718.0], conf=0.9),     # bottom
        Detection(label="car", box=[400.0, 300.0, 550.0, 400.0], conf=0.9),     # interior
    ]
    out = apply_conventions(dets, img_w=1280, img_h=720, edge_margin_px=4.0)
    assert [d.truncated for d in out] == [True, True, True, True, False]


def test_occluded_prediction_by_foreground_y2() -> None:
    # Background car: [100, 200, 200, 300] -> area 10000, y2=300
    # Foreground car: [120, 230, 250, 380] -> y2=380 > 300
    # Intersection: [120, 230, 200, 300] -> 80 * 70 = 5600 (56% of background car)
    bg_car = Detection(label="car", box=[100.0, 200.0, 200.0, 300.0], conf=0.85)
    fg_car = Detection(label="car", box=[120.0, 230.0, 250.0, 380.0], conf=0.95)

    # Slightly overlapping car (< 30%): [300, 200, 400, 300], covered 10*50=500 (5%)
    slightly_overlapped = Detection(
        label="car", box=[300.0, 200.0, 400.0, 300.0], conf=0.9
    )
    small_fg = Detection(label="car", box=[390.0, 250.0, 480.0, 350.0], conf=0.9)

    out = apply_conventions(
        [bg_car, fg_car, slightly_overlapped, small_fg],
        occluded_iou_thresh=0.30,
    )
    assert out[0].occluded is True
    assert out[1].occluded is False
    assert out[2].occluded is False
    assert out[3].occluded is False


def test_compute_z_orders() -> None:
    dets = [
        Detection(label="car", box=[100.0, 200.0, 200.0, 400.0], conf=0.9),
        Detection(label="car", box=[300.0, 200.0, 400.0, 300.0], conf=0.9),
        Detection(label="car", box=[500.0, 200.0, 600.0, 500.0], conf=0.9),
    ]
    assert compute_z_orders(dets) == [0, 0, 0]
    assert compute_z_orders(dets, by_bottom_y=True) == [1, 0, 2]


def test_dedupe_against_human() -> None:
    dets = [
        Detection(label="car", box=[100.0, 200.0, 200.0, 300.0], conf=0.9),  # matches human
        Detection(label="pedestrian", box=[400.0, 250.0, 440.0, 350.0], conf=0.88),  # new
        Detection(label="car", box=[600.0, 200.0, 700.0, 300.0], conf=0.92),  # only polygon overlaps
    ]
    human_shapes = [
        {"type": "rectangle", "points": [102.0, 202.0, 198.0, 298.0], "label_id": 1},
        {
            "type": "polygon",
            "points": [600.0, 200.0, 700.0, 200.0, 700.0, 300.0, 600.0, 300.0],
            "label_id": 2,
        },
    ]
    kept = dedupe_against_human(dets, human_shapes, iou_thresh=0.45)
    assert len(kept) == 2
    assert kept[0].label == "pedestrian"
    assert kept[1].box == [600.0, 200.0, 700.0, 300.0]


def test_degenerate_and_out_of_bounds_boxes_filtered() -> None:
    dets = [
        Detection(label="car", box=[100.0, 100.0, 100.0, 200.0], conf=0.9),  # zero width
        Detection(label="car", box=[-50.0, 100.0, -5.0, 200.0], conf=0.9),  # outside left
        Detection(label="car", box=[1300.0, 100.0, 1400.0, 200.0], conf=0.9),  # outside right
        Detection(label="car", box=[200.0, 200.0, 300.0, 300.0], conf=0.9),  # valid
    ]
    out = apply_conventions(dets, img_w=1280, img_h=720)
    assert len(out) == 1
    assert out[0].box == [200.0, 200.0, 300.0, 300.0]
