"""Unit tests for cvat_tools.learning.cvat_learner."""
from __future__ import annotations

from pathlib import Path

import pytest

from cvat_tools.learning.cvat_learner import (
    LearnedProfile,
    apply_profile_filter,
    learn_from_cvat_shapes,
    load_profile,
    save_profile,
)
from cvat_tools.learning.types import Detection


def _rect(frame: int, label_id: int, box: list[float], outside: bool = False) -> dict:
    return {
        "type": "rectangle",
        "frame": frame,
        "label_id": label_id,
        "points": box,
        "outside": outside,
    }


def test_learn_ignores_polygons_and_learns_safe_bounds() -> None:
    id_to_label = {1: "car", 2: "road"}
    shapes = [
        _rect(0, 1, [100.0, 300.0, 200.0, 380.0]),  # w=100, h=80, area=8000
        _rect(1, 1, [500.0, 320.0, 540.0, 350.0]),  # w=40, h=30, area=1200
        _rect(1, 1, [10.0, 10.0, 50.0, 50.0], outside=True),
        {"type": "polygon", "frame": 0, "label_id": 2, "points": [0, 400, 1280, 400, 1280, 720, 0, 720]},
    ]
    profile = learn_from_cvat_shapes(shapes, id_to_label)

    assert "car" in profile.min_w
    assert "road" not in profile.min_w
    assert profile.frame_stats["total_rectangles"] == 2
    assert profile.frame_stats["ignored_non_rectangles"] == 1
    assert profile.min_w["car"] == pytest.approx(30.0)   # 40 * 0.75
    assert profile.min_h["car"] == pytest.approx(22.5)   # 30 * 0.75
    assert profile.min_area["car"] == pytest.approx(780.0)  # 1200 * 0.65

    # Human boxes themselves must survive apply_profile_filter
    gt_dets = [
        Detection("car", [100.0, 300.0, 200.0, 380.0], conf=0.90),
        Detection("car", [500.0, 320.0, 540.0, 350.0], conf=0.90),
    ]
    kept = apply_profile_filter(gt_dets, profile, is_night=False)
    assert len(kept) == 2

    # Tiny noise box should be pruned
    noise = [Detection("car", [500.0, 320.0, 510.0, 330.0], conf=0.95)]
    assert apply_profile_filter(noise, profile, is_night=False) == []


def test_day_night_conf_sweep_maximizes_f1() -> None:
    id_to_label = {1: "car"}
    shapes = [
        _rect(0, 1, [100.0, 300.0, 200.0, 400.0]),  # day frame 0
        _rect(1, 1, [100.0, 300.0, 200.0, 400.0]),  # night frame 1
    ]
    frame_luma = {0: 140.0, 1: 45.0}
    preds = {
        0: [
            Detection("car", [100.0, 300.0, 200.0, 400.0], conf=0.75),  # TP
            Detection("car", [400.0, 300.0, 500.0, 400.0], conf=0.45),  # FP
            Detection("car", [600.0, 300.0, 700.0, 400.0], conf=0.35),  # FP
        ],
        1: [
            Detection("car", [100.0, 300.0, 200.0, 400.0], conf=0.25),  # TP at night
        ],
    }

    profile = learn_from_cvat_shapes(shapes, id_to_label, frame_luma=frame_luma, model_preds_by_frame=preds)
    assert profile.conf_day["car"] >= 0.50
    assert profile.conf_night["car"] <= 0.25
    assert profile.frame_stats["day_frames"] == 1
    assert profile.frame_stats["night_frames"] == 1


def test_edge_truncation_sensitivity_night() -> None:
    id_to_label = {1: "car"}
    shapes = [
        _rect(0, 1, [0.0, 300.0, 80.0, 400.0]),      # edge-truncated at night
        _rect(0, 1, [400.0, 300.0, 500.0, 400.0]),   # interior at night
    ]
    frame_luma = {0: 40.0}
    preds = {
        0: [
            Detection("car", [0.0, 300.0, 80.0, 400.0], conf=0.20, truncated=True),
            Detection("car", [400.0, 300.0, 500.0, 400.0], conf=0.60),
            Detection("car", [700.0, 300.0, 800.0, 400.0], conf=0.35),  # interior FP
        ]
    }

    profile = learn_from_cvat_shapes(shapes, id_to_label, frame_luma=frame_luma, model_preds_by_frame=preds)
    assert profile.edge_conf_night["car"] <= 0.20
    assert profile.conf_night["car"] >= 0.40

    filtered = apply_profile_filter(preds[0], profile, is_night=True)
    assert len(filtered) == 2
    assert {round(d.conf, 2) for d in filtered} == {0.20, 0.60}


def test_class_confusion_remap_and_high_conf_preservation() -> None:
    id_to_label = {1: "car", 2: "truck"}
    shapes = [
        _rect(0, 1, [100.0, 300.0, 220.0, 400.0]),
        _rect(1, 1, [300.0, 300.0, 420.0, 400.0]),
    ]
    preds = {
        0: [Detection("truck", [100.0, 300.0, 220.0, 400.0], conf=0.42)],
        1: [Detection("truck", [300.0, 300.0, 420.0, 400.0], conf=0.49)],
    }

    profile = learn_from_cvat_shapes(shapes, id_to_label, model_preds_by_frame=preds)
    assert "truck" in profile.class_remaps
    assert profile.class_remaps["truck"]["target"] == "car"
    assert profile.class_remaps["truck"]["max_conf"] == pytest.approx(0.54)

    test_dets = [
        Detection("truck", [100.0, 300.0, 220.0, 400.0], conf=0.45),
        Detection("truck", [500.0, 300.0, 650.0, 450.0], conf=0.85),
    ]
    out = apply_profile_filter(test_dets, profile, is_night=False)
    assert [d.label for d in out] == ["car", "truck"]
    # Input Detection objects must not be mutated
    assert test_dets[0].label == "truck"


def test_box_expand_ratio_and_ignore_labels() -> None:
    id_to_label = {1: "car"}
    shapes = [
        _rect(0, 1, [95.0, 295.0, 205.0, 405.0]),  # 110x110 vs pred 100x100 -> 1.10x
    ]
    preds = {
        0: [
            Detection("car", [100.0, 300.0, 200.0, 400.0], conf=0.80),
            *[Detection("boat", [10.0 * i, 100.0, 10.0 * i + 50.0, 150.0], conf=0.70) for i in range(5)],
        ]
    }

    profile = learn_from_cvat_shapes(shapes, id_to_label, model_preds_by_frame=preds)
    assert profile.box_expand_ratio["car"] == pytest.approx(1.10, abs=0.01)
    assert "boat" in profile.ignore_labels

    out = apply_profile_filter(
        [
            Detection("car", [100.0, 300.0, 200.0, 400.0], conf=0.80),
            Detection("boat", [10.0, 100.0, 60.0, 150.0], conf=0.90),
        ],
        profile,
        is_night=False,
    )
    assert len(out) == 1
    assert out[0].label == "car"
    assert out[0].width == pytest.approx(110.0, abs=1.0)


def test_save_and_load_profile_roundtrip(tmp_path: Path) -> None:
    profile = LearnedProfile(
        min_w={"car": 20.0},
        min_h={"car": 15.0},
        min_area={"car": 300.0},
        max_area_ratio={"car": 0.45},
        conf_day={"car": 0.35},
        conf_night={"car": 0.25},
        edge_conf_night={"car": 0.18},
        y_min_ratio={"car": 0.20},
        y_max_ratio={"car": 0.95},
        box_expand_ratio={"car": 1.05},
        class_remaps={"truck": {"target": "car", "max_conf": 0.55, "count": 3}},
        ignore_labels=["train"],
        frame_stats={"total_frames": 20},
    )
    out_path = tmp_path / "nested" / "learned_profile.json"
    save_profile(profile, out_path)
    loaded = load_profile(out_path)
    assert loaded == profile
