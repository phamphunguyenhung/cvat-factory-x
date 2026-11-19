"""Unit tests for scripts/eval_all_bbox.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_all_bbox as ev  # noqa: E402


def test_compute_iou_cases() -> None:
    a = ev.Box(frame=60, label="car", x1=0, y1=0, x2=10, y2=10)
    b_same = ev.Box(frame=60, label="car", x1=0, y1=0, x2=10, y2=10)
    b_half = ev.Box(frame=60, label="car", x1=5, y1=0, x2=15, y2=10)
    b_disjoint = ev.Box(frame=60, label="car", x1=20, y1=20, x2=30, y2=30)
    b_zero = ev.Box(frame=60, label="car", x1=0, y1=0, x2=0, y2=0)

    assert ev.compute_iou(a, b_same) == pytest.approx(1.0)
    # intersection = 5*10 = 50, union = 100 + 100 - 50 = 150 -> 1/3
    assert ev.compute_iou(a, b_half) == pytest.approx(1.0 / 3.0)
    assert ev.compute_iou(a, b_disjoint) == 0.0
    assert ev.compute_iou(a, b_zero) == 0.0


def test_match_frame_boxes_confidence_order_and_label_mismatch() -> None:
    gt = [
        ev.Box(frame=61, label="car", x1=100, y1=100, x2=200, y2=200),
    ]
    # High-conf pred has IoU ~0.64 with wrong label 'truck'; low-conf has IoU 1.0 with 'car'
    preds = [
        ev.Box(frame=61, label="car", x1=100, y1=100, x2=200, y2=200, conf=0.4),
        ev.Box(frame=61, label="truck", x1=110, y1=110, x2=200, y2=200, conf=0.9),
    ]
    matches, fps, fns = ev.match_frame_boxes(gt, preds, iou_thresh=0.5)
    assert len(matches) == 1
    assert matches[0].pred.label == "truck"
    assert not matches[0].label_correct
    assert len(fps) == 1
    assert fps[0].conf == pytest.approx(0.4)
    assert len(fns) == 0


def test_evaluate_day_night_tiny_edge_and_per_label() -> None:
    # Frame 61 = SANG (day), Frame 60 = TOI (night)
    gt = [
        # SANG: 1 normal car (matched, label ok), 1 edge-truncated car (matched, wrong label 'truck')
        ev.Box(frame=61, label="car", x1=300, y1=300, x2=400, y2=400),
        ev.Box(frame=61, label="car", x1=2.0, y1=300, x2=102, y2=400),  # x1 <= 5 -> edge
        # TOI: 1 tiny traffic light (20x20=400 < 1024, matched), 1 tiny car (10x10=100 < 1024, missed)
        ev.Box(frame=60, label="traffic light", x1=500, y1=200, x2=520, y2=220),
        ev.Box(frame=60, label="car", x1=600, y1=400, x2=610, y2=410),
        # TOI: 1 bottom-edge truck (y2=717 >= 720-5=715, missed)
        ev.Box(frame=60, label="truck", x1=400, y1=600, x2=500, y2=717),
    ]
    pred = [
        ev.Box(frame=61, label="car", x1=300, y1=300, x2=400, y2=400, conf=0.95),
        ev.Box(frame=61, label="truck", x1=2.0, y1=300, x2=102, y2=400, conf=0.85),
        ev.Box(frame=60, label="traffic light", x1=500, y1=200, x2=520, y2=220, conf=0.80),
        # 1 FP in TOI
        ev.Box(frame=60, label="car", x1=50, y1=50, x2=120, y2=120, conf=0.60),
    ]
    latencies = {61: 1.5, 60: 2.5}
    reports = ev.evaluate_predictions(
        gt_boxes=gt,
        pred_boxes=pred,
        day_night={61: "sang", 60: "toi"},
        latencies=latencies,
        frames=[60, 61],
    )

    # SANG checks
    sang = reports["sang"]
    assert sang.overall.gt_count == 2
    assert sang.overall.pred_count == 2
    assert sang.overall.matched == 2
    assert sang.overall.label_correct == 1
    assert sang.overall.recall == pytest.approx(1.0)
    assert sang.overall.precision == pytest.approx(1.0)
    assert sang.overall.label_accuracy == pytest.approx(0.5)
    assert sang.overall.mean_iou == pytest.approx(1.0)
    assert sang.overall.edge_gt == 1
    assert sang.overall.edge_matched == 1
    assert sang.overall.edge_recall == pytest.approx(1.0)
    assert sang.overall.tiny_gt == 0
    assert sang.overall.tiny_recall is None
    assert sang.overall.latency_s_per_frame == pytest.approx(1.5)

    # TOI checks
    toi = reports["toi"]
    assert toi.overall.gt_count == 3
    assert toi.overall.pred_count == 2
    assert toi.overall.matched == 1
    assert toi.overall.fp == 1
    assert toi.overall.fn == 2
    assert toi.overall.recall == pytest.approx(1.0 / 3.0)
    assert toi.overall.precision == pytest.approx(0.5)
    assert toi.overall.label_accuracy == pytest.approx(1.0)
    assert toi.overall.edge_gt == 1
    assert toi.overall.edge_matched == 0
    assert toi.overall.edge_recall == pytest.approx(0.0)
    assert toi.overall.tiny_gt == 2
    assert toi.overall.tiny_matched == 1
    assert toi.overall.tiny_recall == pytest.approx(0.5)
    assert toi.overall.latency_s_per_frame == pytest.approx(2.5)

    # ALL checks
    all_rep = reports["all"]
    assert all_rep.overall.gt_count == 5
    assert all_rep.overall.matched == 3
    assert all_rep.overall.edge_gt == 2
    assert all_rep.overall.edge_matched == 1
    assert all_rep.overall.edge_recall == pytest.approx(0.5)
    assert all_rep.overall.tiny_gt == 2
    assert all_rep.overall.tiny_matched == 1
    assert all_rep.overall.tiny_recall == pytest.approx(0.5)
    assert all_rep.overall.latency_s_per_frame == pytest.approx(2.0)

    # Per-label checks on ALL
    assert all_rep.per_label["car"].gt_count == 3
    assert all_rep.per_label["car"].matched == 2
    assert all_rep.per_label["car"].label_correct == 1
    assert all_rep.per_label["car"].label_accuracy == pytest.approx(0.5)
    assert all_rep.per_label["traffic light"].recall == pytest.approx(1.0)
    assert all_rep.per_label["truck"].gt_count == 1
    assert all_rep.per_label["truck"].matched == 0


def test_load_boxes_json_formats(tmp_path: Path) -> None:
    gt_file = tmp_path / "gt.json"
    gt_file.write_text(
        json.dumps([
            {"frame": 60, "label": "car", "type": "rectangle", "points": [10, 20, 50, 60]},
            {"frame": 60, "label": "road", "type": "polygon", "points": [0, 0, 100, 0, 100, 100]},
        ]),
        encoding="utf-8",
    )
    gt_boxes, _, _ = ev.load_boxes_json(gt_file)
    assert len(gt_boxes) == 1
    assert gt_boxes[0].label == "car"

    pred_file = tmp_path / "pred.json"
    pred_file.write_text(
        json.dumps({
            "latencies": {"60": 0.42},
            "shapes": [
                {"frame": 60, "_label": "car", "type": "rectangle", "points": [50, 60, 10, 20], "_conf": 0.77}
            ],
        }),
        encoding="utf-8",
    )
    pred_boxes, lats, _ = ev.load_boxes_json(pred_file)
    assert len(pred_boxes) == 1
    assert pred_boxes[0].x1 == 10.0 and pred_boxes[0].x2 == 50.0
    assert pred_boxes[0].conf == pytest.approx(0.77)
    assert lats[60] == pytest.approx(0.42)


def test_xe_report_baseline_and_cli_out_json(tmp_path: Path) -> None:
    csv_path = ROOT / "data" / "jobs" / "285" / "previews" / "cars" / "xe_report.csv"
    baseline, dn = ev.load_xe_report_baseline(csv_path)
    assert len(dn) == 20
    assert baseline["all"].gt_count == 182
    assert baseline["all"].pred_count == 176
    assert baseline["all"].matched == 169
    assert baseline["all"].label_correct == 161
    assert baseline["sang"].gt_count == 85
    assert baseline["toi"].gt_count == 97

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "pred.json"
    out_file = tmp_path / "report.json"
    gt_file.write_text(
        json.dumps([{"frame": 61, "label": "car", "type": "rectangle", "points": [10, 10, 110, 110]}]),
        encoding="utf-8",
    )
    pred_file.write_text(
        json.dumps({"latencies": {"61": 0.5}, "shapes": [{"frame": 61, "label": "car", "points": [10, 10, 110, 110]}]}),
        encoding="utf-8",
    )
    rc = ev.main(["--gt", str(gt_file), "--pred", str(pred_file), "--baseline", str(csv_path), "--out-json", str(out_file)])
    assert rc == 0
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["baseline"]["all"]["gt_count"] == 182
    assert report["evaluation"]["sang"]["overall"]["recall"] == pytest.approx(1.0)
    assert report["evaluation"]["sang"]["overall"]["latency_s_per_frame"] == pytest.approx(0.5)
