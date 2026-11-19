"""Unit tests for YOLO dataset exporter, training config, and CLI script."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from cvat_tools.learning.yolo_exporter import CVAT_TO_YOLO_CLASSES, export_cvat_to_yolo


def _make_img(path: Path, width: int = 200, height: int = 100, luma: int = 150) -> Path:
    img = np.full((height, width, 3), luma, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_class_list_order():
    assert CVAT_TO_YOLO_CLASSES == [
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


def test_export_aliases_skips_dedupe_and_format(tmp_path: Path):
    img_dir = tmp_path / "src_imgs"
    img_dir.mkdir()
    frame_images = {
        0: _make_img(img_dir / "f0.jpg", width=200, height=100, luma=160),
        1: _make_img(img_dir / "f1.jpg", width=200, height=100, luma=160),
    }
    id_to_label = {
        1: "car",
        2: "person",
        3: "traffic_light",
        4: "sky",
        5: "area/direct",
    }
    shapes = [
        # Valid car
        {"type": "rectangle", "frame": 0, "label_id": 1, "points": [20.0, 10.0, 60.0, 50.0], "source": "manual"},
        # Near-identical duplicate car on same frame -> should be deduped
        {"type": "rectangle", "frame": 0, "label_id": 1, "points": [21.0, 10.5, 60.5, 50.0], "source": "auto"},
        # Alias person -> pedestrian (class index 5)
        {"type": "rectangle", "frame": 0, "label_id": 2, "points": [100.0, 20.0, 120.0, 80.0]},
        # Alias traffic_light -> traffic light (class index 7)
        {"type": "rectangle", "frame": 1, "label_id": 3, "points": [80.0, 10.0, 100.0, 30.0]},
        # Polygon -> should be skipped
        {"type": "polygon", "frame": 0, "label_id": 5, "points": [0.0, 0.0, 100.0, 0.0, 100.0, 100.0]},
        # Stuff rectangle -> should be skipped
        {"type": "rectangle", "frame": 0, "label_id": 4, "points": [0.0, 0.0, 200.0, 30.0]},
        # Degenerate zero-width box -> should be skipped
        {"type": "rectangle", "frame": 1, "label_id": 1, "points": [50.0, 10.0, 50.0, 40.0]},
    ]

    out_dir = tmp_path / "yolo_out"
    summary = export_cvat_to_yolo(
        shapes=shapes,
        id_to_label=id_to_label,
        frame_images=frame_images,
        out_dir=out_dir,
        val_ratio=0.5,
        seed=42,
    )

    assert summary["total_frames"] == 2
    assert summary["train_frames"] == 1
    assert summary["val_frames"] == 1
    assert summary["total_boxes"] == 3
    assert summary["deduped_boxes"] == 1
    assert summary["skipped_shapes"] == 3
    assert summary["class_counts"] == {"car": 1, "pedestrian": 1, "traffic light": 1}

    data_yaml = yaml.safe_load((out_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["nc"] == 10
    assert data_yaml["names"][0] == "car"
    assert data_yaml["names"][5] == "pedestrian"
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"

    # Check normalized coordinates in label files
    all_label_files = list((out_dir / "labels").rglob("*.txt"))
    assert len(all_label_files) == 2
    lines = []
    for lf in all_label_files:
        lines.extend(lf.read_text(encoding="utf-8").strip().splitlines())
    assert len(lines) == 3

    # Verify car box [20, 10, 60, 50] on 200x100 -> xc=0.2, yc=0.3, w=0.2, h=0.4
    car_line = next(ln for ln in lines if ln.startswith("0 "))
    cls_id, xc, yc, bw, bh = map(float, car_line.split())
    assert int(cls_id) == 0
    assert abs(xc - 0.2) < 1e-5
    assert abs(yc - 0.3) < 1e-5
    assert abs(bw - 0.2) < 1e-5
    assert abs(bh - 0.4) < 1e-5


def test_day_night_stratified_split(tmp_path: Path):
    img_dir = tmp_path / "src_imgs"
    img_dir.mkdir()
    frame_images = {}
    shapes = []
    for i in range(5):
        frame_images[i] = _make_img(img_dir / f"day_{i}.jpg", luma=180)
        shapes.append({"type": "rectangle", "frame": i, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]})
    for i in range(5, 10):
        frame_images[i] = _make_img(img_dir / f"night_{i}.jpg", luma=25)
        shapes.append({"type": "rectangle", "frame": i, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]})

    out_dir = tmp_path / "yolo_strat"
    summary = export_cvat_to_yolo(
        shapes=shapes,
        id_to_label={1: "car"},
        frame_images=frame_images,
        out_dir=out_dir,
        val_ratio=0.2,
        seed=42,
    )

    assert summary["day_frames"] == {"train": 4, "val": 1}
    assert summary["night_frames"] == {"train": 4, "val": 1}
    assert summary["train_frames"] == 8
    assert summary["val_frames"] == 2


def test_training_config_and_cli_export_only(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "training" / "job285_yolo.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["imgsz"] == 960
    assert cfg["epochs"] == 50
    assert cfg["batch"] == 4
    assert cfg["mosaic"] == 0.5
    assert cfg["hsv_v"] == 0.4

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_img(frames_dir / "frame_000010.jpg", luma=170)
    _make_img(frames_dir / "frame_000011.jpg", luma=30)

    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "id_to_label": {"1": "car", "2": "person"},
                "shapes": [
                    {"type": "rectangle", "frame": 10, "label_id": 1, "points": [10, 10, 80, 60]},
                    {"type": "rectangle", "frame": 11, "label_id": 2, "points": [30, 20, 60, 90]},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "cli_out"
    import scripts.train_from_cvat as cli_mod

    rc = cli_mod.main(
        [
            "--job",
            "285",
            "--export-only",
            "--annotations-json",
            str(ann_path),
            "--frames-dir",
            str(frames_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "data.yaml").exists()
    assert (out_dir / "export_summary.json").exists()
