"""Integration tests for scripts/bbox_all_objects.py."""
from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = WORKSPACE / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bbox_all_objects as pipeline  # noqa: E402
from cvat_tools.core import paths  # noqa: E402
from cvat_tools.core import transactions as tx  # noqa: E402
from cvat_tools.core.client import WriteNotAuthorized  # noqa: E402
from cvat_tools.learning.types import Detection  # noqa: E402


def _make_fake_cvat(frames: list[int], write_enabled: bool = False) -> MagicMock:
    """Create a mock Cvat client with 20 frames (or given frames) and human shapes."""
    img = np.full((720, 1280, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpg_bytes = buf.tobytes()

    labels = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"},
        {"id": 3, "name": "traffic light"},
        {"id": 4, "name": "traffic sign"},
    ]
    # Human has a car on every frame at [100, 100, 300, 250]
    shapes = [
        {
            "id": 1000 + i,
            "type": "rectangle",
            "frame": f,
            "label_id": 1,
            "points": [100.0, 100.0, 300.0, 250.0],
            "occluded": False,
            "z_order": 0,
            "source": "manual",
            "attributes": [],
        }
        for i, f in enumerate(frames)
    ]

    cv = MagicMock()
    cv.write_enabled = write_enabled
    cv.labels.return_value = labels
    cv.annotations.side_effect = lambda job_id: {"shapes": list(shapes), "tracks": [], "tags": []}
    cv.frame.side_effect = lambda job_id, n: jpg_bytes

    def _create_shapes(job_id: int, new_shapes: list[dict]) -> dict:
        if not cv.write_enabled:
            raise WriteNotAuthorized("not writable")
        base_id = 5000 + len(shapes)
        for idx, s in enumerate(new_shapes):
            shapes.append({**s, "id": base_id + idx})
        return {"shapes": new_shapes}

    cv.create_shapes.side_effect = _create_shapes
    return cv


def _install_mock_submodules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install/patch Subagent 2-6 modules so heavy model inference is mocked."""
    def _ensure_mod(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        try:
            __import__(name)
            return sys.modules[name]
        except ImportError:
            mod = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

    learner = _ensure_mod("cvat_tools.learning.cvat_learner")
    monkeypatch.setattr(
        learner,
        "learn_from_cvat_shapes",
        lambda shapes, id_to_name: {"learned_count": len(shapes), "labels": sorted(set(id_to_name.values()))},
        raising=False,
    )
    monkeypatch.setattr(
        learner,
        "save_profile",
        lambda profile, path: Path(path).write_text(json.dumps(profile), encoding="utf-8"),
        raising=False,
    )
    monkeypatch.setattr(
        learner,
        "load_profile",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        raising=False,
    )
    monkeypatch.setattr(
        learner,
        "apply_profile_filter",
        lambda dets, profile, is_night=False: list(dets),
        raising=False,
    )

    night_mod = _ensure_mod("cvat_tools.detection.night_enhance")
    monkeypatch.setattr(night_mod, "is_night_frame", lambda img: (float(img.mean()) < 70.0, float(img.mean())), raising=False)
    monkeypatch.setattr(night_mod, "get_detection_variants", lambda img: [("orig", img)], raising=False)

    all_obj = _ensure_mod("cvat_tools.detection.all_objects")
    monkeypatch.setattr(all_obj, "load_detector", lambda backend="auto", model_name="yolo26s": {"backend": backend, "model_name": model_name}, raising=False)
    monkeypatch.setattr(
        all_obj,
        "detect_objects_multiscale",
        lambda img, detector, is_night=False: [
            # Matches human car
            Detection(label="car", box=[102.0, 102.0, 298.0, 248.0], conf=0.91, source_model="yolo"),
            # Extra new pedestrian
            Detection(label="pedestrian", box=[400.0, 200.0, 460.0, 360.0], conf=0.82, source_model="yolo"),
        ],
        raising=False,
    )

    sl_mod = _ensure_mod("cvat_tools.detection.signs_lights")
    monkeypatch.setattr(
        sl_mod,
        "detect_signs_and_lights",
        lambda img, is_night=False, yolo_model=None: [
            Detection(label="traffic light", box=[700.0, 60.0, 740.0, 140.0], conf=0.77, source_model="signs"),
        ],
        raising=False,
    )

    conv_mod = _ensure_mod("cvat_tools.learning.convention_engine")
    monkeypatch.setattr(
        conv_mod,
        "ALL_OBJECT_LABELS",
        ("car", "pedestrian", "traffic light", "traffic sign"),
        raising=False,
    )
    monkeypatch.setattr(conv_mod, "apply_conventions", lambda dets: list(dets), raising=False)

    def _dedupe_against_human(dets: list[Detection], human_shapes: list[dict]) -> list[Detection]:
        out = []
        for d in dets:
            if any(pipeline.compute_iou(d.box, h["points"]) >= 0.5 for h in human_shapes):
                continue
            out.append(d)
        return out

    monkeypatch.setattr(conv_mod, "dedupe_against_human", _dedupe_against_human, raising=False)


def test_cli_defaults():
    parser = pipeline.build_parser()
    args = parser.parse_args([])
    assert args.job == 285
    assert args.frames == "60-79"
    assert args.backend == "auto"
    assert args.dry_run is True
    assert args.push is False
    assert args.learn is False


def test_dry_run_end_to_end_20_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_mock_submodules(monkeypatch)
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(paths, "COPILOT", tmp_path / "copilot_missing")
    monkeypatch.setattr(tx, "BACKUPS", tmp_path / "data" / "backups")
    monkeypatch.setattr(tx, "RUNS", tmp_path / "data" / "runs")

    frames = list(range(60, 80))
    fake_cv = _make_fake_cvat(frames, write_enabled=False)
    monkeypatch.setattr(pipeline.Cvat, "from_env", classmethod(lambda cls, profile="local", write_enabled=False: fake_cv))

    args = pipeline.build_parser().parse_args(["--job", "285", "--frames", "60-79", "--learn", "--backend", "openvino"])
    res = pipeline.run_pipeline(args)

    assert res["dry_run"] is True
    fake_cv.create_shapes.assert_not_called()

    # Profile learned and saved
    profile_file = tmp_path / "data" / "jobs" / "285" / "learned_profile.json"
    assert profile_file.exists()

    # Per-frame previews + 4x5 contact sheet
    out_dir = Path(res["out_dir"])
    for f in frames:
        assert (out_dir / f"f{f}_all.jpg").exists()

    sheet = cv2.imread(str(out_dir / "all_bbox_contact_sheet.jpg"))
    assert sheet is not None
    assert sheet.shape == (5 * 360, 4 * 640, 3)

    # CSV & MD reports
    csv_path = out_dir / "all_bbox_report.csv"
    md_path = out_dir / "all_bbox_report.md"
    assert csv_path.exists()
    assert md_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20
    assert rows[0]["khop_IoU0.5"] == "1"
    assert rows[0]["may_thua"] == "2"

    md_text = md_path.read_text(encoding="utf-8")
    assert "## Per-Frame Metrics" in md_text
    assert "## Per-Label Metrics" in md_text
    assert "pedestrian" in md_text

    # Draft JSON: only 2 new boxes per frame (pedestrian + traffic light) = 40 shapes, all source='manual'
    draft_path = Path(res["draft_path"])
    assert draft_path.exists()
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert draft["count"] == 40
    assert all(s["source"] == "manual" for s in draft["shapes"])


def test_push_mode_creates_backup_and_run_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_mock_submodules(monkeypatch)
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(paths, "COPILOT", tmp_path / "copilot_missing")
    monkeypatch.setattr(tx, "BACKUPS", tmp_path / "data" / "backups")
    monkeypatch.setattr(tx, "RUNS", tmp_path / "data" / "runs")

    frames = [60, 61]
    fake_cv = _make_fake_cvat(frames, write_enabled=True)
    monkeypatch.setattr(
        pipeline.Cvat,
        "from_env",
        classmethod(lambda cls, profile="local", write_enabled=False: fake_cv),
    )

    args = pipeline.build_parser().parse_args(["--job", "285", "--frames", "60-61", "--push"])
    res = pipeline.run_pipeline(args)

    assert res["dry_run"] is False
    assert res["push_result"] is not None
    assert res["push_result"]["pushed"] == 4
    assert res["push_result"]["verified"] is True
    assert Path(res["push_result"]["backup"]).exists()
    assert Path(res["push_result"]["run_log"]).exists()
    fake_cv.create_shapes.assert_called_once()
    pushed_payload = fake_cv.create_shapes.call_args[0][1]
    assert all(s["source"] == "manual" for s in pushed_payload)
