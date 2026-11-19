"""Shared pytest fixtures for the 25-item CVAT Autolabel Factory V3 test matrix.

All destructive / network tests use local temp directories, SQLite databases,
and in-memory mocks. Never use live CVAT as a test database.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from cvat_tools.factory.core.models import (
    LabelSpec,
    SkeletonKeypointSpec,
    SkeletonSchema,
    TaskKind,
    TaskSchema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_1623_DIR = REPO_ROOT / "data" / "jobs" / "edufun" / "1623"
JOB_1975_DIR = REPO_ROOT / "data" / "jobs" / "edufun" / "1975"


def _make_pose17_schema() -> TaskSchema:
    kps = tuple(
        SkeletonKeypointSpec(id=i, name=str(i), label_id=12322 + i)
        for i in range(1, 18)
    )
    skel = SkeletonSchema(schema_name="person_vf17", keypoints=kps, root_label_id=12322)
    lbl = LabelSpec(
        id=12322,
        name="person_vf17",
        type="skeleton",
        sublabels=tuple(LabelSpec(id=12322 + i, name=str(i), type="points") for i in range(1, 18)),
        skeleton=skel,
    )
    return TaskSchema(
        task_id=472,
        job_id=1623,
        task_name="week2-human-pose",
        task_kind=TaskKind.SKELETON,
        annotation_types=("skeleton",),
        labels=(lbl,),
        skeleton_schemas={"person_vf17": skel},
        frame_count=30,
        image_width=1280,
        image_height=720,
    )


def _make_face50_schema() -> TaskSchema:
    groups = [
        (17732, "longmaytrai", range(0, 5)),
        (17738, "longmayphai", range(5, 10)),
        (17744, "songmui", range(10, 14)),
        (17749, "mattrai", range(14, 22)),
        (17758, "matphai", range(22, 30)),
        (17767, "moingoai", range(30, 42)),
        (17780, "moitrong", range(42, 50)),
    ]
    labels: list[LabelSpec] = []
    schemas: dict[str, SkeletonSchema] = {}
    for root_id, name, idx_range in groups:
        kps = tuple(
            SkeletonKeypointSpec(id=idx, name=str(idx), label_id=root_id + 1 + offset)
            for offset, idx in enumerate(idx_range)
        )
        skel = SkeletonSchema(schema_name=name, keypoints=kps, root_label_id=root_id)
        schemas[name] = skel
        labels.append(
            LabelSpec(
                id=root_id,
                name=name,
                type="skeleton",
                sublabels=tuple(
                    LabelSpec(id=kp.label_id or 0, name=kp.name, type="points")
                    for kp in kps
                ),
                skeleton=skel,
            )
        )
    return TaskSchema(
        task_id=473,
        job_id=1975,
        task_name="week2-human-face",
        task_kind=TaskKind.SKELETON,
        annotation_types=("skeleton",),
        labels=tuple(labels),
        skeleton_schemas=schemas,
        frame_count=30,
        image_width=1280,
        image_height=720,
    )


@pytest.fixture
def pose17_schema() -> TaskSchema:
    return _make_pose17_schema()


@pytest.fixture
def face50_schema() -> TaskSchema:
    return _make_face50_schema()


@pytest.fixture
def bbox_schema() -> TaskSchema:
    return TaskSchema(
        task_id=285,
        job_id=285,
        task_name="traffic-bbox",
        task_kind=TaskKind.DETECTION,
        annotation_types=("rectangle",),
        labels=(
            LabelSpec(id=1, name="car", type="rectangle"),
            LabelSpec(id=2, name="person", type="rectangle"),
            LabelSpec(id=3, name="motorbike", type="rectangle"),
        ),
        frame_count=10,
        image_width=1280,
        image_height=720,
    )


@pytest.fixture
def seg_schema() -> TaskSchema:
    return TaskSchema(
        task_id=286,
        job_id=286,
        task_name="road-segmentation",
        task_kind=TaskKind.SEGMENTATION,
        annotation_types=("polygon", "mask"),
        labels=(
            LabelSpec(id=10, name="road", type="polygon"),
            LabelSpec(id=11, name="sidewalk", type="polygon"),
            LabelSpec(id=12, name="car", type="mask"),
        ),
        frame_count=10,
        image_width=1280,
        image_height=720,
    )


@pytest.fixture
def tracking_schema() -> TaskSchema:
    return TaskSchema(
        task_id=287,
        job_id=287,
        task_name="vehicle-tracking",
        task_kind=TaskKind.TRACKING,
        annotation_types=("rectangle", "track"),
        labels=(
            LabelSpec(id=20, name="car", type="rectangle"),
            LabelSpec(id=21, name="person", type="rectangle"),
        ),
        frame_count=10,
        image_width=1280,
        image_height=720,
    )


@pytest.fixture
def mixed_quality_images(tmp_path: Path) -> dict[str, Path]:
    """Create representative easy (bright/sharp) and hard (dark/low-contrast/blurry) images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Easy image: high-contrast synthetic scene with bright rectangle & circle
    easy = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(easy, (120, 100), (320, 380), (40, 40, 200), -1)
    cv2.circle(easy, (450, 200), 60, (30, 180, 30), -1)
    easy_path = img_dir / "easy_bright.jpg"
    cv2.imwrite(str(easy_path), easy)

    # Hard image 1: very dark & low-contrast
    dark = np.full((480, 640, 3), 14, dtype=np.uint8)
    cv2.rectangle(dark, (150, 120), (340, 360), (26, 22, 24), -1)
    dark_path = img_dir / "hard_dark.jpg"
    cv2.imwrite(str(dark_path), dark)

    # Hard image 2: motion/gaussian blur + noise
    blur = cv2.GaussianBlur(easy, (21, 21), 5.0)
    rng = np.random.default_rng(42)
    noise = rng.integers(-15, 16, size=blur.shape, dtype=np.int16)
    blur = np.clip(blur.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    blur_path = img_dir / "hard_blur.jpg"
    cv2.imwrite(str(blur_path), blur)

    return {"easy": easy_path, "dark": dark_path, "blur": blur_path}


@pytest.fixture
def dummy_weights(tmp_path: Path) -> Path:
    """Create a small dummy weight file for ModelRegistry tests without loading torch."""
    w = tmp_path / "models" / "yolov8n.pt"
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b"FAKE_YOLO_WEIGHTS_V3_TEST_DATA_" * 64)
    return w


def load_real_labels_json(job_id: int) -> list[dict[str, Any]]:
    """Helper to load local labels.json for Job 1623 or 1975 if available."""
    path = REPO_ROOT / "data" / "jobs" / "edufun" / str(job_id) / "labels.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
