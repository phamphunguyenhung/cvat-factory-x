"""Test Matrix Items 12-16: Pose17, Face50, BBox conversion, Segmentation conversion, Tracking consistency."""
from __future__ import annotations

import numpy as np

from cvat_tools.factory.adapters.detection import DetectionAdapter
from cvat_tools.factory.adapters.segmentation import SegmentationAdapter
from cvat_tools.factory.adapters.skeleton import SkeletonAdapter, SkeletonProfile, detect_skeleton_profile
from cvat_tools.factory.core.models import RawPrediction, TaskSchema
from cvat_tools.factory.quality import validate_tracking


def test_12_pose17_schema_isolation(pose17_schema: TaskSchema) -> None:
    """Item 12: Pose17 schema converts 17 keypoints cleanly and never mixes Face50."""
    adapter = SkeletonAdapter()
    assert adapter.supports(pose17_schema) is True

    profile, schemas = detect_skeleton_profile(pose17_schema)
    assert profile in {SkeletonProfile.POSE17, SkeletonProfile.CUSTOM}
    assert profile != SkeletonProfile.FACE50
    assert "person_vf17" in schemas
    assert schemas["person_vf17"].landmark_count == 17

    kps = [
        {"id": i, "points": [100.0 + i * 5, 200.0 + i * 3], "confidence": 0.9}
        for i in range(1, 18)
    ]
    pred = RawPrediction(
        frame_id=90,
        label="person_vf17",
        confidence=0.92,
        data={"keypoints": kps},
        model_id="vitpose-base",
    )
    proposals = adapter.convert_to_cvat([pred], pose17_schema, run_id="run_pose17")
    assert len(proposals) == 1
    p = proposals[0]
    assert p.shape_type == "skeleton"
    assert len(p.geometry["keypoints"]) == 17
    assert p.geometry["root_label_id"] == 12322
    # Must not contain any Face50 group names
    assert p.label == "person_vf17"


def test_13_face50_schema_isolation(face50_schema: TaskSchema) -> None:
    """Item 13: Face50 schema preserves 7 groups (50 landmarks) and never mixes Pose17."""
    adapter = SkeletonAdapter()
    assert adapter.supports(face50_schema) is True

    profile, schemas = detect_skeleton_profile(face50_schema)
    assert profile == SkeletonProfile.FACE50
    assert sum(s.landmark_count for s in schemas.values()) == 50

    preds = []
    for group_name, skel in schemas.items():
        kps = [
            {"id": kp.id, "points": [300.0 + kp.id * 2, 250.0 + kp.id], "confidence": 0.88}
            for kp in skel.keypoints
        ]
        preds.append(
            RawPrediction(
                frame_id=91,
                label=group_name,
                confidence=0.88,
                data={"keypoints": kps},
                model_id="face50-hybrid",
            )
        )

    proposals = adapter.convert_to_cvat(preds, face50_schema, run_id="run_face50")
    assert len(proposals) == 7
    assert sum(len(p.geometry["keypoints"]) for p in proposals) == 50
    assert all(p.label != "person_vf17" for p in proposals)


def test_14_bbox_conversion(bbox_schema: TaskSchema) -> None:
    """Item 14: DetectionAdapter converts raw boxes to CVAT rectangle Proposals with label_id."""
    adapter = DetectionAdapter()
    assert adapter.supports(bbox_schema) is True

    pred = RawPrediction(
        frame_id=0,
        label="car",
        confidence=0.85,
        data={"box": [100.5, 200.25, 320.75, 410.0], "image_width": 1280, "image_height": 720},
        model_id="yolo26n",
    )
    proposals = adapter.convert_to_cvat([pred], bbox_schema, run_id="run_bbox")
    assert len(proposals) == 1
    p = proposals[0]
    assert p.shape_type == "rectangle"
    assert p.label == "car"
    assert p.geometry["label_id"] == 1
    assert p.geometry["points"] == [100.5, 200.25, 320.75, 410.0]


def test_15_segmentation_conversion(seg_schema: TaskSchema) -> None:
    """Item 15: SegmentationAdapter converts binary mask into simplified CVAT polygon."""
    adapter = SegmentationAdapter()
    assert adapter.supports(seg_schema) is True

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:90] = 1
    pred = RawPrediction(
        frame_id=0,
        label="road",
        confidence=0.91,
        data={"mask": mask, "image_width": 100, "image_height": 100},
        model_id="segformer-b2",
    )
    proposals = adapter.convert_to_cvat([pred], seg_schema, run_id="run_seg")
    assert len(proposals) == 1
    p = proposals[0]
    assert p.shape_type == "polygon"
    assert p.label == "road"
    assert p.geometry["label_id"] == 10
    assert len(p.geometry["points"]) >= 6


def test_16_tracking_consistency() -> None:
    """Item 16: Tracking validator passes consistent tracks and flags broken/jumping tracks."""
    consistent = [
        {"track_id": 1, "frame": 0, "points": [100.0, 100.0, 150.0, 150.0]},
        {"track_id": 1, "frame": 1, "points": [108.0, 102.0, 158.0, 152.0]},
        {"track_id": 1, "frame": 2, "points": [116.0, 104.0, 166.0, 154.0]},
    ]
    assert validate_tracking(consistent) == []

    broken_and_jump = [
        {"track_id": 2, "frame": 0, "points": [10.0, 10.0, 50.0, 50.0]},
        {"track_id": 2, "frame": 4, "points": [900.0, 600.0, 940.0, 640.0]},
    ]
    issues = validate_tracking(broken_and_jump)
    codes = {i.code for i in issues}
    assert "TRACK_BROKEN" in codes
    assert "TRACK_TEMPORAL_JUMP" in codes
