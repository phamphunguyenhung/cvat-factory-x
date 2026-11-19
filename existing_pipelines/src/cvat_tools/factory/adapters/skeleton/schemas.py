"""Dynamic skeleton schema detection & strict Pose17 / Face50 / Custom isolation.

Never hardcodes landmark counts for custom schemas and never mixes Pose17 and Face50 mappings.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from cvat_tools.factory.core.models import (
    LabelSpec,
    SkeletonKeypointSpec,
    SkeletonSchema,
    TaskKind,
    TaskSchema,
)

FACE50_GROUP_NAMES: frozenset[str] = frozenset(
    {"longmaytrai", "longmayphai", "songmui", "mattrai", "matphai", "moingoai", "moitrong"}
)

POSE17_COCO_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class SkeletonProfile(str, Enum):
    POSE17 = "pose17"
    FACE50 = "face50"
    CUSTOM = "custom"


def extract_skeleton_schemas(task_schema: TaskSchema) -> dict[str, SkeletonSchema]:
    """Extract or synthesize SkeletonSchema objects from TaskSchema without hardcoding counts."""
    if task_schema.skeleton_schemas:
        return dict(task_schema.skeleton_schemas)

    extracted: dict[str, SkeletonSchema] = {}
    for lbl in task_schema.labels:
        if lbl.skeleton is not None:
            extracted[lbl.name] = lbl.skeleton
        elif lbl.type == "skeleton" or lbl.sublabels:
            kps = tuple(
                SkeletonKeypointSpec(
                    id=idx,
                    name=sub.name,
                    label_id=sub.id,
                )
                for idx, sub in enumerate(lbl.sublabels)
            )
            extracted[lbl.name] = SkeletonSchema(
                schema_name=lbl.name,
                schema_version=task_schema.schema_version,
                keypoints=kps,
                edges=(),
                coordinate_system="pixel_xy",
                root_label_id=lbl.id,
            )
    return extracted


def detect_skeleton_profile(task_schema: TaskSchema) -> tuple[SkeletonProfile, dict[str, SkeletonSchema]]:
    """Classify a TaskSchema into FACE50, POSE17, or CUSTOM without mixing mappings."""
    schemas = extract_skeleton_schemas(task_schema)
    label_names = {lbl.name for lbl in task_schema.labels}

    # Check Face50: either all 7 Face50 group labels are present or a 50-kp face schema
    if FACE50_GROUP_NAMES.issubset(label_names):
        return SkeletonProfile.FACE50, schemas
    for name, skel in schemas.items():
        if name.lower().startswith("face") and skel.landmark_count == 50:
            return SkeletonProfile.FACE50, schemas

    # Check Pose17: 17-keypoint COCO person skeleton
    coco_set = set(POSE17_COCO_NAMES)
    for name, skel in schemas.items():
        kp_names = {kp.name.lower() for kp in skel.keypoints}
        if skel.landmark_count == 17 and (kp_names == coco_set or name.lower() in {"person", "pose17", "human"}):
            return SkeletonProfile.POSE17, schemas

    return SkeletonProfile.CUSTOM, schemas


def is_skeleton_task(task_schema: TaskSchema) -> bool:
    """Return True if TaskSchema represents any skeleton task (Pose17, Face50, or custom)."""
    if task_schema.task_kind == TaskKind.SKELETON:
        return True
    if "skeleton" in task_schema.annotation_types:
        return True
    if task_schema.skeleton_schemas:
        return True
    return any(lbl.type == "skeleton" or lbl.skeleton is not None or bool(lbl.sublabels) for lbl in task_schema.labels)


def normalize_keypoint_record(
    kp_spec: SkeletonKeypointSpec,
    x: float,
    y: float,
    *,
    occluded: bool = False,
    outside: bool = False,
    confidence: float | None = None,
    visibility: str | None = None,
    sublabel_id: int | None = None,
) -> dict[str, Any]:
    """Build a canonical keypoint dict preserving order, parent-child, visibility, and occlusion."""
    if visibility is None:
        if occluded and outside:
            visibility = "Q+O"
        elif outside:
            visibility = "O"
        elif occluded:
            visibility = "Q"
        else:
            visibility = "V"
    return {
        "id": kp_spec.id,
        "name": kp_spec.name,
        "parent_id": kp_spec.parent_id,
        "parent_name": kp_spec.parent_name,
        "region": kp_spec.region,
        "label_id": sublabel_id if sublabel_id is not None else kp_spec.label_id,
        "points": [float(x), float(y)],
        "occluded": bool(occluded),
        "outside": bool(outside),
        "visibility": visibility,
        "confidence": float(confidence) if confidence is not None else None,
    }


def build_skeleton_geometry(
    keypoints: Sequence[dict[str, Any]],
    schema: SkeletonSchema,
    *,
    profile: SkeletonProfile,
) -> dict[str, Any]:
    """Package ordered keypoints + topology + coordinate system + schema version into Proposal.geometry."""
    by_id = {int(kp["id"]): kp for kp in keypoints if "id" in kp}
    ordered: list[dict[str, Any]] = []
    for spec in schema.keypoints:
        if spec.id in by_id:
            item = dict(by_id[spec.id])
            item.setdefault("name", spec.name)
            item.setdefault("parent_id", spec.parent_id)
            item.setdefault("parent_name", spec.parent_name)
            item.setdefault("label_id", spec.label_id)
            ordered.append(item)

    # Preserve any extra keypoints if schema had no pre-declared keypoints
    if not schema.keypoints:
        ordered = [dict(kp) for kp in keypoints]

    return {
        "profile": profile.value,
        "schema_name": schema.schema_name,
        "schema_version": schema.schema_version,
        "coordinate_system": schema.coordinate_system,
        "landmark_count": len(ordered),
        "edges": [list(e) for e in schema.edges],
        "keypoints": ordered,
        "elements": [
            {
                "type": "points",
                "label_id": kp.get("label_id"),
                "points": list(kp["points"]),
                "occluded": bool(kp.get("occluded", False)),
                "outside": bool(kp.get("outside", False)),
            }
            for kp in ordered
        ],
    }


def sublabel_lookup_from_task(task_schema: TaskSchema) -> tuple[dict[int, str], dict[int, int]]:
    """Build (lbl_id_to_name, sub_id_to_kp_index) from TaskSchema labels."""
    lbl_id_to_name: dict[int, str] = {}
    sub_id_to_kp: dict[int, int] = {}
    for lbl in task_schema.labels:
        lbl_id_to_name[lbl.id] = lbl.name
        for idx, sub in enumerate(lbl.sublabels):
            if str(sub.name).isdigit():
                sub_id_to_kp[sub.id] = int(sub.name)
            else:
                sub_id_to_kp[sub.id] = idx
    return lbl_id_to_name, sub_id_to_kp
