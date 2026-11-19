"""Shared domain models for the CVAT Annotation Factory.

Pure standard-library dataclasses and enums so core/ has zero heavyweight imports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format with microsecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class TaskKind(str, Enum):
    SKELETON = "skeleton"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    UNKNOWN = "unknown"


class RevisionStage(str, Enum):
    ORIGINAL_ANNOTATION = "original_annotation"
    MODEL_PROPOSAL = "model_proposal"
    REFINED_PROPOSAL = "refined_proposal"
    USER_REVIEWED_REVISION = "user_reviewed_revision"
    CVAT_SYNCHRONIZED_REVISION = "cvat_synchronized_revision"


class QualityVerdict(str, Enum):
    PASS = "PASS"
    REFINE = "REFINE"
    REVIEW = "REVIEW"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SkeletonKeypointSpec:
    """Single keypoint specification within a skeleton schema."""

    id: int
    name: str
    parent_id: int | None = None
    parent_name: str | None = None
    symmetric_pair: str | None = None
    region: str | None = None
    label_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkeletonKeypointSpec:
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            parent_id=int(data["parent_id"]) if data.get("parent_id") is not None else None,
            parent_name=str(data["parent_name"]) if data.get("parent_name") is not None else None,
            symmetric_pair=str(data["symmetric_pair"]) if data.get("symmetric_pair") is not None else None,
            region=str(data["region"]) if data.get("region") is not None else None,
            label_id=int(data["label_id"]) if data.get("label_id") is not None else None,
        )


@dataclass(frozen=True)
class SkeletonSchema:
    """Dynamic skeleton schema preserving ordering, hierarchy, topology, and coordinates."""

    schema_name: str
    schema_version: str = "1.0"
    keypoints: tuple[SkeletonKeypointSpec, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()
    coordinate_system: str = "pixel_xy"
    root_label_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def landmark_count(self) -> int:
        return len(self.keypoints)

    def keypoint_by_id(self, kp_id: int) -> SkeletonKeypointSpec | None:
        for kp in self.keypoints:
            if kp.id == kp_id:
                return kp
        return None

    def keypoint_by_name(self, name: str) -> SkeletonKeypointSpec | None:
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "edges": [list(e) for e in self.edges],
            "coordinate_system": self.coordinate_system,
            "root_label_id": self.root_label_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkeletonSchema:
        return cls(
            schema_name=str(data["schema_name"]),
            schema_version=str(data.get("schema_version", "1.0")),
            keypoints=tuple(SkeletonKeypointSpec.from_dict(k) for k in data.get("keypoints", ())),
            edges=tuple((int(a), int(b)) for a, b in data.get("edges", ())),
            coordinate_system=str(data.get("coordinate_system", "pixel_xy")),
            root_label_id=int(data["root_label_id"]) if data.get("root_label_id") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LabelSpec:
    """CVAT label specification including optional sublabels and skeleton definition."""

    id: int
    name: str
    type: str = "any"
    color: str = ""
    attributes: tuple[dict[str, Any], ...] = ()
    sublabels: tuple[LabelSpec, ...] = ()
    skeleton: SkeletonSchema | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "color": self.color,
            "attributes": list(self.attributes),
            "sublabels": [s.to_dict() for s in self.sublabels],
            "skeleton": self.skeleton.to_dict() if self.skeleton is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelSpec:
        skel_raw = data.get("skeleton")
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            type=str(data.get("type", "any")),
            color=str(data.get("color", "")),
            attributes=tuple(data.get("attributes") or ()),
            sublabels=tuple(cls.from_dict(s) for s in data.get("sublabels") or ()),
            skeleton=SkeletonSchema.from_dict(skel_raw) if skel_raw else None,
        )


@dataclass(frozen=True)
class TaskSchema:
    """Normalized CVAT task/job schema passed to adapters and model selection."""

    task_id: int | None = None
    job_id: int | None = None
    task_name: str = ""
    task_kind: TaskKind = TaskKind.UNKNOWN
    annotation_types: tuple[str, ...] = ()
    labels: tuple[LabelSpec, ...] = ()
    skeleton_schemas: dict[str, SkeletonSchema] = field(default_factory=dict)
    frame_count: int = 0
    image_width: int | None = None
    image_height: int | None = None
    schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def label_by_name(self, name: str) -> LabelSpec | None:
        for lbl in self.labels:
            if lbl.name == name:
                return lbl
        return None

    def label_by_id(self, label_id: int) -> LabelSpec | None:
        for lbl in self.labels:
            if lbl.id == label_id:
                return lbl
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "task_name": self.task_name,
            "task_kind": self.task_kind.value,
            "annotation_types": list(self.annotation_types),
            "labels": [l.to_dict() for l in self.labels],
            "skeleton_schemas": {k: v.to_dict() for k, v in self.skeleton_schemas.items()},
            "frame_count": self.frame_count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSchema:
        kind_raw = data.get("task_kind", TaskKind.UNKNOWN.value)
        return cls(
            task_id=int(data["task_id"]) if data.get("task_id") is not None else None,
            job_id=int(data["job_id"]) if data.get("job_id") is not None else None,
            task_name=str(data.get("task_name", "")),
            task_kind=TaskKind(kind_raw),
            annotation_types=tuple(str(x) for x in data.get("annotation_types", ())),
            labels=tuple(LabelSpec.from_dict(l) for l in data.get("labels", ())),
            skeleton_schemas={
                str(k): SkeletonSchema.from_dict(v)
                for k, v in (data.get("skeleton_schemas") or {}).items()
            },
            frame_count=int(data.get("frame_count", 0)),
            image_width=int(data["image_width"]) if data.get("image_width") is not None else None,
            image_height=int(data["image_height"]) if data.get("image_height") is not None else None,
            schema_version=str(data.get("schema_version", "1.0")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Provenance:
    """Full audit provenance attached to every annotation revision."""

    model_id: str = ""
    image_hash: str = ""
    preprocess_params: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    timestamp: str = field(default_factory=utc_now_iso)
    rule_decisions: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    adapter_name: str = ""
    schema_version: str = "1.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        return cls(
            model_id=str(data.get("model_id", "")),
            image_hash=str(data.get("image_hash", "")),
            preprocess_params=dict(data.get("preprocess_params") or {}),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            timestamp=str(data.get("timestamp") or utc_now_iso()),
            rule_decisions=list(data.get("rule_decisions") or []),
            run_id=str(data.get("run_id", "")),
            adapter_name=str(data.get("adapter_name", "")),
            schema_version=str(data.get("schema_version", "1.0")),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class ValidationIssue:
    """Structured finding emitted during geometry/schema/image validation."""

    code: str
    severity: str = "warning"  # "error" | "warning" | "info"
    message: str = ""
    region: str | None = None
    affected_points: list[int] = field(default_factory=list)
    rule_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationIssue:
        return cls(
            code=str(data["code"]),
            severity=str(data.get("severity", "warning")),
            message=str(data.get("message", "")),
            region=str(data["region"]) if data.get("region") is not None else None,
            affected_points=[int(x) for x in data.get("affected_points") or []],
            rule_id=str(data["rule_id"]) if data.get("rule_id") is not None else None,
            details=dict(data.get("details") or {}),
        )


@dataclass
class Proposal:
    """Unified annotation proposal/revision across all stages and task types."""

    job_id: int
    frame_id: int
    label: str
    shape_type: str
    geometry: dict[str, Any]
    provenance: Provenance = field(default_factory=Provenance)
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stage: RevisionStage = RevisionStage.MODEL_PROPOSAL
    confidence: float | None = None
    cvat_shape_id: int | None = None
    track_id: int | None = None
    parent_proposal_id: str | None = None
    verdict: QualityVerdict | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    attributes: list[dict[str, Any]] = field(default_factory=list)
    z_order: int = 0
    occluded: bool = False
    outside: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "job_id": self.job_id,
            "frame_id": self.frame_id,
            "label": self.label,
            "shape_type": self.shape_type,
            "geometry": dict(self.geometry),
            "stage": self.stage.value,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "cvat_shape_id": self.cvat_shape_id,
            "track_id": self.track_id,
            "parent_proposal_id": self.parent_proposal_id,
            "verdict": self.verdict.value if self.verdict is not None else None,
            "issues": [i.to_dict() for i in self.issues],
            "attributes": list(self.attributes),
            "z_order": self.z_order,
            "occluded": self.occluded,
            "outside": self.outside,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        verdict_raw = data.get("verdict")
        return cls(
            proposal_id=str(data.get("proposal_id") or uuid.uuid4().hex),
            job_id=int(data["job_id"]),
            frame_id=int(data["frame_id"]),
            label=str(data["label"]),
            shape_type=str(data["shape_type"]),
            geometry=dict(data.get("geometry") or {}),
            stage=RevisionStage(data.get("stage", RevisionStage.MODEL_PROPOSAL.value)),
            provenance=Provenance.from_dict(data.get("provenance") or {}),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            cvat_shape_id=int(data["cvat_shape_id"]) if data.get("cvat_shape_id") is not None else None,
            track_id=int(data["track_id"]) if data.get("track_id") is not None else None,
            parent_proposal_id=str(data["parent_proposal_id"]) if data.get("parent_proposal_id") is not None else None,
            verdict=QualityVerdict(verdict_raw) if verdict_raw is not None else None,
            issues=[ValidationIssue.from_dict(i) for i in data.get("issues") or []],
            attributes=list(data.get("attributes") or []),
            z_order=int(data.get("z_order", 0)),
            occluded=bool(data.get("occluded", False)),
            outside=bool(data.get("outside", False)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class EvidencePacket:
    """Visual + diagnostic evidence packet for VS Code / human review."""

    job_id: int
    frame_id: int
    packet_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    proposal_ids: list[str] = field(default_factory=list)
    verdict: QualityVerdict = QualityVerdict.REVIEW
    priority_score: float = 0.0
    affected_regions: list[str] = field(default_factory=list)
    affected_points: list[int] = field(default_factory=list)
    image_paths: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "job_id": self.job_id,
            "frame_id": self.frame_id,
            "proposal_ids": list(self.proposal_ids),
            "verdict": self.verdict.value,
            "priority_score": self.priority_score,
            "affected_regions": list(self.affected_regions),
            "affected_points": list(self.affected_points),
            "image_paths": dict(self.image_paths),
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidencePacket:
        return cls(
            packet_id=str(data.get("packet_id") or uuid.uuid4().hex),
            job_id=int(data["job_id"]),
            frame_id=int(data["frame_id"]),
            proposal_ids=[str(x) for x in data.get("proposal_ids") or []],
            verdict=QualityVerdict(data.get("verdict", QualityVerdict.REVIEW.value)),
            priority_score=float(data.get("priority_score", 0.0)),
            affected_regions=[str(x) for x in data.get("affected_regions") or []],
            affected_points=[int(x) for x in data.get("affected_points") or []],
            image_paths={str(k): str(v) for k, v in (data.get("image_paths") or {}).items()},
            summary=str(data.get("summary", "")),
            issues=[ValidationIssue.from_dict(i) for i in data.get("issues") or []],
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or utc_now_iso()),
        )


@dataclass
class RunState:
    """Resumable pipeline run state persisted with atomic checkpoints."""

    job_id: int
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: int | None = None
    adapter_name: str = ""
    status: RunStatus = RunStatus.PENDING
    current_stage: str = "init"
    total_frames: int = 0
    completed_frames: list[int] = field(default_factory=list)
    failed_frames: dict[int, str] = field(default_factory=dict)
    last_checkpoint_frame: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "adapter_name": self.adapter_name,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "total_frames": self.total_frames,
            "completed_frames": list(self.completed_frames),
            "failed_frames": {str(k): v for k, v in self.failed_frames.items()},
            "last_checkpoint_frame": self.last_checkpoint_frame,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        return cls(
            run_id=str(data.get("run_id") or uuid.uuid4().hex),
            job_id=int(data["job_id"]),
            task_id=int(data["task_id"]) if data.get("task_id") is not None else None,
            adapter_name=str(data.get("adapter_name", "")),
            status=RunStatus(data.get("status", RunStatus.PENDING.value)),
            current_stage=str(data.get("current_stage", "init")),
            total_frames=int(data.get("total_frames", 0)),
            completed_frames=[int(x) for x in data.get("completed_frames") or []],
            failed_frames={int(k): str(v) for k, v in (data.get("failed_frames") or {}).items()},
            last_checkpoint_frame=(
                int(data["last_checkpoint_frame"]) if data.get("last_checkpoint_frame") is not None else None
            ),
            config=dict(data.get("config") or {}),
            metrics=dict(data.get("metrics") or {}),
            started_at=str(data.get("started_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            error=str(data["error"]) if data.get("error") is not None else None,
        )


@dataclass
class AdapterContext:
    """Execution context passed to AnnotationAdapter.select_model and preprocess."""

    task_schema: TaskSchema
    hardware: dict[str, Any] = field(default_factory=dict)
    available_models: list[dict[str, Any]] = field(default_factory=list)
    user_model_override: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSelection:
    """Outcome of adapter model selection with explicit capability reporting."""

    model_id: str | None
    supported: bool
    reason: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    fallback_available: bool = False


@dataclass
class ImageInput:
    """Single frame input passed to AnnotationAdapter.preprocess."""

    frame_id: int
    image_path: str | None = None
    image_bytes: bytes | None = None
    image_hash: str = ""
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedBatch:
    """Batch produced by preprocess(), separating inference inputs from display images."""

    items: list[dict[str, Any]] = field(default_factory=list)
    preprocess_params: dict[str, Any] = field(default_factory=dict)
    display_paths: dict[int, str] = field(default_factory=dict)


@dataclass
class RawPrediction:
    """Model or rule prediction prior to CVAT schema conversion."""

    frame_id: int
    label: str
    data: dict[str, Any]
    confidence: float | None = None
    model_id: str = ""
    image_hash: str = ""
    preprocess_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """3-level quality gate report for a single Proposal."""

    proposal_id: str
    verdict: QualityVerdict
    geometry_valid: bool
    image_alignment_good: bool
    schema_valid: bool
    confidence_sufficient: bool
    no_regression: bool
    human_approved: bool = False
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
