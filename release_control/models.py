"""Pydantic v2 domain models for CVAT Factory X Data Release Control Tower."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReleaseStatus(str, Enum):
    """Lifecycle states for a dataset release."""

    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    HOLD = "HOLD"
    REWORK = "REWORK"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"
    ACCEPTED = "ACCEPTED"


class FaultType(str, Enum):
    """Taxonomy of dataset release faults (T1-T8)."""

    T1_MISSING_IMAGE = "T1_MISSING_IMAGE"
    T2_WRONG_HASH = "T2_WRONG_HASH"
    T3_INVALID_CLASS = "T3_INVALID_CLASS"
    T4_BBOX_OUT_OF_BOUNDS = "T4_BBOX_OUT_OF_BOUNDS"
    T5_BBOX_DRIFT = "T5_BBOX_DRIFT"
    T6_CLASS_SWAP = "T6_CLASS_SWAP"
    T7_SCENE_LEAK = "T7_SCENE_LEAK"
    T8_GUIDELINE_MISMATCH = "T8_GUIDELINE_MISMATCH"


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class ImageRecord(BaseModel):
    """Metadata and integrity record for a single image asset."""

    image_id: str
    file_name: str
    relative_path: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(ge=0)
    split: str | None = None
    scene_id: str | None = None
    source_profile: str | None = None


class AnnotationBox(BaseModel):
    """Normalized bounding box annotation (YOLO-style center-xywh by default)."""

    class_id: int
    class_name: str | None = None
    x_center: float
    y_center: float
    width: float
    height: float
    bbox_format: str = "yolo_norm"
    confidence: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """Canonical dataset manifest describing images, annotations, and metadata."""

    dataset_name: str
    version: str
    profile: str = "coco80"
    schema_version: str = "3.2.0"
    created_at: datetime = Field(default_factory=_utc_now)
    source_uri: str | None = None
    split_counts: dict[str, int] = Field(default_factory=dict)
    images: list[ImageRecord] = Field(default_factory=list)
    annotations: dict[str, list[AnnotationBox]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    """Single contract or integrity validation finding."""

    issue_id: str
    code: str
    fault_type: FaultType | None = None
    severity: Severity
    message: str
    image_id: str | None = None
    file_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Aggregated validation result for a snapshot or release candidate."""

    report_id: str
    release_id: str | None = None
    snapshot_id: str | None = None
    passed: bool
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=_utc_now)


class FaultRecord(BaseModel):
    """Record of a single injected or detected fault."""

    fault_id: str
    fault_type: FaultType
    severity: Severity = Severity.CRITICAL
    target_image: str | None = None
    target_label: str | None = None
    description: str
    before_value: Any = None
    after_value: Any = None


class FaultManifest(BaseModel):
    """Manifest describing a fault-injection run against a snapshot."""

    run_id: str
    base_snapshot_id: str
    injected_at: datetime = Field(default_factory=_utc_now)
    faults: list[FaultRecord] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class SnapshotRecord(BaseModel):
    """Immutable dataset snapshot index record."""

    snapshot_id: str
    dataset_name: str
    version: str
    source_path: str
    snapshot_path: str
    manifest_sha256: str
    image_count: int = Field(ge=0)
    label_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalRecord(BaseModel):
    """Governance sign-off record for a release."""

    approval_id: str
    release_id: str
    role: str
    approver: str
    decision: Literal["APPROVED", "REJECTED", "HOLD"]
    comment: str = ""
    created_at: datetime = Field(default_factory=_utc_now)


class ConsumerAck(BaseModel):
    """Downstream consumer acceptance acknowledgment."""

    ack_id: str
    release_id: str
    consumer_name: str
    consumer_version: str | None = None
    status: Literal["ACCEPTED", "REJECTED"]
    verified_sha256: str | None = None
    comment: str = ""
    acked_at: datetime = Field(default_factory=_utc_now)


class LineageEdge(BaseModel):
    """Directed provenance/lineage relationship between artifacts or events."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    actor: str = "system"
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReleaseRecord(BaseModel):
    """Tracked dataset release state in the Control Tower."""

    release_id: str
    dataset_name: str
    version: str
    snapshot_id: str
    status: ReleaseStatus = ReleaseStatus.DRAFT
    validation_report_id: str | None = None
    passport_path: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReleasePassport(BaseModel):
    """Signed handover passport bundling release state, validation, approvals, and lineage."""

    release_id: str
    dataset_name: str
    version: str
    status: ReleaseStatus
    snapshot: SnapshotRecord
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    consumer_acks: list[ConsumerAck] = Field(default_factory=list)
    lineage: list[LineageEdge] = Field(default_factory=list)
    published_is_not_accepted: bool = True
    issued_at: datetime = Field(default_factory=_utc_now)
    passport_sha256: str | None = None
