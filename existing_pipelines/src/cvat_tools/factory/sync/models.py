"""Data models and exceptions for Phase 5 Safe Sync Controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SyncError(RuntimeError):
    """Base error for factory sync operations."""


class PermissionDeniedError(SyncError):
    """Raised when CVAT client or user lacks write permissions."""


class StaleSnapshotError(SyncError):
    """Raised when live CVAT annotations changed since snapshot was taken."""


class QualityGateError(SyncError):
    """Raised when quality gate blocks syncing."""


class ReadBackMismatchError(SyncError):
    """Raised when post-write read-back verification fails."""


class SyncConflictError(SyncError):
    """Raised when a concurrent edit conflict prevents safe mutation or rollback."""


class QualityGateStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class OperationKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded network retry configuration."""
    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 4.0
    backoff_factor: float = 2.0
    retry_statuses: tuple[int, ...] = (429, 502, 503, 504)


@dataclass
class SyncOperation:
    """A single intended annotation mutation."""
    op_id: str
    kind: OperationKind
    frame: int
    label: str
    shape_type: str
    payload: dict[str, Any]
    target_cvat_id: int | None = None
    before_state: dict[str, Any] | None = None
    local_row_id: int | None = None
    quality_status: QualityGateStatus = QualityGateStatus.PASS
    approved: bool = True
    preserve_points: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkippedItem:
    """An operation skipped during planning or execution."""
    op_id: str
    reason: str
    frame: int | None = None
    label: str | None = None
    target_cvat_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShapeDiff:
    """Exact diff for an update or create operation."""
    op_id: str
    kind: OperationKind
    frame: int
    label: str
    target_cvat_id: int | None = None
    changed_fields: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    points_changed: bool = False
    vertex_deviation_px: float = 0.0
    element_diffs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SyncPlan:
    """Dry-run / pre-flight plan produced by SyncController.plan()."""
    run_id: str
    job_id: int
    live_fingerprint: str
    live_version: int | None
    n_existing: int
    creates: list[SyncOperation] = field(default_factory=list)
    updates: list[SyncOperation] = field(default_factory=list)
    deletes: list[SyncOperation] = field(default_factory=list)
    skipped: list[SkippedItem] = field(default_factory=list)
    diffs: list[ShapeDiff] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_mutations(self) -> int:
        return len(self.creates) + len(self.updates) + len(self.deletes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "live_fingerprint": self.live_fingerprint,
            "live_version": self.live_version,
            "n_existing": self.n_existing,
            "counts": {
                "creates": len(self.creates),
                "updates": len(self.updates),
                "deletes": len(self.deletes),
                "skipped": len(self.skipped),
                "total_mutations": self.total_mutations,
            },
            "diffs": [
                {
                    "op_id": d.op_id,
                    "kind": d.kind.value,
                    "frame": d.frame,
                    "label": d.label,
                    "target_cvat_id": d.target_cvat_id,
                    "changed_fields": {k: list(v) for k, v in d.changed_fields.items()},
                    "points_changed": d.points_changed,
                    "vertex_deviation_px": round(d.vertex_deviation_px, 4),
                    "element_diffs": d.element_diffs,
                }
                for d in self.diffs
            ],
            "skipped": [
                {
                    "op_id": s.op_id,
                    "reason": s.reason,
                    "frame": s.frame,
                    "label": s.label,
                    "target_cvat_id": s.target_cvat_id,
                    "details": s.details,
                }
                for s in self.skipped
            ],
            "warnings": list(self.warnings),
        }


@dataclass
class SyncResult:
    """Result of SyncController.sync() or resume()."""
    run_id: str
    job_id: int
    dry_run: bool
    status: str  # "dry_run" | "completed" | "no_op" | "partial" | "failed" | "rolled_back"
    before_fingerprint: str
    after_fingerprint: str
    n_before: int
    n_after: int
    created_ids: list[int] = field(default_factory=list)
    updated_ids: list[int] = field(default_factory=list)
    deleted_ids: list[int] = field(default_factory=list)
    skipped: list[SkippedItem] = field(default_factory=list)
    verified: bool = False
    verification_errors: list[str] = field(default_factory=list)
    backup_path: str | None = None
    mutation_log_path: str | None = None
    idempotency_key: str | None = None
    resumed_from_checkpoint: bool = False
    plan: SyncPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "dry_run": self.dry_run,
            "status": self.status,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "created_ids": list(self.created_ids),
            "updated_ids": list(self.updated_ids),
            "deleted_ids": list(self.deleted_ids),
            "n_skipped": len(self.skipped),
            "verified": self.verified,
            "verification_errors": list(self.verification_errors),
            "backup_path": self.backup_path,
            "mutation_log_path": self.mutation_log_path,
            "idempotency_key": self.idempotency_key,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }


@dataclass
class RollbackResult:
    """Result of conflict-aware rollback."""
    run_id: str
    job_id: int
    dry_run: bool
    deleted_ids: list[int] = field(default_factory=list)
    restored_ids: list[int] = field(default_factory=list)
    kept_due_to_conflict: list[dict[str, Any]] = field(default_factory=list)
    kept_protected: list[int] = field(default_factory=list)
    verified: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "dry_run": self.dry_run,
            "deleted_ids": list(self.deleted_ids),
            "restored_ids": list(self.restored_ids),
            "kept_due_to_conflict": list(self.kept_due_to_conflict),
            "kept_protected": list(self.kept_protected),
            "verified": self.verified,
            "errors": list(self.errors),
        }
