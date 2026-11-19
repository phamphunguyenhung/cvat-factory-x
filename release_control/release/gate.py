"""Release Gate state machine enforcing contract validation, dual approvals, and consumer ACK."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from release_control.contracts.engine import ContractEngine
from release_control.db import ControlTowerDB
from release_control.lineage.events import LineageTracker
from release_control.models import (
    ApprovalRecord,
    ConsumerAck,
    ReleaseRecord,
    ReleaseStatus,
    ValidationReport,
)


_VALID_APPROVAL_TYPES = {"qc", "privacy"}
_VERDICT_MAP: dict[str, Literal["APPROVED", "REJECTED", "HOLD"]] = {
    "approved": "APPROVED",
    "rejected": "REJECTED",
    "hold": "HOLD",
}


class ReleaseGate:
    """Governance state machine for dataset releases."""

    def __init__(
        self,
        db: ControlTowerDB,
        contract_engine: ContractEngine | None = None,
        lineage_tracker: LineageTracker | None = None,
    ) -> None:
        self.db = db
        self.contract_engine = contract_engine or ContractEngine()
        self.lineage_tracker = lineage_tracker

    def _require_release(self, release_id: str) -> ReleaseRecord:
        release = self.db.get_release(release_id)
        if release is None:
            raise KeyError(f"Release not found: {release_id}")
        return release

    def _resolve_snapshot_id(
        self, snapshot_digest: str, snapshot_id: str | None = None
    ) -> str:
        if snapshot_id is not None:
            snap = self.db.get_snapshot(snapshot_id)
            if snap is None:
                raise ValueError(f"Snapshot not found: {snapshot_id}")
            return snap.snapshot_id

        for snap in self.db.list_snapshots():
            if snap.manifest_sha256 == snapshot_digest or snap.snapshot_id == snapshot_digest:
                return snap.snapshot_id

        raise ValueError(
            f"No snapshot found matching digest or id: {snapshot_digest!r}"
        )

    def create_release(
        self,
        release_id: str,
        dataset_id: str,
        version_tag: str,
        snapshot_digest: str,
        contract_version: str = "1.0.0",
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReleaseRecord:
        """Create a new release in DRAFT state bound to a recorded snapshot."""
        if self.db.get_release(release_id) is not None:
            raise ValueError(f"Release already exists: {release_id}")

        resolved_snapshot_id = self._resolve_snapshot_id(
            snapshot_digest=snapshot_digest, snapshot_id=snapshot_id
        )
        merged_meta: dict[str, Any] = {
            **(metadata or {}),
            "contract_version": contract_version,
            "snapshot_digest": snapshot_digest,
        }
        record = ReleaseRecord(
            release_id=release_id,
            dataset_name=dataset_id,
            version=version_tag,
            snapshot_id=resolved_snapshot_id,
            status=ReleaseStatus.DRAFT,
            metadata=merged_meta,
        )
        self.db.upsert_release(record)

        if self.lineage_tracker is not None:
            self.lineage_tracker.record_edge(
                source_type="snapshot",
                source_id=resolved_snapshot_id,
                target_type="release",
                target_id=release_id,
                relation="released_as",
                evidence_status="verified",
                metadata={"version": version_tag, "contract_version": contract_version},
            )
        return record

    def submit_validation(
        self,
        release_id: str,
        validation_report: ValidationReport,
        requires_human_review: bool = False,
    ) -> ReleaseRecord:
        """Transition DRAFT/REWORK -> VALIDATED or HOLD based on validation report."""
        release = self._require_release(release_id)
        if release.status not in {ReleaseStatus.DRAFT, ReleaseStatus.REWORK}:
            raise ValueError(
                f"Cannot submit validation when release is in state {release.status.value}"
            )

        human_review = requires_human_review or bool(
            validation_report.metrics.get("requires_human_review", False)
        )
        target = (
            ReleaseStatus.VALIDATED
            if (validation_report.passed and not human_review)
            else ReleaseStatus.HOLD
        )

        updated_meta = {
            **release.metadata,
            "validation_summary": {
                "report_id": validation_report.report_id,
                "passed": validation_report.passed,
                "critical_count": validation_report.critical_count,
                "warning_count": validation_report.warning_count,
                "info_count": validation_report.info_count,
                "requires_human_review": human_review,
                "metrics": validation_report.metrics,
                "checked_at": validation_report.checked_at.isoformat(),
            },
        }
        updated_record = release.model_copy(
            update={
                "status": target,
                "validation_report_id": validation_report.report_id,
                "metadata": updated_meta,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.db.upsert_release(updated_record)
        return self._require_release(release_id)

    def submit_approval(
        self,
        release_id: str,
        approval_type: str,
        actor: str,
        verdict: str,
        note: str = "",
    ) -> ReleaseRecord:
        """Record a real qc or privacy ApprovalRecord for a release."""
        self._require_release(release_id)

        role = approval_type.strip().lower()
        if role not in _VALID_APPROVAL_TYPES:
            raise ValueError(
                f"Invalid approval_type {approval_type!r}; expected one of {_VALID_APPROVAL_TYPES}"
            )
        if not actor or not actor.strip():
            raise ValueError("Approval actor must be non-empty")

        normalized_verdict = _VERDICT_MAP.get(verdict.strip().lower())
        if normalized_verdict is None:
            raise ValueError(
                f"Invalid approval verdict {verdict!r}; expected one of {list(_VERDICT_MAP)}"
            )

        approval = ApprovalRecord(
            approval_id=str(uuid.uuid4()),
            release_id=release_id,
            role=role,
            approver=actor.strip(),
            decision=normalized_verdict,
            comment=note,
        )
        self.db.add_approval(approval)
        return self._require_release(release_id)

    def approve_release(self, release_id: str) -> ReleaseRecord:
        """Transition VALIDATED -> APPROVED only when both qc and privacy approvals are APPROVED."""
        release = self._require_release(release_id)
        if release.status != ReleaseStatus.VALIDATED:
            raise ValueError(
                f"Release {release_id} must be VALIDATED before approval (current: {release.status.value})"
            )
        if not self.contract_engine.can_transition(release.status, ReleaseStatus.APPROVED):
            raise ValueError(
                f"Transition {release.status.value} -> APPROVED is not allowed by contract"
            )

        approvals = self.db.list_approvals(release_id)
        latest_by_role: dict[str, ApprovalRecord] = {}
        for record in approvals:
            latest_by_role[record.role.lower()] = record

        mandatory_roles = [r.lower() for r in self.contract_engine.mandatory_approval_roles()]
        missing_or_unapproved = [
            role
            for role in mandatory_roles
            if role not in latest_by_role or latest_by_role[role].decision != "APPROVED"
        ]
        if missing_or_unapproved:
            raise ValueError(
                f"Cannot approve release {release_id}; missing or non-approved mandatory sign-offs: {missing_or_unapproved}"
            )

        self.db.update_release_status(release_id, ReleaseStatus.APPROVED)
        return self._require_release(release_id)

    def publish_release(self, release_id: str) -> ReleaseRecord:
        """Transition APPROVED -> PUBLISHED (published is explicitly NOT accepted)."""
        release = self._require_release(release_id)
        if release.status != ReleaseStatus.APPROVED:
            raise ValueError(
                f"Release {release_id} must be APPROVED before publishing (current: {release.status.value})"
            )
        if not self.contract_engine.can_transition(release.status, ReleaseStatus.PUBLISHED):
            raise ValueError(
                f"Transition {release.status.value} -> PUBLISHED is not allowed by contract"
            )

        self.db.update_release_status(release_id, ReleaseStatus.PUBLISHED)
        return self._require_release(release_id)

    def accept_release(
        self,
        release_id: str,
        consumer: str,
        note: str = "",
        consumer_version: str | None = None,
        verified_sha256: str | None = None,
    ) -> ReleaseRecord:
        """Transition PUBLISHED -> ACCEPTED only upon explicit ConsumerAck."""
        release = self._require_release(release_id)
        if release.status != ReleaseStatus.PUBLISHED:
            raise ValueError(
                f"Release {release_id} must be PUBLISHED before acceptance (current: {release.status.value})"
            )
        if not consumer or not consumer.strip():
            raise ValueError("Consumer name must be non-empty")
        if not self.contract_engine.can_transition(release.status, ReleaseStatus.ACCEPTED):
            raise ValueError(
                f"Transition {release.status.value} -> ACCEPTED is not allowed by contract"
            )

        ack = ConsumerAck(
            ack_id=str(uuid.uuid4()),
            release_id=release_id,
            consumer_name=consumer.strip(),
            consumer_version=consumer_version,
            status="ACCEPTED",
            verified_sha256=verified_sha256,
            comment=note,
        )
        self.db.add_consumer_ack(ack)
        self.db.update_release_status(release_id, ReleaseStatus.ACCEPTED)

        if self.lineage_tracker is not None:
            self.lineage_tracker.record_edge(
                source_type="release",
                source_id=release_id,
                target_type="consumer",
                target_id=consumer.strip(),
                relation="accepted_by",
                evidence_status="verified",
                metadata={"ack_id": ack.ack_id},
                actor=consumer.strip(),
            )
        return self._require_release(release_id)

    def rework_release(self, release_id: str, reason: str) -> ReleaseRecord:
        """Transition HOLD -> REWORK and record the rework reason."""
        release = self._require_release(release_id)
        if release.status != ReleaseStatus.HOLD:
            raise ValueError(
                f"Release {release_id} must be in HOLD to move to REWORK (current: {release.status.value})"
            )
        if not self.contract_engine.can_transition(release.status, ReleaseStatus.REWORK):
            raise ValueError(
                f"Transition {release.status.value} -> REWORK is not allowed by contract"
            )

        updated_meta = {
            **release.metadata,
            "rework_reason": reason,
        }
        updated_record = release.model_copy(
            update={
                "status": ReleaseStatus.REWORK,
                "metadata": updated_meta,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.db.upsert_release(updated_record)
        return self._require_release(release_id)
