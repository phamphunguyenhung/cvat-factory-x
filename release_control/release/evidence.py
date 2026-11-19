"""Release Evidence Passport builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from release_control.contracts.engine import ContractEngine
from release_control.db import ControlTowerDB
from release_control.models import LineageEdge, ReleasePassport, ValidationReport


class EvidenceBuilder:
    """Assembles and signs ReleasePassport evidence bundles from ControlTowerDB."""

    def __init__(
        self,
        db: ControlTowerDB,
        contract_engine: ContractEngine | None = None,
    ) -> None:
        self.db = db
        self.contract_engine = contract_engine or ContractEngine()

    def build_passport(
        self,
        release_id: str,
        output_path: Path | str | None = Path("reports/RELEASE_EVIDENCE.json"),
        validation_report: ValidationReport | None = None,
    ) -> ReleasePassport:
        """Assemble a ReleasePassport for release_id and optionally persist it as JSON."""
        release = self.db.get_release(release_id)
        if release is None:
            raise KeyError(f"Release not found: {release_id}")

        snapshot = self.db.get_snapshot(release.snapshot_id)
        if snapshot is None:
            raise ValueError(
                f"Snapshot {release.snapshot_id!r} not found for release {release_id}"
            )

        approvals = self.db.list_approvals(release_id)
        consumer_acks = self.db.list_consumer_acks(release_id)

        # Gather lineage edges touching either the release or its underlying snapshot.
        edges_by_id: dict[str, LineageEdge] = {}
        for entity_id in (release_id, release.snapshot_id):
            for edge in self.db.list_lineage_events(entity_id):
                edges_by_id[edge.edge_id] = edge
        lineage = list(edges_by_id.values())

        validation_summary: dict[str, Any] = {
            "contract_version": release.metadata.get("contract_version", "1.0.0"),
            "snapshot_digest": snapshot.manifest_sha256,
            "validation_report_id": release.validation_report_id,
        }
        stored_summary = release.metadata.get("validation_summary")
        if isinstance(stored_summary, dict):
            validation_summary.update(stored_summary)
        if validation_report is not None:
            validation_summary.update(
                {
                    "report_id": validation_report.report_id,
                    "passed": validation_report.passed,
                    "critical_count": validation_report.critical_count,
                    "warning_count": validation_report.warning_count,
                    "info_count": validation_report.info_count,
                    "metrics": validation_report.metrics,
                    "checked_at": validation_report.checked_at.isoformat(),
                }
            )

        passport = ReleasePassport(
            release_id=release.release_id,
            dataset_name=release.dataset_name,
            version=release.version,
            status=release.status,
            snapshot=snapshot,
            validation_summary=validation_summary,
            approvals=approvals,
            consumer_acks=consumer_acks,
            lineage=lineage,
            published_is_not_accepted=self.contract_engine.published_is_not_accepted(),
        )

        canonical_payload = json.dumps(
            passport.model_dump(mode="json", exclude={"passport_sha256"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical_payload).hexdigest()
        passport = passport.model_copy(update={"passport_sha256": digest})

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(passport.model_dump_json(indent=2), encoding="utf-8")
            self.db.update_release_status(
                release_id,
                release.status,
                passport_path=str(out_file),
            )

        return passport
