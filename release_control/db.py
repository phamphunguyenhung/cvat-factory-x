"""SQLite WAL-mode persistence layer for CVAT Factory X Control Tower."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from release_control.models import (
    ApprovalRecord,
    ConsumerAck,
    FaultManifest,
    LineageEdge,
    ReleaseRecord,
    ReleaseStatus,
    SnapshotRecord,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


class ControlTowerDB:
    """SQLite index for snapshots, releases, approvals, lineage events, and fault runs."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    image_count INTEGER NOT NULL,
                    label_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS releases (
                    release_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    validation_report_id TEXT,
                    passport_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    approver TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (release_id) REFERENCES releases(release_id)
                );

                CREATE TABLE IF NOT EXISTS lineage_events (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fault_runs (
                    run_id TEXT PRIMARY KEY,
                    base_snapshot_id TEXT NOT NULL,
                    injected_at TEXT NOT NULL,
                    fault_count INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS consumer_acks (
                    ack_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    consumer_name TEXT NOT NULL,
                    consumer_version TEXT,
                    status TEXT NOT NULL,
                    verified_sha256 TEXT,
                    comment TEXT NOT NULL,
                    acked_at TEXT NOT NULL,
                    FOREIGN KEY (release_id) REFERENCES releases(release_id)
                );

                CREATE INDEX IF NOT EXISTS idx_releases_status ON releases(status);
                CREATE INDEX IF NOT EXISTS idx_approvals_release ON approvals(release_id);
                CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage_events(source_id);
                CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage_events(target_id);
                """
            )

    def upsert_snapshot(self, record: SnapshotRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (
                    snapshot_id, dataset_name, version, source_path, snapshot_path,
                    manifest_sha256, image_count, label_count, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    dataset_name=excluded.dataset_name,
                    version=excluded.version,
                    source_path=excluded.source_path,
                    snapshot_path=excluded.snapshot_path,
                    manifest_sha256=excluded.manifest_sha256,
                    image_count=excluded.image_count,
                    label_count=excluded.label_count,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record.snapshot_id,
                    record.dataset_name,
                    record.version,
                    record.source_path,
                    record.snapshot_path,
                    record.manifest_sha256,
                    record.image_count,
                    record.label_count,
                    _iso(record.created_at),
                    json.dumps(record.metadata),
                ),
            )

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def list_snapshots(self) -> list[SnapshotRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def upsert_release(self, record: ReleaseRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO releases (
                    release_id, dataset_name, version, snapshot_id, status,
                    validation_report_id, passport_path, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id) DO UPDATE SET
                    status=excluded.status,
                    validation_report_id=excluded.validation_report_id,
                    passport_path=excluded.passport_path,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record.release_id,
                    record.dataset_name,
                    record.version,
                    record.snapshot_id,
                    record.status.value,
                    record.validation_report_id,
                    record.passport_path,
                    _iso(record.created_at),
                    _iso(record.updated_at),
                    json.dumps(record.metadata),
                ),
            )

    def get_release(self, release_id: str) -> ReleaseRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        return self._row_to_release(row) if row else None

    def list_releases(self, status: ReleaseStatus | None = None) -> list[ReleaseRecord]:
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM releases WHERE status = ? ORDER BY updated_at DESC",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM releases ORDER BY updated_at DESC"
                ).fetchall()
        return [self._row_to_release(r) for r in rows]

    def update_release_status(
        self,
        release_id: str,
        status: ReleaseStatus,
        validation_report_id: str | None = None,
        passport_path: str | None = None,
    ) -> None:
        now = _iso(datetime.now(timezone.utc))
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE releases
                SET status = ?,
                    validation_report_id = COALESCE(?, validation_report_id),
                    passport_path = COALESCE(?, passport_path),
                    updated_at = ?
                WHERE release_id = ?
                """,
                (status.value, validation_report_id, passport_path, now, release_id),
            )

    def add_approval(self, record: ApprovalRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approvals (
                    approval_id, release_id, role, approver, decision, comment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.approval_id,
                    record.release_id,
                    record.role,
                    record.approver,
                    record.decision,
                    record.comment,
                    _iso(record.created_at),
                ),
            )

    def list_approvals(self, release_id: str) -> list[ApprovalRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE release_id = ? ORDER BY created_at ASC",
                (release_id,),
            ).fetchall()
        return [
            ApprovalRecord(
                approval_id=r["approval_id"],
                release_id=r["release_id"],
                role=r["role"],
                approver=r["approver"],
                decision=r["decision"],
                comment=r["comment"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def add_consumer_ack(self, record: ConsumerAck) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO consumer_acks (
                    ack_id, release_id, consumer_name, consumer_version,
                    status, verified_sha256, comment, acked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.ack_id,
                    record.release_id,
                    record.consumer_name,
                    record.consumer_version,
                    record.status,
                    record.verified_sha256,
                    record.comment,
                    _iso(record.acked_at),
                ),
            )

    def list_consumer_acks(self, release_id: str) -> list[ConsumerAck]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM consumer_acks WHERE release_id = ? ORDER BY acked_at ASC",
                (release_id,),
            ).fetchall()
        return [
            ConsumerAck(
                ack_id=r["ack_id"],
                release_id=r["release_id"],
                consumer_name=r["consumer_name"],
                consumer_version=r["consumer_version"],
                status=r["status"],
                verified_sha256=r["verified_sha256"],
                comment=r["comment"],
                acked_at=datetime.fromisoformat(r["acked_at"]),
            )
            for r in rows
        ]

    def add_lineage_event(self, edge: LineageEdge) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lineage_events (
                    edge_id, source_id, target_id, relation, actor, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.actor,
                    _iso(edge.created_at),
                    json.dumps(edge.metadata),
                ),
            )

    def list_lineage_events(self, entity_id: str | None = None) -> list[LineageEdge]:
        with self._connect() as conn:
            if entity_id is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM lineage_events
                    WHERE source_id = ? OR target_id = ?
                    ORDER BY created_at ASC
                    """,
                    (entity_id, entity_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM lineage_events ORDER BY created_at ASC"
                ).fetchall()
        return [
            LineageEdge(
                edge_id=r["edge_id"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation=r["relation"],
                actor=r["actor"],
                created_at=datetime.fromisoformat(r["created_at"]),
                metadata=json.loads(r["metadata_json"]),
            )
            for r in rows
        ]

    def add_fault_run(self, manifest: FaultManifest) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fault_runs (
                    run_id, base_snapshot_id, injected_at, fault_count, summary_json, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.base_snapshot_id,
                    _iso(manifest.injected_at),
                    len(manifest.faults),
                    json.dumps(manifest.summary),
                    manifest.model_dump_json(),
                ),
            )

    def list_fault_runs(self, base_snapshot_id: str | None = None) -> list[FaultManifest]:
        with self._connect() as conn:
            if base_snapshot_id is not None:
                rows = conn.execute(
                    "SELECT manifest_json FROM fault_runs WHERE base_snapshot_id = ? ORDER BY injected_at DESC",
                    (base_snapshot_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT manifest_json FROM fault_runs ORDER BY injected_at DESC"
                ).fetchall()
        return [FaultManifest.model_validate_json(r["manifest_json"]) for r in rows]

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> SnapshotRecord:
        return SnapshotRecord(
            snapshot_id=row["snapshot_id"],
            dataset_name=row["dataset_name"],
            version=row["version"],
            source_path=row["source_path"],
            snapshot_path=row["snapshot_path"],
            manifest_sha256=row["manifest_sha256"],
            image_count=row["image_count"],
            label_count=row["label_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_release(row: sqlite3.Row) -> ReleaseRecord:
        return ReleaseRecord(
            release_id=row["release_id"],
            dataset_name=row["dataset_name"],
            version=row["version"],
            snapshot_id=row["snapshot_id"],
            status=ReleaseStatus(row["status"]),
            validation_report_id=row["validation_report_id"],
            passport_path=row["passport_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata_json"]),
        )
