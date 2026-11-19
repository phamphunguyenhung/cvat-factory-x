"""SQLite state and draft store for CVAT Annotation Factory.

Strictly separates the 5 revision stages:
  - original_annotation
  - model_proposal
  - refined_proposal
  - user_reviewed_revision
  - cvat_synchronized_revision

Persists full provenance per revision and atomic, resumable run checkpoints.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from cvat_tools.core import paths
from cvat_tools.factory.core.models import (
    Proposal,
    Provenance,
    QualityVerdict,
    RevisionStage,
    RunState,
    RunStatus,
    ValidationIssue,
    utc_now_iso,
)

DEFAULT_FACTORY_DB = paths.DATA / "store" / "factory_drafts.sqlite"

REVISION_STAGES: tuple[str, ...] = tuple(s.value for s in RevisionStage)
REVISION_STATUSES: tuple[str, ...] = ("active", "superseded", "rejected")

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL UNIQUE,
    job_id INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    shape_type TEXT NOT NULL,
    stage TEXT NOT NULL CHECK(stage IN (
        'original_annotation',
        'model_proposal',
        'refined_proposal',
        'user_reviewed_revision',
        'cvat_synchronized_revision'
    )),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'rejected')),
    parent_proposal_id TEXT,
    cvat_shape_id INTEGER,
    track_id INTEGER,
    geometry_json TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '[]',
    z_order INTEGER NOT NULL DEFAULT 0,
    occluded INTEGER NOT NULL DEFAULT 0,
    outside INTEGER NOT NULL DEFAULT 0,
    verdict TEXT,
    issues_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    model_id TEXT NOT NULL DEFAULT '',
    image_hash TEXT NOT NULL DEFAULT '',
    preprocess_params_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL,
    provenance_ts TEXT NOT NULL,
    rule_decisions_json TEXT NOT NULL DEFAULT '[]',
    run_id TEXT NOT NULL DEFAULT '',
    adapter_name TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL DEFAULT '1.0',
    provenance_extra_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_rev_jfs ON revisions(job_id, frame_id, stage, status);
CREATE INDEX IF NOT EXISTS ix_rev_run ON revisions(run_id, stage);
CREATE INDEX IF NOT EXISTS ix_rev_cvat ON revisions(job_id, cvat_shape_id);
CREATE INDEX IF NOT EXISTS ix_rev_parent ON revisions(parent_proposal_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    job_id INTEGER NOT NULL,
    task_id INTEGER,
    adapter_name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'paused', 'completed', 'failed')),
    current_stage TEXT NOT NULL DEFAULT 'init',
    total_frames INTEGER NOT NULL DEFAULT 0,
    completed_frames_json TEXT NOT NULL DEFAULT '[]',
    failed_frames_json TEXT NOT NULL DEFAULT '{}',
    last_checkpoint_frame INTEGER,
    config_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
);

CREATE INDEX IF NOT EXISTS ix_runs_job_status ON runs(job_id, status, updated_at);

CREATE TABLE IF NOT EXISTS run_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    frame_id INTEGER,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_ckpt_run ON run_checkpoints(run_id, id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    job_id INTEGER,
    run_id TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_events_job ON events(job_id, id);
"""


def _coerce_stage(stage: RevisionStage | str) -> RevisionStage:
    if isinstance(stage, RevisionStage):
        return stage
    return RevisionStage(stage)


def _coerce_run_status(status: RunStatus | str) -> RunStatus:
    if isinstance(status, RunStatus):
        return status
    return RunStatus(status)


def _row_to_proposal(row: sqlite3.Row) -> Proposal:
    prov = Provenance(
        model_id=row["model_id"],
        image_hash=row["image_hash"],
        preprocess_params=json.loads(row["preprocess_params_json"]),
        confidence=row["confidence"],
        timestamp=row["provenance_ts"],
        rule_decisions=json.loads(row["rule_decisions_json"]),
        run_id=row["run_id"],
        adapter_name=row["adapter_name"],
        schema_version=row["schema_version"],
        extra=json.loads(row["provenance_extra_json"]),
    )
    verdict_raw = row["verdict"]
    return Proposal(
        proposal_id=row["proposal_id"],
        job_id=int(row["job_id"]),
        frame_id=int(row["frame_id"]),
        label=row["label"],
        shape_type=row["shape_type"],
        geometry=json.loads(row["geometry_json"]),
        provenance=prov,
        stage=RevisionStage(row["stage"]),
        confidence=row["confidence"],
        cvat_shape_id=row["cvat_shape_id"],
        track_id=row["track_id"],
        parent_proposal_id=row["parent_proposal_id"],
        verdict=QualityVerdict(verdict_raw) if verdict_raw is not None else None,
        issues=[ValidationIssue.from_dict(x) for x in json.loads(row["issues_json"])],
        attributes=json.loads(row["attributes_json"]),
        z_order=int(row["z_order"]),
        occluded=bool(row["occluded"]),
        outside=bool(row["outside"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _row_to_run_state(row: sqlite3.Row) -> RunState:
    return RunState(
        run_id=row["run_id"],
        job_id=int(row["job_id"]),
        task_id=int(row["task_id"]) if row["task_id"] is not None else None,
        adapter_name=row["adapter_name"],
        status=RunStatus(row["status"]),
        current_stage=row["current_stage"],
        total_frames=int(row["total_frames"]),
        completed_frames=[int(x) for x in json.loads(row["completed_frames_json"])],
        failed_frames={int(k): str(v) for k, v in json.loads(row["failed_frames_json"]).items()},
        last_checkpoint_frame=row["last_checkpoint_frame"],
        config=json.loads(row["config_json"]),
        metrics=json.loads(row["metrics_json"]),
        started_at=row["started_at"],
        updated_at=row["updated_at"],
        error=row["error"],
    )


class FactoryDraftStore:
    """SQLite persistence layer for revisions, full provenance, and resumable run checkpoints."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = Path(db_path) if db_path is not None else DEFAULT_FACTORY_DB
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30.0)
        self.db.row_factory = sqlite3.Row
        if str(self.path) != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA_SQL)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> FactoryDraftStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Revision storage (5 isolated stages + full provenance)
    # ------------------------------------------------------------------
    def _insert_revision_row(self, proposal: Proposal, *, status: str = "active") -> int:
        prov = proposal.provenance
        conf = proposal.confidence if proposal.confidence is not None else prov.confidence
        cur = self.db.execute(
            """
            INSERT INTO revisions (
                proposal_id, job_id, frame_id, label, shape_type, stage, status,
                parent_proposal_id, cvat_shape_id, track_id,
                geometry_json, attributes_json, z_order, occluded, outside,
                verdict, issues_json, metadata_json,
                model_id, image_hash, preprocess_params_json, confidence,
                provenance_ts, rule_decisions_json, run_id, adapter_name,
                schema_version, provenance_extra_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                proposal.proposal_id,
                int(proposal.job_id),
                int(proposal.frame_id),
                proposal.label,
                proposal.shape_type,
                proposal.stage.value,
                status,
                proposal.parent_proposal_id,
                proposal.cvat_shape_id,
                proposal.track_id,
                json.dumps(proposal.geometry, ensure_ascii=False),
                json.dumps(proposal.attributes, ensure_ascii=False),
                int(proposal.z_order),
                int(bool(proposal.occluded)),
                int(bool(proposal.outside)),
                proposal.verdict.value if proposal.verdict is not None else None,
                json.dumps([i.to_dict() for i in proposal.issues], ensure_ascii=False),
                json.dumps(proposal.metadata, ensure_ascii=False),
                prov.model_id,
                prov.image_hash,
                json.dumps(prov.preprocess_params, ensure_ascii=False),
                conf,
                prov.timestamp or utc_now_iso(),
                json.dumps(prov.rule_decisions, ensure_ascii=False),
                prov.run_id,
                prov.adapter_name,
                prov.schema_version,
                json.dumps(prov.extra, ensure_ascii=False),
                utc_now_iso(),
            ),
        )
        return int(cur.lastrowid)

    def save_revision(
        self,
        proposal: Proposal,
        *,
        status: str = "active",
        supersede_previous: bool = False,
    ) -> int:
        """Atomically persist a single Proposal revision."""
        if status not in REVISION_STATUSES:
            raise ValueError(f"Invalid revision status: {status!r}")
        with self.db:
            if supersede_previous:
                self.db.execute(
                    """
                    UPDATE revisions
                    SET status = 'superseded'
                    WHERE job_id = ? AND frame_id = ? AND label = ? AND stage = ? AND status = 'active'
                    """,
                    (proposal.job_id, proposal.frame_id, proposal.label, proposal.stage.value),
                )
            row_id = self._insert_revision_row(proposal, status=status)
            self._log_event(
                "save_revision",
                proposal.job_id,
                proposal.provenance.run_id or None,
                {
                    "proposal_id": proposal.proposal_id,
                    "frame_id": proposal.frame_id,
                    "stage": proposal.stage.value,
                    "model_id": proposal.provenance.model_id,
                },
            )
            return row_id

    def save_revisions_batch(
        self,
        proposals: Sequence[Proposal],
        *,
        status: str = "active",
        supersede_previous: bool = False,
    ) -> list[int]:
        """Atomically persist a batch of Proposal revisions in a single transaction."""
        if status not in REVISION_STATUSES:
            raise ValueError(f"Invalid revision status: {status!r}")
        if not proposals:
            return []
        row_ids: list[int] = []
        with self.db:
            if supersede_previous:
                seen_keys: set[tuple[int, int, str, str]] = set()
                for p in proposals:
                    key = (p.job_id, p.frame_id, p.label, p.stage.value)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        self.db.execute(
                            """
                            UPDATE revisions
                            SET status = 'superseded'
                            WHERE job_id = ? AND frame_id = ? AND label = ? AND stage = ? AND status = 'active'
                            """,
                            key,
                        )
            for p in proposals:
                row_ids.append(self._insert_revision_row(p, status=status))
        return row_ids

    def get_revision(self, proposal_id: str) -> Proposal | None:
        row = self.db.execute(
            "SELECT * FROM revisions WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return _row_to_proposal(row) if row is not None else None

    def list_revisions(
        self,
        job_id: int,
        *,
        frame_id: int | None = None,
        stage: RevisionStage | str | None = None,
        run_id: str | None = None,
        status: str | None = "active",
    ) -> list[Proposal]:
        clauses = ["job_id = ?"]
        params: list[Any] = [int(job_id)]
        if frame_id is not None:
            clauses.append("frame_id = ?")
            params.append(int(frame_id))
        if stage is not None:
            clauses.append("stage = ?")
            params.append(_coerce_stage(stage).value)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        sql = f"SELECT * FROM revisions WHERE {' AND '.join(clauses)} ORDER BY frame_id ASC, id ASC"
        rows = self.db.execute(sql, params).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def get_latest_by_stage(
        self,
        job_id: int,
        frame_id: int,
        stage: RevisionStage | str,
    ) -> list[Proposal]:
        return self.list_revisions(job_id, frame_id=frame_id, stage=stage, status="active")

    def mark_status(self, proposal_id: str, status: str, note: str = "") -> None:
        if status not in REVISION_STATUSES:
            raise ValueError(f"Invalid revision status: {status!r}")
        with self.db:
            row = self.db.execute(
                "SELECT job_id, run_id, status FROM revisions WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise KeyError(proposal_id)
            self.db.execute(
                "UPDATE revisions SET status = ? WHERE proposal_id = ?",
                (status, proposal_id),
            )
            self._log_event(
                "revision_status",
                row["job_id"],
                row["run_id"] or None,
                {"proposal_id": proposal_id, "old": row["status"], "new": status, "note": note},
            )

    def record_cvat_sync(
        self,
        proposal_id: str,
        cvat_shape_id: int | None,
        *,
        sync_proposal_id: str | None = None,
        note: str = "",
    ) -> Proposal:
        """Create a cvat_synchronized_revision child record without mutating earlier stages."""
        source = self.get_revision(proposal_id)
        if source is None:
            raise KeyError(proposal_id)
        synced = Proposal.from_dict(source.to_dict())
        if sync_proposal_id:
            synced.proposal_id = sync_proposal_id
        else:
            import uuid

            synced.proposal_id = uuid.uuid4().hex
        synced.stage = RevisionStage.CVAT_SYNCHRONIZED_REVISION
        synced.parent_proposal_id = source.proposal_id
        synced.cvat_shape_id = cvat_shape_id
        synced.provenance = Provenance.from_dict(source.provenance.to_dict())
        synced.provenance.timestamp = utc_now_iso()
        if note:
            synced.metadata["sync_note"] = note
        self.save_revision(synced, supersede_previous=True)
        return synced

    def stage_counts(self, job_id: int, *, status: str | None = "active") -> dict[str, int]:
        counts = {s: 0 for s in REVISION_STAGES}
        if status is None:
            rows = self.db.execute(
                "SELECT stage, COUNT(*) AS cnt FROM revisions WHERE job_id = ? GROUP BY stage",
                (int(job_id),),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT stage, COUNT(*) AS cnt FROM revisions WHERE job_id = ? AND status = ? GROUP BY stage",
                (int(job_id), status),
            ).fetchall()
        for r in rows:
            counts[r["stage"]] = int(r["cnt"])
        return counts

    # ------------------------------------------------------------------
    # Atomic resumable run checkpoints
    # ------------------------------------------------------------------
    def create_run(self, run_state: RunState) -> None:
        now = utc_now_iso()
        run_state.started_at = run_state.started_at or now
        run_state.updated_at = now
        with self.db:
            self.db.execute(
                """
                INSERT INTO runs (
                    run_id, job_id, task_id, adapter_name, status, current_stage,
                    total_frames, completed_frames_json, failed_frames_json,
                    last_checkpoint_frame, config_json, metrics_json,
                    started_at, updated_at, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_state.run_id,
                    int(run_state.job_id),
                    run_state.task_id,
                    run_state.adapter_name,
                    run_state.status.value,
                    run_state.current_stage,
                    int(run_state.total_frames),
                    json.dumps(sorted(set(run_state.completed_frames))),
                    json.dumps({str(k): v for k, v in run_state.failed_frames.items()}),
                    run_state.last_checkpoint_frame,
                    json.dumps(run_state.config, ensure_ascii=False),
                    json.dumps(run_state.metrics, ensure_ascii=False),
                    run_state.started_at,
                    run_state.updated_at,
                    run_state.error,
                ),
            )
            self._log_event("run_created", run_state.job_id, run_state.run_id, {"stage": run_state.current_stage})

    def checkpoint_run(
        self,
        run_state: RunState,
        *,
        frame_id: int | None = None,
        stage: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Atomically update run state and append a checkpoint row in a single transaction."""
        now = utc_now_iso()
        run_state.updated_at = now
        if frame_id is not None:
            run_state.last_checkpoint_frame = int(frame_id)
        if stage is not None:
            run_state.current_stage = stage
        with self.db:
            cur = self.db.execute(
                """
                UPDATE runs
                SET task_id = ?, adapter_name = ?, status = ?, current_stage = ?,
                    total_frames = ?, completed_frames_json = ?, failed_frames_json = ?,
                    last_checkpoint_frame = ?, config_json = ?, metrics_json = ?,
                    updated_at = ?, error = ?
                WHERE run_id = ?
                """,
                (
                    run_state.task_id,
                    run_state.adapter_name,
                    run_state.status.value,
                    run_state.current_stage,
                    int(run_state.total_frames),
                    json.dumps(sorted(set(run_state.completed_frames))),
                    json.dumps({str(k): v for k, v in run_state.failed_frames.items()}),
                    run_state.last_checkpoint_frame,
                    json.dumps(run_state.config, ensure_ascii=False),
                    json.dumps(run_state.metrics, ensure_ascii=False),
                    run_state.updated_at,
                    run_state.error,
                    run_state.run_id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Unknown run_id: {run_state.run_id}")
            self.db.execute(
                """
                INSERT INTO run_checkpoints (run_id, frame_id, stage, payload_json, created_at)
                VALUES (?,?,?,?,?)
                """,
                (
                    run_state.run_id,
                    int(frame_id) if frame_id is not None else None,
                    stage or run_state.current_stage,
                    json.dumps(payload or {}, ensure_ascii=False),
                    now,
                ),
            )

    def get_run(self, run_id: str) -> RunState | None:
        row = self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_run_state(row) if row is not None else None

    def get_latest_run(
        self,
        job_id: int,
        *,
        status: RunStatus | str | None = None,
    ) -> RunState | None:
        if status is not None:
            st = _coerce_run_status(status).value
            row = self.db.execute(
                "SELECT * FROM runs WHERE job_id = ? AND status = ? ORDER BY updated_at DESC LIMIT 1",
                (int(job_id), st),
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM runs WHERE job_id = ? ORDER BY updated_at DESC LIMIT 1",
                (int(job_id),),
            ).fetchone()
        return _row_to_run_state(row) if row is not None else None

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM run_checkpoints WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "run_id": r["run_id"],
                "frame_id": r["frame_id"],
                "stage": r["stage"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def list_pending_frames(self, run_id: str, all_frames: Sequence[int]) -> list[int]:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        done = set(run.completed_frames)
        return [int(f) for f in all_frames if int(f) not in done]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def _log_event(
        self,
        kind: str,
        job_id: int | None,
        run_id: str | None,
        detail: dict[str, Any],
    ) -> None:
        self.db.execute(
            "INSERT INTO events (ts, kind, job_id, run_id, detail_json) VALUES (?,?,?,?,?)",
            (utc_now_iso(), kind, job_id, run_id, json.dumps(detail, ensure_ascii=False)),
        )

    def list_events(self, job_id: int | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
        if job_id is not None:
            rows = self.db.execute(
                "SELECT * FROM events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (int(job_id), int(limit)),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "ts": r["ts"],
                "kind": r["kind"],
                "job_id": r["job_id"],
                "run_id": r["run_id"],
                "detail": json.loads(r["detail_json"]),
            }
            for r in rows
        ]
