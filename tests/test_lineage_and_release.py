"""Targeted unit tests for Lineage Engine and Release Gate & Evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from release_control.adapters.mlflow_adapter import MlflowTracker
from release_control.db import ControlTowerDB
from release_control.lineage import ImpactAnalyzer, LineageTracker
from release_control.models import (
    ReleaseStatus,
    SnapshotRecord,
    ValidationReport,
)
from release_control.release import EvidenceBuilder, ReleaseGate


def _seed_snapshot(db: ControlTowerDB, snapshot_id: str = "snap-1", digest: str = "sha256-snap-1") -> SnapshotRecord:
    snap = SnapshotRecord(
        snapshot_id=snapshot_id,
        dataset_name="road_objects",
        version="v1.0.0",
        source_path="/data/raw",
        snapshot_path="/artifacts/snapshots/snap-1",
        manifest_sha256=digest,
        image_count=10,
        label_count=25,
    )
    db.upsert_snapshot(snap)
    return snap


def test_lineage_tracker_and_impact_analyzer(tmp_path: Path) -> None:
    db = ControlTowerDB(tmp_path / "tower.db")
    ol_path = tmp_path / "openlineage.jsonl"
    tracker = LineageTracker(db=db, openlineage_path=ol_path)

    snap = _seed_snapshot(db, snapshot_id="snap-1", digest="digest-abc")
    tracker.record_edge("raw", "raw-batch-1", "snapshot", snap.snapshot_id, "ingested_into", evidence_status="verified")
    tracker.record_edge("snapshot", snap.snapshot_id, "release", "rel-1", "released_as", evidence_status="verified")
    tracker.record_edge("snapshot", snap.snapshot_id, "release", "rel-unverified", "guessed_link", evidence_status="unverified")

    events = tracker.read_events()
    assert len(events) == 3
    for ev in events:
        assert {"eventTime", "eventType", "run", "job", "inputs", "outputs", "evidence_status"} <= set(ev.keys())

    mlflow_tracker = MlflowTracker(tracking_uri=tmp_path / "mlruns.db")
    mlflow_tracker.log_simulated_training_run(
        release_id="rel-1",
        dataset_digest="digest-abc",
        model_name="yolov8n-road",
        metrics={"mAP50": 0.88},
        run_origin="simulated_demo",
    )
    mlflow_tracker.log_simulated_training_run(
        release_id="rel-1",
        dataset_digest="digest-abc",
        model_name="yolov8m-road",
        metrics={"mAP50": 0.92},
        run_origin="actual_mlflow",
    )

    analyzer = ImpactAnalyzer(db=db, mlflow_tracker=mlflow_tracker)
    fwd = analyzer.trace_forward("raw-batch-1")
    fwd_ids = {n["id"] for n in fwd["nodes"]}
    assert "snap-1" in fwd_ids
    assert "rel-1" in fwd_ids
    assert "rel-unverified" not in fwd_ids  # unverified omitted by default

    bwd = analyzer.trace_backward("rel-1")
    bwd_ids = {n["id"] for n in bwd["nodes"]}
    assert "snap-1" in bwd_ids
    assert "raw-batch-1" in bwd_ids

    blast = analyzer.get_blast_radius("digest-abc")
    assert len(blast["snapshots"]) == 1
    assert len(blast["training_runs"]["actual_mlflow"]) == 1
    assert len(blast["training_runs"]["simulated_demo"]) == 1
    assert {m["model_name"] for m in blast["models"]} == {"yolov8n-road", "yolov8m-road"}
    assert len(blast["graph"]["nodes"]) > 0
    assert len(blast["graph"]["edges"]) > 0


def test_release_gate_lifecycle_and_evidence(tmp_path: Path) -> None:
    db = ControlTowerDB(tmp_path / "tower.db")
    tracker = LineageTracker(db=db, openlineage_path=tmp_path / "ol.jsonl")
    gate = ReleaseGate(db=db, lineage_tracker=tracker)
    snap = _seed_snapshot(db, snapshot_id="snap-42", digest="digest-42")

    rel = gate.create_release(
        release_id="rel-42",
        dataset_id="road_objects",
        version_tag="v1.0.0",
        snapshot_digest=snap.manifest_sha256,
        contract_version="1.0.0",
    )
    assert rel.status == ReleaseStatus.DRAFT

    # Failed validation -> HOLD
    fail_report = ValidationReport(
        report_id="rep-fail",
        release_id="rel-42",
        snapshot_id="snap-42",
        passed=False,
        critical_count=1,
    )
    rel = gate.submit_validation("rel-42", fail_report)
    assert rel.status == ReleaseStatus.HOLD

    # HOLD -> REWORK
    rel = gate.rework_release("rel-42", reason="Fix out-of-bounds bbox")
    assert rel.status == ReleaseStatus.REWORK

    # REWORK -> HOLD when requires_human_review is True even if passed=True
    review_report = ValidationReport(
        report_id="rep-review",
        release_id="rel-42",
        snapshot_id="snap-42",
        passed=True,
        metrics={"requires_human_review": True},
    )
    rel = gate.submit_validation("rel-42", review_report)
    assert rel.status == ReleaseStatus.HOLD

    rel = gate.rework_release("rel-42", reason="Human review cleared")
    pass_report = ValidationReport(
        report_id="rep-pass",
        release_id="rel-42",
        snapshot_id="snap-42",
        passed=True,
    )
    rel = gate.submit_validation("rel-42", pass_report)
    assert rel.status == ReleaseStatus.VALIDATED

    # Invalid approvals raise
    with pytest.raises(ValueError):
        gate.submit_approval("rel-42", "qc", "", "approved")
    with pytest.raises(ValueError):
        gate.submit_approval("rel-42", "qc", "alice", "maybe")

    # Cannot approve with only qc
    gate.submit_approval("rel-42", "qc", "alice", "approved", note="QC looks good")
    with pytest.raises(ValueError):
        gate.approve_release("rel-42")

    # Privacy approval completes dual sign-off
    gate.submit_approval("rel-42", "privacy", "bob", "approved", note="PII blurred")
    rel = gate.approve_release("rel-42")
    assert rel.status == ReleaseStatus.APPROVED

    # Publish transitions to PUBLISHED (not ACCEPTED)
    rel = gate.publish_release("rel-42")
    assert rel.status == ReleaseStatus.PUBLISHED

    # Accept requires explicit consumer ACK
    rel = gate.accept_release("rel-42", consumer="training-pipeline-v2", note="Checksum verified")
    assert rel.status == ReleaseStatus.ACCEPTED

    # Build ReleasePassport
    evidence_path = tmp_path / "reports" / "RELEASE_EVIDENCE.json"
    builder = EvidenceBuilder(db=db)
    passport = builder.build_passport("rel-42", output_path=evidence_path, validation_report=pass_report)

    assert evidence_path.exists()
    raw = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert raw["release_id"] == "rel-42"
    assert raw["status"] == "ACCEPTED"
    assert raw["published_is_not_accepted"] is True
    assert raw["validation_summary"]["contract_version"] == "1.0.0"
    assert raw["validation_summary"]["snapshot_digest"] == "digest-42"
    assert len(raw["approvals"]) == 2
    assert len(raw["consumer_acks"]) == 1
    assert len(raw["lineage"]) >= 2
    assert passport.passport_sha256 == raw["passport_sha256"]
