"""Test Matrix Items 17-23: Human edit preservation, concurrent edit, failed/partial sync,
read-back mismatch, rollback conflict, resume after crash, repeated run idempotence.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from cvat_tools.factory.core.models import (
    Proposal,
    Provenance,
    RevisionStage,
    RunState,
    RunStatus,
)
from cvat_tools.factory.storage import FactoryDraftStore
from cvat_tools.factory.sync import (
    OperationKind,
    QualityGateStatus,
    ReadBackMismatchError,
    RetryPolicy,
    StaleSnapshotError,
    SyncController,
    SyncOperation,
    compute_annotations_fingerprint,
)


class _FakeSyncClient:
    """In-memory mock CVAT client supporting read-back mismatch, transient failures, and concurrent edits."""

    def __init__(
        self,
        *,
        shapes: list[dict[str, Any]] | None = None,
        labels: list[dict[str, Any]] | None = None,
        write_enabled: bool = True,
        corrupt_readback: bool = False,
        fail_update_times: int = 0,
    ) -> None:
        self.write_enabled = write_enabled
        self._shapes: list[dict[str, Any]] = copy.deepcopy(shapes or [])
        self._labels: list[dict[str, Any]] = copy.deepcopy(
            labels or [{"id": 1, "name": "car", "type": "rectangle"}]
        )
        self._next_id = 1000
        self.corrupt_readback = corrupt_readback
        self.fail_update_times = fail_update_times
        self.create_calls = 0
        self.update_calls = 0
        self.delete_calls = 0

    def labels(self, job_id: int) -> list[dict[str, Any]]:
        return copy.deepcopy(self._labels)

    def annotations(self, job_id: int) -> dict[str, Any]:
        return {"version": 1, "shapes": copy.deepcopy(self._shapes), "tracks": [], "tags": []}

    def create_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> dict[str, Any]:
        self.create_calls += 1
        created = []
        for s in shapes:
            item = copy.deepcopy(s)
            item["id"] = self._next_id
            self._next_id += 1
            if self.corrupt_readback and item.get("points"):
                item["points"] = [p + 25.0 for p in item["points"]]
            self._shapes.append(item)
            created.append(item)
        return {"shapes": created}

    def update_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> dict[str, Any]:
        self.update_calls += 1
        if self.fail_update_times > 0:
            self.fail_update_times -= 1
            raise TimeoutError("503 Service Unavailable (simulated transient error)")
        by_id = {s["id"]: s for s in self._shapes if "id" in s}
        updated = []
        for s in shapes:
            sid = s["id"]
            if sid in by_id:
                by_id[sid].update(copy.deepcopy(s))
                updated.append(by_id[sid])
        return {"shapes": updated}

    def delete_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> None:
        self.delete_calls += 1
        del_ids = {s["id"] for s in shapes if "id" in s}
        self._shapes = [s for s in self._shapes if s.get("id") not in del_ids]


def test_17_human_edit_preservation(tmp_path: Path) -> None:
    """Item 17: Protected human edits are never overwritten in FactoryDraftStore or SyncController."""
    with FactoryDraftStore(tmp_path / "drafts.sqlite") as store:
        ai_prop = Proposal(
            proposal_id="p_ai_1",
            job_id=285,
            frame_id=0,
            label="car",
            shape_type="rectangle",
            geometry={"points": [10.0, 10.0, 50.0, 50.0], "label_id": 1},
            stage=RevisionStage.MODEL_PROPOSAL,
            provenance=Provenance(model_id="yolo26n"),
        )
        human_prop = Proposal(
            proposal_id="p_human_1",
            job_id=285,
            frame_id=0,
            label="car",
            shape_type="rectangle",
            geometry={"points": [12.0, 11.0, 52.0, 51.0], "label_id": 1},
            stage=RevisionStage.USER_REVIEWED_REVISION,
            cvat_shape_id=101,
            provenance=Provenance(model_id="human"),
        )
        store.save_revision(ai_prop)
        store.save_revision(human_prop)
        assert store.get_revision("p_human_1") is not None
        counts = store.stage_counts(285)
        assert counts["model_proposal"] == 1
        assert counts["user_reviewed_revision"] == 1

    client = _FakeSyncClient(
        shapes=[{"id": 101, "type": "rectangle", "frame": 0, "label_id": 1, "points": [12.0, 11.0, 52.0, 51.0]}]
    )
    ctrl = SyncController(backup_dir=tmp_path / "backups", run_dir=tmp_path / "runs")
    op = SyncOperation(
        op_id="op_17",
        kind=OperationKind.UPDATE,
        frame=0,
        label="car",
        shape_type="rectangle",
        target_cvat_id=101,
        payload={"id": 101, "type": "rectangle", "frame": 0, "label_id": 1, "points": [99.0, 99.0, 150.0, 150.0]},
        quality_status=QualityGateStatus.PASS,
    )
    res = ctrl.sync(client, 285, [op], dry_run=False, protected_shape_ids={101})
    assert len(res.skipped) >= 1
    assert client.update_calls == 0
    assert client._shapes[0]["points"] == [12.0, 11.0, 52.0, 51.0]


def test_18_concurrent_cvat_edit_raises_stale_snapshot(tmp_path: Path) -> None:
    """Item 18: Concurrent remote edit invalidates expected_fingerprint and blocks write."""
    client = _FakeSyncClient(
        shapes=[{"id": 101, "type": "rectangle", "frame": 0, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]}]
    )
    stale_fp = compute_annotations_fingerprint(client.annotations(285))

    # Simulate concurrent human edit on CVAT server
    client._shapes[0]["points"] = [20.0, 20.0, 60.0, 60.0]

    ctrl = SyncController(backup_dir=tmp_path / "backups", run_dir=tmp_path / "runs")
    op = SyncOperation(
        op_id="op_18",
        kind=OperationKind.CREATE,
        frame=1,
        label="car",
        shape_type="rectangle",
        payload={"type": "rectangle", "frame": 1, "label_id": 1, "points": [1.0, 2.0, 3.0, 4.0]},
        quality_status=QualityGateStatus.PASS,
    )
    with pytest.raises(StaleSnapshotError):
        ctrl.sync(client, 285, [op], dry_run=False, expected_fingerprint=stale_fp)
    assert client.create_calls == 0


def test_19_failed_or_partial_sync_retries_and_recovers(tmp_path: Path) -> None:
    """Item 19: Transient update failure retries cleanly under RetryPolicy; backup preserved."""
    client = _FakeSyncClient(
        shapes=[{"id": 101, "type": "rectangle", "frame": 0, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]}],
        fail_update_times=1,
    )
    ctrl = SyncController(
        backup_dir=tmp_path / "backups",
        run_dir=tmp_path / "runs",
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0),
    )
    op = SyncOperation(
        op_id="op_19",
        kind=OperationKind.UPDATE,
        frame=0,
        label="car",
        shape_type="rectangle",
        target_cvat_id=101,
        payload={"id": 101, "type": "rectangle", "frame": 0, "label_id": 1, "points": [15.0, 15.0, 55.0, 55.0]},
        quality_status=QualityGateStatus.PASS,
        preserve_points=False,
    )
    res = ctrl.sync(client, 285, [op], dry_run=False)
    assert res.verified is True
    assert client.update_calls == 2
    assert res.backup_path is not None and Path(res.backup_path).exists()


def test_20_readback_mismatch_detected(tmp_path: Path) -> None:
    """Item 20: Corrupted server geometry after create raises ReadBackMismatchError."""
    client = _FakeSyncClient(corrupt_readback=True)
    ctrl = SyncController(backup_dir=tmp_path / "backups", run_dir=tmp_path / "runs")
    op = SyncOperation(
        op_id="op_20",
        kind=OperationKind.CREATE,
        frame=0,
        label="car",
        shape_type="rectangle",
        payload={"type": "rectangle", "frame": 0, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]},
        quality_status=QualityGateStatus.PASS,
    )
    with pytest.raises(ReadBackMismatchError):
        ctrl.sync(client, 285, [op], dry_run=False, raise_on_readback_mismatch=True)


def test_21_rollback_conflict_preserves_newer_human_edit(tmp_path: Path) -> None:
    """Item 21: Rollback detects concurrent edit on created shape and refuses to delete it."""
    client = _FakeSyncClient()
    ctrl = SyncController(backup_dir=tmp_path / "backups", run_dir=tmp_path / "runs")
    op = SyncOperation(
        op_id="op_21",
        kind=OperationKind.CREATE,
        frame=0,
        label="car",
        shape_type="rectangle",
        payload={"type": "rectangle", "frame": 0, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]},
        quality_status=QualityGateStatus.PASS,
    )
    res = ctrl.sync(client, 285, [op], dry_run=False)
    assert res.mutation_log_path is not None

    # Human edits the newly created shape on CVAT after sync
    client._shapes[0]["points"] = [14.0, 14.0, 54.0, 54.0]

    rb = ctrl.rollback(client, res.mutation_log_path, dry_run=False)
    assert len(rb.kept_due_to_conflict) == 1
    assert client.delete_calls == 0
    assert len(client._shapes) == 1


def test_22_resume_after_crash(tmp_path: Path) -> None:
    """Item 22: SQLite run checkpoints allow resuming only pending frames after crash."""
    with FactoryDraftStore(tmp_path / "drafts.sqlite") as store:
        run = RunState(
            run_id="run_crash_1",
            job_id=1623,
            task_id=472,
            adapter_name="skeleton",
            status=RunStatus.RUNNING,
            current_stage="inference",
            total_frames=5,
        )
        store.create_run(run)
        run.completed_frames.extend([90, 91])
        store.checkpoint_run(run, frame_id=91, stage="inference")

        pending = store.list_pending_frames("run_crash_1", [90, 91, 92, 93, 94])
        assert pending == [92, 93, 94]
        latest = store.get_latest_run(1623)
        assert latest is not None
        assert latest.run_id == "run_crash_1"


def test_23_repeated_run_idempotence(tmp_path: Path) -> None:
    """Item 23: Re-running sync with same idempotency_key is a safe no-op without duplicate creates."""
    client = _FakeSyncClient()
    ctrl = SyncController(backup_dir=tmp_path / "backups", run_dir=tmp_path / "runs")
    op = SyncOperation(
        op_id="op_23",
        kind=OperationKind.CREATE,
        frame=0,
        label="car",
        shape_type="rectangle",
        payload={"type": "rectangle", "frame": 0, "label_id": 1, "points": [10.0, 10.0, 50.0, 50.0]},
        quality_status=QualityGateStatus.PASS,
    )
    res1 = ctrl.sync(client, 285, [op], dry_run=False, idempotency_key="job285_frame0_v1")
    res2 = ctrl.sync(client, 285, [op], dry_run=False, idempotency_key="job285_frame0_v1")

    assert res1.verified is True
    assert res2.status == "no_op"
    assert client.create_calls == 1
    assert len(client._shapes) == 1
