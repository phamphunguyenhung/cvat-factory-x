"""Single Safe Sync Controller for CVAT Factory (Phase 5)."""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from cvat_tools.core import paths
from cvat_tools.core.client import WriteNotAuthorized

from .fingerprint import (
    compute_annotations_fingerprint,
    compute_minimal_patch,
    compute_shape_fingerprint,
    geometry_matches,
    vertex_deviation,
)
from .models import (
    OperationKind,
    PermissionDeniedError,
    QualityGateStatus,
    ReadBackMismatchError,
    RetryPolicy,
    RollbackResult,
    ShapeDiff,
    SkippedItem,
    StaleSnapshotError,
    SyncOperation,
    SyncPlan,
    SyncResult,
)
from .retry import retry_call


class CvatSyncClient(Protocol):
    """Minimal protocol required from a CVAT client (supports real Cvat or test fakes)."""
    write_enabled: bool

    def labels(self, job_id: int) -> list[dict[str, Any]]: ...
    def annotations(self, job_id: int) -> dict[str, Any]: ...
    def create_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> dict[str, Any]: ...
    def update_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> dict[str, Any]: ...
    def delete_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> Any: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_manual_source(shape: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(shape)
    out["source"] = "manual"
    if isinstance(out.get("elements"), list):
        for el in out["elements"]:
            el["source"] = "manual"
    return out


class SyncController:
    """Single safe sync controller enforcing dry-run default, backups, staleness guard,
    quality gates, human-edit preservation, read-back verification, and conflict-aware rollback.
    """

    def __init__(
        self,
        backup_dir: Path | None = None,
        run_dir: Path | None = None,
        retry_policy: RetryPolicy | None = None,
        dedupe_px: float = 1.0,
        batch_size: int = 50,
        permission_checker: Callable[[CvatSyncClient, int], bool] | None = None,
    ) -> None:
        self.backup_dir = Path(backup_dir) if backup_dir else (paths.DATA / "backups")
        self.run_dir = Path(run_dir) if run_dir else (paths.DATA / "runs")
        self.retry_policy = retry_policy or RetryPolicy()
        self.dedupe_px = dedupe_px
        self.batch_size = max(1, batch_size)
        self.permission_checker = permission_checker

    def _check_write_permission(self, client: CvatSyncClient, job_id: int) -> None:
        if not getattr(client, "write_enabled", False):
            raise WriteNotAuthorized("Sync write requires client.write_enabled=True and dry_run=False.")
        if self.permission_checker is not None and not self.permission_checker(client, job_id):
            raise PermissionDeniedError(f"User/token lacks annotation write permission for job {job_id}.")

    def _fetch_live(self, client: CvatSyncClient, job_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ann = retry_call(lambda: client.annotations(job_id), self.retry_policy)
        labels = retry_call(lambda: client.labels(job_id), self.retry_policy)
        return ann, labels

    def _save_backup(self, job_id: int, run_id: str, annotations: dict[str, Any]) -> str:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.backup_dir / f"factory_job{job_id}_{run_id}_{ts}.json"
        path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _run_log_path(self, job_id: int, run_id: str) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / f"sync_job{job_id}_{run_id}.json"

    def _append_mutation_event(self, event: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        jsonl = self.run_dir / "mutations.jsonl"
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_run_log(self, path: Path, record: dict[str, Any]) -> None:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_completed_idempotent_run(self, job_id: int, idempotency_key: str) -> dict[str, Any] | None:
        if not self.run_dir.exists():
            return None
        for p in sorted(self.run_dir.glob(f"sync_job{job_id}_*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("idempotency_key") == idempotency_key and data.get("status") == "completed":
                return data
        return None

    def plan(
        self,
        client: CvatSyncClient,
        job_id: int,
        operations: Iterable[SyncOperation],
        *,
        expected_fingerprint: str | None = None,
        expected_version: int | None = None,
        protected_shape_ids: set[int] | None = None,
        pass_region_ids: set[int] | None = None,
        run_id: str | None = None,
    ) -> SyncPlan:
        """Build a verified, read-only SyncPlan without mutating CVAT."""
        rid = run_id or uuid.uuid4().hex[:12]
        live_ann, live_labels = self._fetch_live(client, job_id)
        live_fp = compute_annotations_fingerprint(live_ann)
        live_ver = live_ann.get("version")

        if expected_fingerprint is not None and live_fp != expected_fingerprint:
            raise StaleSnapshotError(
                f"Live annotations fingerprint {live_fp[:12]} != expected {expected_fingerprint[:12]}."
            )
        if expected_version is not None and live_ver != expected_version:
            raise StaleSnapshotError(f"Live annotations version {live_ver} != expected {expected_version}.")

        label_map = {lbl["name"]: lbl["id"] for lbl in live_labels}
        live_shapes = live_ann.get("shapes") or []
        live_by_id = {s["id"]: s for s in live_shapes if s.get("id") is not None}
        protected = set(protected_shape_ids or set()) | set(pass_region_ids or set())

        plan = SyncPlan(
            run_id=rid,
            job_id=job_id,
            live_fingerprint=live_fp,
            live_version=live_ver,
            n_existing=len(live_shapes),
        )
        accepted_creates: list[dict[str, Any]] = []

        for op in operations:
            # 1. Quality gate check
            if op.quality_status in (QualityGateStatus.FAIL, QualityGateStatus.BLOCKED):
                plan.skipped.append(SkippedItem(
                    op_id=op.op_id, reason="quality_gate_blocked",
                    frame=op.frame, label=op.label, target_cvat_id=op.target_cvat_id,
                    details={"quality_status": op.quality_status.value},
                ))
                continue
            if op.quality_status == QualityGateStatus.REVIEW and not op.approved:
                plan.skipped.append(SkippedItem(
                    op_id=op.op_id, reason="review_unapproved",
                    frame=op.frame, label=op.label, target_cvat_id=op.target_cvat_id,
                ))
                continue

            # 2. Label ID verification
            if op.label not in label_map:
                plan.skipped.append(SkippedItem(
                    op_id=op.op_id, reason="missing_label",
                    frame=op.frame, label=op.label, target_cvat_id=op.target_cvat_id,
                ))
                continue
            label_id = label_map[op.label]

            if op.kind == OperationKind.CREATE:
                payload = _ensure_manual_source({
                    "type": op.shape_type,
                    "frame": op.frame,
                    "label_id": label_id,
                    "points": op.payload.get("points", []),
                    "z_order": op.payload.get("z_order", 0),
                    "occluded": op.payload.get("occluded", False),
                    "outside": op.payload.get("outside", False),
                    "rotation": op.payload.get("rotation", 0.0),
                    "group": op.payload.get("group", 0),
                    "attributes": op.payload.get("attributes", []),
                    **({"elements": op.payload["elements"]} if "elements" in op.payload else {}),
                })
                # Dedupe against live CVAT shapes
                dup_live = next(
                    (
                        s for s in live_shapes
                        if s.get("frame") == op.frame
                        and s.get("label_id") == label_id
                        and s.get("type") == op.shape_type
                        and len(s.get("points") or []) == len(payload["points"])
                        and vertex_deviation(s.get("points"), payload["points"]) <= self.dedupe_px
                    ),
                    None,
                )
                if dup_live is not None:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="duplicate_in_cvat",
                        frame=op.frame, label=op.label, target_cvat_id=dup_live["id"],
                    ))
                    continue
                # Dedupe within batch
                dup_batch = any(
                    x["frame"] == op.frame
                    and x["label_id"] == label_id
                    and x["type"] == op.shape_type
                    and len(x["points"]) == len(payload["points"])
                    and vertex_deviation(x["points"], payload["points"]) <= self.dedupe_px
                    for x in accepted_creates
                )
                if dup_batch:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="duplicate_in_batch",
                        frame=op.frame, label=op.label,
                    ))
                    continue

                accepted_creates.append(payload)
                plan.creates.append(SyncOperation(
                    op_id=op.op_id, kind=OperationKind.CREATE, frame=op.frame,
                    label=op.label, shape_type=op.shape_type, payload=payload,
                    local_row_id=op.local_row_id, quality_status=op.quality_status,
                    approved=op.approved, preserve_points=op.preserve_points,
                    metadata=dict(op.metadata),
                ))
                plan.diffs.append(ShapeDiff(
                    op_id=op.op_id, kind=OperationKind.CREATE, frame=op.frame,
                    label=op.label, changed_fields={"create": (None, payload)},
                ))

            elif op.kind == OperationKind.UPDATE:
                cid = op.target_cvat_id
                if cid is None or cid not in live_by_id:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="missing_shape_id",
                        frame=op.frame, label=op.label, target_cvat_id=cid,
                    ))
                    continue
                if cid in protected:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="human_edit_or_pass_preserved",
                        frame=op.frame, label=op.label, target_cvat_id=cid,
                    ))
                    continue
                live_shape = live_by_id[cid]
                if op.before_state is not None:
                    if compute_shape_fingerprint(op.before_state) != compute_shape_fingerprint(live_shape):
                        plan.skipped.append(SkippedItem(
                            op_id=op.op_id, reason="stale_shape",
                            frame=op.frame, label=op.label, target_cvat_id=cid,
                        ))
                        continue
                desired = dict(op.payload)
                desired.setdefault("label_id", label_id)
                patch, changed_fields, element_diffs = compute_minimal_patch(
                    live_shape, desired, preserve_points=op.preserve_points,
                )
                if not changed_fields and not element_diffs:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="no_op_unchanged",
                        frame=op.frame, label=op.label, target_cvat_id=cid,
                    ))
                    continue
                pts_changed = "points" in changed_fields
                vdev = vertex_deviation(live_shape.get("points"), patch.get("points")) if pts_changed else 0.0
                plan.updates.append(SyncOperation(
                    op_id=op.op_id, kind=OperationKind.UPDATE, frame=live_shape["frame"],
                    label=op.label, shape_type=live_shape["type"], payload=patch,
                    target_cvat_id=cid, before_state=copy.deepcopy(live_shape),
                    local_row_id=op.local_row_id, quality_status=op.quality_status,
                    approved=op.approved, preserve_points=op.preserve_points,
                    metadata=dict(op.metadata),
                ))
                plan.diffs.append(ShapeDiff(
                    op_id=op.op_id, kind=OperationKind.UPDATE, frame=live_shape["frame"],
                    label=op.label, target_cvat_id=cid, changed_fields=changed_fields,
                    points_changed=pts_changed, vertex_deviation_px=vdev, element_diffs=element_diffs,
                ))

            elif op.kind == OperationKind.DELETE:
                cid = op.target_cvat_id
                if cid is None or cid not in live_by_id:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="missing_shape_id",
                        frame=op.frame, label=op.label, target_cvat_id=cid,
                    ))
                    continue
                if cid in protected:
                    plan.skipped.append(SkippedItem(
                        op_id=op.op_id, reason="human_edit_or_pass_preserved",
                        frame=op.frame, label=op.label, target_cvat_id=cid,
                    ))
                    continue
                live_shape = live_by_id[cid]
                if op.before_state is not None:
                    if compute_shape_fingerprint(op.before_state) != compute_shape_fingerprint(live_shape):
                        plan.skipped.append(SkippedItem(
                            op_id=op.op_id, reason="stale_shape",
                            frame=op.frame, label=op.label, target_cvat_id=cid,
                        ))
                        continue
                plan.deletes.append(SyncOperation(
                    op_id=op.op_id, kind=OperationKind.DELETE, frame=live_shape["frame"],
                    label=op.label, shape_type=live_shape["type"], payload={"id": cid},
                    target_cvat_id=cid, before_state=copy.deepcopy(live_shape),
                    local_row_id=op.local_row_id, quality_status=op.quality_status,
                    approved=op.approved, metadata=dict(op.metadata),
                ))
                plan.diffs.append(ShapeDiff(
                    op_id=op.op_id, kind=OperationKind.DELETE, frame=live_shape["frame"],
                    label=op.label, target_cvat_id=cid, changed_fields={"delete": (live_shape, None)},
                ))

        return plan

    def _verify_readback(
        self,
        before_shapes: list[dict[str, Any]],
        after_shapes: list[dict[str, Any]],
        executed_creates: list[SyncOperation],
        executed_updates: list[SyncOperation],
        executed_deletes: list[SyncOperation],
    ) -> tuple[bool, list[str], list[int]]:
        errors: list[str] = []
        before_ids = {s["id"] for s in before_shapes if s.get("id") is not None}
        after_by_id = {s["id"]: s for s in after_shapes if s.get("id") is not None}

        expected_count = len(before_shapes) + len(executed_creates) - len(executed_deletes)
        if len(after_shapes) != expected_count:
            errors.append(f"Shape count mismatch: expected {expected_count}, got {len(after_shapes)}.")

        # Match created shapes
        new_shapes = [s for s in after_shapes if s.get("id") not in before_ids]
        free = list(new_shapes)
        created_ids: list[int] = []
        for op in executed_creates:
            pl = op.payload
            candidates = [
                s for s in free
                if s.get("frame") == pl["frame"]
                and s.get("label_id") == pl["label_id"]
                and s.get("type") == pl["type"]
            ]
            if not candidates:
                errors.append(f"Created shape missing in read-back for op {op.op_id} (frame={pl['frame']}).")
                continue
            best = min(candidates, key=lambda s: vertex_deviation(s.get("points"), pl.get("points")))
            free.remove(best)
            if not geometry_matches(best, pl, tol_px=0.1):
                errors.append(f"Geometry mismatch in read-back for created shape id={best.get('id')} (op {op.op_id}).")
            else:
                created_ids.append(best["id"])

        # Verify updates
        for op in executed_updates:
            cid = op.target_cvat_id
            if cid not in after_by_id:
                errors.append(f"Updated shape id={cid} missing in read-back.")
                continue
            live = after_by_id[cid]
            if not geometry_matches(live, op.payload, tol_px=0.1):
                errors.append(f"Updated shape id={cid} geometry mismatch in read-back.")

        # Verify deletes
        for op in executed_deletes:
            cid = op.target_cvat_id
            if cid in after_by_id:
                errors.append(f"Deleted shape id={cid} still present in read-back.")

        return len(errors) == 0, errors, created_ids

    def sync(
        self,
        client: CvatSyncClient,
        job_id: int,
        operations: Iterable[SyncOperation],
        *,
        dry_run: bool = True,
        expected_fingerprint: str | None = None,
        expected_version: int | None = None,
        protected_shape_ids: set[int] | None = None,
        pass_region_ids: set[int] | None = None,
        idempotency_key: str | None = None,
        run_id: str | None = None,
        raise_on_readback_mismatch: bool = False,
    ) -> SyncResult:
        """Execute (or dry-run) a safe sync for a CVAT job."""
        ops_list = list(operations)
        if idempotency_key:
            prev = self._find_completed_idempotent_run(job_id, idempotency_key)
            if prev is not None:
                return SyncResult(
                    run_id=prev["run_id"],
                    job_id=job_id,
                    dry_run=dry_run,
                    status="no_op",
                    before_fingerprint=prev["after_fingerprint"],
                    after_fingerprint=prev["after_fingerprint"],
                    n_before=prev["n_after"],
                    n_after=prev["n_after"],
                    verified=True,
                    backup_path=prev.get("backup_path"),
                    mutation_log_path=str(self._run_log_path(job_id, prev["run_id"])),
                    idempotency_key=idempotency_key,
                )

        plan = self.plan(
            client,
            job_id,
            ops_list,
            expected_fingerprint=expected_fingerprint,
            expected_version=expected_version,
            protected_shape_ids=protected_shape_ids,
            pass_region_ids=pass_region_ids,
            run_id=run_id,
        )

        if dry_run:
            return SyncResult(
                run_id=plan.run_id,
                job_id=job_id,
                dry_run=True,
                status="dry_run",
                before_fingerprint=plan.live_fingerprint,
                after_fingerprint=plan.live_fingerprint,
                n_before=plan.n_existing,
                n_after=plan.n_existing,
                skipped=plan.skipped,
                verified=True,
                idempotency_key=idempotency_key,
                plan=plan,
            )

        self._check_write_permission(client, job_id)

        if plan.total_mutations == 0:
            return SyncResult(
                run_id=plan.run_id,
                job_id=job_id,
                dry_run=False,
                status="no_op",
                before_fingerprint=plan.live_fingerprint,
                after_fingerprint=plan.live_fingerprint,
                n_before=plan.n_existing,
                n_after=plan.n_existing,
                skipped=plan.skipped,
                verified=True,
                idempotency_key=idempotency_key,
                plan=plan,
            )

        # Backup live state before any write
        before_ann, _ = self._fetch_live(client, job_id)
        before_fp = compute_annotations_fingerprint(before_ann)
        if before_fp != plan.live_fingerprint:
            raise StaleSnapshotError("Live CVAT state changed between planning and write.")
        backup_path = self._save_backup(job_id, plan.run_id, before_ann)

        log_path = self._run_log_path(job_id, plan.run_id)
        log_record: dict[str, Any] = {
            "run_id": plan.run_id,
            "idempotency_key": idempotency_key,
            "job_id": job_id,
            "timestamp": _now_iso(),
            "status": "in_progress",
            "before_fingerprint": before_fp,
            "after_fingerprint": before_fp,
            "n_before": len(before_ann.get("shapes") or []),
            "n_after": len(before_ann.get("shapes") or []),
            "backup_path": backup_path,
            "completed_op_ids": [],
            "created_ids": [],
            "created_records": [],
            "updated_ids": [],
            "updated_records": [],
            "deleted_ids": [],
            "deleted_records": [],
            "verified": False,
            "verification_errors": [],
        }
        self._write_run_log(log_path, log_record)

        return self._execute_plan(
            client=client,
            job_id=job_id,
            plan=plan,
            before_ann=before_ann,
            log_path=log_path,
            log_record=log_record,
            resumed=False,
            raise_on_readback_mismatch=raise_on_readback_mismatch,
        )

    def _execute_plan(
        self,
        *,
        client: CvatSyncClient,
        job_id: int,
        plan: SyncPlan,
        before_ann: dict[str, Any],
        log_path: Path,
        log_record: dict[str, Any],
        resumed: bool,
        raise_on_readback_mismatch: bool,
    ) -> SyncResult:
        completed_ids = set(log_record.get("completed_op_ids") or [])
        executed_creates: list[SyncOperation] = []
        executed_updates: list[SyncOperation] = []
        executed_deletes: list[SyncOperation] = []

        try:
            # Execute creates in batches
            pending_creates = [op for op in plan.creates if op.op_id not in completed_ids]
            for i in range(0, len(pending_creates), self.batch_size):
                batch = pending_creates[i : i + self.batch_size]
                retry_call(
                    lambda b=batch: client.create_shapes(job_id, [op.payload for op in b]),
                    self.retry_policy,
                )
                executed_creates.extend(batch)
                for op in batch:
                    log_record["completed_op_ids"].append(op.op_id)
                self._write_run_log(log_path, log_record)

            # Execute updates in batches (minimal PATCH)
            pending_updates = [op for op in plan.updates if op.op_id not in completed_ids]
            for i in range(0, len(pending_updates), self.batch_size):
                batch = pending_updates[i : i + self.batch_size]
                retry_call(
                    lambda b=batch: client.update_shapes(job_id, [op.payload for op in b]),
                    self.retry_policy,
                )
                executed_updates.extend(batch)
                for op in batch:
                    log_record["completed_op_ids"].append(op.op_id)
                    log_record["updated_ids"].append(op.target_cvat_id)
                    log_record["updated_records"].append({
                        "id": op.target_cvat_id,
                        "before_state": op.before_state,
                        "patch": op.payload,
                    })
                self._write_run_log(log_path, log_record)

            # Execute deletes in batches
            pending_deletes = [op for op in plan.deletes if op.op_id not in completed_ids]
            for i in range(0, len(pending_deletes), self.batch_size):
                batch = pending_deletes[i : i + self.batch_size]
                retry_call(
                    lambda b=batch: client.delete_shapes(job_id, [op.payload for op in b]),
                    self.retry_policy,
                )
                executed_deletes.extend(batch)
                for op in batch:
                    log_record["completed_op_ids"].append(op.op_id)
                    log_record["deleted_ids"].append(op.target_cvat_id)
                    log_record["deleted_records"].append({
                        "id": op.target_cvat_id,
                        "before_state": op.before_state,
                    })
                self._write_run_log(log_path, log_record)

        except Exception as exc:
            log_record["status"] = "partial" if log_record["completed_op_ids"] else "failed"
            log_record["error"] = str(exc)
            self._write_run_log(log_path, log_record)
            self._append_mutation_event({
                "timestamp": _now_iso(),
                "run_id": plan.run_id,
                "job_id": job_id,
                "status": log_record["status"],
                "error": str(exc),
            })
            raise

        # Read-back verification
        after_ann, _ = self._fetch_live(client, job_id)
        after_shapes = after_ann.get("shapes") or []
        after_fp = compute_annotations_fingerprint(after_ann)
        after_by_id = {s["id"]: s for s in after_shapes if s.get("id") is not None}

        verified, errors, new_created_ids = self._verify_readback(
            before_ann.get("shapes") or [],
            after_shapes,
            executed_creates,
            executed_updates,
            executed_deletes,
        )

        # Record post-sync fingerprints for conflict-aware rollback
        for cid in new_created_ids:
            if cid not in log_record["created_ids"]:
                log_record["created_ids"].append(cid)
            if cid in after_by_id:
                log_record["created_records"].append({
                    "id": cid,
                    "after_fingerprint": compute_shape_fingerprint(after_by_id[cid]),
                })
        for rec in log_record["updated_records"]:
            cid = rec["id"]
            if cid in after_by_id and "after_fingerprint" not in rec:
                rec["after_fingerprint"] = compute_shape_fingerprint(after_by_id[cid])

        log_record["after_fingerprint"] = after_fp
        log_record["n_after"] = len(after_shapes)
        log_record["verified"] = verified
        log_record["verification_errors"] = errors
        log_record["status"] = "completed" if verified else "failed"
        self._write_run_log(log_path, log_record)
        self._append_mutation_event({
            "timestamp": _now_iso(),
            "run_id": plan.run_id,
            "job_id": job_id,
            "status": log_record["status"],
            "created": len(log_record["created_ids"]),
            "updated": len(log_record["updated_ids"]),
            "deleted": len(log_record["deleted_ids"]),
            "verified": verified,
        })

        if not verified and raise_on_readback_mismatch:
            raise ReadBackMismatchError("; ".join(errors))

        return SyncResult(
            run_id=plan.run_id,
            job_id=job_id,
            dry_run=False,
            status=log_record["status"],
            before_fingerprint=log_record["before_fingerprint"],
            after_fingerprint=after_fp,
            n_before=log_record["n_before"],
            n_after=len(after_shapes),
            created_ids=list(log_record["created_ids"]),
            updated_ids=list(log_record["updated_ids"]),
            deleted_ids=list(log_record["deleted_ids"]),
            skipped=plan.skipped,
            verified=verified,
            verification_errors=errors,
            backup_path=log_record["backup_path"],
            mutation_log_path=str(log_path),
            idempotency_key=log_record.get("idempotency_key"),
            resumed_from_checkpoint=resumed,
            plan=plan,
        )

    def resume(
        self,
        client: CvatSyncClient,
        run_log_path: Path | str,
        operations: Iterable[SyncOperation],
        *,
        dry_run: bool = False,
        raise_on_readback_mismatch: bool = False,
    ) -> SyncResult:
        """Resume an interrupted/partial sync run from its mutation log checkpoint."""
        lp = Path(run_log_path)
        log_record = json.loads(lp.read_text(encoding="utf-8"))
        job_id = int(log_record["job_id"])
        run_id = log_record["run_id"]
        completed_ids = set(log_record.get("completed_op_ids") or [])

        if log_record.get("status") == "completed":
            return SyncResult(
                run_id=run_id,
                job_id=job_id,
                dry_run=dry_run,
                status="no_op",
                before_fingerprint=log_record["after_fingerprint"],
                after_fingerprint=log_record["after_fingerprint"],
                n_before=log_record["n_after"],
                n_after=log_record["n_after"],
                created_ids=list(log_record.get("created_ids") or []),
                updated_ids=list(log_record.get("updated_ids") or []),
                deleted_ids=list(log_record.get("deleted_ids") or []),
                verified=bool(log_record.get("verified", True)),
                backup_path=log_record.get("backup_path"),
                mutation_log_path=str(lp),
                idempotency_key=log_record.get("idempotency_key"),
                resumed_from_checkpoint=True,
            )

        remaining_ops = [op for op in operations if op.op_id not in completed_ids]
        plan = self.plan(client, job_id, remaining_ops, run_id=run_id)
        if dry_run:
            return SyncResult(
                run_id=run_id,
                job_id=job_id,
                dry_run=True,
                status="dry_run",
                before_fingerprint=plan.live_fingerprint,
                after_fingerprint=plan.live_fingerprint,
                n_before=plan.n_existing,
                n_after=plan.n_existing,
                skipped=plan.skipped,
                verified=True,
                backup_path=log_record.get("backup_path"),
                mutation_log_path=str(lp),
                resumed_from_checkpoint=True,
                plan=plan,
            )

        self._check_write_permission(client, job_id)
        live_ann, _ = self._fetch_live(client, job_id)
        log_record["status"] = "in_progress"
        self._write_run_log(lp, log_record)
        return self._execute_plan(
            client=client,
            job_id=job_id,
            plan=plan,
            before_ann=live_ann,
            log_path=lp,
            log_record=log_record,
            resumed=True,
            raise_on_readback_mismatch=raise_on_readback_mismatch,
        )

    def rollback(
        self,
        client: CvatSyncClient,
        run_log_path: Path | str,
        *,
        dry_run: bool = True,
        force: bool = False,
        protected_ids: set[int] | None = None,
    ) -> RollbackResult:
        """Conflict-aware rollback: never blindly overwrites or deletes concurrently edited shapes."""
        lp = Path(run_log_path)
        log_record = json.loads(lp.read_text(encoding="utf-8"))
        job_id = int(log_record["job_id"])
        run_id = log_record["run_id"]
        protected = set(protected_ids or set())

        live_ann, _ = self._fetch_live(client, job_id)
        live_by_id = {s["id"]: s for s in (live_ann.get("shapes") or []) if s.get("id") is not None}

        to_delete: list[dict[str, Any]] = []
        to_restore: list[dict[str, Any]] = []
        kept_conflict: list[dict[str, Any]] = []
        kept_protected: list[int] = []

        # Check created shapes
        created_records = log_record.get("created_records") or [
            {"id": cid} for cid in (log_record.get("created_ids") or [])
        ]
        for rec in created_records:
            cid = rec["id"]
            if cid not in live_by_id:
                continue
            if cid in protected:
                kept_protected.append(cid)
                continue
            live_shape = live_by_id[cid]
            expected_fp = rec.get("after_fingerprint")
            if not force and expected_fp and compute_shape_fingerprint(live_shape) != expected_fp:
                kept_conflict.append({
                    "id": cid,
                    "kind": "create_rollback",
                    "reason": "concurrent_edit_detected",
                })
                continue
            to_delete.append({"id": cid})

        # Check updated shapes
        for rec in log_record.get("updated_records") or []:
            cid = rec["id"]
            if cid not in live_by_id:
                continue
            if cid in protected:
                kept_protected.append(cid)
                continue
            live_shape = live_by_id[cid]
            expected_fp = rec.get("after_fingerprint")
            if not force and expected_fp and compute_shape_fingerprint(live_shape) != expected_fp:
                kept_conflict.append({
                    "id": cid,
                    "kind": "update_rollback",
                    "reason": "concurrent_edit_detected",
                })
                continue
            if rec.get("before_state"):
                to_restore.append(_ensure_manual_source(rec["before_state"]))

        result = RollbackResult(
            run_id=run_id,
            job_id=job_id,
            dry_run=dry_run,
            deleted_ids=[d["id"] for d in to_delete],
            restored_ids=[r["id"] for r in to_restore],
            kept_due_to_conflict=kept_conflict,
            kept_protected=kept_protected,
            verified=dry_run,
        )
        if dry_run:
            return result

        self._check_write_permission(client, job_id)
        if to_delete:
            retry_call(lambda: client.delete_shapes(job_id, to_delete), self.retry_policy)
        if to_restore:
            retry_call(lambda: client.update_shapes(job_id, to_restore), self.retry_policy)

        after_ann, _ = self._fetch_live(client, job_id)
        after_ids = {s["id"] for s in (after_ann.get("shapes") or []) if s.get("id") is not None}
        still_present = [d["id"] for d in to_delete if d["id"] in after_ids]
        if still_present:
            result.errors.append(f"Shapes still present after rollback delete: {still_present}")
            result.verified = False
        else:
            result.verified = True

        log_record["rollback"] = {
            "timestamp": _now_iso(),
            "deleted_ids": result.deleted_ids,
            "restored_ids": result.restored_ids,
            "kept_due_to_conflict": kept_conflict,
            "kept_protected": kept_protected,
            "verified": result.verified,
        }
        if result.verified and not kept_conflict:
            log_record["status"] = "rolled_back"
        self._write_run_log(lp, log_record)
        return result


def sync_job(
    client: CvatSyncClient,
    job_id: int,
    operations: Iterable[SyncOperation] = (),
    *,
    yes: bool = False,
    dry_run: bool | None = None,
    **kwargs: Any,
) -> SyncResult:
    """Convenience entrypoint for `cvat-factory sync --job <id>` (dry-run default unless yes=True)."""
    effective_dry_run = (not yes) if dry_run is None else dry_run
    ctrl = SyncController()
    return ctrl.sync(client, job_id, operations, dry_run=effective_dry_run, **kwargs)


def rollback_sync(
    client: CvatSyncClient,
    run_log_path: Path | str,
    *,
    yes: bool = False,
    dry_run: bool | None = None,
    force: bool = False,
    protected_ids: set[int] | None = None,
) -> RollbackResult:
    """Convenience entrypoint for `cvat-factory sync rollback` (dry-run default unless yes=True)."""
    effective_dry_run = (not yes) if dry_run is None else dry_run
    ctrl = SyncController()
    return ctrl.rollback(client, run_log_path, dry_run=effective_dry_run, force=force, protected_ids=protected_ids)
