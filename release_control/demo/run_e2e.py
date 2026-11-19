"""Reproducible 5-step Dataset Time Machine End-to-End Demo for CVAT Factory X."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from release_control.adapters import (
    CvatConnector,
    DatumaroRoundTripAdapter,
    MlflowTracker,
)
from release_control.config import ControlTowerConfig
from release_control.contracts.engine import ContractEngine
from release_control.db import ControlTowerDB
from release_control.fault_lab import FaultBenchmarker, FaultInjector
from release_control.fault_lab.inject import normalize_fault_type
from release_control.lineage import ImpactAnalyzer, LineageTracker
from release_control.models import (
    ReleaseStatus,
    Severity,
    ValidationIssue as ModelValidationIssue,
    ValidationReport as ModelValidationReport,
)
from release_control.release import EvidenceBuilder, ReleaseGate
from release_control.validators.gate import (
    ValidationEngine,
    ValidationReport as GateValidationReport,
)
from release_control.versions import RestoreManager, SnapshotManager


def _normalize_exported_manifest(
    dataset_dir: Path,
    dataset_id: str,
    guideline_version: str,
    coco_path: Path,
) -> dict[str, Any]:
    """Ensure exported manifest.json matches ValidationEngine & FaultInjector schema."""
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coco_data = json.loads(coco_path.read_text(encoding="utf-8"))

    images = manifest.get("images") or manifest.get("items") or []
    classes: dict[str, str] = {
        str(cat["id"]): str(cat["name"]) for cat in coco_data.get("categories", [])
    }

    manifest["dataset_id"] = dataset_id
    manifest["guideline_version"] = guideline_version
    manifest["images"] = images
    manifest["classes"] = classes

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _to_model_validation_report(
    gate_report: GateValidationReport,
    release_id: str,
    snapshot_id: str,
) -> ModelValidationReport:
    """Convert validators.gate.ValidationReport dataclass into models.ValidationReport."""
    converted_issues: list[ModelValidationIssue] = []
    for raw in gate_report.issues:
        raw_ft = raw.get("fault_type")
        parsed_ft = None
        if isinstance(raw_ft, str) and raw_ft not in {"SCHEMA", "PRIVACY", ""}:
            try:
                parsed_ft = normalize_fault_type(raw_ft)
            except ValueError:
                parsed_ft = None

        converted_issues.append(
            ModelValidationIssue(
                issue_id=str(uuid.uuid4()),
                code=str(raw.get("code", "VALIDATION_ISSUE")),
                fault_type=parsed_ft,
                severity=Severity(str(raw.get("severity", "CRITICAL"))),
                message=str(raw.get("message", "")),
                image_id=(
                    str(raw["image_id"]) if raw.get("image_id") is not None else None
                ),
                file_path=raw.get("file_path"),
                details=raw.get("details") or {},
            )
        )

    return ModelValidationReport(
        report_id=f"val-{uuid.uuid4().hex[:12]}",
        release_id=release_id,
        snapshot_id=snapshot_id,
        passed=gate_report.passed,
        critical_count=gate_report.severity_counts.get("CRITICAL", 0),
        warning_count=gate_report.severity_counts.get("WARNING", 0),
        info_count=gate_report.severity_counts.get("INFO", 0),
        issues=converted_issues,
        metrics={
            "counts_by_fault_type": gate_report.counts_by_fault_type,
            "recommended_status": gate_report.recommended_status,
            "requires_human_review": gate_report.requires_human_review,
        },
    )


def _reset_demo_workspace(config: ControlTowerConfig) -> None:
    """Clean demo-specific working directories and SQLite DB for reproducible runs."""
    for folder_name in (
        "demo_v1_clean",
        "demo_v1_corrupt",
        "demo_v2_restored",
        "demo_cvat_job_1975",
    ):
        target = config.working_dir / folder_name
        if target.exists():
            shutil.rmtree(target)

    if config.db_path.exists():
        config.db_path.unlink()
    openlineage_file = config.lineage_dir / "openlineage.jsonl"
    if openlineage_file.exists():
        openlineage_file.unlink()


def run_e2e_demo(
    project_root: Path | None = None,
    subset_limit: int = 30,
    seed: int = 42,
    reset_state: bool = True,
) -> dict[str, Any]:
    """Run the 5-step Dataset Time Machine end-to-end demo and return a structured summary."""
    config = ControlTowerConfig.auto_discover(project_root=project_root)
    config.ensure_dirs()

    if reset_state:
        _reset_demo_workspace(config)

    db = ControlTowerDB(config.db_path)
    contract_engine = ContractEngine(config.contracts_dir)
    lineage_tracker = LineageTracker(
        db=db,
        openlineage_path=config.lineage_dir / "openlineage.jsonl",
    )
    mlflow_tracker = MlflowTracker(tracking_uri=config.mlruns_dir)
    impact_analyzer = ImpactAnalyzer(db=db, mlflow_tracker=mlflow_tracker)
    snapshot_mgr = SnapshotManager(config=config)
    restore_mgr = RestoreManager(config=config)
    gate = ReleaseGate(
        db=db,
        contract_engine=contract_engine,
        lineage_tracker=lineage_tracker,
    )
    evidence_builder = EvidenceBuilder(db=db, contract_engine=contract_engine)

    dataset_id = "coco250_demo"
    guideline_version = "v2.0"
    contract = {"expected_guideline_version": guideline_version, "num_classes": 80}

    connector = CvatConnector(
        coco250_dir=config.day10_data_source / "coco250",
        coco250_calib_dir=config.day10_data_source / "coco250_calib",
        cvat_jobs_dir=config.cvat_data_dir / "jobs" / "edufun",
    )

    # Probe live CVAT (read-only) and local cached job 1975 (non-fatal if absent)
    cvat_probe = connector.probe_cvat_live()
    cvat_job_1975_probe: dict[str, Any]
    try:
        cvat_job_1975_probe = connector.export_cvat_job_local(
            1975,
            config.working_dir / "demo_cvat_job_1975",
            guideline_version=guideline_version,
        )
        cvat_job_1975_probe["available"] = True
    except FileNotFoundError as exc:
        cvat_job_1975_probe = {"available": False, "job_id": "1975", "error": str(exc)}

    # ------------------------------------------------------------------
    # STEP 1: Clean v1
    # ------------------------------------------------------------------
    v1_dir = config.working_dir / "demo_v1_clean"
    export_info = connector.export_coco250_subset(
        dest_dir=v1_dir,
        limit=subset_limit,
        seed=seed,
        include_calib=False,
        guideline_version=guideline_version,
    )
    _normalize_exported_manifest(
        dataset_dir=v1_dir,
        dataset_id=dataset_id,
        guideline_version=guideline_version,
        coco_path=Path(export_info["coco_path"]),
    )

    roundtrip_report = DatumaroRoundTripAdapter().verify_roundtrip(
        export_info["coco_path"]
    )
    if not roundtrip_report.passed:
        raise RuntimeError(f"Datumaro roundtrip failed: {roundtrip_report.details}")

    snap_v1 = snapshot_mgr.create_snapshot(
        v1_dir,
        dataset_id=dataset_id,
        version_tag="v1",
        db=db,
    )
    v1_digest = snap_v1.snapshot_id

    lineage_tracker.record_edge(
        source_type="dataset_source",
        source_id="coco250",
        target_type="snapshot",
        target_id=v1_digest,
        relation="exported_to",
        evidence_status="verified",
        metadata={"limit": subset_limit, "seed": seed},
    )

    rel_v1_id = "rel-coco250-v1"
    gate.create_release(
        release_id=rel_v1_id,
        dataset_id=dataset_id,
        version_tag="v1",
        snapshot_digest=v1_digest,
        contract_version="1.0.0",
    )

    gate_report_v1 = ValidationEngine.validate_dataset_report(
        v1_dir,
        contract=contract,
        reference_dir=v1_dir,
    )
    model_report_v1 = _to_model_validation_report(
        gate_report_v1, release_id=rel_v1_id, snapshot_id=v1_digest
    )
    rel_v1 = gate.submit_validation(
        rel_v1_id,
        model_report_v1,
        requires_human_review=gate_report_v1.requires_human_review,
    )
    if rel_v1.status != ReleaseStatus.VALIDATED:
        raise RuntimeError(
            f"Expected v1 to be VALIDATED, got {rel_v1.status.value}: {gate_report_v1.issues}"
        )

    gate.submit_approval(
        rel_v1_id,
        approval_type="qc",
        actor="qc_lead_demo",
        verdict="approved",
        note="COCO250 v1 QC verified",
    )
    gate.submit_approval(
        rel_v1_id,
        approval_type="privacy",
        actor="qc_lead_demo",
        verdict="approved",
        note="COCO250 v1 privacy cleared",
    )
    gate.approve_release(rel_v1_id)
    gate.publish_release(rel_v1_id)
    rel_v1 = gate.accept_release(
        rel_v1_id,
        consumer="yolo_trainer_demo",
        verified_sha256=v1_digest,
        note="Downstream trainer verified and accepted v1",
    )

    mlflow_run = mlflow_tracker.log_simulated_training_run(
        release_id=rel_v1_id,
        dataset_digest=snap_v1.manifest_sha256,
        model_name="yolov8n-coco250-demo",
        metrics={"mAP50": 0.612, "mAP50_95": 0.418, "precision": 0.74, "recall": 0.66},
        params={"epochs": 10, "imgsz": 640, "snapshot_digest": v1_digest},
        run_origin="simulated_demo",
    )
    lineage_tracker.record_edge(
        source_type="snapshot",
        source_id=v1_digest,
        target_type="mlflow_run",
        target_id=f"mlflow_run:{mlflow_run['run_id']}",
        relation="trained_on",
        evidence_status="verified",
        metadata={"run_origin": "simulated_demo", "release_id": rel_v1_id},
    )
    lineage_tracker.record_edge(
        source_type="mlflow_run",
        source_id=f"mlflow_run:{mlflow_run['run_id']}",
        target_type="model",
        target_id="model:yolov8n-coco250-demo",
        relation="produced_model",
        evidence_status="verified",
        metadata={"run_origin": "simulated_demo"},
    )

    # ------------------------------------------------------------------
    # STEP 2: Inject Fault (T6_CLASS_SWAP + T5_BBOX_DRIFT)
    # ------------------------------------------------------------------
    corrupt_dir = config.working_dir / "demo_v1_corrupt"
    fault_manifest = FaultInjector(seed=seed).inject(
        source_dir=v1_dir,
        dest_dir=corrupt_dir,
        fault_types=["T6_CLASS_SWAP", "T5_BBOX_DRIFT"],
        seed=seed,
    )
    snap_corrupt = snapshot_mgr.create_snapshot(
        corrupt_dir,
        dataset_id=dataset_id,
        version_tag="v1-corrupt",
        db=db,
        parent_digest=v1_digest,
    )
    lineage_tracker.record_edge(
        source_type="snapshot",
        source_id=v1_digest,
        target_type="snapshot",
        target_id=snap_corrupt.snapshot_id,
        relation="corrupted_copy",
        evidence_status="verified",
        metadata={"injected_faults": ["T6_CLASS_SWAP", "T5_BBOX_DRIFT"]},
    )

    rel_corrupt_id = "rel-coco250-v1-corrupt"
    gate.create_release(
        release_id=rel_corrupt_id,
        dataset_id=dataset_id,
        version_tag="v1-corrupt",
        snapshot_digest=snap_corrupt.snapshot_id,
        contract_version="1.0.0",
    )

    # ------------------------------------------------------------------
    # STEP 3: Gate HOLD
    # ------------------------------------------------------------------
    gate_report_corrupt = ValidationEngine.validate_dataset_report(
        corrupt_dir,
        contract=contract,
        reference_dir=v1_dir,
    )
    model_report_corrupt = _to_model_validation_report(
        gate_report_corrupt,
        release_id=rel_corrupt_id,
        snapshot_id=snap_corrupt.snapshot_id,
    )
    rel_corrupt = gate.submit_validation(
        rel_corrupt_id,
        model_report_corrupt,
        requires_human_review=gate_report_corrupt.requires_human_review,
    )
    if rel_corrupt.status != ReleaseStatus.HOLD:
        raise RuntimeError(
            f"Expected v1-corrupt to be HOLD, got {rel_corrupt.status.value}"
        )

    approval_blocked = False
    try:
        gate.approve_release(rel_corrupt_id)
    except ValueError:
        approval_blocked = True
    if not approval_blocked:
        raise RuntimeError("ReleaseGate allowed approval of a HOLD release!")

    # ------------------------------------------------------------------
    # STEP 4: Trace Blast Radius
    # ------------------------------------------------------------------
    blast_radius = impact_analyzer.get_blast_radius(v1_digest)

    # ------------------------------------------------------------------
    # STEP 5: Time Machine Restore & v2
    # ------------------------------------------------------------------
    v2_dir = config.working_dir / "demo_v2_restored"
    restore_result = restore_mgr.restore_snapshot(v1_digest, v2_dir, db=db)
    if restore_result["restored_digest"] != v1_digest:
        raise RuntimeError(
            f"Restore digest mismatch: {restore_result['restored_digest']} != {v1_digest}"
        )

    snap_v2 = snapshot_mgr.create_snapshot(
        v2_dir,
        dataset_id=dataset_id,
        version_tag="v2",
        db=db,
        parent_digest=v1_digest,
    )

    rel_v2_id = "rel-coco250-v2"
    gate.create_release(
        release_id=rel_v2_id,
        dataset_id=dataset_id,
        version_tag="v2",
        snapshot_digest=snap_v2.snapshot_id,
        contract_version="1.0.0",
    )
    gate_report_v2 = ValidationEngine.validate_dataset_report(
        v2_dir,
        contract=contract,
        reference_dir=v1_dir,
    )
    model_report_v2 = _to_model_validation_report(
        gate_report_v2, release_id=rel_v2_id, snapshot_id=snap_v2.snapshot_id
    )
    rel_v2 = gate.submit_validation(
        rel_v2_id,
        model_report_v2,
        requires_human_review=gate_report_v2.requires_human_review,
    )
    if rel_v2.status != ReleaseStatus.VALIDATED:
        raise RuntimeError(f"Expected v2 to be VALIDATED, got {rel_v2.status.value}")

    gate.submit_approval(
        rel_v2_id,
        approval_type="qc",
        actor="qc_lead_demo",
        verdict="approved",
        note="Time Machine restored v2 QC verified",
    )
    gate.submit_approval(
        rel_v2_id,
        approval_type="privacy",
        actor="qc_lead_demo",
        verdict="approved",
        note="Time Machine restored v2 privacy cleared",
    )
    gate.approve_release(rel_v2_id)
    gate.publish_release(rel_v2_id)
    rel_v2 = gate.accept_release(
        rel_v2_id,
        consumer="yolo_trainer_demo",
        verified_sha256=v1_digest,
        note="Downstream trainer verified restored v2 matches v1 digest",
    )

    evidence_path = config.reports_dir / "RELEASE_EVIDENCE.json"
    passport = evidence_builder.build_passport(
        rel_v2_id,
        output_path=evidence_path,
        validation_report=model_report_v2,
    )

    benchmark_path = config.reports_dir / "FAULT_BENCHMARK.json"
    benchmark_report = FaultBenchmarker().run(
        source_dir=v1_dir,
        contract=contract,
        output_path=benchmark_path,
    )

    return {
        "dataset_id": dataset_id,
        "probes": {
            "cvat_live": cvat_probe,
            "cvat_job_1975": cvat_job_1975_probe,
            "datumaro_roundtrip": roundtrip_report.to_dict(),
        },
        "step_1_v1": {
            "release_id": rel_v1.release_id,
            "status": rel_v1.status.value,
            "snapshot_digest": v1_digest,
            "image_count": snap_v1.image_count,
            "mlflow_run": mlflow_run,
        },
        "step_2_fault_injection": {
            "release_id": rel_corrupt_id,
            "injected_faults": fault_manifest.summary,
            "corrupt_digest": snap_corrupt.snapshot_id,
        },
        "step_3_gate_hold": {
            "release_id": rel_corrupt.release_id,
            "status": rel_corrupt.status.value,
            "counts_by_fault_type": gate_report_corrupt.counts_by_fault_type,
            "approval_blocked": approval_blocked,
        },
        "step_4_blast_radius": {
            "affected_snapshots": len(blast_radius["snapshots"]),
            "affected_releases": [r["release_id"] for r in blast_radius["releases"]],
            "simulated_demo_runs": len(blast_radius["training_runs"]["simulated_demo"]),
            "actual_mlflow_runs": len(blast_radius["training_runs"]["actual_mlflow"]),
            "models": [m["model_name"] for m in blast_radius["models"]],
            "edge_count": len(blast_radius["graph"]["edges"]),
        },
        "step_5_restore_v2": {
            "release_id": rel_v2.release_id,
            "status": rel_v2.status.value,
            "restored_digest": restore_result["restored_digest"],
            "digest_matches_v1": restore_result["restored_digest"] == v1_digest,
            "passport_sha256": passport.passport_sha256,
        },
        "reports": {
            "release_evidence": str(evidence_path),
            "fault_benchmark": str(benchmark_path),
            "overall_critical_fault_recall": benchmark_report.overall_critical_fault_recall,
            "overall_fault_recall": benchmark_report.overall_fault_recall,
            "false_hold_rate": benchmark_report.false_hold_rate,
            "false_release_rate": benchmark_report.false_release_rate,
        },
    }
