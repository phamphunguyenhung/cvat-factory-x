"""FastAPI Control Tower Dashboard & REST API for CVAT Factory X."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from release_control.adapters.cvat_adapter import CvatConnector
from release_control.adapters.mlflow_adapter import MlflowTracker
from release_control.config import ControlTowerConfig
from release_control.contracts.engine import ContractEngine
from release_control.db import ControlTowerDB
from release_control.fault_lab.benchmark import FaultBenchmarker
from release_control.fault_lab.inject import FaultInjector, normalize_fault_type
from release_control.lineage.events import LineageTracker
from release_control.lineage.impact import ImpactAnalyzer
from release_control.models import (
    FaultType,
    ReleaseStatus,
    Severity,
    SnapshotRecord,
    ValidationIssue as ModelValidationIssue,
    ValidationReport as ModelValidationReport,
)
from release_control.release.evidence import EvidenceBuilder
from release_control.release.gate import ReleaseGate
from release_control.validators.gate import ValidationEngine
from release_control.versions.restore import RestoreManager, SnapshotIntegrityError
from release_control.versions.snapshot import SnapshotManager

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class BootstrapRequest(BaseModel):
    dataset_id: str = "coco250_demo"
    limit: int = Field(default=30, ge=5, le=250)
    seed: int = 42
    guideline_version: str = "v2.0"


class FaultInjectRequest(BaseModel):
    source_digest: str
    fault_types: list[str] = Field(min_length=1)
    seed: int = 42
    dataset_id: str | None = None


class RestoreRequest(BaseModel):
    snapshot_digest: str
    dataset_id: str | None = None
    version_tag: str = "v2"
    compare_digest: str | None = None


class ApproveRequest(BaseModel):
    actor: str = Field(min_length=1)
    approval_type: str = Field(pattern="^(qc|privacy|QC|PRIVACY)$")
    verdict: str = "APPROVED"
    note: str = ""


class AcceptRequest(BaseModel):
    consumer: str = Field(min_length=1)
    note: str = ""
    consumer_version: str | None = "1.0.0"
    verified_sha256: str | None = None


class BenchmarkRequest(BaseModel):
    source_digest: str | None = None


def _to_model_fault_type(raw: str | None) -> FaultType | None:
    if not raw:
        return None
    try:
        return normalize_fault_type(raw)
    except ValueError:
        return None


def _to_model_report(
    report_dict: dict[str, Any],
    release_id: str,
    snapshot_id: str,
) -> ModelValidationReport:
    issues: list[ModelValidationIssue] = []
    for idx, item in enumerate(report_dict.get("issues", [])):
        sev_raw = str(item.get("severity", "INFO")).upper()
        try:
            sev = Severity(sev_raw)
        except ValueError:
            sev = Severity.INFO
        issues.append(
            ModelValidationIssue(
                issue_id=f"iss-{idx+1}-{uuid.uuid4().hex[:6]}",
                code=str(item.get("code", "VALIDATION_ISSUE")),
                fault_type=_to_model_fault_type(item.get("fault_type")),
                severity=sev,
                message=str(item.get("message", "")),
                image_id=item.get("image_id"),
                file_path=item.get("file_path"),
                details={
                    "raw_fault_type": item.get("fault_type"),
                    "requires_human_review": bool(item.get("requires_human_review", False)),
                    **(item.get("details") or {}),
                },
            )
        )

    sev_counts = report_dict.get("severity_counts", {})
    return ModelValidationReport(
        report_id=f"val-{uuid.uuid4().hex[:10]}",
        release_id=release_id,
        snapshot_id=snapshot_id,
        passed=bool(report_dict.get("passed", False)),
        critical_count=int(sev_counts.get("CRITICAL", 0)),
        warning_count=int(sev_counts.get("WARNING", 0)),
        info_count=int(sev_counts.get("INFO", 0)),
        issues=issues,
        metrics={
            "requires_human_review": bool(report_dict.get("requires_human_review", False)),
            "recommended_status": report_dict.get("recommended_status", "HOLD"),
            "counts_by_fault_type": report_dict.get("counts_by_fault_type", {}),
        },
    )


def _normalize_exported_manifest(
    dataset_dir: Path,
    dataset_id: str,
    guideline_version: str,
) -> None:
    """Ensure exported manifest.json conforms to ValidationEngine & FaultInjector schema."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "images" not in data and isinstance(data.get("items"), list):
        images = []
        for idx, item in enumerate(data["items"]):
            entry = dict(item)
            entry["image_id"] = str(entry.get("image_id", f"img_{idx}"))
            images.append(entry)
        data["images"] = images
    data.setdefault("dataset_id", dataset_id)
    data.setdefault("guideline_version", guideline_version)
    if "classes" not in data:
        data["classes"] = {str(i): f"class_{i}" for i in range(80)}
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_snapshot(db: ControlTowerDB, ref: str) -> SnapshotRecord:
    snapshots = db.list_snapshots()
    if not snapshots:
        raise HTTPException(status_code=404, detail="No snapshots exist yet. Run bootstrap first.")
    if ref == "latest":
        return snapshots[0]
    for snap in snapshots:
        if snap.snapshot_id == ref or snap.manifest_sha256 == ref or snap.snapshot_id.startswith(ref):
            return snap
    raise HTTPException(status_code=404, detail=f"Snapshot not found: {ref}")


def create_app(config: ControlTowerConfig | None = None) -> FastAPI:
    """Create and configure the CVAT Factory X Control Tower FastAPI application."""
    cfg = config or ControlTowerConfig.auto_discover()
    cfg.ensure_dirs()

    db = ControlTowerDB(cfg.db_path)
    contract_engine = ContractEngine(contracts_dir=cfg.contracts_dir)
    snapshot_mgr = SnapshotManager(config=cfg)
    restore_mgr = RestoreManager(config=cfg)
    lineage_tracker = LineageTracker(
        db=db,
        openlineage_path=cfg.lineage_dir / "openlineage.jsonl",
    )
    mlflow_tracker = MlflowTracker(tracking_uri=cfg.mlruns_dir / "mlflow.db")
    impact_analyzer = ImpactAnalyzer(db=db, mlflow_tracker=mlflow_tracker)
    gate = ReleaseGate(db=db, contract_engine=contract_engine, lineage_tracker=lineage_tracker)
    evidence_builder = EvidenceBuilder(db=db, contract_engine=contract_engine)

    app = FastAPI(title="CVAT Factory X — Data Release Control Tower")
    app.state.config = cfg
    app.state.db = db

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard() -> Any:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Dashboard static/index.html not found.")
        return FileResponse(index_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "cvat-factory-x-control-tower",
            "db_path": str(cfg.db_path),
            "artifacts_dir": str(cfg.artifacts_dir),
            "reports_dir": str(cfg.reports_dir),
        }

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        snapshots = [s.model_dump(mode="json") for s in db.list_snapshots()]
        releases = [r.model_dump(mode="json") for r in db.list_releases()]
        approvals_by_release = {
            r["release_id"]: [a.model_dump(mode="json") for a in db.list_approvals(r["release_id"])]
            for r in releases
        }
        acks_by_release = {
            r["release_id"]: [c.model_dump(mode="json") for c in db.list_consumer_acks(r["release_id"])]
            for r in releases
        }
        fault_runs = [f.model_dump(mode="json") for f in db.list_fault_runs()]
        edges = [e.model_dump(mode="json") for e in db.list_lineage_events()]

        node_ids: set[str] = set()
        for e in edges:
            node_ids.add(e["source_id"])
            node_ids.add(e["target_id"])

        return {
            "snapshots": snapshots,
            "releases": releases,
            "active_release": releases[0] if releases else None,
            "approvals": approvals_by_release,
            "consumer_acks": acks_by_release,
            "fault_runs": fault_runs,
            "lineage_summary": {
                "edge_count": len(edges),
                "node_count": len(node_ids),
                "edges": edges,
            },
            "fault_benchmark": _read_json_if_exists(cfg.reports_dir / "FAULT_BENCHMARK.json"),
            "release_evidence": _read_json_if_exists(cfg.reports_dir / "RELEASE_EVIDENCE.json"),
        }

    @app.post("/api/demo/bootstrap")
    def bootstrap_demo(req: BootstrapRequest | None = None) -> dict[str, Any]:
        body = req or BootstrapRequest()
        connector = CvatConnector(
            coco250_dir=cfg.day10_data_source / "coco250",
            coco250_calib_dir=cfg.day10_data_source / "coco250_calib",
        )
        v1_dir = cfg.working_dir / f"{body.dataset_id}_v1_{uuid.uuid4().hex[:6]}"

        try:
            export_info = connector.export_coco250_subset(
                dest_dir=v1_dir,
                limit=body.limit,
                seed=body.seed,
                guideline_version=body.guideline_version,
            )
            _normalize_exported_manifest(
                dataset_dir=v1_dir,
                dataset_id=body.dataset_id,
                guideline_version=body.guideline_version,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        snap = snapshot_mgr.create_snapshot(
            dataset_dir=v1_dir,
            dataset_id=body.dataset_id,
            version_tag="v1",
            db=db,
        )

        lineage_tracker.record_edge(
            source_type="dataset",
            source_id=body.dataset_id,
            target_type="snapshot",
            target_id=snap.snapshot_id,
            relation="snapshotted_as",
            evidence_status="verified",
            metadata={"version": "v1", "image_count": snap.image_count},
        )

        contract = {"expected_guideline_version": body.guideline_version, "num_classes": 80}
        val_report_raw = ValidationEngine.validate_dataset_report(
            snap.snapshot_path,
            contract=contract,
            reference_dir=snap.snapshot_path,
        ).to_dict()

        release_id = f"rel-{body.dataset_id}-v1-{uuid.uuid4().hex[:6]}"
        gate.create_release(
            release_id=release_id,
            dataset_id=body.dataset_id,
            version_tag="v1",
            snapshot_digest=snap.snapshot_id,
            snapshot_id=snap.snapshot_id,
            metadata={"source": "demo_bootstrap", "export_info": export_info},
        )
        model_report = _to_model_report(val_report_raw, release_id=release_id, snapshot_id=snap.snapshot_id)
        release = gate.submit_validation(
            release_id=release_id,
            validation_report=model_report,
            requires_human_review=bool(val_report_raw.get("requires_human_review", False)),
        )

        mlflow_run = mlflow_tracker.log_simulated_training_run(
            release_id=release_id,
            dataset_digest=snap.manifest_sha256,
            model_name="yolo11n-coco250",
            metrics={"mAP50": 0.642, "mAP50_95": 0.431},
            params={"epochs": 10, "imgsz": 640, "version": "v1"},
            run_origin="simulated_demo",
        )
        lineage_tracker.record_edge(
            source_type="release",
            source_id=release_id,
            target_type="training_run",
            target_id=f"mlflow_run:{mlflow_run['run_id']}",
            relation="trained_in",
            evidence_status="verified",
            metadata={"run_origin": "simulated_demo", "model_name": "yolo11n-coco250"},
        )
        lineage_tracker.record_edge(
            source_type="training_run",
            source_id=f"mlflow_run:{mlflow_run['run_id']}",
            target_type="model",
            target_id="model:yolo11n-coco250",
            relation="produces_model",
            evidence_status="verified",
            metadata={"run_origin": "simulated_demo"},
        )

        passport = evidence_builder.build_passport(
            release_id=release_id,
            output_path=cfg.reports_dir / "RELEASE_EVIDENCE.json",
            validation_report=model_report,
        )

        return {
            "dataset_id": body.dataset_id,
            "snapshot": snap.model_dump(mode="json"),
            "release": release.model_dump(mode="json"),
            "validation_report": val_report_raw,
            "mlflow_run": mlflow_run,
            "passport_sha256": passport.passport_sha256,
        }

    @app.post("/api/faults/inject")
    def inject_faults_endpoint(req: FaultInjectRequest) -> dict[str, Any]:
        source_snap = _resolve_snapshot(db, req.source_digest)
        source_dir = Path(source_snap.snapshot_path)
        if not source_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Snapshot directory missing: {source_dir}")

        run_suffix = uuid.uuid4().hex[:6]
        corrupt_dir = cfg.working_dir / f"fault_inject_{req.seed}_{run_suffix}"

        injector = FaultInjector(seed=req.seed)
        try:
            fault_manifest = injector.inject(
                source_dir=source_dir,
                dest_dir=corrupt_dir,
                fault_types=req.fault_types,
                seed=req.seed,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Fault injection failed: {exc}") from exc

        db.add_fault_run(fault_manifest)

        # Remove fault_manifest.json before validation/snapshotting? No, keep it or ignore;
        # ValidationEngine checks manifest.json, images, labels.
        val_report_raw = ValidationEngine.validate_dataset_report(
            corrupt_dir,
            contract={"expected_guideline_version": "v2.0", "num_classes": 80},
            reference_dir=source_dir,
        ).to_dict()

        dataset_id = req.dataset_id or source_snap.dataset_name
        corrupt_snap = snapshot_mgr.create_snapshot(
            dataset_dir=corrupt_dir,
            dataset_id=dataset_id,
            version_tag=f"corrupt-{run_suffix}",
            db=db,
            parent_digest=source_snap.snapshot_id,
        )

        lineage_tracker.record_edge(
            source_type="snapshot",
            source_id=source_snap.snapshot_id,
            target_type="snapshot",
            target_id=corrupt_snap.snapshot_id,
            relation="fault_injected",
            evidence_status="verified",
            metadata={
                "fault_types": req.fault_types,
                "seed": req.seed,
                "run_id": fault_manifest.run_id,
            },
        )

        release_id = f"rel-{dataset_id}-corrupt-{run_suffix}"
        gate.create_release(
            release_id=release_id,
            dataset_id=dataset_id,
            version_tag=f"corrupt-{run_suffix}",
            snapshot_digest=corrupt_snap.snapshot_id,
            snapshot_id=corrupt_snap.snapshot_id,
            metadata={
                "fault_run_id": fault_manifest.run_id,
                "injected_faults": req.fault_types,
                "seed": req.seed,
                "source_snapshot_id": source_snap.snapshot_id,
            },
        )
        model_report = _to_model_report(
            val_report_raw, release_id=release_id, snapshot_id=corrupt_snap.snapshot_id
        )
        # Ensure injected faults result in HOLD if any issues were found
        release = gate.submit_validation(
            release_id=release_id,
            validation_report=model_report,
            requires_human_review=bool(val_report_raw.get("requires_human_review", False)),
        )

        return {
            "fault_manifest": fault_manifest.model_dump(mode="json"),
            "validation_report": val_report_raw,
            "corrupt_snapshot": corrupt_snap.model_dump(mode="json"),
            "release": release.model_dump(mode="json"),
        }

    @app.get("/api/lineage/{node_id}")
    def get_lineage(node_id: str) -> dict[str, Any]:
        if node_id == "all":
            edges = [e.model_dump(mode="json") for e in db.list_lineage_events()]
            nodes_map: dict[str, dict[str, Any]] = {}
            for e in edges:
                src = e["source_id"]
                tgt = e["target_id"]
                if src not in nodes_map:
                    nodes_map[src] = {
                        "id": src,
                        "type": e.get("metadata", {}).get("source_type", "entity"),
                        "depth": 0,
                    }
                if tgt not in nodes_map:
                    nodes_map[tgt] = {
                        "id": tgt,
                        "type": e.get("metadata", {}).get("target_type", "entity"),
                        "depth": 1,
                    }
            return {
                "node_id": "all",
                "nodes": list(nodes_map.values()),
                "edges": edges,
                "blast_radius": None,
            }

        blast = impact_analyzer.get_blast_radius(node_id, include_unverified=True)
        return {
            "node_id": node_id,
            "nodes": blast.get("graph", {}).get("nodes", []),
            "edges": blast.get("graph", {}).get("edges", []),
            "blast_radius": blast,
        }

    @app.post("/api/versions/restore")
    def restore_version(req: RestoreRequest) -> dict[str, Any]:
        clean_snap = _resolve_snapshot(db, req.snapshot_digest)
        dataset_id = req.dataset_id or clean_snap.dataset_name
        restore_dir = cfg.working_dir / f"{dataset_id}_{req.version_tag}_{uuid.uuid4().hex[:6]}"

        try:
            restore_proof = restore_mgr.restore_snapshot(
                snapshot_digest=clean_snap.snapshot_id,
                target_dir=restore_dir,
                db=db,
            )
        except (SnapshotIntegrityError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        val_report_raw = ValidationEngine.validate_dataset_report(
            restore_dir,
            contract={"expected_guideline_version": "v2.0", "num_classes": 80},
            reference_dir=restore_dir,
        ).to_dict()

        restored_snap = snapshot_mgr.create_snapshot(
            dataset_dir=restore_dir,
            dataset_id=dataset_id,
            version_tag=req.version_tag,
            db=db,
            parent_digest=clean_snap.snapshot_id,
        )

        lineage_tracker.record_edge(
            source_type="snapshot",
            source_id=clean_snap.snapshot_id,
            target_type="snapshot",
            target_id=restored_snap.snapshot_id,
            relation="restored_from",
            evidence_status="verified",
            metadata={"version_tag": req.version_tag, "restore_integrity": True},
        )

        release_id = f"rel-{dataset_id}-{req.version_tag}-{uuid.uuid4().hex[:6]}"
        gate.create_release(
            release_id=release_id,
            dataset_id=dataset_id,
            version_tag=req.version_tag,
            snapshot_digest=restored_snap.snapshot_id,
            snapshot_id=restored_snap.snapshot_id,
            metadata={"restored_from": clean_snap.snapshot_id},
        )
        model_report = _to_model_report(
            val_report_raw, release_id=release_id, snapshot_id=restored_snap.snapshot_id
        )
        release = gate.submit_validation(
            release_id=release_id,
            validation_report=model_report,
            requires_human_review=bool(val_report_raw.get("requires_human_review", False)),
        )

        diff_result: dict[str, Any] | None = None
        compare_target = req.compare_digest
        if not compare_target:
            # Auto-pick latest corrupt snapshot if one exists
            for s in db.list_snapshots():
                if s.version.startswith("corrupt-"):
                    compare_target = s.snapshot_id
                    break
        if compare_target:
            try:
                diff_result = snapshot_mgr.diff_snapshots(compare_target, restored_snap.snapshot_id)
            except Exception:
                diff_result = None

        passport = evidence_builder.build_passport(
            release_id=release_id,
            output_path=cfg.reports_dir / "RELEASE_EVIDENCE.json",
            validation_report=model_report,
        )

        return {
            "restore_proof": restore_proof,
            "validation_report": val_report_raw,
            "snapshot": restored_snap.model_dump(mode="json"),
            "release": release.model_dump(mode="json"),
            "diff": diff_result,
            "passport_sha256": passport.passport_sha256,
        }

    @app.post("/api/releases/{release_id}/approve")
    def approve_release_endpoint(release_id: str, req: ApproveRequest) -> dict[str, Any]:
        try:
            release = gate.submit_approval(
                release_id=release_id,
                approval_type=req.approval_type,
                actor=req.actor,
                verdict=req.verdict,
                note=req.note,
            )
            auto_approved = False
            if release.status == ReleaseStatus.VALIDATED:
                approvals = db.list_approvals(release_id)
                latest_by_role = {a.role.lower(): a.decision for a in approvals}
                if (
                    latest_by_role.get("qc") == "APPROVED"
                    and latest_by_role.get("privacy") == "APPROVED"
                ):
                    release = gate.approve_release(release_id)
                    auto_approved = True
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        evidence_builder.build_passport(
            release_id=release_id,
            output_path=cfg.reports_dir / "RELEASE_EVIDENCE.json",
        )
        return {
            "release": release.model_dump(mode="json"),
            "approvals": [a.model_dump(mode="json") for a in db.list_approvals(release_id)],
            "auto_transitioned_to_approved": auto_approved,
        }

    @app.post("/api/releases/{release_id}/publish")
    def publish_release_endpoint(release_id: str) -> dict[str, Any]:
        try:
            release = gate.publish_release(release_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        evidence_builder.build_passport(
            release_id=release_id,
            output_path=cfg.reports_dir / "RELEASE_EVIDENCE.json",
        )
        return {
            "release": release.model_dump(mode="json"),
            "published_is_not_accepted": contract_engine.published_is_not_accepted(),
        }

    @app.post("/api/releases/{release_id}/accept")
    def accept_release_endpoint(release_id: str, req: AcceptRequest) -> dict[str, Any]:
        try:
            release = gate.accept_release(
                release_id=release_id,
                consumer=req.consumer,
                note=req.note,
                consumer_version=req.consumer_version,
                verified_sha256=req.verified_sha256,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        evidence_builder.build_passport(
            release_id=release_id,
            output_path=cfg.reports_dir / "RELEASE_EVIDENCE.json",
        )
        return {
            "release": release.model_dump(mode="json"),
            "consumer_acks": [c.model_dump(mode="json") for c in db.list_consumer_acks(release_id)],
        }

    @app.post("/api/benchmark/run")
    def run_benchmark(req: BenchmarkRequest | None = None) -> dict[str, Any]:
        body = req or BenchmarkRequest()
        source_dir: Path | None = None

        if body.source_digest:
            snap = _resolve_snapshot(db, body.source_digest)
            source_dir = Path(snap.snapshot_path)
        else:
            for snap in db.list_snapshots():
                if not snap.version.startswith("corrupt-") and Path(snap.snapshot_path).is_dir():
                    source_dir = Path(snap.snapshot_path)
                    break

        if source_dir is None or not source_dir.is_dir():
            connector = CvatConnector(
                coco250_dir=cfg.day10_data_source / "coco250",
                coco250_calib_dir=cfg.day10_data_source / "coco250_calib",
            )
            bench_dir = cfg.working_dir / f"bench_src_{uuid.uuid4().hex[:6]}"
            connector.export_coco250_subset(dest_dir=bench_dir, limit=20, seed=42, guideline_version="v2.0")
            _normalize_exported_manifest(
                dataset_dir=bench_dir,
                dataset_id="coco250_bench",
                guideline_version="v2.0",
            )
            source_dir = bench_dir

        report = FaultBenchmarker().run(
            source_dir=source_dir,
            contract={"expected_guideline_version": "v2.0", "num_classes": 80},
            output_path=cfg.reports_dir / "FAULT_BENCHMARK.json",
        )
        return report.to_dict()

    return app


app = create_app()
