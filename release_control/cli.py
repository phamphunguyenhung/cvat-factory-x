"""Command-line interface for CVAT Factory X Data Release Control Tower."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from release_control.adapters import CvatConnector
from release_control.config import ControlTowerConfig, ReadOnlyError, load_dotenv, read_only
from release_control.db import ControlTowerDB
from release_control.demo.run_e2e import _normalize_exported_manifest, run_e2e_demo
from release_control.fault_lab import FaultBenchmarker, FaultInjector
from release_control.validators.gate import ValidationEngine
from release_control.versions import RestoreManager, SnapshotManager


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="release-control",
        description="CVAT Factory X Data Release Control Tower CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. demo
    demo_p = subparsers.add_parser(
        "demo",
        help="Run the 5-step COCO250 + CVAT Dataset Time Machine end-to-end demo.",
    )
    demo_p.add_argument("--limit", type=int, default=30, help="COCO250 subset size (default: 30).")
    demo_p.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42).")
    demo_p.add_argument("--legacy", action="store_true", help="Chay demo COCO250 cu (run_e2e).")
    demo_p.add_argument("--source-dir", type=Path, default=None, help="Dataset sach (mac dinh demo/coco128_clean).")
    demo_p.add_argument("--approver", default=None, help="Nguoi duyet THAT (qc+privacy). Khong co -> dung o VALIDATED.")
    demo_p.add_argument("--consumer", default=None, help="Consumer THAT de ACK. Khong co -> khong ACCEPTED.")
    demo_p.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not reset existing demo DB/working directories before running.",
    )

    # 2. benchmark
    bench_p = subparsers.add_parser(
        "benchmark",
        help="Run FaultBenchmarker (dev seeds, held-out seeds, metamorphic tests).",
    )
    bench_p.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Clean dataset directory to benchmark (defaults to exporting a 30-image COCO250 subset).",
    )
    bench_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path (default: reports/FAULT_BENCHMARK.json).",
    )
    bench_p.add_argument("--limit", type=int, default=30, help="COCO250 subset size if --source-dir is omitted.")
    bench_p.add_argument("--seed", type=int, default=42, help="COCO250 export seed if --source-dir is omitted.")

    # 3. validate
    val_p = subparsers.add_parser(
        "validate",
        help="Run ValidationEngine against contracts and print JSON report.",
    )
    val_p.add_argument("dataset_dir", type=Path, help="Dataset directory containing manifest.json.")
    val_p.add_argument("--reference-dir", type=Path, default=None, help="Optional clean reference directory.")
    val_p.add_argument(
        "--guideline-version",
        default="v2.0",
        help="Expected guideline version in contract (default: v2.0).",
    )

    # 4. inject
    inj_p = subparsers.add_parser(
        "inject",
        help="Inject deterministic faults (T1..T8) into an isolated copy of source_dir.",
    )
    inj_p.add_argument("source_dir", type=Path, help="Clean source dataset directory.")
    inj_p.add_argument("dest_dir", type=Path, help="Destination directory for corrupted copy.")
    inj_p.add_argument(
        "--faults",
        nargs="+",
        required=True,
        help="Fault types to inject (e.g. T5_BBOX_DRIFT T6_CLASS_SWAP).",
    )
    inj_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # 5. snapshot
    snap_p = subparsers.add_parser(
        "snapshot",
        help="Create an immutable content-addressed dataset snapshot.",
    )
    snap_p.add_argument("dataset_dir", type=Path, help="Dataset directory to snapshot.")
    snap_p.add_argument("--dataset-id", required=True, help="Logical dataset identifier.")
    snap_p.add_argument("--tag", required=True, help="Version tag (e.g. v1, v2).")
    snap_p.add_argument("--parent-digest", default=None, help="Optional parent snapshot digest.")

    # 6. restore
    rest_p = subparsers.add_parser(
        "restore",
        help="Restore an immutable snapshot and verify its cryptographic digest.",
    )
    rest_p.add_argument("digest", help="Snapshot SHA-256 digest (or snapshot directory).")
    rest_p.add_argument("target_dir", type=Path, help="Target directory for restored dataset.")

    # 7. dashboard
    dash_p = subparsers.add_parser(
        "dashboard",
        help="Launch the FastAPI/Uvicorn Control Tower dashboard.",
    )
    dash_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    dash_p.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    dash_p.add_argument("--reload", action="store_true", help="Enable Uvicorn auto-reload.")

    lin_p = subparsers.add_parser("lineage-impact", help="Blast radius cua dataset/snapshot/release.")
    lin_p.add_argument("node", help="dataset_id, digest, snapshot_id hoac release_id.")
    lin_p.add_argument("--include-unverified", action="store_true", help="Gom canh simulated/unverified (gan nhan).")

    rel_p = subparsers.add_parser("release", help="Vong doi release: create/validate/approve/publish/accept.")
    rel_p.add_argument("action", choices=["create", "validate", "approve", "publish", "accept"])
    rel_p.add_argument("release_id")
    rel_p.add_argument("--dataset-id")
    rel_p.add_argument("--tag")
    rel_p.add_argument("--digest", help="Snapshot digest/id (create).")
    rel_p.add_argument("--dataset-dir", type=Path, help="Thu muc dataset de validate.")
    rel_p.add_argument("--guideline-version", default="v2.0")
    rel_p.add_argument("--approver", help="Nguoi duyet that (approve): ghi qc+privacy APPROVED.")
    rel_p.add_argument("--consumer", help="Consumer that (accept).")

    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning exit code 0 on success."""
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    config = ControlTowerConfig.auto_discover()
    config.ensure_dirs()
    try:
        return _dispatch(args, parser, config)
    except ReadOnlyError as exc:
        print(f"[CHAN] {exc}")
        return 3


def _vi_report(r: Any) -> str:
    return f"passed={r.passed} status={r.recommended_status} severity={r.severity_counts}"


def _top_evidence(r: Any, n: int = 3) -> list[dict[str, Any]]:
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    crit = sorted(r.issues, key=lambda i: order.get(i.get("severity"), 3))
    return [
        {k: i.get(k) for k in ("code", "fault_type", "severity", "image_id", "file_path", "message")}
        for i in crit[:n]
    ]


def _make_gate_objs(config: ControlTowerConfig):
    from release_control.adapters import MlflowTracker
    from release_control.contracts.engine import ContractEngine
    from release_control.lineage.events import LineageTracker
    from release_control.lineage.impact import ImpactAnalyzer
    from release_control.release.evidence import EvidenceBuilder
    from release_control.release.gate import ReleaseGate

    db = ControlTowerDB(config.db_path)
    ce = ContractEngine(config.contracts_dir)
    lt = LineageTracker(db=db, openlineage_path=config.lineage_dir / "openlineage.jsonl")
    gate = ReleaseGate(db=db, contract_engine=ce, lineage_tracker=lt)
    impact = ImpactAnalyzer(db=db, mlflow_tracker=MlflowTracker(tracking_uri=config.mlruns_dir))
    return db, ce, lt, gate, impact, EvidenceBuilder(db=db, contract_engine=ce)


def _run_demo(args: Any, config: ControlTowerConfig) -> int:
    import os
    import shutil
    import subprocess
    import sys

    from release_control.demo.run_e2e import _to_model_validation_report

    ns = os.environ.get("RC_DEMO_NAMESPACE", "demo_cli")
    root = config.artifacts_dir / ns
    if root.exists() and not args.no_reset:
        shutil.rmtree(root)
    cfg = config.model_copy(update={"artifacts_dir": root, "db_path": root / "control_tower.sqlite"})
    cfg.ensure_dirs()
    src = args.source_dir or (config.project_root / "demo" / "coco128_clean")
    if not (src / "manifest.json").is_file():
        mk = config.project_root / "scripts" / "make_demo_data.py"
        if mk.is_file():
            subprocess.run([sys.executable, "-X", "utf8", str(mk)], check=False)
    if not (src / "manifest.json").is_file():
        print(f"[LOI] Khong co dataset sach tai {src}")
        return 2
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dsid = manifest.get("dataset_id", "demo")
    gv = manifest.get("guideline_version", "v1.0")
    contract = {"expected_guideline_version": gv, "num_classes": 80}
    db, ce, lt, gate, impact, evb = _make_gate_objs(cfg)
    sm, rm = SnapshotManager(config=cfg), RestoreManager(config=cfg)
    ok = True

    # (1)
    r1 = ValidationEngine.validate_dataset_report(src, contract=contract)
    snap1 = sm.create_snapshot(src, dsid, "v1", db=db)
    rel1 = gate.create_release(f"{ns}-rel-v1", dsid, "v1", snap1.manifest_sha256, snapshot_id=snap1.snapshot_id)
    rel1 = gate.submit_validation(rel1.release_id, _to_model_validation_report(r1, rel1.release_id, snap1.snapshot_id))
    print(f"[1] Validate sach: {_vi_report(r1)} -> release v1 = {rel1.status.value}; snapshot {snap1.snapshot_id[:12]}")
    ok &= rel1.status.value == "VALIDATED"

    # (2)
    bad = root / "working" / "corrupt"
    FaultInjector(seed=42).inject(src, bad, ["T6", "T5"], seed=42)
    parse_ok = bool(json.loads((bad / "manifest.json").read_text(encoding="utf-8")))
    r2 = ValidationEngine.validate_dataset_report(bad, contract=contract, reference_dir=src)
    snap2 = sm.create_snapshot(bad, dsid, "v1-corrupt", db=db, parent_digest=snap1.snapshot_id)
    rel2 = gate.create_release(f"{ns}-rel-corrupt", dsid, "v1-corrupt", snap2.manifest_sha256, snapshot_id=snap2.snapshot_id)
    rel2 = gate.submit_validation(rel2.release_id, _to_model_validation_report(r2, rel2.release_id, snap2.snapshot_id))
    ev = _top_evidence(r2)
    print(f"[2] Inject T6+T5: parse duoc={parse_ok}; {_vi_report(r2)} -> {rel2.status.value}")
    for e in ev:
        print(f"    bang chung: {e['code']} {e['image_id']} {e['file_path']}")
    ok &= parse_ok and rel2.status.value == "HOLD" and bool(ev)

    # (3)
    lt.record_edge(source_type="snapshot", source_id=snap1.snapshot_id, target_type="model",
                   target_id="model:demo_detector", relation="trained_into",
                   evidence_status="unverified", metadata={"simulated": True, "note": "canh gia lap de demo"})
    br = impact.get_blast_radius(snap1.manifest_sha256, include_unverified=True)
    edges = br["graph"]["edges"]
    nsim = sum(e["evidence_status"] != "verified" for e in edges)
    print(f"[3] Lineage impact: {len(br['snapshots'])} snapshot, {len(br['releases'])} release, "
          f"{len(edges)} canh ({len(edges) - nsim} verified co bang chung, {nsim} simulated)")
    for e in edges:
        if e["evidence_status"] != "verified":
            print(f"    [simulated] {e['source']} -{e['relation']}-> {e['target']}")

    # (4)
    out = root / "working" / "restored_v2"
    res = rm.restore_snapshot(snap1.snapshot_id, out, db=db)
    digest_ok = res["restored_digest"] == snap1.snapshot_id
    r4 = ValidationEngine.validate_dataset_report(out, contract=contract)
    snap4 = sm.create_snapshot(out, dsid, "v2", db=db, parent_digest=snap1.snapshot_id)
    rel4 = gate.create_release(f"{ns}-rel-v2", dsid, "v2", snap4.manifest_sha256, snapshot_id=snap4.snapshot_id)
    mr4 = _to_model_validation_report(r4, rel4.release_id, snap4.snapshot_id)
    rel4 = gate.submit_validation(rel4.release_id, mr4)
    print(f"[4] Restore: digest khop={digest_ok}; validate lai: {_vi_report(r4)} -> v2 = {rel4.status.value}")
    ok &= digest_ok and rel4.status.value == "VALIDATED"
    if args.approver:
        for role in ("qc", "privacy"):
            gate.submit_approval(rel4.release_id, role, args.approver, "approved", "CLI --approver")
        gate.approve_release(rel4.release_id)
        rel4 = gate.publish_release(rel4.release_id)
        if args.consumer:
            rel4 = gate.accept_release(rel4.release_id, args.consumer, verified_sha256=snap4.snapshot_id)
        print(f"    approver that '{args.approver}' -> v2 = {rel4.status.value}")
        ok &= rel4.status.value in {"PUBLISHED", "ACCEPTED"}
    else:
        print("    KHONG co approval that (--approver) -> DUNG o VALIDATED, khong PUBLISHED.")

    # (5)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    bench = FaultBenchmarker().run(source_dir=src, contract=contract, output_path=cfg.reports_dir / "FAULT_BENCHMARK.json")
    passport = evb.build_passport(rel4.release_id, output_path=cfg.reports_dir / "RELEASE_EVIDENCE.json", validation_report=mr4)
    print(f"[5] Ghi reports/FAULT_BENCHMARK.json (critical recall={bench.overall_critical_fault_recall}, "
          f"false_release={bench.false_release_rate}) va RELEASE_EVIDENCE.json (sha {passport.passport_sha256[:12]})")
    print("KET QUA: " + ("DAT" if ok else "KHONG DAT"))
    return 0 if ok else 1


def _dispatch(args: Any, parser: argparse.ArgumentParser, config: ControlTowerConfig) -> int:
    if args.command == "lineage-impact":
        _, _, _, _, impact, _ = _make_gate_objs(config)
        _print_json(impact.get_blast_radius(args.node, include_unverified=args.include_unverified))
        return 0

    if args.command == "release":
        db, _, _, gate, _, _ = _make_gate_objs(config)
        rid = args.release_id
        if args.action == "create":
            if not (args.dataset_id and args.tag and args.digest):
                parser.error("create can --dataset-id --tag --digest")
            rec = gate.create_release(rid, args.dataset_id, args.tag, args.digest)
        elif args.action == "validate":
            from release_control.demo.run_e2e import _to_model_validation_report
            if not args.dataset_dir:
                parser.error("validate can --dataset-dir")
            rep = ValidationEngine.validate_dataset_report(
                args.dataset_dir, contract={"expected_guideline_version": args.guideline_version, "num_classes": 80})
            cur = db.get_release(rid)
            rec = gate.submit_validation(rid, _to_model_validation_report(rep, rid, cur.snapshot_id))
        elif args.action == "approve":
            if not args.approver:
                print("[CHAN] Can --approver that; khong tu bia approval.")
                return 2
            for role in ("qc", "privacy"):
                gate.submit_approval(rid, role, args.approver, "approved", "CLI --approver")
            rec = gate.approve_release(rid)
        elif args.action == "publish":
            rec = gate.publish_release(rid)
        else:
            if not args.consumer:
                print("[CHAN] Can --consumer that; khong tu bia ACK.")
                return 2
            rec = gate.accept_release(rid, args.consumer)
        _print_json(rec.model_dump(mode="json"))
        return 0

    if args.command == "demo" and not args.legacy:
        return _run_demo(args, config)

    if args.command == "demo":
        summary = run_e2e_demo(
            subset_limit=args.limit,
            seed=args.seed,
            reset_state=not args.no_reset,
        )
        _print_json(summary)
        return 0

    if args.command == "benchmark":
        source_dir = args.source_dir
        if source_dir is None:
            source_dir = config.working_dir / "benchmark_source"
            connector = CvatConnector(
                coco250_dir=config.day10_data_source / "coco250",
                coco250_calib_dir=config.day10_data_source / "coco250_calib",
                cvat_jobs_dir=config.cvat_data_dir / "jobs" / "edufun",
            )
            export_info = connector.export_coco250_subset(
                dest_dir=source_dir,
                limit=args.limit,
                seed=args.seed,
                guideline_version="v2.0",
            )
            _normalize_exported_manifest(
                dataset_dir=source_dir,
                dataset_id="coco250_benchmark",
                guideline_version="v2.0",
                coco_path=Path(export_info["coco_path"]),
            )
        output_path = args.output or (config.reports_dir / "FAULT_BENCHMARK.json")
        report = FaultBenchmarker().run(
            source_dir=source_dir,
            contract={"expected_guideline_version": "v2.0", "num_classes": 80},
            output_path=output_path,
        )
        _print_json(report.to_dict())
        return 0

    if args.command == "validate":
        contract = {"expected_guideline_version": args.guideline_version, "num_classes": 80}
        report = ValidationEngine.validate_dataset_report(
            args.dataset_dir,
            contract=contract,
            reference_dir=args.reference_dir,
        )
        _print_json(report.to_dict())
        return 0 if report.passed else 1

    if args.command == "inject":
        manifest = FaultInjector(seed=args.seed).inject(
            source_dir=args.source_dir,
            dest_dir=args.dest_dir,
            fault_types=args.faults,
            seed=args.seed,
        )
        _print_json(manifest.model_dump(mode="json"))
        return 0

    if args.command == "snapshot":
        db = ControlTowerDB(config.db_path)
        record = SnapshotManager(config=config).create_snapshot(
            dataset_dir=args.dataset_dir,
            dataset_id=args.dataset_id,
            version_tag=args.tag,
            db=db,
            parent_digest=args.parent_digest,
        )
        _print_json(record.model_dump(mode="json"))
        return 0

    if args.command == "restore":
        db = ControlTowerDB(config.db_path)
        result = RestoreManager(config=config).restore_snapshot(
            snapshot_digest=args.digest,
            target_dir=args.target_dir,
            db=db,
        )
        _print_json(result)
        return 0

    if args.command == "dashboard":
        import uvicorn

        uvicorn.run(
            "release_control.dashboard.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
