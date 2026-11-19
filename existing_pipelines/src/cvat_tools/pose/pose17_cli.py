"""CLI for Pose17 5-Layer Validator, Audit, Case Miner, and Guideline Engine.

Read-only on CVAT. Supports both `python -m cvat_tools pose17 ...` and `python -m tools.pose17 ...`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cvat_tools.core import paths
from cvat_tools.pose.pose17_validate import (
    DEFAULT_CASES_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RULES_PATH,
    VALIDATOR_VERSION,
    _read_csv_rows,
    cases_for_rule,
    get_structured_logger,
    load_job_snapshot,
    lookup_rule,
    open_case,
    render_before_after_cases,
    validate_job,
    write_validation_outputs,
)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        else:
            print(f"{k}: {v}")


def cmd_validate(args: argparse.Namespace) -> int:
    log = get_structured_logger()
    snap = load_job_snapshot(args.job, offline=args.offline, profile=args.profile)
    res = validate_job(snap, profile=args.profile)
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else DEFAULT_OUTPUT_DIR
    val_path, cov_path = write_validation_outputs(res, output_dir=out_dir)

    by_layer = dict(Counter(f["layer"] for f in res["findings"]))
    by_sev = dict(Counter(f["severity"] for f in res["findings"]))
    log.info("pose17.validate.done", extra={"structured": {
        "job_id": args.job, "findings": res["finding_count"], "by_layer": by_layer,
    }})
    payload = {
        "command": "validate",
        "job_id": args.job,
        "validator_version": VALIDATOR_VERSION,
        "snapshot_sha256": res["snapshot_sha256"],
        "sequence_mode": res["sequence_mode"],
        "resolutions": res["resolutions"],
        "frame_count": res["frame_count"],
        "finding_count": res["finding_count"],
        "by_layer": by_layer,
        "by_severity": by_sev,
        "validation_results_csv": str(val_path),
        "frame_coverage_csv": str(cov_path),
    }
    _emit(payload, getattr(args, "json", False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    snap = load_job_snapshot(args.job, offline=args.offline, profile=args.profile)
    res = validate_job(snap, profile=args.profile)
    by_layer = dict(Counter(f["layer"] for f in res["findings"]))
    by_rule = dict(Counter(f["rule_id"] for f in res["findings"]))
    by_issue = dict(Counter(f["issue_type"] for f in res["findings"]))
    status_counts = dict(Counter(c["status"] for c in res["coverage_rows"]))
    l5_flags = [f for f in res["findings"] if f["layer"] == "L5_EVIDENCE_QA"]
    payload = {
        "command": "audit",
        "job_id": args.job,
        "snapshot_sha256": res["snapshot_sha256"],
        "sequence_mode": res["sequence_mode"],
        "resolutions": res["resolutions"],
        "frame_count": res["frame_count"],
        "frame_status": status_counts,
        "finding_count": res["finding_count"],
        "by_layer": by_layer,
        "by_rule": by_rule,
        "by_issue_type": by_issue,
        "l5_advisory_count": len(l5_flags),
        "l5_auto_promoted_count": sum(1 for f in l5_flags if f.get("auto_promoted")),
    }
    _emit(payload, getattr(args, "json", False))
    return 0


def cmd_mine_cases(args: argparse.Namespace) -> int:
    snap = load_job_snapshot(args.job, offline=args.offline, profile=args.profile)
    res = validate_job(snap, profile=args.profile)
    existing_cases = _read_csv_rows(DEFAULT_CASES_PATH)
    existing_by_frame = {int(c["frame_index"]): c for c in existing_cases if str(c.get("job_id")) == str(args.job)}

    mined: list[dict[str, Any]] = []
    for cov in res["coverage_rows"]:
        fidx = int(cov["frame_index"])
        fr_findings = [f for f in res["findings"] if int(f["frame_index"]) == fidx and not f["advisory_only"]]
        if not fr_findings and fidx not in existing_by_frame:
            continue
        case_row = existing_by_frame.get(fidx, {})
        mined.append({
            "frame_index": fidx,
            "image_name": cov["image_name"],
            "case_id": case_row.get("case_id", ""),
            "review_status": case_row.get("review_status", "unregistered"),
            "error_count": cov["error_count"],
            "warning_count": cov["warning_count"],
            "detected_rules": sorted({f["rule_id"] for f in fr_findings}),
            "detected_issues": sorted({f["issue_type"] for f in fr_findings}),
        })

    payload = {
        "command": "mine-cases",
        "job_id": args.job,
        "total_existing_cases": len(existing_by_frame),
        "mined_frames_count": len(mined),
        "cases": mined,
    }
    _emit(payload, getattr(args, "json", False))
    return 0


def cmd_build_guideline(args: argparse.Namespace) -> int:
    rules = _read_csv_rows(DEFAULT_RULES_PATH)
    cases = _read_csv_rows(DEFAULT_CASES_PATH)
    status_counts = dict(Counter(r.get("status", "") for r in rules))
    missing_ev = []
    for c in cases:
        ev = (c.get("evidence_path") or "").strip()
        if not ev or not (paths.WORKSPACE / ev).exists():
            missing_ev.append(c.get("case_id"))

    rendered: list[str] = []
    if getattr(args, "render_before_after", False):
        snap = load_job_snapshot(args.job, offline=args.offline, profile=args.profile)
        out_files = render_before_after_cases(snap, profile=args.profile)
        rendered = [str(p) for p in out_files]

    payload = {
        "command": "build-guideline",
        "job_id": args.job,
        "rules_csv": str(DEFAULT_RULES_PATH),
        "rules_count": len(rules),
        "rule_status_counts": status_counts,
        "cases_csv": str(DEFAULT_CASES_PATH),
        "cases_count": len(cases),
        "missing_evidence_cases": missing_ev,
        "rendered_before_after": rendered,
    }
    _emit(payload, getattr(args, "json", False))
    return 0


def cmd_lookup_rule(args: argparse.Namespace) -> int:
    rid = args.rule_id or args.rule_id_pos
    if not rid:
        raise ValueError("Provide --rule-id or positional rule_id")
    rule = lookup_rule(rid)
    if rule is None:
        raise KeyError(f"Rule not found: {rid}")
    linked = cases_for_rule(rid)
    payload = {
        "command": "lookup-rule",
        "rule": rule,
        "matching_cases_count": len(linked),
        "matching_case_ids": [c["case_id"] for c in linked],
    }
    _emit(payload, getattr(args, "json", False))
    return 0


def cmd_open_case(args: argparse.Namespace) -> int:
    cid = args.case_id or args.case_id_pos
    if not cid:
        raise ValueError("Provide --case-id or positional case_id")
    case = open_case(cid)
    if case is None:
        raise KeyError(f"Case not found: {cid}")
    payload = {
        "command": "open-case",
        "case": case,
    }
    _emit(payload, getattr(args, "json", False))
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    from cvat_tools.pose.pose17_sync import build_proposals, open_in_vscode, render_bold_proposals

    payload = build_proposals(
        job_id=args.job,
        offline=args.offline,
        profile=args.profile,
        include_missing_drafts=not getattr(args, "no_missing_drafts", False),
    )
    rendered = render_bold_proposals(payload, profile=args.profile)
    opened = False
    if getattr(args, "open", False):
        to_open = []
        if rendered.get("contact_sheet"):
            to_open.append(rendered["contact_sheet"])
        to_open.extend(rendered.get("files", []))
        opened = open_in_vscode(to_open)
    out = {
        "command": "propose",
        "job_id": args.job,
        "proposal_count": payload["proposal_count"],
        "frames": [p["frame_index"] for p in payload["proposals"]],
        "contact_sheet": rendered.get("contact_sheet"),
        "rendered_count": len(rendered.get("files", [])),
        "opened_in_vscode": opened,
    }
    _emit(out, getattr(args, "json", False))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    from cvat_tools.pose.pose17_sync import sync_proposals

    frames = None
    if getattr(args, "frames", None):
        frames = [int(x.strip()) for x in str(args.frames).split(",") if x.strip()]
    out = sync_proposals(
        job_id=args.job,
        frames=frames,
        profile=args.profile,
        yes=bool(getattr(args, "yes", False)),
    )
    out["command"] = "sync"
    _emit(out, getattr(args, "json", False))
    return 0


def _add_common_job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--job", type=int, default=1623, help="CVAT job ID (default: 1623)")
    p.add_argument("--profile", default="edufun", help="CVAT profile (default: edufun)")
    p.add_argument("--offline", action="store_true", help="Read local snapshot instead of CVAT")
    p.add_argument("--json", action="store_true", help="Output clean JSON")


def _register_subcommands(psub: argparse._SubParsersAction, add_fn) -> None:
    s = add_fn(psub, "validate", cmd_validate, help="Run L1-L5 validation and write CSVs")
    _add_common_job_args(s)
    s.add_argument("--out-dir", default=None, help="Output directory for CSVs")

    s = add_fn(psub, "audit", cmd_audit, help="Print L1-L5 audit summary")
    _add_common_job_args(s)

    s = add_fn(psub, "mine-cases", cmd_mine_cases, help="Mine difficult cases from job")
    _add_common_job_args(s)

    s = add_fn(psub, "build-guideline", cmd_build_guideline, help="Verify guideline registry & optionally render before/after")
    _add_common_job_args(s)
    s.add_argument("--render-before-after", action="store_true", help="Render before/after comparison images")

    s = add_fn(psub, "propose", cmd_propose, help="Build bold color-coded proposal images and optionally open in VS Code")
    _add_common_job_args(s)
    s.add_argument("--open", action="store_true", help="Open contact sheet and proposal images in VS Code")
    s.add_argument("--no-missing-drafts", action="store_true", help="Skip AI draft proposals for missing frames")

    s = add_fn(psub, "sync", cmd_sync, help="Preview (dry-run) or sync approved proposals to CVAT (--yes)")
    p_sync_job = s
    p_sync_job.add_argument("--job", type=int, default=1623, help="CVAT job ID (default: 1623)")
    p_sync_job.add_argument("--profile", default="edufun", help="CVAT profile (default: edufun)")
    p_sync_job.add_argument("--frames", default=None, help="Comma-separated frame indices to sync (default: all proposed)")
    p_sync_job.add_argument("--yes", action="store_true", help="Actually write approved changes to CVAT (with backup)")
    p_sync_job.add_argument("--json", action="store_true", help="Output clean JSON")

    s = add_fn(psub, "lookup-rule", cmd_lookup_rule, help="Lookup rule metadata and linked cases")
    s.add_argument("rule_id_pos", nargs="?", default=None, help="Rule ID (e.g. P17-G11)")
    s.add_argument("--rule-id", default=None, help="Rule ID (e.g. P17-G11)")
    s.add_argument("--json", action="store_true", help="Output clean JSON")

    s = add_fn(psub, "open-case", cmd_open_case, help="Open case metadata and verify evidence_path")
    s.add_argument("case_id_pos", nargs="?", default=None, help="Case ID (e.g. CASE-1623-F093)")
    s.add_argument("--case-id", default=None, help="Case ID (e.g. CASE-1623-F093)")
    s.add_argument("--json", action="store_true", help="Output clean JSON")


def register(sub: argparse._SubParsersAction, add_fn) -> None:
    """Register `pose17` subcommand group on the main cvat-tools parser."""
    p17 = add_fn(sub, "pose17", lambda a: 0, help="Pose17 5-layer validator & guideline engine")
    psub = p17.add_subparsers(dest="p17cmd", required=True)
    _register_subcommands(psub, add_fn)


def main(argv: list[str] | None = None) -> int:
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="tools.pose17", description="Pose17 Validator & Guideline CLI")
    parser.add_argument("--json", action="store_true", help="Output clean JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add(parent, name, fn, **kw):
        s = parent.add_parser(name, **kw)
        s.set_defaults(fn=fn)
        return s

    _register_subcommands(sub, _add)

    args = parser.parse_args(argv)
    try:
        return args.fn(args) or 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
