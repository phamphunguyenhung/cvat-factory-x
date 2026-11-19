"""Unified CLI for CVAT Autolabel Factory V2 (`python -m cvat_tools.autolabel ...`)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from cvat_tools.autolabel.cache import cache_dir, load_checkpoint, load_or_fetch_snapshot, save_checkpoint
from cvat_tools.autolabel.qc import evaluate_frame_qc
from cvat_tools.autolabel.refine import selective_refine_frame
from cvat_tools.autolabel.sync import safe_auto_sync
from cvat_tools.core import paths
from cvat_tools.pose.face50_engine import render_face50_proposals
from cvat_tools.pose.pose17_sync import open_in_vscode


def _parse_job_maps(state: dict[str, Any]) -> tuple[dict[int, str], dict[int, int], dict[int, list[dict[str, Any]]]]:
    labels = state["labels"]
    lbl_name = {int(l["id"]): str(l["name"]) for l in labels}
    sub_kp = {int(s["id"]): int(s["name"]) for l in labels for s in (l.get("sublabels") or []) if str(s["name"]).isdigit()}
    by_f: dict[int, list[dict[str, Any]]] = {}
    for sh in state["annotations"].get("shapes") or []:
        by_f.setdefault(int(sh["frame"]), []).append(sh)
    return lbl_name, sub_kp, by_f


def run_pipeline(
    job_id: int,
    mode: str = "auto",
    profile: str = "edufun",
    offline: bool = False,
    yes: bool = False,
    open_vscode: bool = False,
    force_refresh: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    state = load_or_fetch_snapshot(job_id=job_id, profile=profile, offline=offline, force_refresh=force_refresh)
    lbl_name, sub_kp, by_f = _parse_job_maps(state)
    meta = state["meta"]
    fdir = paths.job_dir(profile, job_id) / "frames"

    cp = load_checkpoint(job_id)
    results: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []

    for i, fm in enumerate(meta.get("frames") or []):
        fidx = int(meta["start_frame"]) + i
        img_name = Path(fm["name"]).name
        shapes = by_f.get(fidx, [])
        res = selective_refine_frame(
            frame_idx=fidx,
            shapes=shapes,
            sub_id_to_kp=sub_kp,
            lbl_id_to_name=lbl_name,
            width=int(fm["width"]),
            height=int(fm["height"]),
            img_path=fdir / img_name,
        )
        res["image_name"] = img_name
        results.append(res)
        if res["status"] == "eligible":
            eligible.append(res)
        elif res["status"] == "review":
            review_queue.append(res)
        cp["completed_frames"][str(fidx)] = res["status"]

    cp["last_stage"] = "refine"
    save_checkpoint(job_id, cp)

    # Render Contact Sheet for eligible + review frames
    vis_payload = {
        "job_id": job_id,
        "sublabel_id_to_kid": {},
        "sub_id_to_kp": {str(k): v for k, v in sub_kp.items()},
        "eligible_proposals": [
            {
                "frame_index": r["frame"],
                "image_name": r["image_name"],
                "changed_kpts": r["changed_kpts"],
                "changes": [{"kp": k} for k in r["changed_kpts"]],
                "all_before_shapes": r["all_before_shapes"],
                "all_proposed_shapes": r["all_proposed_shapes"],
            }
            for r in eligible
        ],
        "review_queue": [
            {
                "frame_index": r["frame"],
                "image_name": r["image_name"],
                "reason": "; ".join(f["code"] for f in r["after_qc"]["findings"]) or "manual_review",
            }
            for r in review_queue
        ],
    }
    cdir = cache_dir(job_id)
    vis_dir = cdir / "evidence"
    rendered = render_face50_proposals(vis_payload, profile=profile, output_dir=vis_dir) if (eligible or review_queue) else {"contact_sheet": None, "files": []}

    if (open_vscode or review_queue) and rendered.get("contact_sheet"):
        open_in_vscode([rendered["contact_sheet"]])

    # Sync stage
    if mode == "auto" and yes:
        sync_res = safe_auto_sync(job_id=job_id, eligible_results=eligible, profile=profile, yes=True)
    else:
        sync_res = safe_auto_sync(job_id=job_id, eligible_results=eligible, profile=profile, yes=False)

    elapsed_s = round(time.perf_counter() - t0, 3)
    n_frames = len(meta.get("frames") or [])
    report = {
        "job_id": job_id,
        "mode": mode,
        "elapsed_sec": elapsed_s,
        "frames_per_min": round(n_frames / max(elapsed_s / 60.0, 1e-4), 1),
        "api_calls": state.get("api_calls", 0) + (2 if sync_res.get("written_to_cvat") else 0),
        "agent_calls": 0,
        "tokens_io": 0,
        "cache_read_tokens": 0,
        "frames_total": n_frames,
        "eligible_frames": [r["frame"] for r in eligible],
        "review_frames": [r["frame"] for r in review_queue],
        "total_kp_corrected": sum(len(r["changed_kpts"]) for r in eligible),
        "qc_errors_before": sum(r["before_qc"]["error_count"] for r in results),
        "qc_errors_after": sum(r["after_qc"]["error_count"] for r in results),
        "qc_warnings_before": sum(r["before_qc"]["warning_count"] for r in results),
        "qc_warnings_after": sum(r["after_qc"]["warning_count"] for r in results),
        "contact_sheet": rendered.get("contact_sheet"),
        "sync": sync_res,
    }
    (cdir / "latest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def cmd_benchmark(args: argparse.Namespace) -> int:
    new_rep = run_pipeline(job_id=args.job, mode="dry-run", profile=args.profile, offline=args.offline, yes=False)
    old_baseline = {
        "time_sec": 1273,
        "tokens_io": 1468887,
        "cache_read_tokens": 13116355,
        "agent_calls": 10,
        "frames_per_min": 1.41,
        "source": "Job 1975 F01..F10 subagent transcripts (2026-11-18)",
    }
    out = {
        "command": "benchmark",
        "job_id": args.job,
        "old_10_agent_baseline": old_baseline,
        "new_autolabel_v2": {
            "time_sec": new_rep["elapsed_sec"],
            "tokens_io": new_rep["tokens_io"],
            "cache_read_tokens": new_rep["cache_read_tokens"],
            "agent_calls": new_rep["agent_calls"],
            "frames_per_min": new_rep["frames_per_min"],
            "qc_errors_before": new_rep["qc_errors_before"],
            "qc_errors_after": new_rep["qc_errors_after"],
            "eligible_frames": new_rep["eligible_frames"],
            "review_frames": new_rep["review_frames"],
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(prog="cvat_tools.autolabel", description="CVAT Autolabel Factory V2")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("analyze", "draft", "refine", "validate", "sync", "benchmark", "report", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--job", type=int, default=1975)
        sp.add_argument("--profile", default="edufun")
        sp.add_argument("--mode", default="auto", choices=["auto", "dry-run"])
        sp.add_argument("--offline", action="store_true")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--yes", action="store_true")
        sp.add_argument("--open", action="store_true")
        sp.add_argument("--refresh", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "benchmark":
        return cmd_benchmark(args)
    if args.cmd == "report":
        rp = cache_dir(args.job) / "latest_report.json"
        if not rp.exists():
            rep = run_pipeline(args.job, mode="dry-run", profile=args.profile, offline=args.offline)
        else:
            rep = json.loads(rp.read_text(encoding="utf-8"))
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    yes_flag = bool(args.yes and not args.dry_run and args.mode == "auto" and args.cmd in ("run", "sync"))
    rep = run_pipeline(
        job_id=args.job,
        mode=args.mode,
        profile=args.profile,
        offline=args.offline,
        yes=yes_flag,
        open_vscode=args.open,
        force_refresh=args.refresh,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0
