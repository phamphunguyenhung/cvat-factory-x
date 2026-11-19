"""CLI integration for V3 Panoptic Harness (`cvat-tools harness <subcommand>`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from cvat_tools.core import paths
from cvat_tools.pipeline.source import FrameSource
from harness.contracts import to_serializable
from harness.cvat_adapter import CvatAdapter
from harness.frame_profiler import assess_continuity
from harness.orchestrator import HarnessOrchestrator
from harness.quality_evaluator import QualityEvaluator
from harness.review_queue import write_review_html


def _parse_frames(spec: str | None) -> list[int]:
    if not spec:
        return list(range(60, 80))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _load_frames(job: int, frames: list[int]) -> dict[int, Any]:
    src = FrameSource(job)
    return {fid: src.image(fid) for fid in frames}


def _emit(payload: Any, as_json: bool = False) -> int:
    if as_json:
        print(json.dumps(to_serializable(payload), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(to_serializable(payload), indent=2, ensure_ascii=False))
    return 0


def register(subparsers, add_fn=None) -> None:
    p = subparsers.add_parser("harness", help="Object-First Panoptic Harness V3 commands")
    hsub = p.add_subparsers(dest="harness_cmd", required=True)

    def _add(name: str, fn, help_text: str):
        sp = hsub.add_parser(name, help=help_text)
        sp.add_argument("--job", type=int, default=285)
        sp.add_argument("--frames", type=str, default="71")
        sp.add_argument("--mode", type=str, default="balanced", choices=["fast", "balanced", "high_recall", "challenger"])
        sp.add_argument("--out", type=Path, default=None)
        sp.set_defaults(func=fn)
        return sp

    _add("audit", _cmd_audit, "Check sequence continuity and frame profiles")
    _add("thing", _cmd_thing, "Run Thing Engine only")
    _add("stuff", _cmd_stuff, "Run Stuff Engine only")
    _add("structure", _cmd_structure, "Run Structure Engine")
    _add("track", _cmd_track, "Run conditional Temporal Association Engine")
    _add("compose", _cmd_compose, "Run full Thing+Stuff+Structure+Compose pipeline")
    _add("eval", _cmd_eval, "Evaluate V3 pipeline against local target")
    _add("preview", _cmd_preview, "Write local HTML review queue and preview images")
    sp_exp = _add("export", _cmd_export, "Export V3 draft to local ShapeStore / JSON (read-only remote)")
    sp_exp.add_argument("--yes", action="store_true", help="Explicitly confirm remote write (requires write_enabled)")


def _cmd_audit(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    rep = assess_continuity(sorted(frames.items()), continuity_hint=orch.router.continuity_hint)
    return _emit({"job": args.job, "continuity": rep, "registry": orch.registry.status()})


def _cmd_thing(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames, run_things=True, run_stuff=False, run_structure=False, enable_tracking=False)
    if args.out:
        CvatAdapter.export_draft_json(scenes, args.out)
    return _emit({"frames": len(scenes), "elements": sum(len(s.elements) for s in scenes), "peak_rss_mb": orch.registry.peak_rss_mb})


def _cmd_stuff(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames, run_things=False, run_stuff=True, run_structure=False, enable_tracking=False)
    if args.out:
        CvatAdapter.export_draft_json(scenes, args.out)
    return _emit({"frames": len(scenes), "elements": sum(len(s.elements) for s in scenes)})


def _cmd_structure(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames, run_things=True, run_stuff=True, run_structure=True, enable_tracking=False)
    if args.out:
        CvatAdapter.export_draft_json(scenes, args.out)
    return _emit({"frames": len(scenes), "elements": sum(len(s.elements) for s in scenes)})


def _cmd_track(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    rep = assess_continuity(sorted(frames.items()), continuity_hint=orch.router.continuity_hint)
    if not rep.is_continuous:
        return _emit({"tracking_enabled": False, "reason": rep.reason})
    scenes = orch.run_batch(frames, run_things=True, run_stuff=False, run_structure=False, enable_tracking=True)
    return _emit({"tracking_enabled": True, "frames": len(scenes)})


def _cmd_compose(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames)
    out = args.out or (paths.DATA / "jobs" / str(args.job) / "previews" / "v3" / "v3_draft.json")
    CvatAdapter.export_draft_json(scenes, out, include_review=True)
    return _emit({"out": str(out), "frames": len(scenes), "elements": sum(len(s.elements) for s in scenes)})


def _cmd_eval(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames, run_things=True, run_stuff=False, run_structure=False, enable_tracking=False)
    ev = QualityEvaluator()
    gate_rep = ev.check_stop_gates(scenes)
    return _emit(gate_rep)


def _cmd_preview(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames)
    base_dir = args.out or (paths.DATA / "jobs" / str(args.job) / "previews" / "v3")
    html_path = write_review_html(scenes, base_dir / "review_v3.html", base_dir / "review_v3.json")
    return _emit({"html": str(html_path), "review_items": sum(len(s.review_items) for s in scenes)})


def _cmd_export(args: argparse.Namespace) -> int:
    frames = _load_frames(args.job, _parse_frames(args.frames))
    orch = HarnessOrchestrator(job_id=args.job, mode=args.mode)
    scenes = orch.run_batch(frames)
    out = args.out or (paths.DATA / "jobs" / str(args.job) / "previews" / "v3" / "v3_export.json")
    CvatAdapter.export_draft_json(scenes, out)
    res = CvatAdapter().save_to_local_store(scenes)
    return _emit({"json": str(out), "local_store": res, "remote_pushed": False})
