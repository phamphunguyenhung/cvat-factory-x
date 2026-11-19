"""CVAT task/job discovery, schema/model inference, local run inspection, and resume-first selection."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from cvat_tools.core import paths
from cvat_tools.factory.cvat.gateway import CvatGateway
from cvat_tools.factory.discovery.models import (
    CoverageSummary,
    DiscoveredJob,
    DiscoveryResult,
    LabelInfo,
    LocalRunStatus,
)
from cvat_tools.job.discover import frame_ids as _frame_ids

_YOLO_TRAFFIC_LABELS = {
    "person",
    "pedestrian",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
    "rider",
}
_SEG_LABELS = {
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "vegetation",
    "terrain",
    "sky",
}
_FACE50_HINTS = {"nose_tip", "left_brow", "right_brow", "upper_lip", "lower_lip", "chin"}
_POSE17_HINTS = {"left_shoulder", "right_shoulder", "left_hip", "right_knee", "left_ankle"}


def _extract_assignee(job: dict[str, Any]) -> str | None:
    assignee = job.get("assignee")
    if isinstance(assignee, dict):
        return assignee.get("username") or assignee.get("name")
    if isinstance(assignee, str):
        return assignee or None
    return None


def _parse_labels(raw_labels: Iterable[dict[str, Any]]) -> tuple[LabelInfo, ...]:
    out: list[LabelInfo] = []
    for item in raw_labels:
        out.append(
            LabelInfo(
                id=int(item["id"]),
                name=str(item["name"]),
                type=str(item.get("type") or "any"),
                attributes=tuple(item.get("attributes") or ()),
                sublabels=tuple(item.get("sublabels") or ()),
            )
        )
    return tuple(out)


def infer_annotation_type(
    labels: Iterable[LabelInfo],
    shapes: Iterable[dict[str, Any]] = (),
    tracks: Iterable[dict[str, Any]] = (),
) -> str:
    """Infer primary annotation type from CVAT label schema and existing shapes/tracks."""
    label_types = {l.type.lower() for l in labels if l.type and l.type.lower() != "any"}
    has_sublabels = any(bool(l.sublabels) for l in labels)
    shape_types = {str(s.get("type") or "").lower() for s in shapes if s.get("type")}
    track_list = list(tracks)
    for t in track_list:
        for s in t.get("shapes", ()):
            if s.get("type"):
                shape_types.add(str(s["type"]).lower())

    if has_sublabels or "skeleton" in label_types or "skeleton" in shape_types:
        return "skeleton"
    if track_list and ("rectangle" in shape_types or "rectangle" in label_types or not label_types):
        return "tracking"

    has_bbox = "rectangle" in label_types or "rectangle" in shape_types
    has_seg = bool(({"polygon", "mask"} & label_types) or ({"polygon", "mask"} & shape_types))
    if has_bbox and has_seg:
        return "mixed"
    if has_seg:
        return "segmentation"
    if has_bbox:
        return "bbox"
    if "polyline" in label_types or "polyline" in shape_types:
        return "polyline"
    if "points" in label_types or "points" in shape_types:
        return "points"
    return "unknown"


def recommend_suitable_model(labels: Iterable[LabelInfo], annotation_type: str) -> str | None:
    """Recommend a suitable model/adapter purely from schema metadata (no ML imports)."""
    label_list = list(labels)
    names = {l.name.strip().lower() for l in label_list}
    sublabel_names = {
        str(sl.get("name") or "").strip().lower()
        for l in label_list
        for sl in l.sublabels
    }
    max_sublabels = max((len(l.sublabels) for l in label_list), default=0)

    if annotation_type == "skeleton":
        if max_sublabels >= 40 or (sublabel_names & _FACE50_HINTS):
            return "face50_engine"
        if max_sublabels == 17 or (sublabel_names & _POSE17_HINTS):
            return "yolo26-pose"
        return "skeleton_custom"

    if annotation_type == "segmentation":
        if any(n.startswith("lane/") or n.startswith("area/") for n in names):
            return "lane_detectors+segformer"
        if names & _SEG_LABELS:
            return "segformer/mask2former"
        return "sam/mask2former"

    if annotation_type in ("bbox", "tracking"):
        suffix = "+bytetrack" if annotation_type == "tracking" else ""
        if names & _YOLO_TRAFFIC_LABELS:
            return f"yolo26s{suffix}"
        return f"yolo26s{suffix}"

    if annotation_type == "mixed":
        return "multi_adapter(yolo26s+segformer)"

    if names & _YOLO_TRAFFIC_LABELS:
        return "yolo26s"
    if names & _SEG_LABELS:
        return "segformer/mask2former"
    return None


def inspect_local_run_status(
    job_id: int,
    *,
    profile: str = "local",
    data_dir: Path | None = None,
    db_path: Path | None = None,
) -> LocalRunStatus:
    """Inspect local checkpoints and SQLite draft store for an existing/resumable run."""
    base_data = Path(data_dir) if data_dir is not None else paths.DATA
    jdir = (
        base_data / "jobs" / (str(job_id) if profile == "local" else f"{profile}/{job_id}")
    )

    for fname in ("run_state.json", "checkpoint.json", "state.json"):
        cand = jdir / fname
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                status = str(data.get("status") or "in_progress").lower()
                resumable = bool(data.get("resumable", status in {"in_progress", "paused", "failed"}))
                return LocalRunStatus(
                    job_id=job_id,
                    run_id=data.get("run_id"),
                    status=status,
                    stage=data.get("stage"),
                    resumable=resumable,
                    updated_at=data.get("updated_at"),
                    shapes_count=int(data.get("shapes_count", 0)),
                    source_path=str(cand),
                )
            except Exception:
                pass

    sqlite_file = Path(db_path) if db_path is not None else (base_data / "store" / "shapes.sqlite")
    if sqlite_file.exists():
        try:
            conn = sqlite3.connect(str(sqlite_file))
            try:
                row = conn.execute(
                    "SELECT COUNT(*), MAX(created_at) FROM shapes WHERE job = ? AND status = 'active'",
                    (job_id,),
                ).fetchone()
                count = int(row[0]) if row and row[0] else 0
                updated = str(row[1]) if row and row[1] else None
                if count > 0:
                    return LocalRunStatus(
                        job_id=job_id,
                        run_id=None,
                        status="in_progress",
                        stage="local_draft",
                        resumable=True,
                        updated_at=updated,
                        shapes_count=count,
                        source_path=str(sqlite_file),
                    )
            finally:
                conn.close()
        except Exception:
            pass

    return LocalRunStatus(job_id=job_id, status="none", resumable=False)


def _compute_coverage(
    labels: tuple[LabelInfo, ...],
    total_frames: int,
    ann: dict[str, Any] | None,
) -> CoverageSummary:
    if not ann:
        return CoverageSummary(
            annotated_frames=0,
            total_frames=total_frames,
            coverage_ratio=0.0,
        )
    by_id = {l.id: l.name for l in labels}
    annotated_frames: set[int] = set()
    per_label: dict[str, int] = defaultdict(int)
    shapes = ann.get("shapes", []) or []
    tracks = ann.get("tracks", []) or []

    for s in shapes:
        if s.get("frame") is not None:
            annotated_frames.add(int(s["frame"]))
        lname = by_id.get(int(s.get("label_id", -1)), f"#{s.get('label_id')}")
        per_label[lname] += 1

    for t in tracks:
        lname = by_id.get(int(t.get("label_id", -1)), f"#{t.get('label_id')}")
        for s in t.get("shapes", []) or []:
            if not s.get("outside"):
                if s.get("frame") is not None:
                    annotated_frames.add(int(s["frame"]))
                per_label[lname] += 1

    ratio = len(annotated_frames) / total_frames if total_frames > 0 else 0.0
    return CoverageSummary(
        annotated_frames=len(annotated_frames),
        total_frames=total_frames,
        coverage_ratio=round(ratio, 4),
        shapes_count=len(shapes),
        tracks_count=len(tracks),
        per_label_counts=dict(per_label),
    )


def discover_job(
    gateway: CvatGateway,
    job_id: int,
    *,
    profile: str = "local",
    include_coverage: bool = True,
    task_cache: dict[int, dict[str, Any]] | None = None,
    job_obj: dict[str, Any] | None = None,
    data_dir: Path | None = None,
    db_path: Path | None = None,
) -> DiscoveredJob:
    """Discover full metadata, schema, coverage, model recommendation, and local run status for one job."""
    job = job_obj if job_obj is not None else gateway.job(job_id)
    task_id = int(job["task_id"])
    task: dict[str, Any] = {}
    if task_cache is not None and task_id in task_cache:
        task = task_cache[task_id]
    else:
        try:
            task = gateway.task(task_id)
        except Exception:
            task = {"id": task_id, "name": f"task-{task_id}"}

    raw_labels = gateway.labels(job_id)
    labels = _parse_labels(raw_labels)

    meta: dict[str, Any] = {}
    try:
        meta = gateway.meta(job_id)
    except Exception:
        meta = {}

    fids = _frame_ids(job, meta) if ("start_frame" in job and "stop_frame" in job) else []
    frame_count = len(fids) if fids else int(meta.get("size") or 0)
    frames = meta.get("frames") or []
    frame_wh: tuple[int, int] | None = None
    if frames and frames[0].get("width") and frames[0].get("height"):
        frame_wh = (int(frames[0]["width"]), int(frames[0]["height"]))

    ann: dict[str, Any] | None = None
    if include_coverage:
        try:
            ann = gateway.annotations(job_id)
        except Exception:
            ann = None

    ann_type = infer_annotation_type(
        labels,
        shapes=(ann or {}).get("shapes", ()),
        tracks=(ann or {}).get("tracks", ()),
    )
    coverage = _compute_coverage(labels, frame_count, ann)
    suitable = recommend_suitable_model(labels, ann_type)
    local_status = inspect_local_run_status(
        job_id, profile=profile, data_dir=data_dir, db_path=db_path
    )

    return DiscoveredJob(
        job_id=int(job["id"]) if "id" in job else job_id,
        task_id=task_id,
        task_name=str(task.get("name") or f"task-{task_id}"),
        status=str(job.get("state") or job.get("status") or "unknown"),
        stage=str(job.get("stage") or "unknown"),
        assignee=_extract_assignee(job),
        frame_count=frame_count,
        frame_wh=frame_wh,
        annotation_type=ann_type,
        labels=labels,
        coverage=coverage,
        suitable_model=suitable,
        last_local_run_status=local_status,
    )


def rank_jobs_for_resume(
    jobs: Iterable[DiscoveredJob],
    username: str | None = None,
) -> list[DiscoveredJob]:
    """Rank jobs with resume-first priority: resumable local runs -> assigned active -> active -> others."""

    def _key(j: DiscoveredJob) -> tuple[int, int, int, int]:
        resumable_rank = 0 if j.last_local_run_status.resumable else 1
        assigned_rank = 0 if (username and j.assignee == username) else 1
        active_rank = 0 if j.status.lower() in {"new", "in progress", "annotation", "validation"} else 1
        return (resumable_rank, assigned_rank, active_rank, -j.job_id)

    return sorted(jobs, key=_key)


def discover_jobs(
    gateway: CvatGateway | None = None,
    *,
    host: str | None = None,
    profile: str = "local",
    task_id: int | None = None,
    status: str | None = None,
    mine: bool = False,
    limit: int | None = None,
    include_coverage: bool = False,
    username: str | None = None,
    data_dir: Path | None = None,
    db_path: Path | None = None,
) -> DiscoveryResult:
    """Discover accessible CVAT jobs and rank them for resume-first selection."""
    if gateway is None:
        from cvat_tools.factory.auth.credentials import discover_credentials

        creds = discover_credentials(profile=profile, host=host, verify=False, interactive=False)
        gateway = creds.to_gateway(write_enabled=False)

    if username is None:
        try:
            username = gateway.me().get("username")
        except Exception:
            username = None

    task_cache: dict[int, dict[str, Any]] = {}
    try:
        for t in gateway.tasks():
            if "id" in t:
                task_cache[int(t["id"])] = t
    except Exception:
        pass

    raw_jobs = gateway.jobs()
    if task_id is not None:
        raw_jobs = [j for j in raw_jobs if int(j.get("task_id", -1)) == int(task_id)]
    if status is not None:
        want = status.strip().lower()
        raw_jobs = [j for j in raw_jobs if str(j.get("state") or j.get("status") or "").strip().lower() == want]
    if mine and username:
        raw_jobs = [j for j in raw_jobs if _extract_assignee(j) == username]
    if limit is not None:
        raw_jobs = raw_jobs[:limit]

    discovered = [
        discover_job(
            gateway,
            int(j["id"]),
            profile=profile,
            include_coverage=include_coverage,
            task_cache=task_cache,
            job_obj=j,
            data_dir=data_dir,
            db_path=db_path,
        )
        for j in raw_jobs
    ]
    ranked = rank_jobs_for_resume(discovered, username=username)
    recommended = ranked[0].job_id if ranked else None
    return DiscoveryResult(
        host=gateway.host,
        org=gateway.org,
        username=username,
        jobs=tuple(ranked),
        recommended_resume_job_id=recommended,
    )


def inspect_job(
    job_id: int,
    *,
    host: str | None = None,
    offline: bool = False,
    profile: str = "local",
    gateway: CvatGateway | None = None,
    data_dir: Path | None = None,
    db_path: Path | None = None,
) -> DiscoveredJob | LocalRunStatus:
    """Inspect a single CVAT job online, or return its local run status when offline=True."""
    if offline:
        return inspect_local_run_status(job_id, profile=profile, data_dir=data_dir, db_path=db_path)
    if gateway is None:
        from cvat_tools.factory.auth.credentials import discover_credentials

        creds = discover_credentials(profile=profile, host=host, verify=False, interactive=False)
        gateway = creds.to_gateway(write_enabled=False)
    return discover_job(
        gateway,
        job_id,
        profile=profile,
        include_coverage=True,
        data_dir=data_dir,
        db_path=db_path,
    )


def resolve_default_job(
    *,
    host: str | None = None,
    profile: str = "local",
    gateway: CvatGateway | None = None,
    data_dir: Path | None = None,
    db_path: Path | None = None,
) -> int | None:
    """Return the top resume-first job ID (local resumable run first, then online discovery)."""
    base_data = Path(data_dir) if data_dir is not None else paths.DATA
    jobs_root = base_data / "jobs" if profile == "local" else (base_data / "jobs" / profile)
    if jobs_root.exists():
        for child in sorted(jobs_root.iterdir()):
            if child.is_dir() and child.name.isdigit():
                st = inspect_local_run_status(int(child.name), profile=profile, data_dir=data_dir, db_path=db_path)
                if st.resumable:
                    return int(child.name)
    try:
        res = discover_jobs(gateway=gateway, host=host, profile=profile, limit=10, data_dir=data_dir, db_path=db_path)
        return res.recommended_resume_job_id
    except Exception:
        return None


def select_job(
    jobs: Iterable[DiscoveredJob],
    *,
    job_id: int | None = None,
    username: str | None = None,
    interactive: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> DiscoveredJob | None:
    """Select a job explicitly, by resume-first ranking, or via interactive menu."""
    ranked = rank_jobs_for_resume(jobs, username=username)
    if not ranked:
        return None

    if job_id is not None:
        for j in ranked:
            if j.job_id == job_id:
                return j
        raise KeyError(f"Job {job_id} not found in discovered jobs.")

    if not interactive:
        return ranked[0]

    output_fn("Available CVAT jobs (resume-first order):")
    for idx, j in enumerate(ranked, start=1):
        tags: list[str] = []
        if j.last_local_run_status.resumable:
            tags.append(f"RESUME:{j.last_local_run_status.status}")
        if username and j.assignee == username:
            tags.append("ASSIGNED")
        tag_str = f" [{' '.join(tags)}]" if tags else ""
        output_fn(
            f"  [{idx}] job={j.job_id} task={j.task_id} ({j.task_name}) "
            f"type={j.annotation_type} frames={j.frame_count} model={j.suitable_model or '-'}{tag_str}"
        )

    choice = input_fn("Select job number [1]: ").strip()
    if not choice:
        return ranked[0]
    idx = int(choice)
    if idx < 1 or idx > len(ranked):
        raise ValueError(f"Selection {idx} out of range 1..{len(ranked)}")
    return ranked[idx - 1]
