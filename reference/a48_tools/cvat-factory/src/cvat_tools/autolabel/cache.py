"""Stage A: Incremental CVAT Snapshot, Image SHA256 Cache, Proposal Cache & Checkpoint."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cvat_tools.core import paths
from cvat_tools.core.client import Cvat


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_dir(job_id: int) -> Path:
    d = paths.WORKSPACE / "data" / "autolabel_cache" / f"job_{job_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_or_fetch_snapshot(
    job_id: int,
    profile: str = "edufun",
    offline: bool = False,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Incremental CVAT snapshot + image cache. Only downloads missing images or changed annotations."""
    cdir = cache_dir(job_id)
    state_p = cdir / "snapshot_state.json"
    cached = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}

    fdir = paths.job_dir(profile, job_id) / "frames"
    fdir.mkdir(parents=True, exist_ok=True)

    if offline:
        if not cached:
            raise FileNotFoundError(f"No cached snapshot for job {job_id} at {state_p}")
        return cached

    cv = Cvat.from_env(profile, write_enabled=False)
    ann = cv.annotations(job_id)
    ann_hash = sha256_obj(ann.get("shapes") or [])

    if not force_refresh and cached.get("ann_hash") == ann_hash and cached.get("meta") and cached.get("labels"):
        # Fast path: annotations unchanged; only verify missing local frames
        meta = cached["meta"]
        labels = cached["labels"]
        api_calls = 1  # only cv.annotations
    else:
        meta = cached.get("meta") or cv.meta(job_id)
        labels = cached.get("labels") or cv.labels(job_id)
        api_calls = 3 if not cached.get("meta") else 1

    # Incremental image cache + SHA256
    img_hashes: dict[str, str] = dict(cached.get("image_hashes") or {})
    downloaded = 0
    for i, fm in enumerate(meta.get("frames") or []):
        fidx = int(meta["start_frame"]) + i
        name = Path(fm["name"]).name
        fp = fdir / name
        if not fp.exists():
            fp.write_bytes(cv.frame(job_id, fidx))
            downloaded += 1
            api_calls += 1
            img_hashes[name] = sha256_bytes(fp.read_bytes())
        elif name not in img_hashes:
            img_hashes[name] = sha256_bytes(fp.read_bytes())

    state = {
        "job_id": job_id,
        "profile": profile,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "ann_hash": ann_hash,
        "version": ann.get("version", 0),
        "api_calls": api_calls,
        "downloaded_frames": downloaded,
        "meta": meta,
        "labels": labels,
        "annotations": ann,
        "image_hashes": img_hashes,
    }
    state_p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return state


def load_checkpoint(job_id: int) -> dict[str, Any]:
    cp = cache_dir(job_id) / "checkpoint.json"
    return json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {"completed_frames": {}, "last_stage": None}


def save_checkpoint(job_id: int, data: dict[str, Any]) -> None:
    cp = cache_dir(job_id) / "checkpoint.json"
    cp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
