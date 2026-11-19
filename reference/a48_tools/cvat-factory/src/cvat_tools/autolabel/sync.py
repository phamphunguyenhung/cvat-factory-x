"""Stage F: Safe Auto-Sync with Concurrent-Edit Conflict Detection, Read-Back & Auto-Rollback."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cvat_tools.autolabel.cache import sha256_obj
from cvat_tools.core.client import Cvat
from cvat_tools.pose import week2


class ConcurrentEditConflict(RuntimeError):
    """Raised when CVAT live state was modified concurrently after snapshot/proposal."""


def _shapes_signature(sh: dict[str, Any]) -> tuple[bool, bool, dict[int, tuple[Any, bool, bool]]]:
    pts = {
        int(e["label_id"]): (
            [round(float(e["points"][0]), 2), round(float(e["points"][1]), 2)],
            bool(e.get("occluded")),
            bool(e.get("outside")),
        )
        for e in (sh.get("elements") or [])
    }
    return (bool(sh.get("occluded")), bool(sh.get("outside")), pts)


def safe_auto_sync(
    job_id: int,
    eligible_results: list[dict[str, Any]],
    profile: str = "edufun",
    yes: bool = False,
) -> dict[str, Any]:
    """Dry-run by default; with yes=True, backup, PATCH minimal updates, read-back, auto-rollback on failure."""
    to_update = [sh for r in eligible_results for sh in (r.get("proposed_shapes") or []) if sh.get("id") is not None]
    to_create = [sh for r in eligible_results for sh in (r.get("proposed_shapes") or []) if sh.get("id") is None]
    before_expected = {int(sh["id"]): sh for r in eligible_results for sh in (r.get("before_shapes") or []) if sh.get("id") is not None}

    if not yes:
        return {
            "job_id": job_id,
            "mode": "dry_run",
            "written_to_cvat": False,
            "eligible_frames": [r["frame"] for r in eligible_results],
            "update_shapes_count": len(to_update),
            "create_shapes_count": len(to_create),
            "total_kp_changed": sum(len(r.get("changed_kpts") or []) for r in eligible_results),
        }

    if not to_update and not to_create:
        return {
            "job_id": job_id,
            "mode": "no_op",
            "written_to_cvat": False,
            "eligible_frames": [],
            "update_shapes_count": 0,
            "create_shapes_count": 0,
            "total_kp_changed": 0,
        }

    cv = Cvat.from_env(profile, write_enabled=True)
    before_live = cv.annotations(job_id)
    live_by_id = {int(s["id"]): s for s in (before_live.get("shapes") or []) if s.get("id") is not None}

    # 1) Concurrent edit check before writing
    for sid, exp_b in before_expected.items():
        live_sh = live_by_id.get(sid)
        if live_sh is None:
            raise ConcurrentEditConflict(f"Shape {sid} missing on CVAT live; concurrent deletion detected.")
        if _shapes_signature(live_sh) != _shapes_signature(exp_b):
            raise ConcurrentEditConflict(f"Concurrent CVAT edit detected on shape {sid} (frame {exp_b.get('frame')}). Aborting sync.")

    # 2) Backup before write
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = week2.job_folder(job_id) / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    backup_file = bdir / f"before_autolabel_sync_{ts}.json"
    backup_file.write_text(json.dumps(before_live, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3) Write with auto-rollback on partial write / network failure / read-back mismatch
    created_ids: list[int] = []
    rolled_back = False
    try:
        if to_update:
            cv.update_shapes(job_id, to_update)
        if to_create:
            for sh in to_create:
                sh["source"] = "manual"
                for el in sh.get("elements") or []:
                    el["source"] = "manual"
            cres = cv.create_shapes(job_id, to_create)
            created_ids = [int(s["id"]) for s in (cres.get("shapes") or []) if s.get("id") is not None]

        # 4) Read-back verification
        after_live = cv.annotations(job_id)
        after_by_id = {int(s["id"]): s for s in (after_live.get("shapes") or []) if s.get("id") is not None}
        mismatches = 0
        for exp_sh in to_update:
            got = after_by_id.get(int(exp_sh["id"]))
            if not got or _shapes_signature(got) != _shapes_signature(exp_sh):
                mismatches += 1
        if mismatches > 0:
            raise RuntimeError(f"Read-back verification failed: {mismatches} shape mismatches")
    except Exception as exc:
        # Attempt safe rollback of this run's shapes
        try:
            restore = [live_by_id[int(s["id"])] for s in to_update if int(s["id"]) in live_by_id]
            if restore:
                cv.update_shapes(job_id, restore)
            if created_ids:
                now_shapes = cv.annotations(job_id).get("shapes") or []
                del_objs = [s for s in now_shapes if int(s["id"]) in set(created_ids)]
                if del_objs:
                    cv.delete_shapes(job_id, del_objs)
            rolled_back = True
        except Exception:
            rolled_back = False
        raise RuntimeError(f"Sync failed ({exc}); rolled_back={rolled_back}, backup={backup_file}") from exc

    run_file = bdir / f"run_autolabel_sync_{ts}.json"
    run_file.write_text(json.dumps({
        "job_id": job_id,
        "timestamp": ts,
        "backup_file": str(backup_file),
        "updated_ids": [int(s["id"]) for s in to_update],
        "created_ids": created_ids,
        "eligible_frames": [r["frame"] for r in eligible_results],
        "readback_mismatches": 0,
        "before_hash": sha256_obj(before_live.get("shapes") or []),
        "after_hash": sha256_obj(after_live.get("shapes") or []),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "job_id": job_id,
        "mode": "committed",
        "written_to_cvat": True,
        "synced_frames": [r["frame"] for r in eligible_results],
        "update_shapes_count": len(to_update),
        "create_shapes_count": len(to_create),
        "total_kp_changed": sum(len(r.get("changed_kpts") or []) for r in eligible_results),
        "readback_mismatches": 0,
        "backup_file": str(backup_file),
        "run_file": str(run_file),
    }
