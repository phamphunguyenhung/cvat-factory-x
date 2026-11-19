"""Ghi nhap track len CVAT an toan: dry-run mac dinh; backup; chi THEM; tu choi trung nguoi da co; doc lai; rollback theo id.

Ghi that chi xay ra khi yes=True VA client write_enabled=True (hai lop khoa).
"""
from __future__ import annotations

from datetime import datetime

from cvat_tools.core import transactions as tx
from cvat_tools.core.client import Cvat
from cvat_tools.tracking import geometry as core


def plan(cv: Cvat, job_id: int, draft: dict) -> dict:
    """Khong ghi de cong viec dang lam: nhap trung nguoi da co track (cung box >= 10 frame) bi bo qua."""
    existing = cv.annotations(job_id)
    stop = cv.job(job_id)["stop_frame"]
    have = [core.expand_cvat_track(t, stop) for t in existing["tracks"]]
    labels = {l["id"] for l in cv.labels(job_id)}
    errors, skipped, new = [], [], []
    for t in draft["tracks"]:
        if t["payload"]["label_id"] not in labels:
            errors.append(f"{t['name']}: label_id {t['payload']['label_id']} khong thuoc job nay (nhap cua job khac?)")
            continue
        mine = core.expand_cvat_track(t["payload"], stop)
        dup = next((i for i, h in enumerate(have) if core.same_person(mine, h)), None)
        if dup is not None:
            skipped.append(f"{t['name']}: da co track #{existing['tracks'][dup].get('id')} cung nguoi, bo qua")
        else:
            new.append(t["payload"])
    return {"existing": existing, "errors": errors, "skipped": skipped, "new": new}


def apply(cv: Cvat, job_id: int, draft: dict, yes: bool = False) -> dict:
    pl = plan(cv, job_id, draft)
    summary = {"job": job_id, "to_create": len(pl["new"]), "keyframes": sum(len(t["shapes"]) for t in pl["new"]),
               "errors": pl["errors"], "skipped": pl["skipped"], "before_tracks": len(pl["existing"]["tracks"]), "applied": False}
    if not yes or pl["errors"] or not pl["new"]:
        return summary
    _, bk = tx.backup_annotations(cv, job_id)
    cv.create_tracks(job_id, pl["new"])  # raise WriteNotAuthorized neu client chi doc
    after = cv.annotations(job_id)
    old_ids = {e["id"] for e in pl["existing"]["tracks"]}
    ok = len(after["tracks"]) == len(pl["existing"]["tracks"]) + len(pl["new"]) and all(
        s.get("source") == "manual" for t in after["tracks"] for s in t["shapes"] if "source" in s)
    run = tx.write_run_log(job_id, {"backup": bk, "created_ids": [t["id"] for t in after["tracks"] if t["id"] not in old_ids],
                                    "model": draft.get("model"), "tracker": draft.get("tracker"), "verified": ok,
                                    "at": datetime.now().isoformat(timespec="seconds")})
    return {**summary, "applied": True, "after_tracks": len(after["tracks"]), "verified": ok, "backup": bk, "run": run}


def rollback(cv: Cvat, run_file) -> dict:
    return tx.rollback_tracks(cv, run_file)
