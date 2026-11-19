"""Ghi an toan dung chung: dry-run mac dinh, backup truoc khi ghi, chi THEM, doc lai, rollback theo id.

Provenance: track ghi len CVAT co source='manual' (de nguoi duyet sua nhu nhan cua minh) NHUNG run log
(data/runs/*.json) luon ghi `provenance: ai_draft` + model/tracker, tach khoi nhan nguoi.
"""
from __future__ import annotations

import json
from datetime import datetime

from cvat_tools.core import paths
from cvat_tools.core.client import Cvat

BACKUPS = paths.DATA / "backups"
RUNS = paths.DATA / "runs"


def backup_annotations(cv: Cvat, job_id: int, tag: str = "") -> tuple[dict, str]:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    ann = cv.annotations(job_id)
    p = BACKUPS / f"job{job_id}{('_' + tag) if tag else ''}_{datetime.now():%Y%m%d_%H%M%S}.json"
    p.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf8")
    return ann, str(p)


def write_run_log(job_id: int, info: dict) -> str:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"job{job_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
    p.write_text(json.dumps({"job": job_id, "provenance": "ai_draft", **info}, ensure_ascii=False), encoding="utf8")
    return str(p)


def rollback_shapes(cv: Cvat, run_file, keep_ids=()) -> dict:
    """Xoa dung shape co id trong run log (`created_ids`); `keep_ids` = id can giu lai (vd nguoi da sua)."""
    run = json.loads(open(run_file, encoding="utf8").read())
    ids = set(run["created_ids"]) - set(keep_ids)
    mine = [s for s in cv.annotations(run["job"])["shapes"] if s["id"] in ids]
    if mine:
        cv.delete_shapes(run["job"], mine)
    return {"job": run["job"], "deleted": len(mine), "deleted_ids": [s["id"] for s in mine],
            "kept": sorted(set(run["created_ids"]) & set(keep_ids)), "remaining": len(cv.annotations(run["job"])["shapes"])}


def rollback_tracks(cv: Cvat, run_file) -> dict:
    run = json.loads(open(run_file, encoding="utf8").read())
    ann = cv.annotations(run["job"])
    mine = [t for t in ann["tracks"] if t["id"] in set(run["created_ids"])]
    cv.delete_tracks(run["job"], mine)
    return {"job": run["job"], "deleted": len(mine), "remaining": len(cv.annotations(run["job"])["tracks"])}
