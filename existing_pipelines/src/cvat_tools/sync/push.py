"""Push: day shape nhap tu kho len CVAT. Dry-run mac dinh; chi THEM; dedupe; backup + run log; doc lai; cvat_id ghi nguoc vao kho.

Ghi that can ca `yes=True` va client `write_enabled=True` (hai khoa). Shape ghi len co source='manual' (quy tac nguoi dung);
provenance nam o kho + run log (`provenance: ai_draft`).
"""
from __future__ import annotations

from cvat_tools.core import transactions as tx
from cvat_tools.core.client import WriteNotAuthorized
from cvat_tools.core.shapestore import TOOL_STAGES, ShapeStore, vertex_deviation

DEDUPE_PX = 1.0


def _dup(a_pts, b_pts, tol) -> bool:
    return len(a_pts) == len(b_pts) and vertex_deviation(a_pts, b_pts) <= tol


def plan(cv, store: ShapeStore, job: int, stages=TOOL_STAGES, dedupe_px: float = DEDUPE_PX) -> dict:
    ids = {l["name"]: l["id"] for l in cv.labels(job)}
    existing = cv.annotations(job)["shapes"]
    todo, skipped, accepted = [], [], []
    for r in store.latest(job):
        if r["stage"] not in stages or r["status"] != "active" or r["cvat_id"] is not None:
            continue
        if r["label"] not in ids:
            skipped.append({"row": r["id"], "reason": "label_khong_co_tren_job", "label": r["label"]})
            continue
        lid = ids[r["label"]]
        hit = next((s for s in existing if s["frame"] == r["frame"] and s["label_id"] == lid and s["type"] == r["shape_type"]
                    and _dup(s["points"], r["points"], dedupe_px)), None)
        if hit:
            skipped.append({"row": r["id"], "reason": "trung_voi_cvat", "cvat_id": hit["id"]})
            continue
        if any(x["frame"] == r["frame"] and x["label_id"] == lid and x["type"] == r["shape_type"] and _dup(x["points"], r["points"], dedupe_px)
               for x in accepted):
            skipped.append({"row": r["id"], "reason": "trung_trong_lo"})
            continue
        payload = {"type": r["shape_type"], "frame": r["frame"], "label_id": lid, "points": r["points"], "z_order": r["z_order"],
                   "occluded": r["occluded"], "source": "manual", "group": 0, "rotation": 0,
                   "attributes": r["attributes"] if isinstance(r["attributes"], list) else []}
        accepted.append(payload)
        todo.append({"row": r["id"], "payload": payload})
    return {"job": job, "to_add": todo, "skipped": skipped, "n_existing": len(existing)}


def push(cv, store: ShapeStore, job: int, yes: bool = False, stages=TOOL_STAGES, dedupe_px: float = DEDUPE_PX) -> dict:
    p = plan(cv, store, job, stages, dedupe_px)
    rep = {"job": job, "dry_run": not yes, "n_to_add": len(p["to_add"]), "skipped": p["skipped"], "n_existing": p["n_existing"]}
    if not yes or not p["to_add"]:
        return rep
    if not cv.write_enabled:
        raise WriteNotAuthorized("push --yes can client write_enabled=True.")
    before, backup = tx.backup_annotations(cv, job, "shapepush")
    before_ids = {s["id"] for s in before["shapes"]}
    cv.create_shapes(job, [t["payload"] for t in p["to_add"]])
    after = cv.annotations(job)["shapes"]
    new = [s for s in after if s["id"] not in before_ids]          # created_ids suy bang so id truoc/sau
    rows, free = {}, list(new)
    for t in p["to_add"]:
        pl = t["payload"]
        cand = [s for s in free if s["frame"] == pl["frame"] and s["label_id"] == pl["label_id"] and s["type"] == pl["type"]]
        if cand:
            m = min(cand, key=lambda s: vertex_deviation(s["points"], pl["points"]) if s["points"] else 1e9)
            free.remove(m)
            rows[m["id"]] = t["row"]
    verified = len(after) == len(before_ids) + len(p["to_add"]) and len(rows) == len(p["to_add"])
    run = tx.write_run_log(job, {"kind": "shapes", "created_ids": [s["id"] for s in new], "rows": {str(k): v for k, v in rows.items()},
                                 "backup": backup, "n_before": len(before_ids), "n_after": len(after), "verified": verified})
    for cid, row in rows.items():
        store.set_cvat_id(row, cid, "push")
    store.log("push", job, {"run": run, "added": len(rows), "verified": verified})
    return {**rep, "dry_run": False, "added": len(rows), "n_after": len(after), "verified": verified, "run_log": run, "backup": backup}


def rollback(cv, store: ShapeStore, run_file) -> dict:
    """Xoa dung shape trong run log; GIU shape nguoi da sua sau push (dong human_final trong kho)."""
    import json
    run = json.loads(open(run_file, encoding="utf8").read())
    job = run["job"]
    edited = {cid for cid in run["created_ids"] if any(r["stage"] == "human_final" for r in store.by_cvat_id(job, cid))}
    res = tx.rollback_shapes(cv, run_file, keep_ids=edited)
    for cid in res["deleted_ids"]:
        for r in store.by_cvat_id(job, cid):
            if str(cid) in run.get("rows", {}) and r["id"] == run["rows"][str(cid)]:
                store.set_cvat_id(r["id"], None, "rollback")
    store.log("rollback", job, {"run": str(run_file), **{k: res[k] for k in ("deleted", "kept")}})
    return res
