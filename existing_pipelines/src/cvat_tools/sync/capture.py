"""Capture: doc annotation job (CHI DOC) va ghep voi kho de suy ra human_final.

- shape tool da dong bo (cvat_id) nhung nguoi sua toa do/nhan/z/occluded -> dong human_final moi (parent = ban truoc).
- shape CVAT chua tung thay (cvat_id la) -> human_final nguon nguoi, khong parent.
- shape tool da dong bo nhung khong con tren CVAT -> dong human_final status='rejected'.
Khong bao gio goi lenh ghi cua client.
"""
from __future__ import annotations

from cvat_tools.core.shapestore import TOOL_STAGES, ShapeStore, vertex_deviation

TOL_PX = 0.5


def _same_points(a, b, tol: float) -> bool:
    return len(a) == len(b) and vertex_deviation(a, b) <= tol


def capture(cv, store: ShapeStore, job: int, tol: float = TOL_PX) -> dict:
    names = {l["id"]: l["name"] for l in cv.labels(job)}
    cur = {s["id"]: s for s in cv.annotations(job)["shapes"]}
    heads: dict[int, dict] = {}
    for r in store.list(job=job, synced=True):          # theo id tang dan -> ban cuoi thang
        heads[r["cvat_id"]] = r
    res = {"job": job, "kept": 0, "edited": 0, "rejected": 0, "new_human": 0, "unchanged_human": 0}

    for cid, head in heads.items():
        if head["status"] == "rejected":
            continue
        s = cur.get(cid)
        if s is None:
            store.add(job, head["frame"], head["label"], head["shape_type"], head["points"], "human_final", z_order=head["z_order"],
                      occluded=head["occluded"], attributes=head["attributes"], parent_id=head["id"], status="rejected",
                      cvat_id=cid, provenance={"source": "human", "event": "deleted_in_cvat"})
            res["rejected"] += 1
            continue
        label = names.get(s["label_id"], f"label#{s['label_id']}")
        changed = (label != head["label"] or not _same_points(s["points"], head["points"], tol)
                   or int(s.get("z_order", 0)) != head["z_order"] or bool(s.get("occluded", False)) != head["occluded"])
        if changed:
            store.add(job, s["frame"], label, s["type"], s["points"], "human_final", z_order=s.get("z_order", 0),
                      occluded=s.get("occluded", False), attributes=s.get("attributes", []), parent_id=head["id"], cvat_id=cid,
                      provenance={"source": "human", "event": "edited_in_cvat", "cvat_source": s.get("source")})
            res["edited"] += 1
        elif head["stage"] in TOOL_STAGES:
            res["kept"] += 1
        else:
            res["unchanged_human"] += 1

    for cid, s in cur.items():
        if cid in heads:
            continue
        label = names.get(s["label_id"], f"label#{s['label_id']}")
        store.add(job, s["frame"], label, s["type"], s["points"], "human_final", z_order=s.get("z_order", 0),
                  occluded=s.get("occluded", False), attributes=s.get("attributes", []), cvat_id=cid,
                  provenance={"source": "human", "event": "new_in_cvat", "cvat_source": s.get("source")})
        res["new_human"] += 1
    store.log("capture", job, res)
    return res


def stats(store: ShapeStore, job: int) -> list[dict]:
    """Thong ke theo label tren shape tool da dong bo: ti le giu nguyen/sua/xoa, do lech dinh trung binh (px, chi shape bi sua toa do)."""
    acc: dict[str, dict] = {}

    def a(label):
        return acc.setdefault(label, {"label": label, "n": 0, "kept": 0, "edited": 0, "deleted": 0, "devs": [], "new_human": 0})

    for r in store.list(job=job, synced=True):
        if r["stage"] in TOOL_STAGES:
            d = store.descendants(r["id"])
            x = a(r["label"])
            x["n"] += 1
            if any(c["status"] == "rejected" for c in d):
                x["deleted"] += 1
                continue
            hum = [c for c in d if c["stage"] == "human_final"]
            if hum:
                x["edited"] += 1
                if hum[-1]["points"] != r["points"]:
                    x["devs"].append(vertex_deviation(r["points"], hum[-1]["points"]))
            else:
                x["kept"] += 1
        elif r["parent_id"] is None and r["stage"] == "human_final":
            a(r["label"])["new_human"] += 1
    rows = []
    for x in sorted(acc.values(), key=lambda v: v["label"]):
        n = max(1, x["n"])
        devs = x.pop("devs")
        rows.append({**x, "kept_pct": 100 * x["kept"] / n, "edited_pct": 100 * x["edited"] / n, "deleted_pct": 100 * x["deleted"] / n,
                     "mean_vertex_dev_px": sum(devs) / len(devs) if devs else None})
    return rows


def format_stats(rows: list[dict]) -> str:
    out = [f"{'label':<16}{'tool':>5}{'giu%':>7}{'sua%':>7}{'xoa%':>7}{'lech dinh px':>14}{'nguoi ve moi':>14}"]
    for r in rows:
        dev = f"{r['mean_vertex_dev_px']:.2f}" if r["mean_vertex_dev_px"] is not None else "-"
        out.append(f"{r['label']:<16}{r['n']:>5}{r['kept_pct']:>7.0f}{r['edited_pct']:>7.0f}{r['deleted_pct']:>7.0f}{dev:>14}{r['new_human']:>14}")
    return "\n".join(out)
