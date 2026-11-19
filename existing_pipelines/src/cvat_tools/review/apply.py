"""Ap dung lenh sua (edits.py) len shape nhap + bam canh tat dinh (geomrefine) + validate lai + nhat ky.

KHONG ghi de nhap goc: save_result chi ghi ban sua + nhat ky ra thu muc rieng. Khong ghi CVAT.
"""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import split as _split
from shapely.ops import unary_union

from cvat_tools.review import edits as E
from cvat_tools.review import validators as V
from cvat_tools.segmentation import geomrefine as G


def _r(v) -> float:
    return round(float(v), 2)


def _clamp(x, y, wh):
    if wh:
        x, y = min(max(x, 0.0), wh[0]), min(max(y, 0.0), wh[1])
    return _r(x), _r(y)


def _set_vertices(sh, vs):
    if sh["type"] == "rectangle":
        xs, ys = [v[0] for v in vs], [v[1] for v in vs]
        sh["points"] = [_r(min(xs)), _r(min(ys)), _r(max(xs)), _r(max(ys))]
    else:
        sh["points"] = [_r(c) for v in vs for c in v]


# ------------------------------------------------------------------ bam canh
def _snap_rect(sh, img, band):
    """Moi canh hop tim vi tri co gradient trung binh manh nhat trong +-band."""
    import cv2
    H, W = img.shape[:2]
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    x0, y0, x1, y1 = sh["points"][:4]
    b = int(round(band))

    def best(coord, lo, hi, vertical):  # canh dung (x=coord) hay ngang (y=coord)
        lo, hi = int(max(0, lo)), int(min((H if vertical else W) - 1, hi))
        if hi <= lo:
            return coord, 0.0
        scores = []
        cand = list(range(max(0, int(round(coord)) - b), min((W if vertical else H) - 1, int(round(coord)) + b) + 1))
        for c in cand:
            scores.append(float(np.abs(gx[lo:hi + 1, c]).mean() if vertical else np.abs(gy[c, lo:hi + 1]).mean()))
        if not scores or max(scores) < 6.0 * 8:  # Sobel ksize=3 chua chuan hoa /8
            return coord, 0.0
        return float(cand[int(np.argmax(scores))]), max(scores) / 8

    nx0, _ = best(x0, y0, y1, True)
    nx1, _ = best(x1, y0, y1, True)
    ny0, _ = best(y0, x0, x1, False)
    ny1, _ = best(y1, x0, x1, False)
    shift = float(np.mean([abs(nx0 - x0), abs(nx1 - x1), abs(ny0 - y0), abs(ny1 - y1)]))
    sh["points"] = [_r(min(nx0, nx1)), _r(min(ny0, ny1)), _r(max(nx0, nx1)), _r(max(ny0, ny1))]
    return shift


def snap_shape(sh, img, band, rs, max_pts=None) -> dict:
    """Bam canh trong dai hep. Tra {'band','mean_shift','status'}; khong lam hong shape (invalid -> bo qua)."""
    if img is None:
        return {"band": band, "status": "no_image", "mean_shift": 0.0}
    if sh["type"] == "rectangle":
        return {"band": band, "status": "ok", "mean_shift": round(_snap_rect(sh, img, band), 2)}
    p = V.poly_of(sh)
    if p.is_empty or not p.is_valid:
        return {"band": band, "status": "skip_invalid", "mean_shift": 0.0}
    wd = 2.0 * p.area / max(p.length, 1e-6)
    k = min(band, max(0.75, 0.5 * wd)) if V.is_narrow(sh["label"], rs) else band
    q, shift = G.snap_to_edges(p, img, k=k)
    q = G.valid_polygon(q.simplify(0.6, preserve_topology=True)) or q
    n0 = len(V.vertices(sh))
    cap = rs["max_vertices"].get(V.group_of(sh["label"], rs)) or max(n0 * 2, 12)
    if len(q.exterior.coords) - 1 > cap:
        q = G._cap_points(q, 0.6, cap)
    if q is None or not q.is_valid or q.is_empty:
        return {"band": band, "status": "skip_invalid", "mean_shift": 0.0}
    sh["points"] = [_r(c) for xy in list(q.exterior.coords)[:-1] for c in xy]
    return {"band": round(k, 2), "status": "ok", "mean_shift": round(float(shift), 2)}


# ------------------------------------------------------------------ lenh
def _next_id(shapes: dict, base: str) -> str:
    n = 1
    while f"{base}.{n}" in shapes:
        n += 1
    return f"{base}.{n}"


def _apply_one(cmd, shapes: dict, img, wh, rs) -> tuple[str, list[str]]:
    """Doi shapes tai cho. Tra (detail, shape_ids_can_snap)."""
    op = cmd["op"]
    if op == "delete_shape":
        del shapes[cmd["shape"]]
        return "deleted", []
    if op == "merge":
        ids = cmd["shapes"]
        first = shapes[ids[0]]
        if first["type"] == "rectangle":
            vs = [v for i in ids for v in V.vertices(shapes[i])]
            _set_vertices(first, vs)
        else:
            u = unary_union([V._valid_geom(shapes[i]) for i in ids])
            if u.geom_type != "Polygon":
                u = u.convex_hull if not hasattr(u, "geoms") else unary_union([g.buffer(2) for g in u.geoms]).buffer(-2)
            if u.geom_type != "Polygon":
                u = max(u.geoms, key=lambda g: g.area)
            first["points"] = [_r(c) for xy in list(u.exterior.coords)[:-1] for c in xy]
        if "label" in cmd:
            first["label"] = V.canon_label(cmd["label"], rs)
            first["z_order"] = V.expected_z(first["label"], rs) if V.expected_z(first["label"], rs) is not None else first["z_order"]
        for i in ids[1:]:
            del shapes[i]
        return f"merged {len(ids)} -> {ids[0]}", [ids[0]]
    sid = cmd["shape"]
    sh = shapes[sid]
    if op == "relabel":
        old = sh["label"]
        sh["label"] = V.canon_label(cmd["label"], rs)
        ez = V.expected_z(sh["label"], rs)
        if ez is not None:
            sh["z_order"] = ez
        return f"{old} -> {sh['label']}", []
    if op == "set_flag":
        sh[cmd["flag"]] = bool(cmd.get("value", True))
        return f"{cmd['flag']}={sh[cmd['flag']]}", []
    if op == "split":
        p = V._valid_geom(sh)
        (ax, ay), (bx, by) = cmd["line"]
        d = math.hypot(bx - ax, by - ay) or 1.0
        ext = 10000.0 / d  # keo dai de cat het
        ln = LineString([(ax - (bx - ax) * ext, ay - (by - ay) * ext), (bx + (bx - ax) * ext, by + (by - ay) * ext)])
        parts = [g for g in _split(p, ln).geoms if g.area >= 1.0]
        if len(parts) < 2:
            raise ValueError("duong cat khong chia duoc polygon")
        parts.sort(key=lambda g: -g.area)
        out = []
        for k, g in enumerate(parts):
            tgt = sh if k == 0 else copy.deepcopy(sh)
            tgt["points"] = [_r(c) for xy in list(g.exterior.coords)[:-1] for c in xy]
            if k:
                tgt["id"] = _next_id(shapes, sid)
                shapes[tgt["id"]] = tgt
            out.append(tgt["id"])
        return f"split -> {out}", out
    if op == "snap_to_edge":
        return "snap", [sid]
    vs = [tuple(v) for v in V.vertices(sh)]
    if op == "move_vertex":
        i = cmd["idx"]
        x, y = vs[i]
        nx, ny = (cmd["to"] if "to" in cmd else (x + cmd.get("dx", 0), y + cmd.get("dy", 0)))
        vs[i] = _clamp(nx, ny, wh)
        if sh["type"] == "rectangle":  # goc doi: keo canh tuong ung, giu hinh chu nhat
            x0, y0, x1, y1 = sh["points"][:4]
            nx, ny = vs[i]
            if i in (0, 3):
                x0 = nx
            else:
                x1 = nx
            if i in (0, 1):
                y0 = ny
            else:
                y1 = ny
            sh["points"] = [_r(min(x0, x1)), _r(min(y0, y1)), _r(max(x0, x1)), _r(max(y0, y1))]
            return f"corner {i} -> ({nx},{ny})", [sid] if cmd.get("snap_band") else []
        _set_vertices(sh, vs)
        return f"v{i} ({x:.1f},{y:.1f}) -> {vs[i]}", [sid] if cmd.get("snap_band") else []
    if op == "delete_vertex":
        del vs[cmd["idx"]]
        _set_vertices(sh, vs)
        return f"deleted v{cmd['idx']}", [sid] if cmd.get("snap_band") else []
    if op == "add_vertex":
        i = cmd["after_idx"]
        a, b = vs[i], vs[(i + 1) % len(vs)]
        pt = cmd["at"] if "at" in cmd else (a[0] + (b[0] - a[0]) * cmd["t"], a[1] + (b[1] - a[1]) * cmd["t"])
        vs.insert(i + 1, _clamp(pt[0], pt[1], wh))
        _set_vertices(sh, vs)
        return f"added v{i + 1} {vs[i + 1]}", [sid] if cmd.get("snap_band") else []
    raise ValueError(f"op chua ho tro: {op}")


def apply_edits(shapes, commands, img=None, frame_wh=None, rs=None, default_snap: float | None = None) -> dict:
    """Ap dung list lenh len COPY cua shapes. -> {shapes, log, issues_before, issues_after, remaining, resolved}.

    default_snap: dai bam canh (px) tu dong sau MOI lenh doi hinh khong tu khai snap_band (None = tat).
    Lenh sai schema / sai thao tac bi tu choi (status rejected/error), cac lenh con lai van chay.
    """
    rs = rs or V.load_ruleset()
    if frame_wh is None and img is not None:
        frame_wh = (img.shape[1], img.shape[0])
    cur = {s["id"]: copy.deepcopy(s) for s in shapes}
    before = V.validate_frame(list(cur.values()), frame_wh, rs)
    log = []
    for n, cmd in enumerate(commands):
        errs = E.validate_command(cmd, cur, rs)
        if errs:
            log.append({"i": n, "op": cmd.get("op") if isinstance(cmd, dict) else None, "status": "rejected", "errors": errs})
            continue
        snapshot = copy.deepcopy(cur)
        ent = {"i": n, "op": cmd["op"], "shape": cmd.get("shape") or cmd.get("shapes"), "status": "ok"}
        try:
            detail, snap_ids = _apply_one(cmd, cur, img, frame_wh, rs)
            ent["detail"] = detail
            band = cmd.get("band_px") or cmd.get("snap_band")
            if band is None and default_snap and cmd["op"] not in ("delete_shape", "relabel", "set_flag"):
                band, snap_ids = default_snap, [i for i in (snap_ids or ([cmd["shape"]] if cmd.get("shape") in cur else [])) if i in cur]
            if band:
                ent["snap"] = []
                for sid in snap_ids:
                    if sid in cur:
                        ent["snap"].append({"shape": sid, **snap_shape(cur[sid], img, band, rs)})
        except Exception as e:  # lenh hong khong lam hong ca ban sua
            cur = snapshot
            ent.update(status="error", errors=[f"{type(e).__name__}: {e}"])
        log.append(ent)
    after_shapes = list(cur.values())
    after = V.validate_frame(after_shapes, frame_wh, rs)
    b_codes, a_codes = V.codes_by_shape(before, "error"), V.codes_by_shape(after, "error")
    touched = {i for e in log if e["status"] == "ok" for i in ([e["shape"]] if isinstance(e["shape"], str) else e["shape"] or [])}
    return {
        "shapes": after_shapes, "log": log, "issues_before": before, "issues_after": after,
        "counts_before": V.count_by_code(before), "counts_after": V.count_by_code(after),
        "remaining": {k: v for k, v in a_codes.items()},                       # shape con loi muc error
        "touched_still_failing": {k: a_codes[k] for k in a_codes if k in touched or k.split(".")[0] in touched},
        "resolved": sorted(k for k in b_codes if k not in a_codes),
        "new_failures": sorted(k for k in a_codes if k not in b_codes),
    }


def to_draft_shapes(shapes) -> list[dict]:
    """Shape noi bo -> shape kieu draft (giu truong goc nhu label_id, _src...)."""
    out = []
    for s in shapes:
        raw = copy.deepcopy(s.get("raw") or {"frame": s.get("frame"), "source": "manual", "attributes": [], "rotation": 0.0})
        raw.update(type=s["type"], points=[_r(v) for v in s["points"]], z_order=s["z_order"],
                   occluded=bool(s["occluded"]), _label=s["label"], _review_id=s["id"])
        if s.get("truncated"):
            raw["_truncated"] = True
        out.append(raw)
    return out


def save_result(result: dict, out_dir, source: str | Path | None = None, tag: str = "review_edit") -> dict:
    """Ghi ban sua + nhat ky vao out_dir (khong bao gio ghi de `source`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_shapes, f_log = out / f"{tag}_{stamp}.json", out / f"{tag}_{stamp}.log.json"
    if source and Path(source).resolve() in (f_shapes.resolve(), f_log.resolve()):
        raise ValueError("khong ghi de nhap goc")
    f_shapes.write_text(json.dumps({"kind": tag, "created": datetime.now().isoformat(timespec="seconds"),
                                    "source": str(source) if source else None,
                                    "shapes": to_draft_shapes(result["shapes"])}, ensure_ascii=False), encoding="utf8")
    f_log.write_text(json.dumps({k: result[k] for k in ("log", "counts_before", "counts_after", "remaining",
                                                       "touched_still_failing", "resolved", "new_failures")},
                                ensure_ascii=False, indent=1), encoding="utf8")
    return {"shapes": str(f_shapes), "log": str(f_log)}
