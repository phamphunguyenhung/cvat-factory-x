"""Review pack moi frame: issues.json + crop phong to quanh shape nghi ngo (luoi toa do goc, danh so dinh) + anh tong overlay.

Luoi ghi so px TOA DO ANH GOC; crop co meta (x0,y0,scale) de doi: goc = x0 + px/scale.
Sap theo muc nghiem trong (error > warn > info), nhieu loi truoc.
"""
from __future__ import annotations

import colorsys
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from cvat_tools.review import edits as E
from cvat_tools.review import validators as V

NICE = (2, 5, 10, 20, 25, 50, 100, 200)


def label_color(label: str) -> tuple[int, int, int]:
    """BGR sang, bao hoa: thay duoc tren nen toi."""
    h = (zlib.crc32(label.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _pts(shape, x0=0.0, y0=0.0, s=1.0):
    return np.array([[(x - x0) * s, (y - y0) * s] for x, y in V.vertices(shape)], np.float32)


def _draw_outline(img, shape, color, x0=0.0, y0=0.0, s=1.0, th=1):
    p = np.round(_pts(shape, x0, y0, s)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], True, (0, 0, 0), th + 2, cv2.LINE_AA)  # vien toi de noi tren nen sang
    cv2.polylines(img, [p], True, color, th, cv2.LINE_AA)


def _text(img, txt, org, color=(255, 255, 255), sc=0.45, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def _grid(img, x0, y0, s, W, H):
    """Luoi toa do goc, buoc 'dep' sao cho cach nhau >= ~70px man hinh."""
    step = next((n for n in NICE if n * s >= 70), NICE[-1])
    ov = img.copy()
    gx = int(np.ceil(x0 / step) * step)
    while (gx - x0) * s < W:
        X = int(round((gx - x0) * s))
        cv2.line(ov, (X, 0), (X, H), (255, 255, 255), 1)
        gx += step
    gy = int(np.ceil(y0 / step) * step)
    while (gy - y0) * s < H:
        Y = int(round((gy - y0) * s))
        cv2.line(ov, (0, Y), (W, Y), (255, 255, 255), 1)
        gy += step
    img[:] = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    gx = int(np.ceil(x0 / step) * step)
    while (gx - x0) * s < W:
        _text(img, str(gx), (int(round((gx - x0) * s)) + 2, 11), (0, 255, 255), 0.38)
        gx += step
    gy = int(np.ceil(y0 / step) * step)
    while (gy - y0) * s < H:
        _text(img, str(gy), (2, int(round((gy - y0) * s)) - 2), (0, 255, 255), 0.38)
        gy += step


def render_crop(img, shape, window, others=(), mark=None, target=900, max_scale=10.0):
    """window=(x0,y0,x1,y1) goc. Tra (anh, meta)."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(window[0])), max(0, int(window[1]))
    x1, y1 = min(W, int(np.ceil(window[2]))), min(H, int(np.ceil(window[3])))
    s = float(min(max(target / max(x1 - x0, y1 - y0, 1), 1.0), max_scale))
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    ch, cw = crop.shape[:2]
    _grid(crop, x0, y0, s, cw, ch)
    for o in others:
        _draw_outline(crop, o, tuple(int(c * 0.6) for c in label_color(o["label"])), x0, y0, s, 1)
    col = label_color(shape["label"])
    _draw_outline(crop, shape, (255, 255, 255), x0, y0, s, 1)
    _draw_outline(crop, shape, col, x0, y0, s, 1)
    vs = _pts(shape, x0, y0, s)
    stride = 1 if len(vs) <= 60 else 2
    for i, (px, py) in enumerate(vs):
        if i % stride:
            continue
        c = (int(round(px)), int(round(py)))
        cv2.circle(crop, c, 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(crop, c, 2, (0, 255, 255), -1, cv2.LINE_AA)
        _text(crop, str(i), (c[0] + 4, c[1] - 4), (0, 255, 255), 0.42)
    if mark is not None:
        c = (int(round((mark[0] - x0) * s)), int(round((mark[1] - y0) * s)))
        cv2.drawMarker(crop, c, (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    _text(crop, f"{shape['id']} {shape['label']}  x{s:.1f}  origin({x0},{y0})", (4, ch - 6), (255, 255, 255), 0.45)
    return crop, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "scale": round(s, 3), "size": [cw, ch]}


def render_overview(img, shapes, flagged_ids, alpha=0.30):
    """Anh tong: to mau theo label (alpha) + vien sang 2 lop; shape nghi ngo co nhan id."""
    out = img.copy()
    ordered = sorted(shapes, key=lambda s: s["z_order"])
    ov = out.copy()
    for s in ordered:
        if s["type"] == "polygon" and len(s["points"]) >= 6:
            cv2.fillPoly(ov, [np.round(_pts(s)).astype(np.int32)], label_color(s["label"]))
    out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)
    for s in ordered:
        _draw_outline(out, s, label_color(s["label"]), th=1)
    for rank, sid in enumerate(flagged_ids, 1):
        s = next(x for x in shapes if x["id"] == sid)
        p = np.round(_pts(s)).astype(np.int32)
        _draw_outline(out, s, (0, 0, 255), th=3)
        cx, cy = p.mean(axis=0).astype(int)
        _text(out, f"{rank}:{sid.split('_')[-1]}", (int(cx), int(cy)), (0, 255, 255), 0.6, 2)
    labs = sorted({s["label"] for s in shapes})
    y = 14
    for lb in labs:
        cv2.rectangle(out, (6, y - 9), (16, y + 1), label_color(lb), -1)
        _text(out, lb, (20, y), (255, 255, 255), 0.4)
        y += 14
    return out


def _focus_points(shape, issues):
    pts = []
    vs = V.vertices(shape)
    for i in issues:
        d = i["data"]
        if d.get("at"):
            pts.append(tuple(d["at"]))
        for k in d.get("idx", [])[:2] if isinstance(d.get("idx"), list) else []:
            if k < len(vs):
                pts.append(vs[k])
    return pts[:2]


def build_pack(frame: int, shapes: list[dict], img: np.ndarray, out_dir, rs: dict | None = None,
               max_shapes: int = 12, min_sev: str = "warn") -> dict:
    """shapes: shape noi bo cua 1 frame. Ghi out_dir/frame_<N>/{issues.json, overview.jpg, crop_*.jpg}."""
    rs = rs or V.load_ruleset()
    H, W = img.shape[:2]
    issues = V.validate_frame(shapes, (W, H), rs)
    by_id: dict = {}
    for i in issues:
        by_id.setdefault(i["shape_id"], []).append(i)
    sev = V.SEV_RANK
    ranked = sorted((sid for sid, lst in by_id.items() if max(sev[i["severity"]] for i in lst) >= sev[min_sev]),
                    key=lambda sid: (-max(sev[i["severity"]] for i in by_id[sid]), -len(by_id[sid]), sid))
    chosen = ranked[:max_shapes]
    fdir = Path(out_dir) / f"frame_{frame}"
    fdir.mkdir(parents=True, exist_ok=True)
    smap = {s["id"]: s for s in shapes}
    entries = []
    for rank, sid in enumerate(chosen, 1):
        sh = smap[sid]
        x0, y0, x1, y1 = V.poly_of(sh).bounds
        m = max(30.0, 0.35 * max(x1 - x0, y1 - y0))
        crops = []
        tasks = [("shape", (x0 - m, y0 - m, x1 + m, y1 + m), None)]
        for k, fp in enumerate(_focus_points(sh, by_id[sid])):
            tasks.append((f"focus{k}", (fp[0] - 60, fp[1] - 45, fp[0] + 60, fp[1] + 45), fp))
        for kind, win, mk in tasks:
            wbox = V.poly_of({"type": "rectangle", "points": list(win)})
            near = [o for o in shapes if o["id"] != sid and V.poly_of(o).intersects(wbox)][:12] if kind != "shape" else []
            im, meta = render_crop(img, sh, win, others=near, mark=mk)
            fn = f"crop_{rank:02d}_{sid}_{kind}.jpg"
            cv2.imwrite(str(fdir / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crops.append({"file": fn, "kind": kind, **meta})
        entries.append({"rank": rank, "shape_id": sid, "label": sh["label"], "type": sh["type"], "points": sh["points"],
                        "z_order": sh["z_order"], "occluded": sh["occluded"], "truncated": sh["truncated"],
                        "worst": max((i["severity"] for i in by_id[sid]), key=lambda s_: sev[s_]),
                        "issues": by_id[sid], "crops": crops})
    cv2.imwrite(str(fdir / "overview.jpg"), render_overview(img, shapes, chosen), [cv2.IMWRITE_JPEG_QUALITY, 88])
    doc = {"frame": frame, "size": [W, H], "n_shapes": len(shapes),
           "counts_by_code": V.count_by_code(issues),
           "counts_by_severity": {k: sum(1 for i in issues if i["severity"] == k) for k in ("error", "warn", "info")},
           "flagged_shapes": len(ranked), "shown": len(chosen), "overview": "overview.jpg",
           "how_to_reply": "Tra list lenh JSON theo EDIT_SCHEMA (khong ghi toa do tuyet doi cho lane/curb/pole). "
                           "Doi px crop -> goc: x = x0 + px/scale.",
           "edit_schema": E.EDIT_SCHEMA, "shapes": entries,
           "other_issues": [i for i in issues if i["shape_id"] not in chosen and V.SEV_RANK[i["severity"]] >= sev[min_sev]][:200]}
    (fdir / "issues.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf8")
    return doc
