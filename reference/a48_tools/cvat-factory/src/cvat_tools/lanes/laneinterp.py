"""Lane anchor interpolation (pure numpy/cv2; no network/CVAT).

CHI DUNG khi job khai bao continuity=video (frame lien tuc, keyframe moi 10 anh). Mac dinh cac frame KHONG lien tuc
(vd job 285): dung framegen.generate_frame (tung frame doc lap), KHONG goi interpolate_* / snap_to_candidates.

A lane is parameterised as (x_bottom at y=H, vanishing point VP, y_top). Anchors are
user-clicked polylines on a few key frames; x_bottom / VP / y_top are interpolated linearly
by frame number, outside the anchor range the nearest anchor is held.

anchors.json: {"job":285,"anchors":[{"frame":60,"label":"lane/single white",
               "id":"a" (optional), "points":[x1,y1,x2,y2,...]}]}   (px, 1280x720)
Lanes: polyline >= 2 points. Crosswalk (label contains "crosswalk"): 4-6 points (polygon).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

W_DEFAULT, H_DEFAULT = 1280, 720
Point = Tuple[float, float]


def is_crosswalk(label: str) -> bool:
    return "crosswalk" in label.lower()


def is_double(label: str) -> bool:
    return "double" in label.lower()


def load_anchors(path) -> Dict[str, Any]:
    """Load and validate anchors.json; returns the dict with anchors sorted by frame."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("anchors"), list):
        raise ValueError("anchors.json: thieu khoa 'anchors'")
    for i, a in enumerate(data["anchors"]):
        if not isinstance(a.get("frame"), int) or not isinstance(a.get("label"), str):
            raise ValueError(f"anchor #{i}: can 'frame' (int) va 'label' (str)")
        pts = a.get("points")
        if not isinstance(pts, list) or len(pts) % 2 or not all(isinstance(v, (int, float)) for v in pts):
            raise ValueError(f"anchor #{i}: 'points' phai la list phang x,y,...")
        n = len(pts) // 2
        if is_crosswalk(a["label"]):
            if not 4 <= n <= 6:
                raise ValueError(f"anchor #{i}: crosswalk can 4-6 diem, co {n}")
        elif n < 2:
            raise ValueError(f"anchor #{i}: lane can >= 2 diem, co {n}")
        if "id" in a:
            a["id"] = str(a["id"])
    data["anchors"].sort(key=lambda a: a["frame"])
    return data


def _pts(points: Sequence[float]) -> np.ndarray:
    return np.asarray(points, float).reshape(-1, 2)


def _fit_x_of_y(p: np.ndarray) -> Tuple[float, float]:
    """Least squares x = a + b*y through the polyline points."""
    if len(p) == 2 or np.ptp(p[:, 1]) < 1e-6:
        (x1, y1), (x2, y2) = p[0], p[-1]
        b = 0.0 if y2 == y1 else (x2 - x1) / (y2 - y1)
        return x1 - b * y1, b
    b, a = np.polyfit(p[:, 1], p[:, 0], 1)
    return float(a), float(b)


def lane_params(points: Sequence[float], H: int = H_DEFAULT,
                vp: Optional[Point] = None) -> Tuple[float, Point]:
    """(x_bottom, vp): polyline extended to y=H; vp given or fallback = top end of the polyline."""
    p = _pts(points)
    a, b = _fit_x_of_y(p)
    xb = a + b * H
    if vp is None:
        top = p[np.argmin(p[:, 1])]
        vp = (float(a + b * top[1]), float(top[1]))
    return float(xb), (float(vp[0]), float(vp[1]))


def _vp_from_lanes(lines: List[Tuple[float, float]], H: int) -> Optional[Point]:
    """Median pairwise intersection of lines x=a+b*y, only if plausible (above the bottom)."""
    pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (a1, b1), (a2, b2) = lines[i], lines[j]
            if abs(b1 - b2) < 1e-3:
                continue
            y = (a2 - a1) / (b1 - b2)
            if -H < y < 0.9 * H:
                pts.append((a1 + b1 * y, y))
    if not pts:
        return None
    arr = np.array(pts)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


def _group(anchors: Sequence[Dict[str, Any]]):
    """key=(label,id) -> {frame: anchor}. Missing id = order index within (frame,label)."""
    seen: Dict[Tuple[int, str], int] = {}
    out: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    for a in anchors:
        k = (a["frame"], a["label"])
        idx = seen.get(k, 0)
        seen[k] = idx + 1
        out.setdefault((a["label"], a.get("id") or str(idx)), {})[a["frame"]] = a
    return out


def _interp_vec(frames_vals: Dict[int, np.ndarray], f: int) -> np.ndarray:
    ks = sorted(frames_vals)
    if f <= ks[0]:
        return frames_vals[ks[0]]
    if f >= ks[-1]:
        return frames_vals[ks[-1]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= f <= hi:
            t = (f - lo) / (hi - lo)
            return (1 - t) * frames_vals[lo] + t * frames_vals[hi]
    return frames_vals[ks[-1]]


def _lane_poly(xb: float, vp: Point, y_top: float, H: int) -> List[float]:
    vx, vy = vp
    if abs(H - vy) < 1e-6:
        xt = xb
    else:
        xt = xb + (y_top - H) * (vx - xb) / (vy - H)
    return [round(xb, 1), float(H), round(float(xt), 1), round(float(y_top), 1)]


def _parallel(poly: List[float], offset: float) -> List[List[float]]:
    p = _pts(poly)
    return [(p + [-offset / 2, 0]).flatten().round(1).tolist(),
            (p + [offset / 2, 0]).flatten().round(1).tolist()]


def interpolate_lanes(anchors, frames: Iterable[int], H: int = H_DEFAULT,
                      double_offset: float = 14.0, use_vp: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    """{frame: [{id,label,polyline,polylines}]} for lane anchors (crosswalks are skipped).

    `anchors` is the loaded dict or its list. VP = median intersection of the lanes of one anchor
    frame when >=2 lanes converge plausibly, else each lane's own top point. Double lanes get
    `polylines` = two parallel polylines `double_offset` px apart (polyline = the first one)."""
    alist = anchors["anchors"] if isinstance(anchors, dict) else list(anchors)
    lane_as = [a for a in alist if not is_crosswalk(a["label"])]
    groups = _group(lane_as)
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for a in lane_as:
        by_frame.setdefault(a["frame"], []).append(a)
    frame_vp: Dict[int, Optional[Point]] = {}
    for fr, items in by_frame.items():
        frame_vp[fr] = _vp_from_lanes([_fit_x_of_y(_pts(a["points"])) for a in items], H) \
            if use_vp and len(items) >= 2 else None
    tracks: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}
    for key, per in groups.items():
        tracks[key] = {}
        for fr, a in per.items():
            p = _pts(a["points"])
            xb, vp = lane_params(a["points"], H, frame_vp.get(fr))
            tracks[key][fr] = np.array([xb, vp[0], vp[1], float(p[:, 1].min())])
    out: Dict[int, List[Dict[str, Any]]] = {}
    for f in frames:
        res = []
        for (label, lid), tr in tracks.items():
            xb, vx, vy, yt = _interp_vec(tr, f)
            poly = _lane_poly(xb, (vx, vy), yt, H)
            polys = _parallel(poly, double_offset) if is_double(label) else [poly]
            res.append({"id": lid, "label": label, "polyline": polys[0], "polylines": polys})
        out[f] = res
    return out


def resample_polygon(points: Sequence[float], n: int) -> np.ndarray:
    """Resample a closed polygon to n points evenly by perimeter. Starts at the vertex nearest
    the top-left corner (min x+y) and goes in the original winding, so two anchors with different
    vertex counts are matched in order."""
    p = _pts(points)
    k = int(np.argmin(p[:, 0] + p[:, 1]))
    p = np.roll(p, -k, axis=0)
    q = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(q, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, q[:, 0]), np.interp(t, cum, q[:, 1])], 1)


def interpolate_quad(anchors, frames: Iterable[int]) -> Dict[int, List[Dict[str, Any]]]:
    """{frame: [{id,label,polygon}]} for crosswalk anchors (4-6 vertices). Same vertex count in all
    anchors of a crosswalk = linear per vertex; different counts = every anchor resampled evenly by
    perimeter to the max count, then linear per vertex."""
    alist = anchors["anchors"] if isinstance(anchors, dict) else list(anchors)
    groups = _group([a for a in alist if is_crosswalk(a["label"])])
    out: Dict[int, List[Dict[str, Any]]] = {}
    for f in frames:
        res = []
        for (label, qid), per in groups.items():
            counts = {len(a["points"]) // 2 for a in per.values()}
            if len(counts) == 1:
                vecs = {fr: np.asarray(a["points"], float) for fr, a in per.items()}
            else:
                m = max(counts)
                vecs = {fr: resample_polygon(a["points"], m).flatten() for fr, a in per.items()}
            v = _interp_vec(vecs, f)
            res.append({"id": qid, "label": label, "polygon": [round(float(x), 1) for x in v]})
        out[f] = res
    return out


def is_night(img_bgr: np.ndarray, thr: float = 70.0) -> bool:
    """Mean grey brightness below thr = night frame."""
    g = img_bgr.mean(axis=2) if img_bgr.ndim == 3 else img_bgr
    return float(g.mean()) < thr


def _cand_line(c) -> Optional[List[float]]:
    if isinstance(c, dict):
        c = c.get("p") or c.get("points")
    if c is None or len(c) < 4:
        return None
    return [float(c[0]), float(c[1]), float(c[-2]), float(c[-1])]


def snap_to_candidates(polyline: Sequence[float], candidates: Optional[Sequence[Any]],
                       max_px: float = 20.0, enabled: bool = True) -> List[float]:
    """Move polyline x's onto the nearest candidate line (lanes.detect_lane_markings output:
    dict with "p"=[x1,y1,x2,y2], or a raw list) when the mean horizontal gap over the shared
    y-range is < max_px. Otherwise (or enabled=False, e.g. night) the polyline is returned unchanged."""
    poly = [float(v) for v in polyline]
    if not enabled or not candidates:
        return poly
    p = _pts(poly)
    y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
    a, b = _fit_x_of_y(p)
    best, best_d = None, max_px
    for c in candidates:
        cl = _cand_line(c)
        if cl is None or cl[3] == cl[1]:
            continue
        lo, hi = max(y0, min(cl[1], cl[3])), min(y1, max(cl[1], cl[3]))
        if hi <= lo or hi - lo < 0.3 * (y1 - y0):
            continue
        ys = np.linspace(lo, hi, 8)
        cx = cl[0] + (ys - cl[1]) * (cl[2] - cl[0]) / (cl[3] - cl[1])
        d = float(np.mean(np.abs(cx - (a + b * ys))))
        if d < best_d:
            best, best_d = cl, d
    if best is None:
        return poly
    x = best[0] + (p[:, 1] - best[1]) * (best[2] - best[0]) / (best[3] - best[1])
    return np.stack([x, p[:, 1]], 1).flatten().round(1).tolist()


def lane_polygon(polyline: Sequence[float], half: float = 6.0, W: int = W_DEFAULT,
                 H: int = H_DEFAULT) -> List[float]:
    """Polyline -> thin polygon (+-half px horizontally; same shape as bbox_all_draft.line_poly
    for 2-point lines, generalised to N vertices; script is not imported to avoid cv2/ultralytics deps)."""
    p = _pts(polyline)
    q = np.vstack([p + [-half, 0], (p + [half, 0])[::-1]])
    q[:, 0] = q[:, 0].clip(0, W)
    q[:, 1] = q[:, 1].clip(0, H)
    return q.flatten().round(1).tolist()
