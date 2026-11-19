"""Dedicated detector for `lane/single white` and `lane/double white` (draft only, never a final label).

Pipeline: white mask (top-hat + low saturation; night = per-frame percentile threshold on a gamma/CLAHE
enhanced copy) confined to the road -> connected blobs -> drop short / fat blobs -> one straight fit per blob
-> merge collinear blobs into ONE marking (a dashed or broken line is one line, never several) ->
exhaustive vanishing-point vote (lines that converge are boosted, outliers are kept with low conf) ->
pair close parallels into `double`. Every marking carries a `conf` in [0, 1] so the user can filter.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cvat_tools.lanes.laneinterp import is_night, lane_polygon

SINGLE = "lane/single white"
DOUBLE = "lane/double white"

CFG: Dict[str, float] = {
    "roi_bottom": 0.84,      # bonnet starts below this fraction of H
    "tophat_k": 51,          # top-hat kernel @720p
    "tophat_day": 30,
    "tophat_night": 22,
    "sat_max": 70,
    "val_min_day": 140,
    "val_min_night": 90,
    "night_pct": 96.0,       # night: pixel must be in the brightest (100-pct)% of the road
    "min_blob_area": 40,     # px^2 @720p
    "min_blob_len": 22,      # px @720p, blob principal length
    "min_elong": 3.0,        # sqrt(l1/l2)
    "max_fat_area": 6000,
    "min_tilt_deg": 14,
    "collinear_px": 8,       # endpoint distance to cluster line
    "collinear_deg": 6,
    "min_line_len": 70,      # painted length of a merged line (px @720p)
    "min_line_span": 45,     # vertical span
    "pair_min_px": 5,
    "pair_max_px": 30,
    "pair_par_deg": 4,
    "vp_tol_deg": 2.5,
    "max_lines": 8,
    "max_warmth": 30.0,      # median (R+G)/2-B of a white blob; above = yellow paint
}


def _road_mask(road_poly: Any, W: int, H: int) -> np.ndarray:
    if isinstance(road_poly, np.ndarray) and road_poly.ndim == 2:
        return (road_poly > 0).astype(np.uint8)
    m = np.zeros((H, W), np.uint8)
    polys = [road_poly] if len(road_poly) and not hasattr(road_poly[0], "__len__") else list(road_poly)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p, np.float32)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def _enhance(img: np.ndarray) -> np.ndarray:
    m = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    g = float(np.clip(np.log(100 / 255.0) / np.log(max(m, 1.0) / 255.0), 0.25, 1.0))
    out = (255.0 * (img.astype(np.float32) / 255.0) ** g).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def white_mask(img_bgr: np.ndarray, road: np.ndarray, night: Optional[bool] = None) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    s = H / 720.0
    night = is_night(img_bgr) if night is None else night
    src = _enhance(img_bgr) if night else img_bgr
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    k = int(CFG["tophat_k"] * s) | 1
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(25 * s) | 1,) * 2)
    allowed = cv2.dilate(cv2.morphologyEx(road, cv2.MORPH_CLOSE, kk),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(9 * s) | 1,) * 2))
    val = hsv[..., 2]
    if night:
        vals = gray[allowed > 0]
        vmin = max(CFG["val_min_night"], float(np.percentile(vals, CFG["night_pct"]))) if vals.size else 255
        m = (th > CFG["tophat_night"]) & (gray >= vmin) & (hsv[..., 1] < CFG["sat_max"] + 20)
    else:
        m = (th > CFG["tophat_day"]) & (hsv[..., 1] < CFG["sat_max"]) & (val > CFG["val_min_day"])
    m = m & (allowed > 0)
    m = m.astype(np.uint8) * 255
    m[int(CFG["roi_bottom"] * H):] = 0
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def _fit(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def _dist(p: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.abs((p[..., 0] - c[0]) * -d[1] + (p[..., 1] - c[1]) * d[0])


def _tilt(d: np.ndarray) -> float:
    a = abs(math.degrees(math.atan2(d[1], d[0])))
    return min(a, 180 - a)


def _ang(d1: np.ndarray, d2: np.ndarray) -> float:
    a = abs(math.degrees(math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1[0] * d2[0] + d1[1] * d2[1])))
    return min(a, 180 - a)


def _yellow_zone(img_bgr: np.ndarray, night: bool = False) -> np.ndarray:
    """Faded / over-exposed yellow paint and its halo (a white fit on it would mislabel a yellow line)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # sodium lamps tint night asphalt orange, so night needs a much higher saturation
    y = cv2.inRange(hsv, (12, 110 if night else 45, 90), (38, 255, 255))
    return cv2.dilate(y, np.ones((13, 13), np.uint8)) > 0


def _warmth(img_bgr: np.ndarray) -> np.ndarray:
    """(R+G)/2 - B per pixel: ~0 for white/grey paint, large for yellow paint even when over-exposed."""
    f = img_bgr.astype(np.float32)
    return (f[..., 2] + f[..., 1]) / 2 - f[..., 0]


def _blobs(mask: np.ndarray, s: float, yellow: Optional[np.ndarray] = None, warm: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i][4]
        if area < CFG["min_blob_area"] * s * s or area > CFG["max_fat_area"] * s * s:
            continue
        ys, xs = np.nonzero(lab[stats[i][1]:stats[i][1] + stats[i][3], stats[i][0]:stats[i][0] + stats[i][2]] == i)
        pts = np.stack([xs + stats[i][0], ys + stats[i][1]], 1).astype(np.float32)
        cov = np.cov((pts - pts.mean(0)).T)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = math.sqrt(ev[0] / max(ev[1], 1e-6))
        c, d = _fit(pts)
        t = (pts - c) @ d
        length = float(t.max() - t.min())
        if elong < CFG["min_elong"] or length < CFG["min_blob_len"] * s or _tilt(d) < CFG["min_tilt_deg"]:
            continue
        if yellow is not None and float(yellow[pts[:, 1].astype(int), pts[:, 0].astype(int)].mean()) > 0.3:
            continue
        if warm is not None and float(np.median(warm[pts[:, 1].astype(int), pts[:, 0].astype(int)])) > CFG["max_warmth"]:
            continue
        out.append({"pts": pts, "len": length})
    return out


def _merge(blobs: List[Dict[str, Any]], s: float) -> List[Dict[str, Any]]:
    """Greedy collinear merge: a blob joins a cluster when both its ends lie near the cluster line."""
    cl: List[Dict[str, Any]] = []
    for b in sorted(blobs, key=lambda b: -b["len"]):
        c, d = _fit(b["pts"])
        for k in cl:
            ends = np.array([c + d * ((b["pts"] - c) @ d).min(), c + d * ((b["pts"] - c) @ d).max()])
            if _ang(d, k["d"]) < CFG["collinear_deg"] and float(_dist(ends, k["c"], k["d"]).max()) < CFG["collinear_px"] * s:
                k["parts"].append(b)
                k["pts"] = np.vstack([k["pts"], b["pts"]])
                k["c"], k["d"] = _fit(k["pts"])
                break
        else:
            cl.append({"parts": [b], "pts": b["pts"], "c": c, "d": d})
    lines = []
    for k in cl:
        d = k["d"] if k["d"][1] >= 0 else -k["d"]
        t = (k["pts"] - k["c"]) @ d
        a, b = k["c"] + d * t.min(), k["c"] + d * t.max()  # a = top, b = bottom (d points down)
        plen = float(sum(p["len"] for p in k["parts"]))
        perp = float(np.mean(_dist(k["pts"], k["c"], d)))
        lines.append({"a": a, "b": b, "d": d, "c": k["c"], "len": plen, "span": float(b[1] - a[1]),
                      "perp": perp, "n": len(k["parts"])})
    return [l for l in lines if l["len"] >= CFG["min_line_len"] * s and l["span"] >= CFG["min_line_span"] * s]


def _vanishing(lines: List[Dict[str, Any]], W: int, H: int) -> Tuple[Optional[np.ndarray], List[int]]:
    best, best_in, best_w = None, [], 0.0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            A, B = lines[i], lines[j]
            den = A["d"][0] * B["d"][1] - A["d"][1] * B["d"][0]
            if abs(den) < 1e-6:
                continue
            t = ((B["c"][0] - A["c"][0]) * B["d"][1] - (B["c"][1] - A["c"][1]) * B["d"][0]) / den
            vp = A["c"] + t * A["d"]
            if not (-0.5 * W <= vp[0] <= 1.5 * W and -0.3 * H <= vp[1] <= 0.7 * H):
                continue
            inl = []
            for k, L in enumerate(lines):
                mid = (L["a"] + L["b"]) / 2
                if _ang(L["d"], vp - mid) < CFG["vp_tol_deg"] and vp[1] < mid[1]:
                    inl.append(k)
            w = sum(lines[k]["len"] for k in inl)
            if len(inl) >= 2 and w > best_w:
                best, best_in, best_w = vp, inl, w
    return best, best_in


def _x_at(L: Dict[str, Any], y: float) -> float:
    return float(L["a"][0] + (y - L["a"][1]) * (L["b"][0] - L["a"][0]) / max(1e-6, L["b"][1] - L["a"][1]))


def _conf(L: Dict[str, Any], s: float, in_vp: bool, has_vp: bool, night: bool) -> float:
    c = 0.35 * min(1.0, L["len"] / (250 * s)) + 0.25 * min(1.0, L["span"] / (150 * s))
    c += 0.15 * max(0.0, 1 - L["perp"] / (8 * s)) + (0.25 if in_vp else 0.1 if not has_vp else 0.0)
    return round(float(min(1.0, c * (0.85 if night else 1.0))), 3)


def detect_white(img_bgr: np.ndarray, road_poly: Any, night: Optional[bool] = None) -> Tuple[List[Dict], List[Dict]]:
    """-> (singles, doubles), each sorted by conf desc."""
    H, W = img_bgr.shape[:2]
    s = H / 720.0
    road = _road_mask(road_poly, W, H)
    if not road.any():
        return [], []
    if night is None:  # street lamps lift the frame mean, so also check the top third (sky / dark buildings)
        night = is_night(img_bgr) or float(cv2.cvtColor(img_bgr[:H // 3], cv2.COLOR_BGR2GRAY).mean()) < 70
    lines = _merge(_blobs(white_mask(img_bgr, road, night), s, _yellow_zone(img_bgr, night), _warmth(img_bgr)), s)
    lines = sorted(lines, key=lambda l: -l["len"])[:int(CFG["max_lines"])]
    vp, inl = _vanishing(lines, W, H)
    for k, L in enumerate(lines):
        L["conf"] = _conf(L, s, k in inl, vp is not None, night)

    used, doubles = set(), []
    for i, A in enumerate(lines):
        for j in range(i + 1, len(lines)):
            B = lines[j]
            if i in used or j in used or _ang(A["d"], B["d"]) > CFG["pair_par_deg"]:
                continue
            y0, y1 = max(A["a"][1], B["a"][1]), min(A["b"][1], B["b"][1])
            if y1 - y0 < 0.4 * min(A["span"], B["span"]):
                continue
            sep = abs(_x_at(A, (y0 + y1) / 2) - _x_at(B, (y0 + y1) / 2))
            if CFG["pair_min_px"] * s <= sep <= CFG["pair_max_px"] * s:
                used.update((i, j))
                ya, yb = min(A["a"][1], B["a"][1]), max(A["b"][1], B["b"][1])
                pl = [(_x_at(A, ya) + _x_at(B, ya)) / 2, ya, (_x_at(A, yb) + _x_at(B, yb)) / 2, yb]
                doubles.append((pl, sep, (A["conf"] + B["conf"]) / 2))

    def clip(pl):
        return [round(float(min(max(v, 0.0), W if k % 2 == 0 else H)), 1) for k, v in enumerate(pl)]

    def extend(L):  # run the line down towards the bonnet when it clearly continues towards the camera
        ybot = CFG["roi_bottom"] * H
        if L["b"][1] > ybot - 0.2 * H and L["b"][1] < ybot:
            return [L["a"][0], L["a"][1], _x_at(L, ybot), ybot]
        return [L["a"][0], L["a"][1], L["b"][0], L["b"][1]]

    singles = []
    for i, L in enumerate(lines):
        if i not in used:
            pl = clip(extend(L))
            singles.append({"label": SINGLE, "polyline": pl, "polygon": lane_polygon(pl, 6 * s, W, H), "conf": L["conf"]})
    dbl = []
    for pl, sep, cf in doubles:
        pl = clip(pl)
        dbl.append({"label": DOUBLE, "polyline": pl, "polygon": lane_polygon(pl, sep / 2 + 4 * s, W, H), "conf": round(float(cf), 3)})
    for tag, group in (("s", singles), ("d", dbl)):
        group.sort(key=lambda d: -d["conf"])
        for k, d in enumerate(group):
            d["id"] = f"white_{tag}{k}"
    return singles, dbl


def detect_single_white(img_bgr: np.ndarray, road_poly: Any) -> List[Dict[str, Any]]:
    return detect_white(img_bgr, road_poly)[0]


def detect_double_white(img_bgr: np.ndarray, road_poly: Any) -> List[Dict[str, Any]]:
    return detect_white(img_bgr, road_poly)[1]
