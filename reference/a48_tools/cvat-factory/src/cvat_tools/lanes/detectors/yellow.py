"""Dedicated detectors for `lane/single yellow` and `lane/double yellow`.

Signature: (img_bgr, road_poly) -> [{"label", "polyline", "polygon", "conf", "id"}]
`road_poly` is the hand-drawn road polygon, flat [x1,y1,...] or [[x,y],...]; paint is only looked for inside it.
`polyline` / `polygon` are lists of [x, y].

Night handling: sodium street lamps tint the whole scene yellow, so absolute hue is useless. Yellow paint is
the part of the road that is *more yellow and brighter than the road around it*: colour is measured as
Lab-b minus a heavily blurred (road-only) local background, and the threshold adapts to the spread of that
score inside the road. Lamp reflections / headlight streaks are dropped by shape (not thin + elongated),
by tilt (paint recedes towards the vanishing point) and by how continuously a candidate line is painted.
When nothing passes the confidence bar the result is an empty list - never a guess.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SINGLE = "lane/single yellow"
DOUBLE = "lane/double yellow"

CFG: Dict[str, Any] = {
    "bottom": 0.84,          # bonnet starts below this fraction of the height
    "min_tilt_deg": 18,      # flatter = stop line / box edge / reflection, not a lane line
    "min_comp_len": 22,      # px @720p, per painted blob (long side)
    "min_elong": 2.5,        # blob long/short side
    "max_comp_width": 60,    # fatter blobs are glare / vehicles
    "db_min": 9.0,           # Lab-b above the local road background
    "dl_min": 6.0,           # and brighter than it
    "hue_lo": 15, "hue_hi": 42,
    "sat_min": 60,
    "cluster_deg": 5.0,
    "cluster_px": 6.0, 
    "min_line_len": 70,      # painted length of a fitted line
    "min_span": 40,          # vertical span
    "min_cover": 0.30,       # share of the line span that really carries paint
    "pair_min_gap": 5.0,     # double yellow = two parallels this far apart (perpendicular px @720p) ...
    "pair_max_gap": 34.0,    # ... and no further
    "pair_min_overlap": 45,  # shared vertical extent
    "pair_parallel_deg": 12.0,
    "min_conf": 0.45,
    "max_doubles": 2,
    "max_singles": 3,
    "max_mask_frac": 0.10,   # more yellow than this inside the road = glare / tint, not paint
}


def _poly_xy(poly) -> np.ndarray:
    a = np.asarray(poly, np.float32)
    return a.reshape(-1, 2)


def _road_mask(shape, road_poly) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    if road_poly is None or len(np.asarray(road_poly).reshape(-1)) < 6:
        return m
    cv2.fillPoly(m, [np.round(_poly_xy(road_poly)).astype(np.int32).reshape(-1, 1, 2)], 255)
    return m


def _local_bg(chan: np.ndarray, road: np.ndarray, sigma: float) -> np.ndarray:
    """Road-only blurred mean of a channel (normalised convolution)."""
    w = (road > 0).astype(np.float32)
    num = cv2.GaussianBlur(chan * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def yellow_mask(img_bgr: np.ndarray, road: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """(mask 0/255, info). Yellow relative to the local road, so a sodium-tinted night scene is not all 'yellow'."""
    H, W = img_bgr.shape[:2]
    scale = H / 720.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Lc, bc = lab[..., 0], lab[..., 2] - 128.0
    sigma = 45 * scale
    db = bc - _local_bg(bc, road, sigma)
    dl = Lc - _local_bg(Lc, road, sigma)
    inside = road > 0
    vals = db[inside]
    info = {"night_tint": 0.0, "db_thr": CFG["db_min"]}
    thr = CFG["db_min"]
    if vals.size > 200:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-3
        thr = min(15.0, max(CFG["db_min"], med + 3.0 * 1.4826 * mad))   # adaptive: noisy / tinted road needs more contrast
        info["night_tint"] = float(np.median(bc[inside]))
    info["db_thr"] = thr
    hue = hsv[..., 0]
    m = (db > thr) & (dl > CFG["dl_min"]) & (hue >= CFG["hue_lo"]) & (hue <= CFG["hue_hi"]) & (hsv[..., 1] >= CFG["sat_min"] * (0.6 if info["night_tint"] > 20 else 0.8))
    m &= inside
    m[int(CFG["bottom"] * H):] = False
    m = m.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    info["frac"] = float((m > 0).sum()) / max(1.0, float(inside.sum()))
    info["contrast"] = float(np.mean(db[m > 0])) if (m > 0).any() else 0.0
    return m, info


def _blob_segments(mask: np.ndarray, scale: float) -> List[Dict[str, Any]]:
    """Straight paint segments (HoughLinesP on the mask). `pts` = mask pixels hugging each segment, so
    thick painted bands and merged blobs still give lines, while round glare blobs give none."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):   # glare / car-body blobs: big and compact
        x, y, w, h, area = stats[i]
        if area > 1500 * scale * scale and area / float(max(1, w * h)) > 0.55:
            continue
        keep[lab == i] = 255
    lines = cv2.HoughLinesP(keep, 1, np.pi / 180, threshold=int(14 * scale), minLineLength=int(CFG["min_comp_len"] * scale),
                            maxLineGap=int(20 * scale))
    out: List[Dict[str, Any]] = []
    if lines is None:
        return out
    ls = sorted(lines[:, 0].tolist(), key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))[:80]
    for x1, y1, x2, y2 in ls:
        if _tilt((x1, y1, x2, y2)) < CFG["min_tilt_deg"]:
            continue
        band = np.zeros_like(mask)
        cv2.line(band, (int(x1), int(y1)), (int(x2), int(y2)), 255, max(3, int(9 * scale)))
        ys, xs = np.nonzero(band & keep)
        if len(xs) < 20:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        p = (x0 + t.min() * vx, y0 + t.min() * vy, x0 + t.max() * vx, y0 + t.max() * vy)
        out.append({"p": p, "pts": pts, "len": float(t.max() - t.min())})
    return out

def _tilt(p) -> float:
    a = abs(math.degrees(math.atan2(p[3] - p[1], p[2] - p[0])))
    return min(a, 180 - a)


def _theta(p) -> float:
    return math.atan2(p[2] - p[0], -(p[3] - p[1])) % math.pi


def _cluster(segs: List[Dict[str, Any]], scale: float) -> List[List[Dict[str, Any]]]:
    segs = sorted(segs, key=lambda s: -s["len"])
    cl: List[Dict[str, Any]] = []
    for s in segs:
        th = _theta(s["p"])
        mx, my = (s["p"][0] + s["p"][2]) / 2, (s["p"][1] + s["p"][3]) / 2
        for c in cl:
            d = abs(th - c["theta"])
            d = min(d, math.pi - d)
            dist = abs(mx * math.cos(c["theta"]) + my * math.sin(c["theta"]) - c["rho"])
            if math.degrees(d) < CFG["cluster_deg"] and dist < CFG["cluster_px"] * scale:
                c["segs"].append(s)
                break
        else:
            cl.append({"theta": th, "rho": mx * math.cos(th) + my * math.sin(th), "segs": [s]})
    return [c["segs"] for c in cl]


def _fit_cluster(segs: List[Dict[str, Any]], scale: float) -> Optional[Dict[str, Any]]:
    pts = np.concatenate([s["pts"] for s in segs])
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    tmin, tmax = float(t.min()), float(t.max())
    p = (x0 + tmin * vx, y0 + tmin * vy, x0 + tmax * vx, y0 + tmax * vy)
    if p[3] < p[1]:
        p = (p[2], p[3], p[0], p[1])
    span = abs(p[3] - p[1])
    length = tmax - tmin
    if _tilt(p) < CFG["min_tilt_deg"] or span < CFG["min_span"] * scale:
        return None
    painted = sum(s["len"] for s in segs)
    if painted < CFG["min_line_len"] * scale:
        return None
    # continuity: fraction of 8px bins along the axis that contain paint
    bins = np.unique(np.floor((t - tmin) / (8 * scale)).astype(int))
    cover = len(bins) / max(1.0, length / (8 * scale) + 1)
    if cover < CFG["min_cover"]:
        return None
    # straightness: residual of blob pixels from the fitted line
    res = float(np.mean(np.abs((pts[:, 0] - x0) * -vy + (pts[:, 1] - y0) * vx)))
    return {"p": p, "len": painted, "cover": float(cover), "res": res, "n": len(segs),
            "area": int(len(pts))}


def _x_at(p, y: float) -> float:
    return p[0] if p[3] == p[1] else p[0] + (y - p[1]) * (p[2] - p[0]) / (p[3] - p[1])


def _conf(f: Dict[str, Any], scale: float, contrast: float) -> float:
    length = min(1.0, f["len"] / (260 * scale))
    straight = max(0.0, 1.0 - f["res"] / (6.0 * scale))
    contr = min(1.0, contrast / 40.0)
    return float(round(0.35 * length + 0.25 * f["cover"] + 0.2 * straight + 0.2 * contr, 3))


def _quad(p, half: float, W: int, H: int) -> List[List[float]]:
    q = [[p[0] - half, p[1]], [p[0] + half, p[1]], [p[2] + half, p[3]], [p[2] - half, p[3]]]
    return [[round(min(max(x, 0.0), W), 1), round(min(max(y, 0.0), H), 1)] for x, y in q]


def _pair(a: Dict[str, Any], b: Dict[str, Any], scale: float) -> Optional[Tuple[float, float]]:
    """(gap_px, overlap_px) when a and b are two close parallels, else None."""
    pa, pb = a["p"], b["p"]
    d = abs(math.degrees(_theta(pa) - _theta(pb)))
    d = min(d, 180 - d)
    if d > CFG["pair_parallel_deg"]:
        return None
    lo = max(min(pa[1], pa[3]), min(pb[1], pb[3]))
    hi = min(max(pa[1], pa[3]), max(pb[1], pb[3]))
    if hi - lo < CFG["pair_min_overlap"] * scale:
        return None
    ym = (lo + hi) / 2
    dx = abs(_x_at(pa, ym) - _x_at(pb, ym))
    gap = dx * math.sin(math.radians(max(_tilt(pa), _tilt(pb))))   # perpendicular distance
    if not (CFG["pair_min_gap"] * scale <= gap <= CFG["pair_max_gap"] * scale):
        return None
    return gap, hi - lo


def _rails_separate(mask: np.ndarray, a, b, scale: float) -> bool:
    """True when the strip midway between two candidate rails is mostly NOT paint (else it is one fat band)."""
    lo = max(min(a[1], a[3]), min(b[1], b[3]))
    hi = min(max(a[1], a[3]), max(b[1], b[3]))
    ys = np.linspace(lo, hi, 12)
    hit = 0
    for y in ys:
        x = int(round((_x_at(a, y) + _x_at(b, y)) / 2))
        if 0 <= x < mask.shape[1] and mask[int(y), max(0, x - 1):x + 2].any():
            hit += 1
    return hit / len(ys) <= 0.4


def detect_yellow(img_bgr: np.ndarray, road_poly) -> Dict[str, Any]:
    """Shared work: returns {"singles": [...], "doubles": [...], "info": {...}} before label filtering."""
    H, W = img_bgr.shape[:2]
    scale = H / 720.0
    road = _road_mask(img_bgr.shape, road_poly)
    empty = {"singles": [], "doubles": [], "info": {}}
    if not (road > 0).any():
        return empty
    mask, info = yellow_mask(img_bgr, road)
    empty["info"] = info
    if info["frac"] > CFG["max_mask_frac"]:
        return empty                                    # tint / glare: refuse rather than guess
    segs = _blob_segments(mask, scale)
    lines = []
    for cl in _cluster(segs, scale):
        f = _fit_cluster(cl, scale)
        if f is not None:
            f["conf"] = _conf(f, scale, info["contrast"])
            lines.append(f)
    lines.sort(key=lambda f: -f["conf"])
    used = set()
    doubles: List[Dict[str, Any]] = []
    for i, a in enumerate(lines):
        if i in used:
            continue
        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            r = _pair(a, lines[j], scale)
            if r is None or not _rails_separate(mask, a["p"], lines[j]["p"], scale):
                continue
            b = lines[j]
            # both rails must be solid, not one real line plus a smear next to it
            c = min(a["conf"], b["conf"])
            if c < CFG["min_conf"] + 0.05:
                continue
            used.update((i, j))
            ya = max(min(a["p"][1], a["p"][3]), min(b["p"][1], b["p"][3]))
            yb = min(max(a["p"][1], a["p"][3]), max(b["p"][1], b["p"][3]))
            mid = (_x_at(a["p"], ya) + _x_at(b["p"], ya)) / 2, ya, (_x_at(a["p"], yb) + _x_at(b["p"], yb)) / 2, yb
            doubles.append({"p": mid, "a": a["p"], "b": b["p"], "conf": c, "gap": r[0]})
            break
    singles = [f for i, f in enumerate(lines) if i not in used and f["conf"] >= CFG["min_conf"]]
    doubles.sort(key=lambda d: -d["conf"])
    return {"singles": singles[:CFG["max_singles"]], "doubles": doubles[:CFG["max_doubles"]], "info": info}


def _emit(label: str, p, polygon, conf: float, idx: int) -> Dict[str, Any]:
    return {"label": label, "polyline": [[round(float(p[0]), 1), round(float(p[1]), 1)], [round(float(p[2]), 1), round(float(p[3]), 1)]],
            "polygon": polygon, "conf": round(float(conf), 3), "id": idx}


def detect_single_yellow(img_bgr: np.ndarray, road_poly) -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    half = 6.0 * H / 720.0
    res = detect_yellow(img_bgr, road_poly)
    return [_emit(SINGLE, f["p"], _quad(f["p"], half, W, H), f["conf"], i) for i, f in enumerate(res["singles"])]


def detect_double_yellow(img_bgr: np.ndarray, road_poly) -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    half = 4.0 * H / 720.0
    res = detect_yellow(img_bgr, road_poly)
    out = []
    for i, d in enumerate(res["doubles"]):
        qa, qb = _quad(d["a"], half, W, H), _quad(d["b"], half, W, H)
        hull = cv2.convexHull(np.array(qa + qb, np.float32)).reshape(-1, 2)
        out.append(_emit(DOUBLE, d["p"], [[round(float(x), 1), round(float(y), 1)] for x, y in hull], d["conf"], i))
    return out




