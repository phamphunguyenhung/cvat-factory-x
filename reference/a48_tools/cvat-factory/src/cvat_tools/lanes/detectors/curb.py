"""Dedicated detector for `lane/road curb` = the kerb line where the road meets the pavement.

Signature: detect_curb(img_bgr, road_poly[, sidewalk_poly]) -> [{"label","polyline","polygon","conf","id"}]
Polygons may be flat [x1,y1,...] or [[x,y],...]; `sidewalk_poly` may also be a list of polygons.

The hand-drawn road polygon already contains the kerb as its boundary, so no search over the image is done:
the road outline is simplified, every straight run is tested, and only runs that are really a road/kerb edge
survive:
  * not on the image border, not the far top cut, not below the bonnet line, not flat (stop line / horizon);
  * outward neighbour is pavement - the share of outside samples that fall on `sidewalk_poly` (when given), or,
    without it, a consistent light/dark step across the edge in the image (|dL| >= threshold);
  * consecutive supported runs that bend gently are chained into one polyline.
Below the support bar nothing is returned.
"""
import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

LABEL = "lane/road curb"

CFG: Dict[str, Any] = {
    "bottom": 0.84,
    "border_px": 6,         # runs hugging the image border are crops, not kerbs
    "min_run_len": 45,      # px @720p
    "min_tilt_deg": 14,
    "eps": 3.0,             # approxPolyDP tolerance @720p
    "off": 9,               # px: how far each side of the edge is sampled
    "sw_support": 0.50,     # share of outside samples on sidewalk
    "grad_min": 10.0,       # |dL| across the edge when no sidewalk is given
    "grad_support": 0.65,   # share of samples agreeing on the step direction
    "chain_deg": 25.0,
    "min_conf": 0.45,
    "max_out": 4,
}


def _polys(p) -> List[np.ndarray]:
    if p is None:
        return []
    try:
        a = np.asarray(p, np.float32)
        if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 2):
            return [a.reshape(-1, 2)] if a.size >= 6 else []
    except ValueError:
        pass
    return [np.asarray(q, np.float32).reshape(-1, 2) for q in p if np.asarray(q).size >= 6]


def _mask(shape, polys: List[np.ndarray]) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for q in polys:
        cv2.fillPoly(m, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 255)
    return m


def _tilt(a, b) -> float:
    ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
    return min(ang, 180 - ang)


def _sample_run(a, b, scale: float, road: np.ndarray, gray: np.ndarray, side: Optional[np.ndarray]) -> Dict[str, Any]:
    """Support statistics of the road outline run a->b."""
    H, W = road.shape
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    n = max(5, int(length / (6 * scale)))
    t = np.linspace(0.05, 0.95, n)
    px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    nx, ny = -(b[1] - a[1]) / length, (b[0] - a[0]) / length
    off = CFG["off"] * scale
    # orient the normal outward from the road
    ox, oy = np.clip(px + nx * off, 0, W - 1).astype(int), np.clip(py + ny * off, 0, H - 1).astype(int)
    ix, iy = np.clip(px - nx * off, 0, W - 1).astype(int), np.clip(py - ny * off, 0, H - 1).astype(int)
    if road[oy, ox].mean() > road[iy, ix].mean():
        nx, ny = -nx, -ny
        ox, oy, ix, iy = ix, iy, ox, oy
    keep = (px > CFG["border_px"]) & (px < W - 1 - CFG["border_px"]) & (py > CFG["border_px"]) & (py < H - 1 - CFG["border_px"]) & (py < CFG["bottom"] * H)
    out = {"ok_frac": float(keep.mean()), "sw": 0.0, "grad": 0.0, "dl": 0.0}
    if keep.sum() < 3:
        return out
    if side is not None:
        out["sw"] = float((side[oy, ox] > 0)[keep].mean())
    # light/dark step across the edge, 3-px averaged strips
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    dl = blur[oy, ox].astype(np.float32) - blur[iy, ix].astype(np.float32)
    d = dl[keep]
    if d.size:
        sgn = np.sign(np.median(d))
        out["grad"] = float(((np.sign(d) == sgn) & (np.abs(d) >= CFG["grad_min"])).mean())
        out["dl"] = float(np.median(np.abs(d)))
    return out


def _outline(road: np.ndarray, scale: float) -> List[np.ndarray]:
    cs, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 500 * scale * scale:
            continue
        out.append(cv2.approxPolyDP(c, CFG["eps"] * scale, True).reshape(-1, 2).astype(np.float32))
    return out


def _poly_quad(pl: List[List[float]], half: float, W: int, H: int) -> List[List[float]]:
    left = [[x - half, y] for x, y in pl]
    right = [[x + half, y] for x, y in reversed(pl)]
    return [[round(min(max(x, 0.0), W), 1), round(min(max(y, 0.0), H), 1)] for x, y in left + right]


def detect_curb(img_bgr: np.ndarray, road_poly, sidewalk_poly=None) -> List[Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    scale = H / 720.0
    road = _mask(img_bgr.shape, _polys(road_poly))
    if not road.any():
        return []
    sw_polys = _polys(sidewalk_poly)
    side = None
    if sw_polys:
        side = cv2.dilate(_mask(img_bgr.shape, sw_polys), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(13 * scale) | 1, int(13 * scale) | 1)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    runs: List[Dict[str, Any]] = []
    for pts in _outline(road, scale):
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if np.hypot(*(b - a)) < CFG["min_run_len"] * scale or _tilt(a, b) < CFG["min_tilt_deg"]:
                continue
            st = _sample_run(a, b, scale, road, gray, side)
            if st["ok_frac"] < 0.6:
                continue
            support = st["sw"] if side is not None else st["grad"]
            need = CFG["sw_support"] if side is not None else CFG["grad_support"]
            if support < need:
                continue
            # a sidewalk-backed run still needs to be a visible edge more often than not (not a pure label border)
            length = float(np.hypot(*(b - a)))
            conf = 0.6 * support + 0.25 * min(1.0, length / (200 * scale)) + 0.15 * min(1.0, st["dl"] / 30.0)
            if side is None:
                conf *= 0.85   # no sidewalk polygon to confirm the other side: less sure
            if side is not None and st["grad"] > 0:
                conf = min(1.0, conf + 0.05 * st["grad"])
            runs.append({"a": a, "b": b, "conf": conf, "len": length})

    # chain end-to-start neighbours that bend gently into one polyline
    chains: List[List[Dict[str, Any]]] = []
    for r in runs:
        for ch in chains:
            last = ch[-1]
            if np.hypot(*(last["b"] - r["a"])) < 4 * scale:
                v1, v2 = last["b"] - last["a"], r["b"] - r["a"]
                ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
                if ang < CFG["chain_deg"]:
                    ch.append(r)
                    break
        else:
            chains.append([r])
    res: List[Dict[str, Any]] = []
    half = 4.0 * scale
    for ch in chains:
        pl = [[float(ch[0]["a"][0]), float(ch[0]["a"][1])]] + [[float(r["b"][0]), float(r["b"][1])] for r in ch]
        tot = sum(r["len"] for r in ch)
        conf = sum(r["conf"] * r["len"] for r in ch) / tot
        if conf < CFG["min_conf"]:
            continue
        pl = [[round(x, 1), round(y, 1)] for x, y in pl]
        res.append({"label": LABEL, "polyline": pl, "polygon": _poly_quad(pl, half, W, H), "conf": round(float(conf), 3), "len": tot})
    res.sort(key=lambda d: -d["len"] * d["conf"])
    res = res[:CFG["max_out"]]
    for i, d in enumerate(res):
        d.pop("len")
        d["id"] = i
    return res

