"""Crosswalk (zebra) detector: repeated bright stripes inside the road polygon.

detect_crosswalk(img_bgr, road_poly) -> [{"label": "lane/crosswalk", "polygon": [x1,y1,...], "conf": float}]

Pipeline: dark frames -> gamma+CLAHE (detection only) -> multi-scale top-hat on grey, adaptive
threshold inside the (slightly dilated) road mask -> scan thin strips across the image in both
orientations (stripes side by side along x = far/ahead crossings, stacked along y = crossings seen
from the side); in every strip look for a run of >= min_stripes bright runs with regular period and
similar width -> merge neighbouring strips that hit -> hull simplified to <= 6 vertices.
Pure cv2/numpy, CPU only, no model, no network.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

LABEL = "lane/crosswalk"


def _enhance(gray: np.ndarray) -> np.ndarray:
    m = float(gray.mean())
    g = gray
    if m < 90:
        gam = float(np.clip(np.log(100 / 255.0) / np.log(max(m, 1.0) / 255.0), 0.25, 1.0))
        g = (255.0 * (gray / 255.0) ** gam).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _binary(gray: np.ndarray, inside: np.ndarray) -> np.ndarray:
    bw = np.zeros(gray.shape, np.uint8)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    for ksz in (7, 21, 45):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k).astype(np.float32)
        vals = th[inside]
        if vals.size < 100:
            continue
        thr = max(20.0, float(np.percentile(vals, 90)), float(vals.mean() + 2.0 * vals.std()))
        bw |= ((th > thr) & inside).astype(np.uint8)
    return bw


def _runs(profile: np.ndarray) -> List[tuple]:
    on = np.concatenate([[0], (profile > 0.5).astype(np.int8), [0]])
    d = np.diff(on)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _regular_seq(runs: List[tuple], min_n: int):
    """Longest consecutive subsequence of runs with similar width and regular centre spacing."""
    best: List[tuple] = []
    n = len(runs)
    for i in range(n):
        seq = [runs[i]]
        for j in range(i + 1, n):
            cand = seq + [runs[j]]
            w = np.array([b - a for a, b in cand], float)
            c = np.array([(a + b) / 2 for a, b in cand])
            gaps = np.diff(c)
            mg, mw = np.median(gaps), np.median(w)
            if (gaps.max() > 1.7 * mg or gaps.min() < 0.55 * mg or w.max() > 2.2 * mw or w.min() < 0.45 * mw
                    or mw < 2 or mg < 3.0):
                break
            seq = cand
        if len(seq) > len(best):
            best = seq
    return best if len(best) >= min_n else []


def _scan(bw: np.ndarray, min_n: int, transpose: bool) -> List[Dict[str, Any]]:
    m = bw.T if transpose else bw
    H, W = m.shape  # strips are rows [y, y+s) of m; stripes repeat along its x axis
    hits = []
    for s in (6, 12, 24):
        for y in range(0, H - s, s // 2):
            prof = m[y:y + s].mean(axis=0)
            seq = _regular_seq(_runs(prof), min_n)
            if seq:
                hits.append({"x0": seq[0][0], "x1": seq[-1][1], "y0": y, "y1": y + s, "n": len(seq)})
    # merge hits whose rects overlap/touch (union-find)
    k = len(hits)
    par = list(range(k))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            a, b = hits[i], hits[j]
            if a["y0"] <= b["y1"] and b["y0"] <= a["y1"] and a["x0"] < b["x1"] and b["x0"] < a["x1"]:
                ov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                if ov > 0.25 * min(a["x1"] - a["x0"], b["x1"] - b["x0"]):
                    par[f(i)] = f(j)
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for i, h in enumerate(hits):
        groups.setdefault(f(i), []).append(h)
    out = []
    for g in groups.values():
        pts = []
        for h in g:
            pts += [(h["x0"], h["y0"]), (h["x1"], h["y0"]), (h["x1"], h["y1"]), (h["x0"], h["y1"])]
        pts = np.array(pts, np.float32)
        if transpose:
            pts = pts[:, ::-1]
        out.append({"pts": pts, "n": max(h["n"] for h in g), "rows": len(g)})
    return out


def _merge_near(cs: List[Dict[str, Any]], gap: float = 30.0) -> List[Dict[str, Any]]:
    """Merge candidates whose bounding boxes are within `gap` px (fragments of one crossing)."""
    cs = list(cs)
    changed = True
    while changed:
        changed = False
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a = cv2.boundingRect(cs[i]["pts"]); b = cv2.boundingRect(cs[j]["pts"])
                if (a[0] - gap < b[0] + b[2] and b[0] - gap < a[0] + a[2]
                        and a[1] - gap < b[1] + b[3] and b[1] - gap < a[1] + a[3]):
                    cs[i] = {"pts": np.concatenate([cs[i]["pts"], cs[j]["pts"]]), "n": max(cs[i]["n"], cs[j]["n"]),
                             "rows": cs[i]["rows"] + cs[j]["rows"]}
                    del cs[j]
                    changed = True
                    break
            if changed:
                break
    return cs


def _simplify(hull: np.ndarray, max_pts: int = 6) -> np.ndarray:
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.12, 40):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) <= max_pts:
            return ap.reshape(-1, 2)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def detect_crosswalk(img_bgr: np.ndarray, road_poly: Sequence[float], min_stripes: int = 5,
                     min_conf: float = 0.0, min_rows: int = 2) -> List[Dict[str, Any]]:
    """Find zebra crossings inside road_poly (flat [x1,y1,...] or list of such polygons)."""
    H, W = img_bgr.shape[:2]
    polys = road_poly if len(road_poly) and isinstance(road_poly[0], (list, tuple, np.ndarray)) else [road_poly]
    road = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(road, [np.round(np.asarray(p, float).reshape(-1, 2)).astype(np.int32)], 255)
    road = cv2.dilate(road, np.ones((5, 5), np.uint8))
    gray = _enhance(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    bw = _binary(gray, road > 0)
    res: List[Dict[str, Any]] = []
    for tr in (False, True):
        for c in _merge_near(_scan(bw, min_stripes, tr)):
            if c["rows"] < min_rows:
                continue
            poly = _simplify(cv2.convexHull(c["pts"]))
            poly[:, 0] = poly[:, 0].clip(0, W)
            poly[:, 1] = poly[:, 1].clip(0, H)
            conf = float(min(1.0, 0.2 + 0.08 * c["n"] + 0.05 * c["rows"]))
            if conf < min_conf:
                continue
            res.append({"label": LABEL, "polygon": [round(float(v), 1) for v in poly.flatten()], "conf": round(conf, 3)})
    res.sort(key=lambda r: -r["conf"])
    return res
