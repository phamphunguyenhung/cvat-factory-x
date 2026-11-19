"""Luat COT (pole) theo QUY_UOC 3.6: moi cot = MOT polygon chu nhat hep (4 dinh) om than cot tu chan toi dinh.

  poles = extract_poles(pole_mask, veg_mask=None, thing_boxes=())   # [{polygon, length, width, fill, n_blobs}]

Quy trinh: blob pole du dien tich -> uoc luong truc (fitLine) -> noi cac manh THANG HANG (lech goc nho, lech ngang nho,
khoang ho doc truc nho) -> moi nhom: truc chung fit tu moi pixel, chieu dai = tam chieu len truc, be rong = TRUNG VI be rong
cac lat cat (khong phinh theo diem nhieu) -> hinh chu nhat xoay. Loai: khong du manh (cao/rong <= min_aspect, ngan, qua rong),
cot nam trong than cay (vegetation phu > veg_frac) hoac de len vat the (hop phu > thing_frac) TRU khi mask ro (fill >= clear_fill).
Cot gap khuc (6-8 dinh) chua lam: chi cot thang.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np

CFG = dict(min_area=30.0, min_length=20.0, min_aspect=3.0, max_width=40.0, max_angle=8.0, max_perp=4.0, max_gap=30.0,
           veg_frac=0.5, ring_px=9, ring_veg=0.5, thing_frac=0.5, clear_fill=0.6, min_width=2.0)


def _blobs(mask: np.ndarray, min_area: float):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return [(lab == i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def _axis(pts: np.ndarray):
    """pts (N,2) -> (centroid, unit direction) bang PCA; huong len tren (dy<0)."""
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    if d[1] > 0:
        d = -d
    return c, d, s


def _ang(d1, d2) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d1, d2))), 0, 1))))


def _group(blobs: List[np.ndarray], cfg: dict) -> List[List[int]]:
    info = []
    for b in blobs:
        ys, xs = np.nonzero(b)
        p = np.stack([xs, ys], 1).astype(np.float32)
        c, d, _ = _axis(p)
        t = (p - c) @ d
        info.append((c, d, float(t.min()), float(t.max()), p))
    par = list(range(len(blobs)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            ci, di, ai, bi, pi = info[i]
            cj, dj, aj, bj, pj = info[j]
            # chi noi manh co truc thuc su dai (blob tron nho khong co huong ro) va cung huong
            if (bi - ai) < 6 or (bj - aj) < 6 or _ang(di, dj) > cfg["max_angle"]:
                continue
            long_i = (bi - ai) >= (bj - aj)
            c0, d = (ci, di) if long_i else (cj, dj)
            other = pj if long_i else pi
            mine = pi if long_i else pj
            perp = np.array([-d[1], d[0]], np.float32)
            if abs(float(np.median((other - c0) @ perp))) > cfg["max_perp"]:
                continue
            t1, t2 = (mine - c0) @ d, (other - c0) @ d
            gap = max(float(t1.min()), float(t2.min())) - min(float(t1.max()), float(t2.max()))
            if gap <= cfg["max_gap"]:
                par[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(blobs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _frac_inside(rect_poly: np.ndarray, region: np.ndarray) -> float:
    m = np.zeros(region.shape, np.uint8)
    cv2.fillPoly(m, [np.round(rect_poly).astype(np.int32)], 1)
    a = float(m.sum())
    return float((m.astype(bool) & region).sum()) / a if a > 0 else 0.0


def _ring_frac(rect_poly: np.ndarray, region: np.ndarray, px: int) -> float:
    """Ti le `region` trong VANH quanh hinh chu nhat (dilate px, tru chinh hinh): cot giua than cay co vanh toan la tan la."""
    m = np.zeros(region.shape, np.uint8)
    cv2.fillPoly(m, [np.round(rect_poly).astype(np.int32)], 1)
    ring = cv2.dilate(m, np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool) & ~m.astype(bool)
    n = float(ring.sum())
    return float((ring & region).sum()) / n if n > 0 else 0.0


def extract_poles(pole_mask: np.ndarray, veg_mask: np.ndarray | None = None, thing_boxes: Sequence[Sequence[float]] = (),
                  cfg: dict | None = None) -> List[dict]:
    c = {**CFG, **(cfg or {})}
    H, W = pole_mask.shape[:2]
    blobs = _blobs(pole_mask, 4.0)
    if not blobs:
        return []
    things = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in thing_boxes:
        things[int(max(0, y1)):int(y2) + 1, int(max(0, x1)):int(x2) + 1] = True
    out = []
    for grp in _group(blobs, c):
        m = np.zeros((H, W), bool)
        for i in grp:
            m |= blobs[i]
        ys, xs = np.nonzero(m)
        if len(xs) < c["min_area"]:
            continue
        p = np.stack([xs, ys], 1).astype(np.float32)
        ctr, d, _ = _axis(p)
        perp = np.array([-d[1], d[0]], np.float32)
        t, s = (p - ctr) @ d, (p - ctr) @ perp
        lo, hi = float(t.min()), float(t.max())
        length = hi - lo
        # be rong = trung vi be rong cac lat cat doc truc (khong phinh theo diem nhieu)
        bins = np.arange(lo, hi + 1.0, 2.0)
        idx = np.digitize(t, bins)
        ws = [float(s[idx == k].max() - s[idx == k].min() + 1) for k in np.unique(idx) if (idx == k).sum() >= 2]
        width = max(float(np.median(ws)) if ws else 1.0, c["min_width"])
        if length < c["min_length"] or length / width <= c["min_aspect"] or width > c["max_width"]:
            continue
        smid = float((s.max() + s.min()) / 2)
        p0 = ctr + d * lo + perp * smid
        p1 = ctr + d * hi + perp * smid
        h = width / 2
        rect = np.array([p0 + perp * h, p1 + perp * h, p1 - perp * h, p0 - perp * h], np.float32)
        rect[:, 0] = rect[:, 0].clip(0, W)
        rect[:, 1] = rect[:, 1].clip(0, H)
        area = float(cv2.contourArea(rect))
        fill = float(m.sum()) / max(area, 1.0)
        clear = fill >= c["clear_fill"]
        if veg_mask is not None and not clear and (_frac_inside(rect, veg_mask) > c["veg_frac"]
                                                    or _ring_frac(rect, veg_mask, c["ring_px"]) > c["ring_veg"]):
            continue
        if thing_boxes and not clear and _frac_inside(rect, things) > c["thing_frac"]:
            continue
        out.append({"polygon": [round(float(v), 1) for v in rect.reshape(-1)], "length": round(length, 1), "width": round(width, 1),
                    "fill": round(fill, 2), "n_blobs": len(grp)})
    out.sort(key=lambda r: -r["length"])
    return out
