"""area/direct + area/alternative tu 2 vach bien lan xe minh (laneinterp) + VP + road tay.

KHONG dung do vach: 2 bien den tu anchors noi suy nen xe truoc che van khong lech.
direct = road ∩ vung giua 2 bien keo toi VP; alternative = road - dilate(direct).
Polygon tra ve la list cac polygon phang [x1,y1,x2,y2,...] (mot mask co the tach nhieu manh).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cvat_tools.lanes.laneinterp import lane_params

Point = Tuple[float, float]


def _line_xy(poly: Sequence[float], H: int) -> Tuple[float, float, float]:
    """(a, b) with x = a + b*(y-H) for the polyline, plus its bottom x (a)."""
    p = np.asarray(poly, float).reshape(-1, 2)
    if len(p) == 2 and p[0][1] == p[1][1]:
        return float(p[0][0]), 0.0, float(p[0][0])
    b, a = np.polyfit(p[:, 1], p[:, 0], 1) if len(p) > 2 else _two(p)
    return float(a + b * H), float(b), float(a + b * H)


def _two(p: np.ndarray):
    (x1, y1), (x2, y2) = p[0], p[-1]
    b = (x2 - x1) / (y2 - y1)
    return b, x1 - b * y1


def vp_from_pair(left: Sequence[float], right: Sequence[float], H: int) -> Optional[Point]:
    xl, bl, _ = _line_xy(left, H)
    xr, br, _ = _line_xy(right, H)
    if abs(bl - br) < 1e-6:
        return None
    dy = (xr - xl) / (bl - br)  # y-H where x equal
    return float(xl + bl * dy), float(H + dy)


def poly_mask(points: Sequence[float], W: int, H: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(points, np.float32)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def mask_polys(m: np.ndarray, eps: float = 3.0, min_area: float = 800) -> List[List[float]]:
    m = cv2.GaussianBlur(m.astype(np.uint8) * 255, (0, 0), 2.0) > 127
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append([round(float(v), 1) for v in a.flatten()])
    return out


def ego_wedge(left_lane_polyline, right_lane_polyline, vp: Optional[Point], H: int, W: int,
              y_top: float) -> np.ndarray:
    """Quad (4x2) between the two border lines, both lines passing through VP, from y=H+margin to y_top."""
    xl, _, _ = _line_xy(left_lane_polyline, H)
    xr, _, _ = _line_xy(right_lane_polyline, H)
    if vp is None:
        vp = vp_from_pair(left_lane_polyline, right_lane_polyline, H)
    if vp is None:
        raise ValueError("khong xac dinh duoc VP (2 bien song song)")
    vx, vy = vp
    yb = float(H)
    y_top = max(y_top, vy + 0.02 * H)

    def x_at(xb, y):
        return xb + (y - H) * (vx - xb) / (vy - H)

    return np.array([[x_at(xl, y_top), y_top], [x_at(xr, y_top), y_top],
                     [x_at(xr, yb + 10), yb + 10], [x_at(xl, yb + 10), yb + 10]], np.float32)


def build_area(road_poly, left_lane_polyline, right_lane_polyline, vp: Optional[Point], H: int, W: int,
               min_direct: float = 800, min_alt: float = 2500):
    """-> (direct_polys, alternative_polys). road_poly: flat polygon, hoac list cac polygon."""
    roads = road_poly if road_poly and isinstance(road_poly[0], (list, tuple, np.ndarray)) else [road_poly]
    road = np.zeros((H, W), bool)
    for p in roads:
        road |= poly_mask(p, W, H) > 0
    if not road.any():
        return [], []
    ys = np.nonzero(road)[0]
    wedge = ego_wedge(left_lane_polyline, right_lane_polyline, vp, H, W, float(ys.min()))
    wm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(wm, [np.round(wedge).astype(np.int32)], 1)
    direct = cv2.morphologyEx((road & (wm > 0)).astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((7, 7), np.uint8)).astype(bool)
    alt = road & ~cv2.dilate(direct.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    alt = cv2.morphologyEx(alt.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
    return mask_polys(direct, 3.0, min_direct), mask_polys(alt, 3.0, min_alt)
