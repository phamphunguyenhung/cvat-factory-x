"""Loc vach lane sai cho: bo vach neu phan lon dien tich chong len vat the (hop xe) hoac nam ngoai road nguoi ve;
ha conf vach khong co bang chung son that (ti le polygon trung mask son < min_paint).

Nguong co dinh, khong chinh theo frame: max_thing=0.5, min_in_road=0.5, min_paint=0.25 (conf *= 0.5).
Chi ap dung cho lane/* mot-vach (khong crosswalk, khong curb: luat rieng; khong area/*).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from cvat_tools.lanes import framegen as FG

FILTERED = {"lane/single white", "lane/double white", "lane/single yellow", "lane/double yellow", "lane/single other", "lane/double other"}
MAX_THING = 0.5
MIN_IN_ROAD = 0.5
MIN_PAINT = 0.25
CONF_PENALTY = 0.5


def _mask(poly, W, H, grow: int = 0) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)], 1)
    if grow:
        m = cv2.dilate(m, np.ones((grow, grow), np.uint8))
    return m.astype(bool)


def paint_evidence(img: np.ndarray, roads: Sequence[Sequence[float]]):
    """Mask son theo luat framegen (sang rieng cho frame toi + top-hat)."""
    H, W = img.shape[:2]
    rm = FG.road_mask(roads, W, H)
    enh, dark, _ = FG.brighten_for_paint(img, rm)
    return FG.paint_mask(enh, rm, dark).astype(bool)


def filter_lanes(lane_shapes: List[dict], img: np.ndarray, roads: Sequence[Sequence[float]],
                 thing_boxes: Sequence[Sequence[float]]) -> Tuple[List[dict], List[dict]]:
    """-> (shapes giu, [{label, why, ...}] bi bo). Shape giu co the them issue 'no_paint_evidence' va conf giam."""
    H, W = img.shape[:2]
    road = FG.road_mask(roads, W, H).astype(bool) if roads else None
    things = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in thing_boxes:
        things[int(max(0, y1)):int(y2) + 1, int(max(0, x1)):int(x2) + 1] = True
    paint = None
    keep, dropped = [], []
    for s in lane_shapes:
        if s["label"] not in FILTERED:
            keep.append(s)
            continue
        m = _mask(s["polygon"], W, H)
        a = float(m.sum())
        if a <= 0:
            continue
        on_thing = float((m & things).sum()) / a
        in_road = float((m & road).sum()) / a if road is not None else 1.0
        if on_thing > MAX_THING:
            dropped.append({"label": s["label"], "why": "tren_vat_the", "ratio": round(on_thing, 2)})
            continue
        if in_road < MIN_IN_ROAD:
            dropped.append({"label": s["label"], "why": "ngoai_road", "ratio": round(in_road, 2)})
            continue
        if paint is None:
            paint = paint_evidence(img, roads) if roads else np.zeros((H, W), bool)
        pr = float((_mask(s["polygon"], W, H, 5) & paint).sum()) / max(1.0, float(_mask(s["polygon"], W, H, 5).sum()))
        s = dict(s)
        if pr < MIN_PAINT:
            s["conf"] = round(float(s.get("conf", 0.5)) * CONF_PENALTY, 3)
            s["_issues"] = list(s.get("_issues", [])) + [f"no_paint_evidence:{pr:.2f}"]
        keep.append(s)
    return keep, dropped
