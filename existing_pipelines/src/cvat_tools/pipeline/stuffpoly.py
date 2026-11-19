"""Class map Cityscapes -> polygon nen (nguyen vung), cot (chu nhat hep), hop bien bao.

Nguong dien tich/sang LAY NGUYEN tu tools/cvat-seg-copilot/scripts/bbox_draft.py (STUFF, MIN_LIGHT), khong chinh theo frame nao.
Polygon chi lay duong bao NGOAI (khong khoet lo, quy uoc 1). Cot: luat hep cao/rong theo ruleset review (pole_max_width,
pole_min_aspect) -> hop xoay nho nhat 4 dinh.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from cvat_tools.segmentation import labelmap as LM
from cvat_tools.segmentation import pole_rule as PR
from cvat_tools.segmentation.onnx_seg import CITYSCAPES

# label -> (dien tich toi thieu theo ti le anh, can du sang, so manh toi da)
STUFF = {"sky": (0.01, False, 4), "building": (0.01, False, 8), "vegetation": (0.004, False, 8),
         "sidewalk": (0.01, False, 4), "terrain": (0.004, True, 4), "wall": (0.004, True, 4), "fence": (0.002, True, 4)}
SIGN = (0.0003, False, 12)   # traffic sign -> hop
POLE_MIN_AREA = 40.0
POLE_MIN_HEIGHT = 20.0
MIN_LIGHT = 40.0             # nhu bbox_draft.py (gray TB cua manh < 40 -> bo, cho label can du sang)
EPS = 2.0


def _components(mask: np.ndarray, min_area: float, close_k: int = 5):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    idx = [i for i in sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA]) if st[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, st, idx


def mask_to_polygon(mask: np.ndarray, eps: float = EPS) -> List[float]:
    """Duong bao ngoai lon nhat -> polygon phang (khong lo)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return [float(v) for v in a.reshape(-1)] if len(a) >= 3 else []


def stuff_polygons(cm: np.ndarray, gray: np.ndarray, labels=None, fmap: np.ndarray | None = None) -> Dict[str, List[dict]]:
    """{label: [{polygon, conf, _src}]} tu BAN DO NHAN DAY DU (labelmap.full_stuff_map: khong ho, lap theo nen lan can)."""
    H, W = cm.shape
    fmap = LM.full_stuff_map(cm) if fmap is None else fmap
    out: Dict[str, List[dict]] = {}
    for lab, (frac, need_light, top) in STUFF.items():
        if labels is not None and lab not in labels:
            continue
        lab_im, st, idx = _components(fmap == CITYSCAPES[lab], frac * W * H)
        res = []
        for i in idx[:top]:
            comp = lab_im == i
            if need_light and float(gray[comp].mean()) < MIN_LIGHT:
                continue
            poly = mask_to_polygon(comp)
            if poly:
                res.append({"polygon": poly, "conf": 0.0, "_src": "b2:" + lab})
        if res:
            out[lab] = res
    return out


def sign_boxes(cm: np.ndarray) -> List[dict]:
    """traffic sign tu phan vung Cityscapes: moi manh -> hop bao (rectangle)."""
    H, W = cm.shape
    lab, st, idx = _components(cm == CITYSCAPES["traffic sign"], SIGN[0] * W * H, 3)
    out = []
    for i in idx[:SIGN[2]]:
        x, y, w, h = (int(st[i, k]) for k in range(4))
        out.append({"box": [float(x), float(y), float(x + w), float(y + h)], "conf": 0.0, "_src": "b2:traffic sign"})
    return out


def pole_polygons(cm: np.ndarray, max_width: float = 40.0, min_aspect: float = 3.0, thing_boxes=()) -> List[dict]:
    """Cot theo segmentation.pole_rule (noi manh thang hang, 1 hinh chu nhat xoay/cot, bo cot trong cay/de vat the)."""
    res = PR.extract_poles(cm == CITYSCAPES["pole"], cm == CITYSCAPES["vegetation"], thing_boxes,
                           {"max_width": max_width, "min_aspect": min_aspect})
    return [{"polygon": r["polygon"], "conf": 0.5, "_src": "b2:pole", "_meta": r} for r in res]
