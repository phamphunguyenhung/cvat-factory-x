"""Ban do nhan DAY DU cua nen (khong de ho giua cac polygon) + do phu / gap map.

full_stuff_map(cm): moi pixel co id Cityscapes (argmax) -> ban do nhan nen {sky, building, vegetation, sidewalk, terrain,
wall, fence, road}; chi pixel cua lop KHONG thuoc bo nhan nen lam lo (cot/den/bien: quy ve nen lan can gan nhat; vat the
va capo: giu trong, vi co hop vat the hoac nam ngoai bo nhan) va manh vun < min_frac*anh duoc LAP theo nhan lan can gan nhat.
Polygon sau do lay duong bao NGOAI cua tung manh (nguyen vung, khong khoet lo).

coverage(shapes, W, H) -> mask bool pixel da co shape (polygon/hop); gap_stats / render_gap de bao % chua phu.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

STUFF_IDS: Dict[str, int] = {"road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4, "vegetation": 8, "terrain": 9, "sky": 10}
# lop mang/nho de len nen -> nen nam sau chung: lap theo nen lan can (pole 5, traffic light 6, traffic sign 7)
FILL_THROUGH = (5, 6, 7)
UNLABELED = 255


def full_stuff_map(cm: np.ndarray, min_frac: float = 0.001, close_k: int = 3) -> np.ndarray:
    """uint8 HxW: id Cityscapes cua nen, UNLABELED(255) = khong co nhan nen (vat the/capo/ngoai bo)."""
    H, W = cm.shape
    stuff_ids = list(STUFF_IDS.values())
    M = np.where(np.isin(cm, stuff_ids), cm, UNLABELED).astype(np.uint8)
    to_fill = np.isin(cm, FILL_THROUGH)
    k = np.ones((close_k, close_k), np.uint8)
    for sid in stuff_ids:   # bo manh vun nho -> lap theo lan can
        m = (M == sid).astype(np.uint8)
        if not m.any():
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < min_frac * W * H])
        M[small] = UNLABELED
        to_fill |= small
    known = M != UNLABELED
    if known.any() and to_fill.any():
        # nhan gan nhat cua pixel da biet cho moi pixel can lap
        dist, lbl = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(known)
        pix_label = np.zeros(lbl.max() + 1, np.uint8)
        pix_label[lbl[ys, xs]] = M[ys, xs]
        filled = pix_label[lbl]
        M = np.where(to_fill & ~known, filled, M).astype(np.uint8)
    return M


def coverage(shapes: Iterable[dict], W: int, H: int) -> np.ndarray:
    cov = np.zeros((H, W), np.uint8)
    for s in shapes:
        p = s["points"]
        if s["type"] == "rectangle":
            x1, y1, x2, y2 = (int(round(v)) for v in p[:4])
            cv2.rectangle(cov, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), 1, -1)
        elif len(p) >= 6:
            cv2.fillPoly(cov, [np.round(np.asarray(p, np.float32).reshape(-1, 2)).astype(np.int32)], 1)
    return cov.astype(bool)


def gap_stats(shapes: Iterable[dict], W: int, H: int) -> dict:
    cov = coverage(shapes, W, H)
    return {"uncovered_pct": round(100.0 * float((~cov).mean()), 2), "covered_pct": round(100.0 * float(cov.mean()), 2)}


def render_gap(img: np.ndarray, shapes: Iterable[dict], title: str) -> Tuple[np.ndarray, dict]:
    """Anh gap map: pixel chua co shape to DO; hien % chua phu."""
    H, W = img.shape[:2]
    cov = coverage(shapes, W, H)
    st = {"uncovered_pct": round(100.0 * float((~cov).mean()), 2)}
    out = (img.astype(np.float32) * 0.55).astype(np.uint8)
    red = out.copy()
    red[~cov] = (0, 0, 255)
    out = cv2.addWeighted(red, 0.6, out, 0.4, 0)
    cv2.putText(out, f"{title}  chua phu {st['uncovered_pct']}%", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, f"{title}  chua phu {st['uncovered_pct']}%", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out, st
