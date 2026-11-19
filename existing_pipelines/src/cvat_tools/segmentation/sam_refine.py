"""SAM (MobileSAM / SAM2.1) qua ultralytics, lazy-load, CPU. Dung cho vat the (xe, cot, bien)
va 'click dung/sai' cuc bo. Weights lay tu tools/cvat-seg-copilot (khong sua repo do).

prompts = {"box": [x1,y1,x2,y2] | None, "points": [[x,y],...], "labels": [1,0,...]}  (1=dung, 0=sai)
Tra polygon phang [x1,y1,...] hoac None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

WEIGHTS = {"mobilesam": "mobile_sam.pt", "sam2": "sam2.1_b.pt"}
_MODELS: Dict[str, object] = {}


def _weights_dir() -> Path:
    for p in Path(__file__).resolve().parents:
        d = p / "tools" / "cvat-seg-copilot"
        if d.exists():
            return d
    raise FileNotFoundError("tools/cvat-seg-copilot")


def load(backend: str = "mobilesam"):
    if backend not in WEIGHTS:
        raise ValueError(f"backend khong ho tro: {backend}")
    if backend not in _MODELS:
        from ultralytics import SAM  # lazy
        w = _weights_dir() / WEIGHTS[backend]
        if not w.exists():
            raise FileNotFoundError(w)
        _MODELS[backend] = SAM(str(w))
    return _MODELS[backend]


def _roi(prompts: dict, shape, margin: float):
    H, W = shape[:2]
    pts = []
    if prompts.get("box"):
        b = prompts["box"]; pts += [[b[0], b[1]], [b[2], b[3]]]
    pts += [list(p) for p in prompts.get("points", [])]
    a = np.asarray(pts, float)
    lo, hi = a.min(0), a.max(0)
    m = max(hi - lo) * margin + 16
    x0, y0 = np.maximum(0, lo - m).astype(int)
    x1, y1 = np.minimum([W, H], hi + m).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def mask_to_polygon(mask: np.ndarray, max_points: int = 40, eps: float = 1.0,
                    must_contain: Optional[List[List[float]]] = None) -> Optional[List[float]]:
    import cv2
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    for x, y in (must_contain or []):  # uu tien thanh phan chua diem duong
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and lab[yi, xi] > 0:
            keep = int(lab[yi, xi]); break
    m = (lab == keep).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    e = eps
    ap = cv2.approxPolyDP(c, e, True)
    while len(ap) > max_points and e < 20:
        e *= 1.3
        ap = cv2.approxPolyDP(c, e, True)
    if len(ap) < 3:
        return None
    return [float(v) for v in ap.reshape(-1)]


def refine_with_sam(img_bgr: np.ndarray, prompts: dict, backend: str = "mobilesam", local: bool = True,
                    margin: float = 0.5, max_points: int = 40, device: str = "cpu") -> Optional[List[float]]:
    """Cat ROI quanh prompt (local=True) de SAM nhin ro hon + nhanh hon, roi quy ve toa do anh goc."""
    model = load(backend)
    x0, y0, x1, y1 = (0, 0, img_bgr.shape[1], img_bgr.shape[0])
    if local:
        x0, y0, x1, y1 = _roi(prompts, img_bgr.shape, margin)
    crop = np.ascontiguousarray(img_bgr[y0:y1, x0:x1])
    off = np.array([x0, y0], float)
    kw = {}
    box = prompts.get("box")
    pts = prompts.get("points") or []
    if box:
        kw["bboxes"] = [[box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]]
    if pts:
        kw["points"] = [[float(p[0] - x0), float(p[1] - y0)] for p in pts]
        kw["labels"] = [int(v) for v in (prompts.get("labels") or [1] * len(pts))]
    if not kw:
        raise ValueError("can box hoac points")
    res = model(crop, device=device, verbose=False, **kw)
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return None
    mask = res[0].masks.data[0].cpu().numpy()
    if mask.shape != crop.shape[:2]:
        import cv2
        mask = cv2.resize(mask.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    pos = [[p[0] - x0, p[1] - y0] for p, l in zip(pts, kw.get("labels", [])) if l == 1]
    poly = mask_to_polygon(mask, max_points, must_contain=pos)
    if poly is None:
        return None
    a = np.asarray(poly).reshape(-1, 2) + off
    return [round(float(v), 2) for v in a.reshape(-1)]
