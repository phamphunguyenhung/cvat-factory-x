"""Per-label lane detectors behind one registry + unified output.

detect_all(img_bgr, road_poly, sidewalk_poly=None, labels=None, min_conf=0.0)
  -> [{"label", "polygon": [x1,y1,...] (flat), "conf", "polyline"?, "id"?}, ...]
road_poly / sidewalk_poly: one flat polygon [x1,y1,...] or a list of flat polygons.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .crosswalk import detect_crosswalk
from .curb import detect_curb
from .white import detect_double_white, detect_single_white
from .yellow import detect_double_yellow, detect_single_yellow


def _polys(p: Any) -> List[List[float]]:
    """-> list of flat polygons (accepts None, flat, [[x,y],..] or list of those)."""
    if p is None:
        return []
    try:
        a = np.asarray(p, np.float32)
        if a.ndim == 1:
            return [a.tolist()] if a.size >= 6 else []
        if a.ndim == 2 and a.shape[1] == 2:
            return [a.reshape(-1).tolist()] if a.size >= 6 else []
    except ValueError:
        pass
    return [np.asarray(q, np.float32).reshape(-1).tolist() for q in p if np.asarray(q).size >= 6]


def flat_polygon(poly: Any) -> List[float]:
    return [round(float(v), 1) for v in np.asarray(poly, np.float32).reshape(-1)]


def _per_road(fn):
    # yellow detectors take a single polygon: run per road polygon and merge
    def run(img, roads, sidewalk):
        out: List[Dict[str, Any]] = []
        for r in roads:
            out.extend(fn(img, r))
        return out
    return run


def _all_roads(fn):
    return lambda img, roads, sidewalk: fn(img, roads if len(roads) > 1 else roads[0])


REGISTRY: Dict[str, Callable] = {
    "lane/crosswalk": _all_roads(detect_crosswalk),
    "lane/single white": _all_roads(detect_single_white),
    "lane/double white": _all_roads(detect_double_white),
    "lane/single yellow": _per_road(detect_single_yellow),
    "lane/double yellow": _per_road(detect_double_yellow),
    "lane/road curb": lambda img, roads, sidewalk: detect_curb(
        img, roads if len(roads) > 1 else roads[0], sidewalk if sidewalk else None),
}


def detect_all(img_bgr: np.ndarray, road_poly: Any, sidewalk_poly: Any = None,
               labels: Optional[Sequence[str]] = None, min_conf: float = 0.0) -> List[Dict[str, Any]]:
    roads = _polys(road_poly)
    if not roads:
        return []
    sw = _polys(sidewalk_poly)
    res: List[Dict[str, Any]] = []
    for label, fn in REGISTRY.items():
        if labels is not None and label not in labels:
            continue
        for d in fn(img_bgr, roads, sw):
            conf = float(d.get("conf", 0.0))
            if conf < min_conf:
                continue
            o = {"label": label, "polygon": flat_polygon(d["polygon"]), "conf": round(conf, 3)}
            if d.get("polyline") is not None:
                o["polyline"] = flat_polygon(d["polyline"])
            if d.get("id") is not None:
                o["id"] = d["id"]
            res.append(o)
    return res
