"""Tang (z_order) + giai chong lan + co che khuat cho segmentation job 285.

Thuan numpy/shapely. Polygon NGUYEN VUNG (khong khoet lo); vat the tren dien bang z_order.
Instance vao: {label, polygon(phang [x0,y0,x1,y1,...]), conf, id}.
Shape ra: {id, label, polygon, z_order, occluded, truncated, conf, _issues,
           occlusion_ratio, truncated_ratio}.
"""
from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as _box
from shapely.ops import unary_union
from shapely.validation import make_valid

# z_order theo CACH NGUOI DUNG VE o frame 71 job 285 (frame71_target.json), chot 14/11/2026:
# sky -9; building/vegetation (+terrain/wall/fence cung nhom nen) -8; sidewalk -7; pole -2; road -1;
# area/direct, area/alternative, lane/* 0; vat the 0. (Cu: road -6, area -5/-4, lane -3, bien -1, xe may +1, nguoi +2.)
LAYER_ORDER: dict[str, int] = {
    "sky": -9, "building": -8, "vegetation": -8, "terrain": -8, "wall": -8, "fence": -8,
    "sidewalk": -7, "pole": -2, "road": -1,
    "area/alternative": 0, "area/direct": 0,
    "lane/crosswalk": 0, "lane/road curb": 0, "lane/single white": 0, "lane/single yellow": 0,
    "lane/double white": 0, "lane/double yellow": 0, "lane/single other": 0, "lane/double other": 0,
    "traffic sign": 0, "car": 0, "truck": 0, "bus": 0, "train": 0, "motorcycle": 0, "bicycle": 0,
    "pedestrian": 0, "rider": 0, "traffic light": 0,
}
ALIASES = {"person": "pedestrian", "traffic_light": "traffic light", "traffic_sign": "traffic sign"}

LANE_LABELS = {k for k in LAYER_ORDER if k.startswith("lane/")}
AREA_LABELS = {"area/direct", "area/alternative"}
OVERLAY = LANE_LABELS | AREA_LABELS  # ve de len road; khong che nen/vat the
INSIDE_ROAD = OVERLAY  # rang buoc: nam trong road
CONF_PENALTY = 0.7


def canon(label: str) -> str:
    return ALIASES.get(label, label)


def z_order(label: str) -> int:
    return LAYER_ORDER.get(canon(label), 0)


def _polys_only(g):
    if g.is_empty:
        return Polygon()
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else Polygon()


def to_geom(flat) -> Polygon:
    pts = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]
    if len(pts) < 3:
        return Polygon()
    g = Polygon(pts)
    if not g.is_valid:
        g = make_valid(g)
    return _polys_only(g)


def _parts(g) -> list[Polygon]:
    if g.is_empty:
        return []
    return [p for p in (g.geoms if isinstance(g, MultiPolygon) else [g]) if p.area > 0]


def _ring_flat(ring) -> list[float]:
    return [round(v, 2) for xy in list(ring.coords)[:-1] for v in xy]


def to_flat(poly: Polygon) -> list[float]:
    return _ring_flat(poly.exterior)


def _bottom(g) -> float:
    return g.bounds[3]


def resolve(instances, frame_wh=None, occ_thresh: float = 0.30, trunc_thresh: float = 0.02,
            min_area: float = 1.0, return_dropped: bool = False):
    """Giai chong lan. frame_wh=(w,h) bat clip khung + truncated; None = khong clip.

    Instance cat/ru ra nhieu manh -> nhieu shape cung id.
    Khuat: chi tinh tang TREN (z lon hon; cung z 0 giua cac vat the thi vat thap hon trong anh = gan hon; vat the luon de len lane/area).
    Lop phu (lane/area) khong che nen/vat the.
    """
    work, dropped = [], []
    for i, ins in enumerate(instances):
        label = canon(ins["label"])
        g = to_geom(ins["polygon"])
        a0 = g.area
        rec = {"id": ins.get("id", i), "label": label, "conf": float(ins.get("conf", 1.0)), "issues": []}
        if a0 < min_area:
            dropped.append({**rec, "why": "degenerate"})
            continue
        trunc = 0.0
        if frame_wh:
            g = _polys_only(g.intersection(_box(0, 0, frame_wh[0], frame_wh[1])))
            if g.area < min_area:
                dropped.append({**rec, "why": "outside_frame"})
                continue
            trunc = max(0.0, 1.0 - g.area / a0)
        rec.update(g=g, trunc=trunc, z=z_order(label))
        work.append(rec)

    # --- rang buoc cau truc ---
    roads = [w["g"] for w in work if w["label"] == "road"]
    if roads:
        road = unary_union(roads)
        for w in work:
            if w["label"] in INSIDE_ROAD:
                a = w["g"].area
                g2 = _polys_only(w["g"].intersection(road))
                if g2.area < min_area:
                    dropped.append({"id": w["id"], "label": w["label"], "conf": w["conf"], "why": "outside_road"})
                    w["g"] = Polygon()
                    continue
                lost = 1 - g2.area / a
                if lost > 0.02:
                    w["conf"] *= CONF_PENALTY
                    w["issues"].append(f"clipped_to_road:{lost:.2f}")
                w["g"] = g2
        work = [w for w in work if not w["g"].is_empty]
    directs = [w["g"] for w in work if w["label"] == "area/direct"]
    if directs:
        direct = unary_union(directs)
        for w in work:
            if w["label"] == "area/alternative":
                a = w["g"].area
                g2 = _polys_only(w["g"].difference(direct))
                lost = 1 - g2.area / a if a else 0
                if lost > 0.02:
                    w["conf"] *= CONF_PENALTY
                    w["issues"].append(f"overlap_direct:{lost:.2f}")
                w["g"] = g2
        work = [w for w in work if w["g"].area >= min_area]

    # --- khuat ---
    out = []
    for w in work:
        occs = []
        for o in work:
            if o is w:
                continue
            if w["label"] not in OVERLAY and o["label"] in OVERLAY:
                continue
            if w["label"] in OVERLAY and o["label"] not in OVERLAY and o["z"] >= 0:
                occs.append(o["g"])   # lane/area cung z 0 voi vat the nhung nam tren mat duong: vat the luon de len
                continue
            if o["z"] > w["z"] or (o["z"] == w["z"] >= 0 and _bottom(o["g"]) > _bottom(w["g"])):
                occs.append(o["g"])
        ratio = 0.0
        if occs:
            ratio = min(1.0, w["g"].intersection(unary_union(occs)).area / w["g"].area)
        for part in _parts(w["g"]):
            out.append({
                "id": w["id"], "label": w["label"], "polygon": to_flat(part), "z_order": w["z"],
                "occluded": ratio >= occ_thresh, "truncated": w["trunc"] >= trunc_thresh,
                "conf": round(w["conf"], 4), "_issues": list(w["issues"]),
                "occlusion_ratio": round(ratio, 4), "truncated_ratio": round(w["trunc"], 4),
            })
    out.sort(key=lambda s: s["z_order"])
    return (out, dropped) if return_dropped else out


def holes_view(shape, higher_layer_shapes):
    """Ban khoet lo DAN XUAT (khong ghi vao shape): tru cac shape tang tren.

    Tra list {"exterior": flat, "holes": [flat, ...]}.
    """
    g = to_geom(shape["polygon"])
    hi = [to_geom(s["polygon"]) for s in higher_layer_shapes]
    if hi:
        g = _polys_only(g.difference(unary_union(hi)))
    return [{"exterior": to_flat(p), "holes": [_ring_flat(r) for r in p.interiors]} for p in _parts(g)]
