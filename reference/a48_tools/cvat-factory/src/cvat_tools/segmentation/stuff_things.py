"""Luật STUFF_OVER_THING: polygon nền (building, sky, vegetation, sidewalk...) đè lên vật thể (car, truck...).

Quy ước (QUY_UOC_VE_CHUNG.md): polygon nguyên vùng, không khoét lỗ. Vật thể phía trên, nên phần nền nằm
dưới vật thể không cần trừ. Nhưng khi model gán NHẦM pixel của vật thể vào nền (ví dụ mui xe tải bị tính là
building) thì biên nền lấn vào vật thể là lỗi. Luật tất định (không dùng VLM):

  nền' = lấp_lỗ_kín( nền − vùng_vật_thể ), bỏ mảnh vụn nhỏ

- Phần vật thể cắt vào BIÊN của nền để lại khuyết (notch): giữ, vì đó là chỗ nền bị cắt thật.
- Phần vật thể nằm trọn trong nền tạo lỗ kín: lấp lại (nguyên vùng), vì nền thật nằm sau vật thể.
- Vùng vật thể: ưu tiên mask thật của instance (SAM / YOLO-seg); không có thì dùng hộp co vào `box_inset`
  và đánh dấu nguồn là 'box' (kém tin cậy hơn).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box as sbox
from shapely.ops import unary_union
from shapely.validation import make_valid

THING_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "pedestrian", "rider"}
STUFF_LABELS = {"building", "sky", "vegetation", "sidewalk", "wall", "fence", "terrain"}


@dataclass
class ThingRegion:
    label: str
    box: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2
    polygon: list[float] | None = None                      # mask instance đã đổi sang polygon phẳng
    source: str = "box"                                     # 'mask' | 'box'

    def geom(self, box_inset: float = 0.03):
        if self.polygon:
            return make_valid(Polygon(np.asarray(self.polygon, float).reshape(-1, 2)))
        x1, y1, x2, y2 = self.box
        dx, dy = (x2 - x1) * box_inset, (y2 - y1) * box_inset
        return sbox(x1 + dx, y1 + dy, x2 - dx, y2 - dy)


@dataclass
class Fix:
    changed: bool
    polygons: list[list[float]]           # polygon phẳng sau sửa (có thể >1 mảnh)
    removed_area: float = 0.0
    overlaps: list[dict] = field(default_factory=list)  # từng vật thể bị đè: label, tỉ lệ, nguồn


def _polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon) or hasattr(g, "geoms"):
        return [p for p in g.geoms if isinstance(p, Polygon)]
    return []


def _flat(p: Polygon) -> list[float]:
    return [round(float(v), 1) for xy in list(p.exterior.coords)[:-1] for v in xy]


def overlap_report(stuff_poly: list[float], things: list[ThingRegion], box_inset: float = 0.03):
    s = make_valid(Polygon(np.asarray(stuff_poly, float).reshape(-1, 2)))
    out = []
    for t in things:
        if t.label not in THING_LABELS:
            continue
        g = t.geom(box_inset)
        inter = s.intersection(g).area
        if g.area > 0 and inter > 0:
            out.append({"label": t.label, "inter_px2": round(inter), "ratio_of_thing": round(inter / g.area, 3),
                        "source": t.source})
    return out


def subtract_things(stuff_poly: list[float], things: list[ThingRegion], *, min_ratio: float = 0.05,
                    box_inset: float = 0.03, min_part_frac: float = 0.005, simplify_px: float = 1.0) -> Fix:
    """Trừ các vật thể bị đè (tỉ lệ giao >= min_ratio diện tích vật thể) khỏi polygon nền."""
    s0 = make_valid(Polygon(np.asarray(stuff_poly, float).reshape(-1, 2)))
    rep = [r for r in overlap_report(stuff_poly, things, box_inset) if r["ratio_of_thing"] >= min_ratio]
    if not rep:
        return Fix(False, [list(map(float, stuff_poly))], 0.0, [])
    cut = unary_union([t.geom(box_inset) for t in things
                       if t.label in THING_LABELS and
                       any(r["label"] == t.label and r["source"] == t.source for r in rep)])
    g = s0.difference(cut)
    parts = [p for p in _polys(g) if p.area >= min_part_frac * s0.area]
    # nguyên vùng: bỏ lỗ kín (nền thật nằm sau vật thể), giữ khuyết ở biên
    parts = [Polygon(p.exterior).simplify(simplify_px) for p in parts]
    parts = [p for p in parts if not p.is_empty and p.is_valid]
    return Fix(True, [_flat(p) for p in parts], round(s0.area - unary_union(parts).area if parts else s0.area, 1), rep)


def notch_report(stuff_poly: list[float], things: list[ThingRegion], box_inset: float = 0.03) -> list[dict]:
    """Từng vật thể: phần nền thật sự BỊ CẮT nếu trừ vật thể đó rồi lấp lỗ kín (= khuyết ở biên), chia diện tích vật thể.

    Vật thể nằm trọn trong nền (lỗ kín) cho cut_ratio = 0: đúng quy ước nguyên vùng, không phải lỗi.
    Mỗi phần tử: idx (vị trí trong `things`), label, source, inter_ratio (giao thô), cut_ratio (hiệu dụng), cut_px2.
    """
    s = make_valid(Polygon(np.asarray(stuff_poly, float).reshape(-1, 2)))
    out = []
    for k, t in enumerate(things):
        if t.label not in THING_LABELS:
            continue
        g = t.geom(box_inset)
        if g.is_empty or g.area <= 0:
            continue
        inter = s.intersection(g).area
        if inter <= 0:
            continue
        rest = [Polygon(p.exterior) for p in _polys(s.difference(g))]
        cut = max(0.0, s.area - unary_union(rest).area) if rest else s.area
        out.append({"idx": k, "label": t.label, "source": t.source, "inter_ratio": round(inter / g.area, 3),
                    "cut_ratio": round(cut / g.area, 3), "cut_px2": round(cut)})
    return out
