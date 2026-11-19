"""Validator shape nhap theo docs/QUY_UOC_VE_CHUNG.md (ruleset: config/rulesets/default.yaml, co fallback hang so).

Shape noi bo: {id, frame, label, type: polygon|rectangle, points(phang), z_order, occluded, truncated, raw}.
Issue: {code, severity: error|warn|info, shape_id, label, frame, message, data}.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from shapely.geometry import Polygon
from shapely.geometry import box as _box
from shapely.ops import unary_union
from shapely.validation import explain_validity

from cvat_tools.segmentation import layers as L
from cvat_tools.segmentation import stuff_things as ST

SEV_RANK = {"error": 3, "warn": 2, "info": 1}

FALLBACK = {
    "occluded_threshold": 0.30,
    "truncated_threshold": 0.02,
    "groups": {
        "object": ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"],
        "stuff": ["road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky"],
        "pole": ["pole"],
        "lane": ["lane/single white", "lane/single yellow", "lane/double white", "lane/double yellow", "lane/single other", "lane/double other"],
        "crosswalk": ["lane/crosswalk"], "curb": ["lane/road curb"], "area": ["area/direct", "area/alternative"],
    },
    "shape_by_group": {"object": "rectangle", "stuff": "polygon", "pole": "polygon", "lane": "polygon",
                       "crosswalk": "polygon", "curb": "polygon", "area": "polygon"},
    "z_order": {"road": -6, "area/alternative": -5, "area/direct": -4, "lane": -3, "pole": -2, "object": 0,
                "other_stuff": -8, "sky": -9},
    "aliases": {"person": "pedestrian", "traffic_light": "traffic light", "traffic_sign": "traffic sign"},
    "max_vertices": {"pole": 8, "crosswalk": 6},
}
REVIEW_DEFAULTS = {  # ghi de duoc bang khoa `review:` trong ruleset
    "lane_max_width": 60.0, "lane_min_aspect": 3.0, "curb_max_width": 80.0,
    "pole_max_width": 40.0, "pole_min_aspect": 3.0,
    "tiny_box_min_side": 4.0, "tiny_box_min_area": 24.0,
    "dup_iou": 0.9, "dup_vertex_dist": 0.5, "min_poly_area": 4.0,
    "area_overlap": 0.02, "frame_margin": 1.0, "edge_touch": 1.5,
    "outside_road": {"lane": 0.10, "crosswalk": 0.10, "area": 0.10, "curb": 0.5}, "road_tol": 2.0,
    "crosswalk_min_vertices": 4,
    "stuff_min_ratio": 0.05, "stuff_box_inset": 0.03,   # STUFF_OVER_THING
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_ruleset(job=None, config_dir: Path | None = None) -> dict:
    """default.yaml (+ jobs/<job>.yaml) neu co, nguoc lai hang so theo QUY_UOC_VE_CHUNG.md."""
    rs = copy.deepcopy(FALLBACK)
    try:
        import yaml

        from cvat_tools.core import paths
        d = (config_dir or paths.CONFIG) / "rulesets"
        f = d / "default.yaml"
        if f.exists():
            rs = _merge(rs, yaml.safe_load(f.read_text(encoding="utf8")) or {})
        if job is not None and (d / "jobs" / f"{job}.yaml").exists():
            rs = _merge(rs, yaml.safe_load((d / "jobs" / f"{job}.yaml").read_text(encoding="utf8")) or {})
    except Exception:  # thieu yaml/file -> dung hang so
        pass
    rs["review"] = _merge(REVIEW_DEFAULTS, rs.get("review") or {})
    return rs


def canon_label(label: str, rs: dict) -> str:
    return rs["aliases"].get(label, label)


def group_of(label: str, rs: dict) -> str | None:
    label = canon_label(label, rs)
    for g, labs in rs["groups"].items():
        if label in labs:
            return g
    return None


def expected_type(label: str, rs: dict) -> str | None:
    g = group_of(label, rs)
    return rs["shape_by_group"].get(g) if g else None


def is_narrow(label: str, rs: dict) -> bool:
    return group_of(label, rs) in ("lane", "curb", "pole")


def expected_z(label: str, rs: dict) -> int | None:
    """None = chi can < z(road) (nen khac)."""
    label = canon_label(label, rs)
    z, g = rs["z_order"], group_of(label, rs)
    if label in z:
        return z[label]
    if g in ("lane", "crosswalk", "curb"):
        return z["lane"]
    if g in z:
        return z[g]
    return None


# ------------------------------------------------------------------ shape io
def _attr_flag(raw: dict, name: str) -> bool:
    if raw.get(name) is True:
        return True
    for a in raw.get("attributes") or []:
        if isinstance(a, dict) and a.get("name") == name and str(a.get("value")).lower() == "true":
            return True
    return False


def shapes_from_draft(draft_shapes, frame: int | None = None, rs: dict | None = None, id_prefix: str = "") -> list[dict]:
    """Shape kieu draft (type/points/_label/z_order...) -> shape noi bo; id on dinh theo (frame, thu tu)."""
    out, cnt = [], {}
    for s in draft_shapes:
        f = s.get("frame")
        if frame is not None and f != frame:
            continue
        k = cnt.get(f, 0)
        cnt[f] = k + 1
        lab = s.get("_label") or s.get("label") or ""
        if rs:
            lab = canon_label(lab, rs)
        out.append({"id": f"{id_prefix}f{f}_{k:03d}", "frame": f, "label": lab, "type": s["type"],
                    "points": [float(v) for v in s["points"]], "z_order": int(s.get("z_order", 0)),
                    "occluded": bool(s.get("occluded")), "truncated": _attr_flag(s, "truncated"),
                    "raw": copy.deepcopy(s)})
    return out


def poly_of(shape) -> Polygon:
    p = shape["points"]
    if shape["type"] == "rectangle":
        x0, y0, x1, y1 = p[:4]
        return _box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    pts = [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]
    return Polygon(pts) if len(pts) >= 3 else Polygon()


def flat_of(shape) -> list[float]:
    if shape["type"] == "rectangle":
        x0, y0, x1, y1 = shape["points"][:4]
        return [x0, y0, x1, y0, x1, y1, x0, y1]
    return list(shape["points"])


def vertices(shape) -> list[tuple[float, float]]:
    p = shape["points"]
    if shape["type"] == "rectangle":
        x0, y0, x1, y1 = p[:4]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]


def _valid_geom(shape):
    g = poly_of(shape)
    if g.is_empty:
        return g
    return g if g.is_valid else g.buffer(0)


def _thin_metrics(g: Polygon) -> dict:
    if g.is_empty or g.length <= 0:
        return {"mean_width": 0.0, "length": 0.0, "aspect": 0.0}
    w = 2.0 * g.area / g.length
    ln = g.length / 2.0
    return {"mean_width": round(w, 2), "length": round(ln, 1), "aspect": round(ln / max(w, 1e-6), 1)}


def _rot_rect_sides(g: Polygon) -> tuple[float, float]:
    r = g.minimum_rotated_rectangle
    if r.is_empty or r.geom_type != "Polygon":
        return 0.0, 0.0
    c = list(r.exterior.coords)
    a, b = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    return max(a, b), min(a, b)


# ------------------------------------------------------------------ nen de vat the
def thing_regions(shapes, rs: dict, thing_masks: dict | None = None) -> list[tuple[str, ST.ThingRegion]]:
    """Vung vat the cua frame: mask instance (thing_masks[shape_id], polygon phang) uu tien hon hop cua shape."""
    out = []
    for s in shapes:
        lab = canon_label(s["label"], rs)
        if lab not in ST.THING_LABELS:
            continue
        m = (thing_masks or {}).get(s["id"])
        if m and len(m) >= 6:
            out.append((s["id"], ST.ThingRegion(lab, polygon=list(m), source="mask")))
        elif s["type"] == "rectangle":
            x0, y0, x1, y1 = s["points"][:4]
            out.append((s["id"], ST.ThingRegion(lab, box=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), source="box")))
        elif len(s["points"]) >= 6:
            out.append((s["id"], ST.ThingRegion(lab, polygon=list(s["points"]), source="mask")))
    return out


def stuff_intrusions(shape, regions, rs: dict) -> list[dict]:
    """Vat the bi polygon nen `shape` de len (cut_ratio >= nguong). Moi phan tu them 'thing_id' va 'region'."""
    cfg = rs["review"]
    if canon_label(shape["label"], rs) not in ST.STUFF_LABELS or shape["type"] != "polygon" or len(shape["points"]) < 6:
        return []
    ids = [r for r in regions if r[0] != shape["id"]]
    rep = ST.notch_report(shape["points"], [r[1] for r in ids], cfg["stuff_box_inset"])
    return [{**r, "thing_id": ids[r["idx"]][0], "region": ids[r["idx"]][1]} for r in rep
            if r["cut_ratio"] >= cfg["stuff_min_ratio"]]


# ------------------------------------------------------------------ validate
def validate_frame(shapes: list[dict], frame_wh: tuple[int, int] | None = None, rs: dict | None = None,
                   thing_masks: dict | None = None) -> list[dict]:
    rs = rs or load_ruleset()
    cfg = rs["review"]
    issues: list[dict] = []

    def add(sh, code, sev, msg, **data):
        issues.append({"code": code, "severity": sev, "shape_id": sh["id"], "label": sh["label"],
                       "frame": sh.get("frame"), "message": msg, "data": data})

    geoms = {}
    for sh in shapes:
        lab = sh["label"]
        grp = group_of(lab, rs)
        if grp is None:
            add(sh, "LABEL_UNKNOWN", "error", f"label '{lab}' khong thuoc ruleset")
            continue
        et = expected_type(lab, rs)
        if et and sh["type"] != et:
            add(sh, "SHAPE_TYPE", "error", f"{lab} phai la {et}, shape la {sh['type']}", expected=et, got=sh["type"])
        ez = expected_z(lab, rs)
        zroad = rs["z_order"]["road"]
        if ez is not None and sh["z_order"] != ez:
            add(sh, "Z_ORDER", "error", f"z_order {sh['z_order']} != {ez} theo bang", expected=ez, got=sh["z_order"])
        elif ez is None and sh["z_order"] >= zroad:
            add(sh, "Z_ORDER", "error", f"nen khac phai duoi road (z<{zroad}), co {sh['z_order']}", got=sh["z_order"])

        vs = vertices(sh)
        if sh["type"] == "polygon":
            if len(vs) < 3:
                add(sh, "POLY_TOO_FEW", "error", f"polygon chi co {len(vs)} dinh")
                continue
            dd = cfg["dup_vertex_dist"]
            dups = [i for i in range(len(vs)) if math.dist(vs[i], vs[(i + 1) % len(vs)]) < dd]
            if dups:
                add(sh, "DUP_VERTEX", "warn", f"dinh trung/lien tiep < {dd}px tai idx {dups[:8]}", idx=dups)
            g0 = poly_of(sh)
            if not g0.is_valid:
                why = explain_validity(g0)
                at = None
                if "[" in why:
                    try:
                        at = [round(float(v), 1) for v in why.split("[")[1].rstrip("]").split()[:2]]
                    except ValueError:
                        at = None
                add(sh, "POLY_INVALID", "error", f"polygon khong hop le: {why}", reason=why, at=at)
            elif list(g0.interiors):
                add(sh, "POLY_HOLE", "error", "polygon co lo")
            if g0.area < cfg["min_poly_area"]:
                add(sh, "POLY_TINY", "error", f"dien tich {g0.area:.1f}px2 qua nho", area=round(g0.area, 1))
            mv = rs["max_vertices"].get(grp)
            if mv and len(vs) > mv:
                add(sh, "VERTEX_MAX", "error", f"{lab} co {len(vs)} dinh > {mv}", n=len(vs), max=mv)
            if grp == "crosswalk" and len(vs) < cfg["crosswalk_min_vertices"]:
                add(sh, "VERTEX_MIN", "error", f"crosswalk {len(vs)} dinh < {cfg['crosswalk_min_vertices']}")
        else:
            x0, y0, x1, y1 = sh["points"][:4]
            w, h = abs(x1 - x0), abs(y1 - y0)
            if min(w, h) < cfg["tiny_box_min_side"] or w * h < cfg["tiny_box_min_area"]:
                add(sh, "BOX_TINY", "error", f"hop {w:.1f}x{h:.1f}px qua nho", w=round(w, 1), h=round(h, 1))
        if frame_wh:
            m = cfg["frame_margin"]
            bad = [i for i, (x, y) in enumerate(vs) if x < -m or y < -m or x > frame_wh[0] + m or y > frame_wh[1] + m]
            if bad:
                add(sh, "OUT_OF_FRAME", "warn", f"dinh {bad[:8]} ngoai khung {frame_wh[0]}x{frame_wh[1]}", idx=bad)

        g = _valid_geom(sh)
        geoms[sh["id"]] = g
        if g.is_empty:
            continue
        if grp in ("lane", "curb"):
            m = _thin_metrics(g)
            mx = cfg["lane_max_width"] if grp == "lane" else cfg["curb_max_width"]
            if m["mean_width"] > mx or m["aspect"] < cfg["lane_min_aspect"]:
                add(sh, "LANE_TOO_WIDE", "error",
                    f"{lab} be rong tb {m['mean_width']}px (<= {mx}), dai/rong {m['aspect']} (>= {cfg['lane_min_aspect']})", **m)
        elif grp == "pole":
            ln, wd = _rot_rect_sides(g)
            asp = ln / max(wd, 1e-6)
            if wd > cfg["pole_max_width"] or asp < cfg["pole_min_aspect"]:
                add(sh, "POLE_SHAPE", "error",
                    f"pole rong {wd:.1f}px (<= {cfg['pole_max_width']}), cao/rong {asp:.1f} (>= {cfg['pole_min_aspect']})",
                    width=round(wd, 1), length=round(ln, 1), aspect=round(asp, 1))

    # --- trong road
    roads = [geoms[s["id"]] for s in shapes if s["id"] in geoms and canon_label(s["label"], rs) == "road"
             and not geoms[s["id"]].is_empty]
    if roads:
        road = unary_union(roads).buffer(cfg["road_tol"])
        for sh in shapes:
            grp = group_of(sh["label"], rs)
            if grp in cfg["outside_road"] and sh["id"] in geoms and geoms[sh["id"]].area > 0:
                g = geoms[sh["id"]]
                out_frac = g.difference(road).area / g.area
                if out_frac > cfg["outside_road"][grp]:
                    add(sh, "NOT_IN_ROAD", "error", f"{sh['label']} nam ngoai road {out_frac:.0%}", outside=round(out_frac, 3))
    else:
        for sh in shapes:
            if group_of(sh["label"], rs) in cfg["outside_road"]:
                add(sh, "NO_ROAD_REF", "info", "frame khong co road de doi chieu")
                break

    # --- direct ∩ alternative
    dirs = [s for s in shapes if canon_label(s["label"], rs) == "area/direct" and s["id"] in geoms]
    alts = [s for s in shapes if canon_label(s["label"], rs) == "area/alternative" and s["id"] in geoms]
    for a in alts:
        ga = geoms[a["id"]]
        for d in dirs:
            gd = geoms[d["id"]]
            if ga.is_empty or gd.is_empty:
                continue
            fr = ga.intersection(gd).area / max(min(ga.area, gd.area), 1e-9)
            if fr > cfg["area_overlap"]:
                add(a, "AREA_OVERLAP", "error", f"alternative chong direct {d['id']} {fr:.0%}", other=d["id"], frac=round(fr, 3))

    # --- trung lap cung label
    by_label: dict = {}
    for sh in shapes:
        if sh["id"] in geoms and not geoms[sh["id"]].is_empty:
            by_label.setdefault(canon_label(sh["label"], rs), []).append(sh)
    for lst in by_label.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = geoms[lst[i]["id"]], geoms[lst[j]["id"]]
                if not a.intersects(b):
                    continue
                iou = a.intersection(b).area / max(a.union(b).area, 1e-9)
                if iou >= cfg["dup_iou"]:
                    add(lst[j], "DUPLICATE", "error", f"trung {lst[i]['id']} (IoU {iou:.2f})", other=lst[i]["id"], iou=round(iou, 3))

    # --- che khuat / cat mep (chi vat the)
    objs = [s for s in shapes if group_of(s["label"], rs) == "object" and s["id"] in geoms and not geoms[s["id"]].is_empty]
    if objs:
        inst = [{"id": s["id"], "label": s["label"], "polygon": flat_of(s)} for s in shapes
                if s["id"] in geoms and not geoms[s["id"]].is_empty and group_of(s["label"], rs) is not None]
        res = L.resolve(inst, None, occ_thresh=rs["occluded_threshold"])
        ratio = {r["id"]: r["occlusion_ratio"] for r in reversed(res)}
        th = rs["occluded_threshold"]
        for s in objs:
            r = ratio.get(s["id"], 0.0)
            if r >= th and not s["occluded"]:
                add(s, "OCC_MISSING", "warn", f"bi che {r:.0%} (>= {th:.0%}) nhung occluded=false", ratio=r)
            elif r < th / 2 and s["occluded"]:
                add(s, "OCC_EXTRA", "info", f"occluded=true nhung chi bi che {r:.0%}", ratio=r)
            if frame_wh:
                x0, y0, x1, y1 = geoms[s["id"]].bounds
                t = cfg["edge_touch"]
                touch = x0 <= t or y0 <= t or x1 >= frame_wh[0] - t or y1 >= frame_wh[1] - t
                if touch and not s["truncated"]:
                    add(s, "TRUNC_MISSING", "warn", "hop cham mep khung nhung truncated=false")
                elif not touch and s["truncated"]:
                    add(s, "TRUNC_EXTRA", "info", "truncated=true nhung hop khong cham mep khung")
    # --- nen lan vao vat the (warn)
    regs = thing_regions(shapes, rs, thing_masks)
    if regs:
        for sh in shapes:
            if sh["id"] not in geoms or geoms[sh["id"]].is_empty:
                continue
            hit = stuff_intrusions(sh, regs, rs)
            if hit:
                things = [{"thing_id": h["thing_id"], "label": h["label"], "cut_ratio": h["cut_ratio"],
                           "inter_ratio": h["inter_ratio"], "source": h["source"]} for h in hit]
                txt = ", ".join(f"{t['label']} {t['thing_id']} {t['cut_ratio']:.0%}" for t in things)
                add(sh, "STUFF_OVER_THING", "warn", f"{sh['label']} lan vao vat the: {txt}", things=things,
                    min_ratio=cfg["stuff_min_ratio"], box_inset=cfg["stuff_box_inset"])
    issues.sort(key=lambda i: (-SEV_RANK[i["severity"]], i["shape_id"], i["code"]))
    return issues


def count_by_code(issues) -> dict:
    out: dict = {}
    for i in issues:
        out[i["code"]] = out.get(i["code"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def codes_by_shape(issues, min_sev: str = "warn") -> dict:
    out: dict = {}
    for i in issues:
        if SEV_RANK[i["severity"]] >= SEV_RANK[min_sev]:
            out.setdefault(i["shape_id"], []).append(i["code"])
    return out
