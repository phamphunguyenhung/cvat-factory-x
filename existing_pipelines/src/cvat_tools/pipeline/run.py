"""Mot lenh sinh nhap cho MOI nhan cua job, tung frame DOC LAP. CHI DOC CVAT (khong ghi CVAT).

Buoc (moi buoc do thoi gian vao timings):
  seg_b2     SegFormer-B2 ONNX qua inputspec -> class map Cityscapes
  yolo       YOLO26l tile (detection.yolo_tile; qua inputspec) -> car/truck/bus/motorcycle/bicycle/pedestrian/rider/traffic light/train
  stuff      class map -> polygon nen (sky, building, vegetation, sidewalk, terrain, wall, fence), pole, traffic sign (hop)
  lanes      lanes.framegen.generate_frame (lane/*, crosswalk, area/direct); area/alternative = road - direct (QUY_UOC 3.4)
  resolve    segmentation.layers.resolve (z_order, occluded/truncated chi 10 nhan vat the)
  subtract   stuff_things.subtract_things (nen de len vat the)         [+ sam: MobileSAM tuy chon]
  refine     geomrefine theo loai vat
  decision   review.validators + decision.engine (tren ban sao, khong ghi CVAT) -> tu sua loi do duoc
  store      shapestore stage rule_draft kem provenance
road: shape nguoi da ve (dau vao, origin=human, khong tinh vao shape tool); thieu thi sinh bang b2.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union

from cvat_tools.decision import engine as ENG
from cvat_tools.detection.yolo_tile import derive_riders, detect_tiled, keep_detection, make_predictor
from cvat_tools.lanes.framegen import generate_frame
from cvat_tools.pipeline import stuffpoly as SP
from cvat_tools.pipeline.lanefilter import filter_lanes
from cvat_tools.pipeline.targeteval import OBJECT_LABELS, geom, iou
from cvat_tools.review import validators as V
from cvat_tools.segmentation import geomrefine as G
from cvat_tools.segmentation import layers as L
from cvat_tools.segmentation import stuff_things as ST
from cvat_tools.segmentation.onnx_seg import OnnxSeg

STEPS = ("seg_b2", "yolo", "stuff", "lanes", "resolve", "subtract", "refine", "decision")
MIN_ALT_AREA = 2500.0   # nhu framegen.CFG["min_alt_area"]
DUP_IOU = 0.5           # co trung shape nguoi


class Models:
    """Nap 1 lan, dung cho nhieu frame. seg/predict tiem duoc de test."""

    def __init__(self, seg=None, predict=None, seg_id: str = "b2", yolo_id: str = "yolo26l"):
        self.seg_id, self.yolo_id = seg_id, yolo_id
        t0 = time.time()
        self.seg = seg or OnnxSeg(seg_id)
        if predict is None:
            from ultralytics import YOLO

            from cvat_tools.core import paths
            predict = make_predictor(YOLO(str(paths.CHECKPOINTS / f"{yolo_id}.pt")))
        self.predict = predict
        self.load_s = round(time.time() - t0, 2)


def _rect_poly(b) -> List[float]:
    x1, y1, x2, y2 = b
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _polys_of(g) -> List[Polygon]:
    if g.is_empty:
        return []
    return [p for p in (g.geoms if hasattr(g, "geoms") else [g]) if isinstance(p, Polygon) and p.area > 0]


def alternative_polygons(road_polys: List[List[float]], direct_polys: List[List[float]], min_area: float = MIN_ALT_AREA,
                         eps: float = 1.5) -> List[List[float]]:
    """area/alternative = road - direct (polygon nguyen vung, bien chung trung khit voi direct)."""
    if not road_polys or not direct_polys:
        return []
    road = unary_union([G.valid_polygon(p) or Polygon() for p in road_polys])
    direct = unary_union([G.valid_polygon(p) or Polygon() for p in direct_polys])
    out = []
    for part in _polys_of(road.difference(direct)):
        part = Polygon(part.exterior).simplify(eps, preserve_topology=True)   # khong khoet lo
        part = part.difference(direct)                                         # simplify khong duoc lan sang direct
        for q in _polys_of(part):
            if q.area >= min_area:
                out.append([round(float(v), 1) for xy in list(q.exterior.coords)[:-1] for v in xy])
    return out


def build_instances(road_human, road_model, stuff, poles, signs, dets, lane_shapes):
    """-> (instances cho layers.resolve, meta{id: {src, conf, human}})."""
    ins, meta = [], {}

    def add(label, polygon, conf, src, human=False):
        i = len(ins)
        ins.append({"id": i, "label": label, "polygon": [float(v) for v in polygon], "conf": conf})
        meta[i] = {"src": src, "conf": conf, "human": human}

    for p in road_human:
        add("road", p, 1.0, "human", True)
    for s in road_model:
        add("road", s["polygon"], 0.5, s["_src"])
    for lab, lst in stuff.items():
        for s in lst:
            add(lab, s["polygon"], 0.5, s["_src"])
    for s in poles:
        add("pole", s["polygon"], 0.5, s["_src"])
    for s in signs:
        add("traffic sign", _rect_poly(s["box"]), 0.5, s["_src"])
    for lab, b, c in dets:
        add(lab, _rect_poly(b), c, "yolo26l")
    for s in lane_shapes:
        add(s["label"], s["polygon"], s.get("conf", 0.5), "framegen:" + str(s.get("_src", "")))
    return ins, meta


def merge_parts(out, meta):
    """Ket qua resolve (co the nhieu manh/id) -> shape tool: vat the = hop bao cac manh; polygon = tung manh."""
    shapes = []
    byid: Dict[int, list] = {}
    for o in out:
        byid.setdefault(o["id"], []).append(o)
    for i, parts in byid.items():
        m = meta[i]
        lab = parts[0]["label"]
        if lab in OBJECT_LABELS:
            xy = np.concatenate([np.asarray(p["polygon"]).reshape(-1, 2) for p in parts])
            pts = [round(float(xy[:, 0].min()), 1), round(float(xy[:, 1].min()), 1),
                   round(float(xy[:, 0].max()), 1), round(float(xy[:, 1].max()), 1)]
            shapes.append({"label": lab, "type": "rectangle", "points": pts, "z_order": parts[0]["z_order"],
                           "occluded": any(p["occluded"] for p in parts), "truncated": any(p["truncated"] for p in parts),
                           "conf": m["conf"], "src": m["src"], "human": m["human"]})
        else:
            for p in parts:
                shapes.append({"label": lab, "type": "polygon", "points": [float(v) for v in p["polygon"]],
                               "z_order": p["z_order"], "occluded": False, "truncated": False,
                               "conf": m["conf"], "src": m["src"], "human": m["human"]})
    return shapes


def _thing_regions(shapes, sam_masks=None):
    th = []
    for s in shapes:
        if s["type"] == "rectangle" and s["label"] in ST.THING_LABELS:
            x1, y1, x2, y2 = s["points"]
            poly = (sam_masks or {}).get(id(s))
            th.append(ST.ThingRegion(s["label"], box=(x1, y1, x2, y2), polygon=poly, source="mask" if poly else "box"))
    return th


def subtract_things_step(shapes, sam_masks=None):
    """Nen de len vat the -> luat STUFF_OVER_THING (stuff_things.subtract_things). Tra (shapes moi, so shape doi)."""
    things = _thing_regions(shapes, sam_masks)
    out, changed = [], 0
    for s in shapes:
        if s["type"] == "polygon" and s["label"] in ST.STUFF_LABELS and not s["human"] and things:
            fx = ST.subtract_things(s["points"], things)
            if fx.changed:
                changed += 1
                for poly in fx.polygons:
                    out.append({**s, "points": poly, "src": s["src"] + "+subtract"})
                continue
        out.append(s)
    return out, changed


def refine_step(shapes, img):
    n = 0
    for s in shapes:
        if s["type"] != "polygon" or s["human"]:
            continue
        try:
            r = G.refine(s["points"], G.kind_for_label(s["label"]), img)
            if len(r) >= 6 and G.valid_polygon(r) is not None:
                s["points"], n = r, n + 1
        except Exception:  # noqa: BLE001 - refine suy bien: giu nguyen ban
            pass
    return shapes, n


def sam_step(img, shapes, max_n: int = 6):
    """MobileSAM cho vat the de len nen (tuy chon, chi khi con ngan sach). -> {id(shape): polygon}"""
    from cvat_tools.segmentation.sam_refine import refine_with_sam

    stuff = [geom(s) for s in shapes if s["type"] == "polygon" and s["label"] in ST.STUFF_LABELS]
    cand = [s for s in shapes if s["type"] == "rectangle" and s["label"] in ST.THING_LABELS
            and any(g.intersects(sbox(*s["points"])) for g in stuff)]
    cand.sort(key=lambda s: -(s["points"][2] - s["points"][0]) * (s["points"][3] - s["points"][1]))
    out = {}
    for s in cand[:max_n]:
        p = refine_with_sam(img, {"box": s["points"]}, "mobilesam")
        if p:
            out[id(s)] = p
    return out


def human_flags(tool_shapes, human_shapes, thr: float = DUP_IOU):
    """Co 'trung shape nguoi' cho tung shape tool: IoU >= thr voi shape nguoi cung nhan."""
    hg = [(h["label"], geom(h)) for h in human_shapes if h["type"] in ("polygon", "rectangle")]
    for s in tool_shapes:
        g = geom(s)
        s["dup_human"] = any(lb == s["label"] and iou(g, hh) >= thr for lb, hh in hg)
    return tool_shapes


def _count(issues):
    return {"error": sum(i["severity"] == "error" for i in issues), "warn": sum(i["severity"] == "warn" for i in issues),
            "total": len(issues)}


def process_frame(img: np.ndarray, frame: int, models: Models, road_human: List[List[float]], human_shapes: List[dict],
                  *, job: int = 285, rs: Optional[dict] = None, use_sam: bool = False, tiles: bool = True,
                  budget_s: float = 10.0, store=None, run_id: str = "") -> Dict[str, Any]:
    rs = rs or V.load_ruleset(job)
    H, W = img.shape[:2]
    T: Dict[str, float] = {}
    t_all = time.time()

    def lap(name, t0):
        T[name] = round(time.time() - t0, 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    cm, seg_info = models.seg.class_map(img)
    lap("seg_b2", t0)

    t0 = time.time()   # (thu chay song song seg+YOLO: cham hon do tranh CPU, nen chay tuan tu)
    dets = detect_tiled(models.predict, img, model_id=models.yolo_id, tiles=tiles)
    lap("yolo", t0)
    dets = derive_riders([d for d in dets if keep_detection(d, W, H)])

    t0 = time.time()
    stuff = SP.stuff_polygons(cm, gray)
    road_model: List[dict] = []
    if not road_human:   # road nguoi ve thieu -> sinh bang b2 (M2F khong chay: de ngan sach)
        lab, st, idx = SP._components((cm == 0), 0.03 * W * H)
        road_model = [{"polygon": SP.mask_to_polygon(lab == i), "_src": "b2:road"} for i in idx[:2]]
        road_model = [r for r in road_model if r["polygon"]]
    signs = SP.sign_boxes(cm)
    rv = rs["review"]
    thing_boxes = [d[1] for d in dets if d[0] in ST.THING_LABELS]
    poles = SP.pole_polygons(cm, rv["pole_max_width"], rv["pole_min_aspect"], thing_boxes)
    lap("stuff", t0)

    t0 = time.time()
    roads = road_human or [r["polygon"] for r in road_model]
    g = generate_frame(img, roads) if roads else {"shapes": [], "vp": None, "vp_src": "no_road", "dark": False, "n_lines": 0}
    lane_shapes = [s for s in g["shapes"] if s["label"] != "area/alternative"]   # alternative theo quy uoc moi
    lane_shapes, lane_dropped = filter_lanes(lane_shapes, img, roads, thing_boxes) if roads else (lane_shapes, [])
    directs = [s["polygon"] for s in lane_shapes if s["label"] == "area/direct"]
    alt = alternative_polygons(roads, directs)
    lane_shapes += [{"label": "area/alternative", "polygon": p, "conf": 0.5, "_src": "road-direct"} for p in alt]
    lap("lanes", t0)

    t0 = time.time()
    ins, meta = build_instances(road_human, road_model, stuff, poles, signs, dets, lane_shapes)
    out = L.resolve(ins, (W, H), occ_thresh=rs["occluded_threshold"], trunc_thresh=rs["truncated_threshold"])
    shapes = merge_parts(out, meta)
    for s in shapes:   # occluded/truncated chi cho 10 nhan vat the co thuoc tinh
        if s["label"] not in OBJECT_LABELS:
            s["occluded"] = s["truncated"] = False
    lap("resolve", t0)

    t0 = time.time()
    sam_masks = None
    if use_sam and (time.time() - t_all) < 0.6 * budget_s:
        sam_masks = sam_step(img, shapes)
    shapes, n_sub = subtract_things_step(shapes, sam_masks)
    lap("subtract", t0)

    t0 = time.time()
    shapes, n_ref = refine_step(shapes, img)
    lap("refine", t0)

    t0 = time.time()
    draft = [{"frame": frame, "type": s["type"], "points": s["points"], "_label": s["label"], "z_order": s["z_order"],
              "occluded": s["occluded"], "truncated": s["truncated"], "_conf": s["conf"], "_src": s["src"]} for s in shapes]
    internal = V.shapes_from_draft(draft, frame, rs)
    for sh, s in zip(internal, shapes):
        if s["human"]:
            sh["origin"] = "human"
    res = ENG.run_frame(job, frame, internal, img=img, rs=rs, store=None, apply=True)
    lap("decision", t0)

    final = []
    for sh in res["shapes"]:
        if sh.get("origin") == "human":
            continue
        raw = sh.get("raw") or {}
        final.append({"label": sh["label"], "type": sh["type"], "points": [round(float(v), 2) for v in sh["points"]],
                      "z_order": sh["z_order"], "occluded": bool(sh["occluded"]), "truncated": bool(sh.get("truncated")),
                      "conf": raw.get("_conf"), "src": raw.get("_src"), "human": False})
    human_flags(final, human_shapes)
    road_shapes = [{"label": "road", "type": "polygon", "points": p, "z_order": L.z_order("road")} for p in road_human]
    T["total"] = round(time.time() - t_all, 3)
    result = {"frame": frame, "shapes": final, "road_input": road_shapes, "timings": T, "over_budget": T["total"] > budget_s,
              "dark": bool(g.get("dark")) or float(gray.mean()) < 80, "mean_gray": round(float(gray.mean()), 1),
              "seg": seg_info, "vp_src": g.get("vp_src"), "n_dets": len(dets), "n_subtracted": n_sub, "n_refined": n_ref,
              "sam": bool(sam_masks), "lane_dropped": lane_dropped, "n_poles": len(poles), "issues_before": _count(res["issues_before"]), "issues_after": _count(res["issues_after"]),
              "codes_before": res["counts_before"], "codes_after": res["counts_after"],
              "decisions": [{"action": r["proposed_action"], "reason": r["reason_code"], "status": r.get("validation_status")}
                            for r in res["records"]]}
    if store is not None:
        t0 = time.time()
        write_store(store, job, frame, final, run_id)
        result["timings"]["store"] = round(time.time() - t0, 3)
    return result


def write_store(store, job: int, frame: int, shapes: List[dict], run_id: str) -> List[int]:
    ids = []
    for s in shapes:
        attrs = [{"name": "truncated", "value": "true"}] if s.get("truncated") else []
        ids.append(store.add(job, frame, s["label"], s["type"], s["points"], "rule_draft", z_order=s["z_order"],
                             occluded=s["occluded"], attributes=attrs, conf=s.get("conf"),
                             provenance={"run": run_id, "src": s.get("src"), "pipeline": "cvat_tools.pipeline.run",
                                         "dup_human": bool(s.get("dup_human"))}))
    return ids


def parse_frames(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
