"""Pipeline `run`: port YOLO tile (anh tong hop), pole_rule, labelmap/gap, loc lane, alternative = road - direct, buoc noi, xuat anh."""
import json

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from cvat_tools.detection import yolo_tile as YT
from cvat_tools.pipeline import export as EX
from cvat_tools.pipeline import lanefilter as LF
from cvat_tools.pipeline import run as R
from cvat_tools.pipeline import stuffpoly as SP
from cvat_tools.pipeline import targeteval as TE
from cvat_tools.pipeline.cli_run import run_job
from cvat_tools.segmentation import labelmap as LM
from cvat_tools.segmentation import layers as L
from cvat_tools.segmentation import pole_rule as PR
from cvat_tools.segmentation.inputspec import UnknownModelSpec

W, H = 1280, 720
ROAD = [0.0, 400.0, 1280.0, 400.0, 1280.0, 720.0, 0.0, 720.0]


def blank(v=120):
    return np.full((H, W, 3), v, np.uint8)


# ------------------------------------------------------------------ YOLO tile
def fake_predict_factory(log):
    """Tra 1 'car' o toa do ANH GOC (100,200,300,260) bat ke canvas/crop: doi nguoc phep letterbox/tile nhu model that."""
    def predict(canvas, conf, imgsz_hw):
        log.append((canvas.shape[:2], tuple(imgsz_hw), conf))
        return []
    return predict


def test_yolo_full_goes_through_inputspec_letterbox():
    log = []
    # model "tra" hop o toa do canvas letterbox: anh 1280x720 -> canvas 736x1280 (pad 8 tren/duoi), hop canvas y 108..168 = goc y 100..160
    def predict(canvas, conf, hw):
        log.append(canvas.shape[:2])
        if canvas.shape[:2] == (736, 1280):
            return [("car", [100, 108, 300, 168], 0.9)]
        return []
    dets = YT.detect_tiled(predict, blank(), tiles=False)
    assert log == [(736, 1280)]
    (lab, box, conf), = dets
    assert lab == "car" and np.allclose(box, [100, 100, 300, 160], atol=1.0) and conf == 0.9


def test_yolo_tile_offset_edge_drop_and_nms():
    calls = []

    def predict(canvas, conf, hw):
        calls.append(canvas.shape[:2])
        if canvas.shape[:2] == (480, 800):   # tile
            n = len(calls)
            return [("person", [10, 100, 60, 200], 0.8), ("car", [2, 100, 50, 150], 0.9)]   # car cham mep trai tile: bo neu tile khong o x=0
        return []
    dets = YT.detect_tiled(predict, blank(), tiles=True)
    assert calls[0] == (736, 1280) and len(calls) == 10
    ped = [d for d in dets if d[0] == "pedestrian"]
    # tile ox=0 va ox>0: cung nguoi o toa do tile (10,100) -> goc khac nhau theo (ox,oy); NMS khong gop vi khac vi tri
    xs = sorted(round(d[1][0]) for d in ped)
    assert 10 in xs and any(x >= 400 for x in xs)
    cars = [d for d in dets if d[0] == "car"]
    assert all(c[1][0] < 6 or c[1][0] >= 400 for c in cars) and len(cars) <= 3   # tile ox=0 giu (mep anh); tile ox>0 bo
    assert not any(d[0] == "car" and 6 <= d[1][0] < 400 for d in dets)


def test_yolo_nms_and_filters_and_riders():
    def predict(canvas, conf, hw):
        if canvas.shape[:2] == (736, 1280):
            return [("car", [100, 108, 300, 168], 0.9), ("car", [102, 108, 302, 168], 0.5),
                    ("person", [500, 108, 520, 200], 0.7), ("motorcycle", [480, 150, 560, 230], 0.7),
                    ("car", [100, 700, 300, 740], 0.9)]   # tam o day anh = phan chieu capo
        return []
    d = YT.detect_tiled(predict, blank(), tiles=False)
    assert sum(x[0] == "car" for x in d) == 2          # NMS gop 2 hop trung, con hop capo
    d = [x for x in d if YT.keep_detection(x, W, H)]
    assert sum(x[0] == "car" for x in d) == 1
    assert "rider" in [x[0] for x in YT.derive_riders(d)]


def test_yolo_night_conf_and_unknown_model_refused():
    seen = []
    YT.detect_tiled(lambda c, cf, hw: seen.append(cf) or [], blank(30), tiles=False, conf=0.25, night_conf=0.2)
    assert seen == [0.2]
    with pytest.raises(UnknownModelSpec):
        YT.detect_tiled(lambda c, cf, hw: [], blank(), model_id="khong_co_spec", tiles=False)


# ------------------------------------------------------------------ pole_rule
def test_pole_merges_collinear_fragments_and_rejects_bad():
    m = np.zeros((H, W), np.uint8)
    for y0, y1 in ((100, 140), (150, 190), (200, 260)):   # 3 manh thang hang, ho 10 px
        m[y0:y1, 300:308] = 1
    m[400:440, 600:640] = 1          # vuong (khong cao)
    m[500:512, 800:802] = 1          # qua ngan
    poles = PR.extract_poles(m)
    assert len(poles) == 1
    p = poles[0]
    assert p["n_blobs"] == 3 and 155 < p["length"] < 165 and 6 <= p["width"] <= 9 and len(p["polygon"]) == 8
    g = Polygon(np.asarray(p["polygon"]).reshape(-1, 2))
    assert g.is_valid and g.bounds[1] < 105 and g.bounds[3] > 255


def test_pole_rotated_and_vegetation_rejected():
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (200, 300), (260, 100), 1, 5)    # nghieng
    p, = PR.extract_poles(m)
    assert p["length"] > 190 and p["width"] < 12
    veg = np.zeros((H, W), bool)
    veg[80:320, 150:330] = True
    m2 = np.zeros((H, W), np.uint8)
    for y in range(100, 300, 16):                   # mask thua (khong ro) nam giua tan la
        m2[y:y + 8, 250:253] = 1
    assert PR.extract_poles(m2, veg_mask=veg) == []
    assert len(PR.extract_poles(m2)) == 1
    thing = [(240, 90, 270, 310)]
    assert PR.extract_poles(m2, thing_boxes=thing) == []


# ------------------------------------------------------------------ labelmap / gap
def test_full_map_fills_pole_and_small_fragment_but_keeps_things_empty():
    cm = np.full((H, W), 2, np.uint8)         # building
    cm[400:, :] = 0                           # road
    cm[100:300, 600:610] = 5                  # cot de len building -> lap thanh building
    cm[500:503, 100:103] = 2                  # manh building vun giua road -> lap thanh road
    cm[420:500, 900:1000] = 13                # xe: giu trong (co hop xe lo)
    fm = LM.full_stuff_map(cm)
    assert (fm[100:300, 600:610] == 2).all() and fm[501, 101] == 0
    assert (fm[420:500, 900:1000] == LM.UNLABELED).all()


def test_coverage_and_gap_stats():
    shapes = [{"type": "polygon", "points": ROAD}, {"type": "rectangle", "points": [0, 0, 640, 360]}]
    st = LM.gap_stats(shapes, W, H)
    assert abs(st["covered_pct"] - (100 * (0.4444 + 0.25))) < 1.0
    img, st2 = LM.render_gap(blank(), shapes, "t")
    assert img.shape == (H, W, 3) and abs(st2["uncovered_pct"] - st["uncovered_pct"]) < 0.01


def test_stuff_polygons_single_outer_ring_no_hole():
    cm = np.full((H, W), 2, np.uint8)
    cm[100:200, 100:200] = 8     # vegetation
    gray = np.full((H, W), 120, np.uint8)
    out = SP.stuff_polygons(cm, gray)
    assert set(out) >= {"building", "vegetation"}
    b = Polygon(np.asarray(out["building"][0]["polygon"]).reshape(-1, 2))
    assert b.is_valid and len(b.interiors) == 0 and b.area > 0.9 * W * H - 100 * 100 - 5000


# ------------------------------------------------------------------ loc lane
def lane_poly(x0, y0, x1, y1, w=3):
    return [x0 - w, y0, x0 + w, y0, x1 + w, y1, x1 - w, y1]


def test_lane_filter_drops_on_vehicle_and_outside_road_keeps_good():
    img = blank(100)
    on_car = {"label": "lane/single white", "polygon": lane_poly(300, 450, 300, 600), "conf": 0.6}
    outside = {"label": "lane/single white", "polygon": lane_poly(300, 100, 300, 300), "conf": 0.6}
    good = {"label": "lane/single white", "polygon": lane_poly(900, 450, 900, 700), "conf": 0.6}
    cw = {"label": "lane/crosswalk", "polygon": lane_poly(300, 450, 300, 600, 40), "conf": 0.5}
    cv2.line(img, (900, 450), (900, 700), (255, 255, 255), 5)   # son that tren duong cho 'good'
    keep, dropped = LF.filter_lanes([on_car, outside, good, cw], img, [ROAD], [(250, 440, 350, 620)])
    assert {d["why"] for d in dropped} == {"tren_vat_the", "ngoai_road"}
    labs = [s["label"] for s in keep]
    assert labs.count("lane/single white") == 1 and "lane/crosswalk" in labs
    g = [s for s in keep if s["label"] == "lane/single white"][0]
    assert g["conf"] == 0.6 and not any("no_paint" in i for i in g.get("_issues", []))


def test_lane_filter_lowers_conf_without_paint():
    keep, dropped = LF.filter_lanes([{"label": "lane/single white", "polygon": lane_poly(900, 450, 900, 700), "conf": 0.6}],
                                    blank(100), [ROAD], [])
    assert not dropped and keep[0]["conf"] == 0.3 and any("no_paint_evidence" in i for i in keep[0]["_issues"])


# ------------------------------------------------------------------ alternative = road - direct, z_order
def test_alternative_is_road_minus_direct():
    direct = [[400, 400, 800, 400, 800, 720, 400, 720]]
    alts = R.alternative_polygons([ROAD], direct)
    assert len(alts) == 2
    ga = [Polygon(np.asarray(a).reshape(-1, 2)) for a in alts]
    gd = Polygon(np.asarray(direct[0]).reshape(-1, 2))
    assert all(g.intersection(gd).area < 1e-6 for g in ga)
    assert abs(sum(g.area for g in ga) + gd.area - Polygon(np.asarray(ROAD).reshape(-1, 2)).area) < 20
    assert R.alternative_polygons([ROAD], []) == []


def test_z_order_theo_cach_nguoi_dung_ve_frame71():
    z = L.z_order
    assert (z("sky"), z("building"), z("vegetation"), z("sidewalk"), z("pole"), z("road")) == (-9, -8, -8, -7, -2, -1)
    assert z("area/direct") == z("area/alternative") == z("lane/single white") == z("lane/crosswalk") == z("car") == z("traffic light") == 0


def test_resolve_object_over_lane_and_flags():
    ins = [{"id": 0, "label": "road", "polygon": ROAD}, {"id": 1, "label": "lane/single white", "polygon": lane_poly(300, 450, 300, 700, 5)},
           {"id": 2, "label": "car", "polygon": [250, 440, 350, 440, 350, 600, 250, 600]}]
    out = L.resolve(ins, (W, H))
    lane = [o for o in out if o["label"] == "lane/single white"][0]
    car = [o for o in out if o["label"] == "car"][0]
    assert lane["occluded"] and not car["occluded"] and car["z_order"] == 0


# ------------------------------------------------------------------ buoc noi + xuat
class FakeSeg:
    def class_map(self, img):
        cm = np.full((H, W), 2, np.uint8)
        cm[:200] = 10
        cm[400:] = 0
        cm[200:400, 100:200] = 8
        cm[150:350, 1000:1006] = 5
        cm[300:340, 700:720] = 7
        return cm, {"s": 0.0}


def fake_predict(canvas, conf, hw):
    if canvas.shape[:2] == (736, 1280):
        return [("car", [400, 408, 600, 520], 0.9), ("person", [800, 308, 830, 380], 0.8)]
    return []


def make_models():
    return R.Models(seg=FakeSeg(), predict=fake_predict)


def test_process_frame_end_to_end_with_fake_models():
    from cvat_tools.core.shapestore import ShapeStore
    st = ShapeStore(":memory:")
    human = [{"label": "car", "type": "rectangle", "points": [400, 400, 600, 512]}]
    r = R.process_frame(blank(), 5, make_models(), [ROAD], human, store=st, run_id="t")
    labs = {s["label"] for s in r["shapes"]}
    assert {"sky", "building", "vegetation", "car", "pedestrian", "pole", "traffic sign"} <= labs
    assert "road" not in labs and r["road_input"][0]["label"] == "road"           # road nguoi ve = dau vao
    car = [s for s in r["shapes"] if s["label"] == "car"][0]
    assert car["dup_human"] and car["type"] == "rectangle"
    sky = [s for s in r["shapes"] if s["label"] == "sky"][0]
    assert sky["z_order"] == -9 and not sky["occluded"] and not sky["truncated"]
    for k in ("seg_b2", "yolo", "stuff", "lanes", "resolve", "subtract", "refine", "decision", "total"):
        assert k in r["timings"]
    assert r["issues_after"]["error"] <= r["issues_before"]["error"]
    rows = st.db.execute("select stage, provenance from shapes").fetchall()
    assert len(rows) == len(r["shapes"]) and all(x["stage"] == "rule_draft" for x in rows)
    assert json.loads(rows[0]["provenance"])["run"] == "t"


def test_process_frame_without_human_road_uses_model_road():
    r = R.process_frame(blank(), 5, make_models(), [], [])
    assert r["road_input"] == [] and any(s["label"] == "road" for s in r["shapes"])


def test_models_without_spec_refused():
    from cvat_tools.segmentation.onnx_seg import OnnxSeg
    with pytest.raises(UnknownModelSpec):
        OnnxSeg("khong_co_spec")


class FakeSource:
    def image(self, f):
        return blank()

    def shapes(self, f):
        return [{"label": "road", "type": "polygon", "points": ROAD}, {"label": "car", "type": "rectangle", "points": [400, 400, 600, 512]}]

    def target(self, f):
        return None


def test_run_job_exports_images_and_report(tmp_path):
    out = run_job(285, [5, 6], tmp_path, store=False, models=make_models(), source=FakeSource())
    for f in ("f5_tool.jpg", "f5_side_by_side.jpg", "f6_tool.jpg", "f5_gapmap.jpg", "contact_sheet.jpg", "run_report.md", "run_report.csv"):
        assert (tmp_path / f).exists(), f
    side = cv2.imread(str(tmp_path / "f5_side_by_side.jpg"))
    tool = cv2.imread(str(tmp_path / "f5_tool.jpg"))
    assert side.shape[1] == 2 * tool.shape[1]
    md = (tmp_path / "run_report.md").read_text(encoding="utf8")
    assert "Thoi gian tung buoc" in md and "validator" in md.lower() and "trung shape nguoi" in md
    assert len(out["results"]) == 2


def test_target_compare_and_weak():
    tgt = [{"label": "car", "type": "rectangle", "points": [0, 0, 100, 100]}, {"label": "sky", "type": "polygon", "points": [0, 0, 100, 0, 100, 100, 0, 100]},
           {"label": "sidewalk", "type": "polygon", "points": [0, 0, 10, 0, 10, 10]}]
    tool = [{"label": "car", "type": "rectangle", "points": [5, 5, 100, 100]}, {"label": "sky", "type": "polygon", "points": [0, 0, 50, 0, 50, 100, 0, 100]},
            {"label": "pole", "type": "polygon", "points": [0, 0, 5, 0, 5, 50]}]
    c = TE.compare(tgt, tool)
    assert c["car"]["matched"] == 1 and c["car"]["recall"] == 1.0 and abs(c["sky"]["union_iou"] - 0.5) < 1e-6
    w = TE.weak(c)
    assert any(x.startswith("sidewalk: THIEU") for x in w) and any(x.startswith("pole: THUA") for x in w)
    assert not any(x.startswith("car") or x.startswith("sky") for x in w)


def test_render_contact_sheet_and_parse_frames():
    sh = [{"label": "car", "type": "rectangle", "points": [10, 10, 100, 100], "z_order": 0},
          {"label": "sky", "type": "polygon", "points": [0, 0, 100, 0, 100, 50, 0, 50], "z_order": -9}]
    im = EX.render(blank(), sh, "t")
    assert im.shape == (H + 26, W, 3) and (im != blank()[0, 0]).any()
    cs = EX.contact_sheet({i: im for i in range(7)}, cols=5, tw=200)
    assert cs.shape[1] == 1000 and cs.shape[0] > 2 * 100
    assert R.parse_frames("60-62,65") == [60, 61, 62, 65]
