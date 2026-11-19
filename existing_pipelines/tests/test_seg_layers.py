from cvat_tools.segmentation import inventory as inv
from cvat_tools.segmentation import layers as L

WH = (100, 100)


def rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def by(shapes, label):
    return [s for s in shapes if s["label"] == label]


def test_car_covers_40pct_road_no_hole():
    road = {"label": "road", "polygon": rect(0, 50, 100, 100), "conf": 0.9, "id": 1}  # 5000
    car = {"label": "car", "polygon": rect(0, 50, 100, 70), "conf": 0.8, "id": 2}  # 2000 = 40%
    out = L.resolve([road, car], WH)
    r, c = by(out, "road")[0], by(out, "car")[0]
    assert len(out) == 2 and len(r["polygon"]) == 8  # nguyen vung, khong khoet
    assert abs(r["occlusion_ratio"] - 0.4) < 1e-6 and r["occluded"]
    assert c["z_order"] > r["z_order"] and not c["occluded"]
    hv = L.holes_view(r, [c])
    assert abs(L.to_geom(hv[0]["exterior"]).area - 3000) < 1e-6
    assert len(r["polygon"]) == 8  # holes_view khong ghi vao shape


def test_holes_view_makes_hole():
    big = {"polygon": rect(0, 0, 100, 100)}
    hv = L.holes_view(big, [{"polygon": rect(40, 40, 60, 60)}])
    assert len(hv) == 1 and len(hv[0]["holes"]) == 1


def test_occlusion_threshold():
    road = {"label": "road", "polygon": rect(0, 0, 100, 100), "id": 1}
    car = {"label": "car", "polygon": rect(0, 0, 100, 25), "id": 2}  # 25%
    assert not by(L.resolve([road, car], WH), "road")[0]["occluded"]
    assert by(L.resolve([road, car], WH, occ_thresh=0.2), "road")[0]["occluded"]
    car2 = {"label": "car", "polygon": rect(0, 0, 100, 30), "id": 3}  # dung nguong 0.30
    assert by(L.resolve([road, car2], WH), "road")[0]["occluded"]


def test_crosswalk_clipped_to_road():
    road = {"label": "road", "polygon": rect(0, 50, 100, 100), "id": 1, "conf": 1.0}
    cw = {"label": "lane/crosswalk", "polygon": rect(10, 30, 50, 70), "id": 2, "conf": 1.0}  # nua tren ngoai road
    s = by(L.resolve([road, cw], WH), "lane/crosswalk")[0]
    assert abs(L.to_geom(s["polygon"]).area - 800) < 1e-6
    assert s["conf"] < 1.0 and any(i.startswith("clipped_to_road") for i in s["_issues"])
    out, dr = L.resolve([road, {**cw, "polygon": rect(0, 0, 20, 20)}], WH, return_dropped=True)
    assert not by(out, "lane/crosswalk") and dr[0]["why"] == "outside_road"


def test_outside_frame_dropped_and_truncated():
    a = {"label": "car", "polygon": rect(200, 200, 300, 300), "id": 1}
    b = {"label": "car", "polygon": rect(80, 10, 120, 30), "id": 2}  # nua ngoai khung
    out, dr = L.resolve([a, b], WH, return_dropped=True)
    assert len(out) == 1 and out[0]["id"] == 2 and out[0]["truncated"]
    assert abs(out[0]["truncated_ratio"] - 0.5) < 1e-6
    assert dr[0]["why"] == "outside_frame"


def test_direct_alternative_disjoint_within_road():
    road = {"label": "road", "polygon": rect(0, 0, 100, 100), "id": 1}
    d = {"label": "area/direct", "polygon": rect(0, 0, 60, 100), "id": 2, "conf": 1.0}
    alt = {"label": "area/alternative", "polygon": rect(40, 0, 100, 100), "id": 3, "conf": 1.0}
    out = L.resolve([road, d, alt], WH)
    a, dd = by(out, "area/alternative")[0], by(out, "area/direct")[0]
    assert L.to_geom(a["polygon"]).intersection(L.to_geom(dd["polygon"])).area < 1e-6
    assert a["conf"] < 1.0 and a["z_order"] == dd["z_order"] == 0   # z moi theo cach nguoi dung ve (frame 71)
    assert not by(out, "road")[0]["occluded"]  # lop phu khong che road


def test_same_layer_nearer_car_occludes():
    far = {"label": "car", "polygon": rect(0, 0, 50, 40), "id": 1}
    near = {"label": "car", "polygon": rect(0, 20, 50, 60), "id": 2}
    out = L.resolve([far, near], WH)
    f = [s for s in out if s["id"] == 1][0]
    assert abs(f["occlusion_ratio"] - 0.5) < 1e-6 and not [s for s in out if s["id"] == 2][0]["occluded"]


def test_inventory():
    road = {"label": "road", "polygon": rect(0, 50, 100, 100), "id": 1, "conf": 0.9}
    cars = [{"label": "car", "polygon": rect(0, 50, 100, 70), "id": 2, "conf": 0.8},
            {"label": "car", "polygon": rect(0, 0, 10, 10), "id": 3, "conf": 0.6}]
    sh = L.resolve([road] + cars, WH)
    t = inv.inventory(sh)
    assert t["car"]["count"] == 2 and t["road"]["occluded_count"] == 1
    assert abs(t["car"]["conf_mean"] - 0.7) < 1e-6 and t["road"]["bbox_list"] == [[0, 50, 100, 100]]
    md = inv.frame_markdown(7, sh, expected_labels=["sky"])
    assert "| car | 2 |" in md and "Thieu: sky" in md
    s = inv.summary_markdown({1: sh, 2: []})
    assert "| car | 1/2 | 2 |" in s
