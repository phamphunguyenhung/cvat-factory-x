import json

import numpy as np
from cvat_tools.review import apply as A
from cvat_tools.review import edits as E
from cvat_tools.review import pack as P
from cvat_tools.review import validators as V

WH = (400, 300)
RS = V.load_ruleset()


def sh(i, label, pts, typ="polygon", z=None, **kw):
    z = V.expected_z(label, RS) if z is None else z
    if z is None:
        z = -8
    return {"id": i, "frame": 1, "label": label, "type": typ, "points": [float(p) for p in pts], "z_order": z,
            "occluded": kw.get("occluded", False), "truncated": kw.get("truncated", False), "raw": {"frame": 1, "type": typ}}


def rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


ROAD = sh("road", "road", rect(0, 150, 400, 300))


def codes(issues, sid=None):
    return {i["code"] for i in issues if sid is None or i["shape_id"] == sid}


def val(shapes):
    return V.validate_frame(shapes, WH, RS)


def test_clean_scene_has_no_errors():
    lane = sh("l", "lane/single white", [100, 160, 104, 160, 130, 290, 126, 290])
    pole = sh("p", "pole", [300, 20, 306, 20, 306, 160, 300, 160])
    car = sh("c", "car", [20, 200, 90, 260], typ="rectangle")
    assert not [i for i in val([ROAD, lane, pole, car]) if i["severity"] == "error"]


def test_known_errors_detected():
    wide_lane = sh("lw", "lane/single white", rect(100, 160, 250, 290))          # be rong tb ~70px
    pole_fat = sh("pf", "pole", rect(300, 20, 340, 60))                         # vuong, khong cao
    pole_many = sh("pm", "pole", [300, 0, 304, 0, 305, 20, 306, 40, 306, 60, 305, 80, 304, 100, 303, 120, 302, 140, 300, 140])
    bow = sh("bow", "area/direct", [10, 160, 200, 290, 200, 160, 10, 290])      # nut co: tu cat
    dupv = sh("dv", "lane/single white", [100, 160, 100, 160, 104, 160, 130, 290, 126, 290])
    outside = sh("out", "lane/single yellow", [100, 20, 104, 20, 104, 100, 100, 100])  # ngoai road
    a1 = sh("a1", "area/direct", rect(0, 160, 200, 290))
    a2 = sh("a2", "area/alternative", rect(150, 160, 399, 290))
    badz = sh("bz", "lane/single white", [200, 160, 203, 160, 203, 290, 200, 290], z=-3)   # z cu (-3) nay sai: lane z=0
    badt = sh("bt", "car", rect(10, 10, 60, 60), typ="polygon")
    tiny = sh("tb", "car", [5, 5, 7, 7], typ="rectangle")
    d1 = sh("d1", "sidewalk", rect(0, 100, 100, 150))
    d2 = sh("d2", "sidewalk", rect(0, 100, 100, 150))
    iss = val([ROAD, wide_lane, pole_fat, pole_many, bow, dupv, outside, a1, a2, badz, badt, tiny, d1, d2])
    assert "LANE_TOO_WIDE" in codes(iss, "lw")
    assert "POLE_SHAPE" in codes(iss, "pf")
    assert "VERTEX_MAX" in codes(iss, "pm")
    assert "POLY_INVALID" in codes(iss, "bow")
    assert "DUP_VERTEX" in codes(iss, "dv")
    assert "NOT_IN_ROAD" in codes(iss, "out")
    assert "AREA_OVERLAP" in codes(iss, "a2")
    assert "Z_ORDER" in codes(iss, "bz")
    assert "SHAPE_TYPE" in codes(iss, "bt")
    assert "BOX_TINY" in codes(iss, "tb")
    assert "DUPLICATE" in codes(iss, "d2")
    assert not codes(iss, "road")


def test_occluded_truncated_flags():
    near = sh("near", "car", [0, 150, 200, 290], typ="rectangle")
    far = sh("far", "car", [50, 160, 150, 240], typ="rectangle")      # bi near che 100%
    edge = sh("edge", "car", [300, 100, 400, 160], typ="rectangle")   # cham mep phai
    iss = val([ROAD, near, far, edge])
    assert "OCC_MISSING" in codes(iss, "far")
    assert "TRUNC_MISSING" in codes(iss, "edge")
    ok = val([ROAD, near, sh("far", "car", [50, 160, 150, 240], typ="rectangle", occluded=True),
              sh("edge", "car", [300, 100, 400, 160], typ="rectangle", truncated=True)])
    assert not codes(ok, "far") & {"OCC_MISSING"} and not codes(ok, "edge") & {"TRUNC_MISSING"}


def test_pack_lists_expected_codes_sorted(tmp_path):
    img = np.full((WH[1], WH[0], 3), 40, np.uint8)
    shapes = [ROAD, sh("lw", "lane/single white", rect(100, 160, 250, 290)),
              sh("pm", "pole", [300, 0, 304, 0, 305, 20, 306, 40, 306, 60, 305, 80, 304, 100, 303, 120, 302, 140, 300, 140]),
              sh("bow", "area/direct", [10, 160, 200, 290, 200, 160, 10, 290])]
    doc = P.build_pack(1, shapes, img, tmp_path, RS)
    d = tmp_path / "frame_1"
    assert (d / "overview.jpg").exists() and (d / "issues.json").exists()
    got = json.loads((d / "issues.json").read_text(encoding="utf8"))
    flagged = {s["shape_id"]: {i["code"] for i in s["issues"]} for s in got["shapes"]}
    assert flagged["lw"] == {"LANE_TOO_WIDE"} and "VERTEX_MAX" in flagged["pm"] and "POLY_INVALID" in flagged["bow"]
    assert all((d / c["file"]).exists() for s in got["shapes"] for c in s["crops"])
    assert any(c["kind"].startswith("focus") for s in got["shapes"] if s["shape_id"] == "bow" for c in s["crops"])
    assert doc["counts_by_code"] == got["counts_by_code"] and got["counts_by_severity"]["error"] >= 3


def test_schema_rejects_absolute_for_narrow_and_bad_ops():
    lane = sh("l", "lane/single white", [100, 160, 104, 160, 130, 290, 126, 290])
    car = sh("c", "car", [20, 200, 90, 260], typ="rectangle")
    m = {s["id"]: s for s in (lane, car)}
    assert E.validate_command({"op": "move_vertex", "shape": "l", "idx": 1, "to": [105, 161]}, m, RS)      # tuyet doi cam
    assert not E.validate_command({"op": "move_vertex", "shape": "l", "idx": 1, "dx": 2, "dy": 0, "snap_band": 4}, m, RS)
    assert E.validate_command({"op": "add_vertex", "shape": "l", "after_idx": 0, "at": [1, 1]}, m, RS)
    assert not E.validate_command({"op": "add_vertex", "shape": "l", "after_idx": 0, "t": 0.5}, m, RS)
    assert E.validate_command({"op": "move_vertex", "shape": "l", "idx": 9, "dx": 1, "dy": 0}, m, RS)      # idx vuot
    assert E.validate_command({"op": "move_vertex", "shape": "l", "idx": 0, "dx": 99, "dy": 0}, m, RS)
    assert E.validate_command({"op": "nope", "shape": "l"}, m, RS)
    assert E.validate_command({"op": "delete_vertex", "shape": "c", "idx": 0}, m, RS)                      # rectangle
    assert E.validate_command({"op": "relabel", "shape": "c", "label": "road"}, m, RS)                     # sai kieu
    assert E.validate_command({"op": "set_flag", "shape": "l", "flag": "truncated"}, m, RS)
    assert E.validate_command({"op": "snap_to_edge", "shape": "l", "band_px": 50}, m, RS)
    assert E.validate_command({"op": "delete_shape", "shape": "zzz"}, m, RS)
    assert not E.validate_commands([{"op": "delete_shape", "shape": "l"}], m, RS)["errors"]


def test_edits_resolve_errors_and_log(tmp_path):
    bow = sh("bow", "area/direct", [10, 160, 200, 290, 200, 160, 10, 290])
    pm = sh("pm", "pole", [300, 0, 304, 0, 305, 20, 306, 40, 306, 60, 305, 80, 304, 100, 303, 120, 302, 140, 300, 140])
    dup = sh("d2", "road", rect(0, 150, 400, 300))
    bad = sh("badpole", "pole", rect(300, 20, 340, 60))
    shapes = [ROAD, dup, bow, pm, bad]
    cmds = [
        {"op": "delete_shape", "shape": "d2"},
        {"op": "move_vertex", "shape": "bow", "idx": 3, "to": [10, 290]},      # idx3=(200,160)->(10,290)?? doi thanh hop le
        {"op": "move_vertex", "shape": "bow", "idx": 2, "to": [200, 290]},
        {"op": "delete_vertex", "shape": "pm", "idx": 8},
        {"op": "delete_vertex", "shape": "pm", "idx": 6},
        {"op": "delete_vertex", "shape": "pm", "idx": 4},
        {"op": "delete_shape", "shape": "badpole"},
        {"op": "move_vertex", "shape": "pm", "idx": 0, "to": [1, 1]},          # cam: pole hep
    ]
    res = A.apply_edits(shapes, cmds, frame_wh=WH, rs=RS)
    assert [e["status"] for e in res["log"]][-1] == "rejected"
    assert "DUPLICATE" in res["counts_before"] and "VERTEX_MAX" in res["counts_before"]
    after_errs = {c for v in res["remaining"].values() for c in v}
    assert not after_errs, res["counts_after"]
    assert "bow" in res["resolved"] and "d2" in res["resolved"]
    assert [s["id"] for s in shapes] == ["road", "d2", "bow", "pm", "badpole"]       # dau vao khong bi doi
    src = tmp_path / "LATEST_x.json"
    src.write_text("{}")
    out = A.save_result(res, tmp_path / "edited", source=src)
    assert (tmp_path / "edited").exists() and json.loads(open(out["log"], encoding="utf8").read())["resolved"]
    assert src.read_text() == "{}"


def test_split_merge_relabel_flag():
    a = sh("a", "sidewalk", rect(0, 0, 100, 100))
    res = A.apply_edits([a], [{"op": "split", "shape": "a", "line": [[50, -10], [50, 110]]}], frame_wh=WH, rs=RS)
    assert len(res["shapes"]) == 2 and abs(sum(V.poly_of(s).area for s in res["shapes"]) - 10000) < 1
    ids = [s["id"] for s in res["shapes"]]
    res2 = A.apply_edits(res["shapes"], [{"op": "merge", "shapes": ids}], frame_wh=WH, rs=RS)
    assert len(res2["shapes"]) == 1 and abs(V.poly_of(res2["shapes"][0]).area - 10000) < 1
    r3 = A.apply_edits(res2["shapes"], [{"op": "relabel", "shape": ids[0], "label": "terrain"}], frame_wh=WH, rs=RS)
    assert r3["shapes"][0]["label"] == "terrain"
    c = sh("c", "car", [0, 0, 50, 50], typ="rectangle")
    r4 = A.apply_edits([c], [{"op": "set_flag", "shape": "c", "flag": "truncated"},
                             {"op": "move_vertex", "shape": "c", "idx": 2, "dx": 5, "dy": 5}], frame_wh=WH, rs=RS)
    assert r4["shapes"][0]["truncated"] and r4["shapes"][0]["points"] == [0, 0, 55, 55]


def test_snap_to_edge_pulls_narrow_lane_onto_stripe():
    img = np.full((120, 200, 3), 30, np.uint8)
    img[10:110, 98:104] = 230                                           # vach sang x 98..104 (rong 6)
    lane = sh("l", "lane/single white", [100, 10, 106, 10, 106, 110, 100, 110])   # lech 2px
    lane["z_order"] = -3
    road = sh("road", "road", rect(0, 0, 200, 120))
    res = A.apply_edits([road, lane], [{"op": "snap_to_edge", "shape": "l", "band_px": 4}], img=img, rs=RS)
    ent = res["log"][0]
    assert ent["status"] == "ok" and ent["snap"][0]["status"] == "ok"
    xs = [res["shapes"][1]["points"][i] for i in range(0, len(res["shapes"][1]["points"]), 2)]
    assert 96.5 <= min(xs) <= 99.5 and 102.5 <= max(xs) <= 105.5, xs


def test_snap_rectangle_to_box_edges():
    img = np.full((200, 300, 3), 30, np.uint8)
    img[50:120, 60:160] = 220
    car = sh("c", "car", [57, 53, 163, 118], typ="rectangle")
    res = A.apply_edits([car], [{"op": "snap_to_edge", "shape": "c", "band_px": 5}], img=img, rs=RS)
    x0, y0, x1, y1 = res["shapes"][0]["points"]
    assert abs(x0 - 60) <= 1 and abs(x1 - 160) <= 1 and abs(y0 - 50) <= 1 and abs(y1 - 120) <= 1


def test_to_draft_keeps_raw_fields():
    s = sh("a", "sidewalk", rect(0, 0, 10, 10))
    s["raw"] = {"frame": 1, "type": "polygon", "label_id": 77, "_src": "x"}
    d = A.to_draft_shapes([s])[0]
    assert d["label_id"] == 77 and d["_label"] == "sidewalk" and d["_review_id"] == "a"
