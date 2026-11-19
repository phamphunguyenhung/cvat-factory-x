"""STUFF_OVER_THING: validator (warn) -> policy (SNAP_EDGE subtract_things | FLAG_REVIEW) -> executor."""
from cvat_tools.decision import action_schema as S
from cvat_tools.decision import engine as EN
from cvat_tools.decision import rule_policy as RP
from cvat_tools.review import validators as V
from shapely.geometry import Polygon

WH = (400, 300)
RS = V.load_ruleset()
BUILDING = [0, 0, 400, 0, 400, 160, 0, 160]


def sh(i, label, pts, typ="polygon", origin="tool"):
    z = V.expected_z(label, RS)
    return {"id": i, "frame": 1, "label": label, "type": typ, "points": [float(p) for p in pts],
            "z_order": -8 if z is None else z, "occluded": False, "truncated": False, "origin": origin,
            "raw": {"frame": 1, "type": typ, "_conf": 0.9}}


def truck(box=(150, 100, 250, 200), i="truck"):
    return sh(i, "truck", box, typ="rectangle")


def codes(issues):
    return [i["code"] for i in issues if i["code"] == "STUFF_OVER_THING"]


def run(shapes, **kw):
    return EN.run_frame(1, 1, shapes, None, RS, None, frame_wh=WH, **kw)


def area(s):
    return Polygon(list(zip(s["points"][0::2], s["points"][1::2]))).area


def test_validator_warns_with_things_and_ratio():
    iss = V.validate_frame([sh("b", "building", BUILDING), truck()], WH, RS)
    hit = [i for i in iss if i["code"] == "STUFF_OVER_THING"]
    assert len(hit) == 1 and hit[0]["severity"] == "warn" and hit[0]["shape_id"] == "b"
    t = hit[0]["data"]["things"][0]
    assert t["label"] == "truck" and t["source"] == "box" and t["cut_ratio"] > 0.2
    assert hit[0]["data"]["min_ratio"] == 0.05 and hit[0]["data"]["box_inset"] == 0.03


def test_validator_ignores_thing_fully_inside_and_below_threshold_and_non_stuff():
    inside = truck((150, 20, 250, 100))                      # tron trong nen: quy uoc nguyen vung
    assert not codes(V.validate_frame([sh("b", "building", BUILDING), inside], WH, RS))
    sliver = truck((150, 158, 250, 258))                     # chi lan 2px vao nen (<5%) sau inset
    assert not codes(V.validate_frame([sh("b", "building", BUILDING), sliver], WH, RS))
    # road khong thuoc STUFF_LABELS; traffic sign khong thuoc THING_LABELS
    assert not codes(V.validate_frame([sh("r", "road", [0, 100, 400, 100, 400, 300, 0, 300]), truck()], WH, RS))
    sign = sh("s", "traffic sign", [150, 100, 250, 200], typ="rectangle")
    assert not codes(V.validate_frame([sh("b", "building", BUILDING), sign], WH, RS))


def test_threshold_from_ruleset_review_key():
    rs = V.load_ruleset()
    rs["review"]["stuff_min_ratio"] = 0.99
    assert not codes(V.validate_frame([sh("b", "building", BUILDING), truck()], WH, rs))


def test_policy_maps_to_snap_subtract_with_box_confidence_lower():
    shapes = [sh("b", "building", BUILDING), truck()]
    iss = V.validate_frame(shapes, WH, RS)
    recs, ctx = RP.decide(shapes, iss, RS, WH, 1, 1)
    r = [x for x in recs if x["reason_code"] == "STUFF_OVER_THING"][0]
    assert r["proposed_action"] == "SNAP_EDGE" and r["params"]["mode"] == "subtract_things"
    assert r["confidence"] == 0.45 and not S.validate_action(S.record_action(r), ctx)
    masks = {"truck": [160, 110, 240, 110, 240, 190, 160, 190]}
    iss = V.validate_frame(shapes, WH, RS, masks)
    recs, _ = RP.decide(shapes, iss, RS, WH, 1, 1, thing_masks=masks)
    assert [x["confidence"] for x in recs if x["reason_code"] == "STUFF_OVER_THING"] == [0.70]


def test_executor_cuts_tool_shape_no_hole_and_marks_box():
    res = run([sh("b", "building", BUILDING), truck()])
    b = [s for s in res["shapes"] if s["label"] == "building"]
    assert len(b) == 1 and area(b[0]) < 400 * 160 - 80 * 40
    assert not Polygon(list(zip(b[0]["points"][0::2], b[0]["points"][1::2]))).interiors
    assert "cat_theo_hop" in b[0]["raw"]["_issues"]
    assert res["counts_before"].get("STUFF_OVER_THING") == 1 and "STUFF_OVER_THING" not in res["counts_after"]
    assert any(r["validation_status"] == "applied_ok" and r["params"].get("mode") == "subtract_things" for r in res["records"])


def test_executor_inside_thing_creates_no_hole_and_no_action():
    res = run([sh("b", "building", BUILDING), truck((150, 20, 250, 100))])
    assert not any(r["reason_code"] == "STUFF_OVER_THING" for r in res["records"])
    assert [s["points"] for s in res["shapes"] if s["id"] == "b"] == [BUILDING]


def test_human_shape_only_flagged_never_edited():
    res = run([sh("b", "building", BUILDING, origin="human"), truck()])
    r = [x for x in res["records"] if x["reason_code"] == "STUFF_OVER_THING"]
    assert r and all(x["proposed_action"] == "FLAG_REVIEW" for x in r)
    assert [s["points"] for s in res["shapes"] if s["id"] == "b"][0] == [float(v) for v in BUILDING]


def test_mask_preferred_over_box_and_no_cut_when_mask_clear():
    # hop de len building nhung mask that nam duoi y=160: khong loi
    masks = {"truck": [160, 170, 240, 170, 240, 195, 160, 195]}
    assert not codes(V.validate_frame([sh("b", "building", BUILDING), truck()], WH, RS, masks))
    # mask lan vao building hep hon hop: cat theo mask, khong theo hop
    masks = {"truck": [180, 120, 220, 120, 220, 200, 180, 200]}
    res = run([sh("b", "building", BUILDING), truck()], thing_masks=masks)
    b = [s for s in res["shapes"] if s["label"] == "building"][0]
    cut = 400 * 160 - area(b)
    assert 0 < cut < 50 * 40   # mask rong 40 (hop rong 100)
    assert "cat_theo_hop" not in b["raw"].get("_issues", [])
