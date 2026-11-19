import json

import numpy as np
import pytest
from cvat_tools.lanes import laneinterp as li

H = 720


def A(frame, pts, label="lane/single white", id=None):
    d = {"frame": frame, "label": label, "points": pts}
    if id is not None:
        d["id"] = id
    return d


def lane_to_vp(xb, vp=(640, 300), ytop=400):
    """Polyline from (xb,H) towards VP, ending at y=ytop."""
    xt = xb + (ytop - H) * (vp[0] - xb) / (vp[1] - H)
    return [xb, H, xt, ytop]


def test_lane_params_bottom_and_vp_fallback():
    xb, vp = li.lane_params([100, 600, 200, 500])  # dx/dy = -1
    assert xb == pytest.approx(100 - (720 - 600))
    assert vp == pytest.approx((200, 500))


def test_interpolate_midpoint_exact():
    anchors = [A(60, lane_to_vp(300), id="L"), A(70, lane_to_vp(500), id="L")]
    res = li.interpolate_lanes(anchors, [60, 65, 70])
    assert res[60][0]["polyline"][:2] == pytest.approx([300, 720], abs=0.2)
    assert res[70][0]["polyline"][:2] == pytest.approx([500, 720], abs=0.2)
    mid = res[65][0]
    assert mid["polyline"][0] == pytest.approx(400, abs=0.2)
    assert mid["polyline"][2] == pytest.approx(400 + (400 - 720) * (640 - 400) / (300 - 720), abs=0.3)
    assert mid["polyline"][3] == 400


def test_outside_range_holds_nearest():
    anchors = [A(60, lane_to_vp(300), id="L"), A(70, lane_to_vp(500), id="L")]
    res = li.interpolate_lanes(anchors, [50, 79])
    assert res[50][0]["polyline"][0] == pytest.approx(300, abs=0.2)
    assert res[79][0]["polyline"][0] == pytest.approx(500, abs=0.2)


def test_two_lanes_shared_vp_and_ids():
    anchors = [A(60, lane_to_vp(300), id="a"), A(60, lane_to_vp(900), id="b"),
               A(70, lane_to_vp(400), id="a"), A(70, lane_to_vp(1000), id="b")]
    res = li.interpolate_lanes(anchors, [65])[65]
    xs = {r["id"]: r["polyline"][0] for r in res}
    assert xs["a"] == pytest.approx(350, abs=1) and xs["b"] == pytest.approx(950, abs=1)


def test_double_gives_two_parallel_polylines():
    r = li.interpolate_lanes([A(60, lane_to_vp(300), "lane/double yellow", "d")], [60], double_offset=10)[60][0]
    a, b = r["polylines"]
    assert len(r["polylines"]) == 2
    assert b[0] - a[0] == pytest.approx(10) and b[2] - a[2] == pytest.approx(10)
    assert a[1] == b[1] and a[3] == b[3]


def test_crosswalk_quad_interp_and_lanes_skipped():
    q1 = [100, 500, 300, 500, 320, 560, 80, 560]
    q2 = [200, 520, 400, 520, 420, 600, 180, 600]
    anchors = [A(60, q1, "crosswalk", "c"), A(70, q2, "crosswalk", "c"), A(60, lane_to_vp(300), id="L")]
    qs = li.interpolate_quad(anchors, [65, 80])
    assert qs[65][0]["polygon"] == pytest.approx([150, 510, 350, 510, 370, 580, 130, 580])
    assert qs[80][0]["polygon"] == pytest.approx(q2)
    assert all(r["label"] != "crosswalk" for r in li.interpolate_lanes(anchors, [65])[65])


def test_crosswalk_mixed_vertex_counts():
    q4 = [100, 500, 300, 500, 300, 560, 100, 560]
    q6 = [200, 520, 300, 520, 400, 520, 400, 600, 300, 600, 200, 600]
    anchors = [A(60, q4, "crosswalk", "c"), A(70, q6, "crosswalk", "c")]
    qs = li.interpolate_quad(anchors, [60, 65, 70])
    assert len(qs[65][0]["polygon"]) == 12 and len(qs[60][0]["polygon"]) == 12
    xs = qs[60][0]["polygon"][0::2]; ys = qs[60][0]["polygon"][1::2]
    assert min(xs) == pytest.approx(100) and max(xs) == pytest.approx(300)
    assert min(ys) == pytest.approx(500) and max(ys) == pytest.approx(560)
    assert qs[70][0]["polygon"] == pytest.approx(
        [v for pt in li.resample_polygon(q6, 6) for v in pt], abs=0.1)


def test_snap_on_off():
    poly = [300, 720, 400, 400]
    cand = [{"color": "white", "p": [310, 720, 410, 400]}]  # 10px off
    snapped = li.snap_to_candidates(poly, cand, max_px=20, enabled=True)
    assert snapped == pytest.approx([310, 720, 410, 400], abs=0.2)
    assert li.snap_to_candidates(poly, cand, max_px=5, enabled=True) == poly
    assert li.snap_to_candidates(poly, cand, max_px=20, enabled=False) == poly
    assert li.snap_to_candidates(poly, [], 20, True) == poly


def test_is_night():
    assert li.is_night(np.full((72, 128, 3), 20, np.uint8))
    assert not li.is_night(np.full((72, 128, 3), 150, np.uint8))


def test_lane_polygon_matches_line_poly_shape():
    q = li.lane_polygon([300, 720, 400, 400], half=6)
    assert len(q) == 8 and min(q[0::2]) == 294 and max(q[0::2]) == 406


def test_load_anchors_schema(tmp_path):
    good = {"job": 285, "anchors": [A(70, [1, 2, 3, 4]), A(60, [1, 2, 3, 4, 5, 6], id="x"),
                                    A(60, [0, 0, 1, 0, 1, 1, 0, 1], "crosswalk")]}
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps(good))
    d = li.load_anchors(p)
    assert d["job"] == 285 and [a["frame"] for a in d["anchors"]] == [60, 60, 70]
    for bad in ({"anchors": [A(60, [1, 2])]}, {"anchors": [A(60, [1, 2, 3, 4], "crosswalk")]},
                {"anchors": [A(60, list(range(14)), "crosswalk")]},
                {"anchors": [A(60, [1, 2, 3])]}, {"nope": 1}):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            li.load_anchors(p)
