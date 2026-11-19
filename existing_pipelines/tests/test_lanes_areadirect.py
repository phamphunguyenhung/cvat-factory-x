import cv2
import numpy as np
import pytest
from cvat_tools.lanes import areadirect as ad
from cvat_tools.lanes import laneinterp as li

H, W = 720, 1280
VP = (640.0, 300.0)


def lane(xb, ytop=400):
    xt = xb + (ytop - H) * (VP[0] - xb) / (VP[1] - H)
    return [xb, H, xt, ytop]


ROAD = [100, H, 1180, H, 700, 330, 580, 330]  # trapezoid road covering the ego lane and sides


def mask(polys):
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        m |= ad.poly_mask(p, W, H)
    return m > 0


def test_direct_between_borders_and_no_overlap_alt():
    d, a = ad.build_area(ROAD, lane(400), lane(880), VP, H, W)
    assert d and a
    dm, am = mask(d), mask(a)
    ys, xs = np.nonzero(dm)
    for y in (700, 550, 450):
        row = xs[ys == y]
        xl = 400 + (y - H) * (VP[0] - 400) / (VP[1] - H)
        xr = 880 + (y - H) * (VP[0] - 880) / (VP[1] - H)
        assert row.min() >= xl - 8 and row.max() <= xr + 8
        assert row.max() - row.min() > 0.85 * (xr - xl)
    assert not (dm & am).any() or (dm & am).sum() < 0.01 * dm.sum()
    assert not (dm & ~mask([ROAD])).any() or (dm & ~mask([ROAD])).sum() < 0.01 * dm.sum()


def test_independent_of_occlusion_via_lane_interp():
    # borders come from laneinterp anchors; a "car" mask over the road changes nothing (no detection used)
    anchors = [{"frame": 60, "label": "lane/single white", "id": "L1", "points": lane(400)},
               {"frame": 60, "label": "lane/single white", "id": "L2", "points": lane(880)}]
    r = li.interpolate_lanes(anchors, [60])[60]
    left = next(x for x in r if x["id"] == "L1")["polyline"]
    right = next(x for x in r if x["id"] == "L2")["polyline"]
    d1, _ = ad.build_area(ROAD, left, right, None, H, W)
    car = np.zeros((H, W), np.uint8)
    cv2.rectangle(car, (500, 420), (780, 600), 1, -1)  # car in front: not an input
    d2, _ = ad.build_area(ROAD, left, right, None, H, W)
    assert d1 == d2
    m = mask(d1)
    assert m[500, 640] and m[600, 640]  # area under the car stays direct
    assert not m[600, 300] and not m[600, 1000]


def test_vp_from_pair():
    vp = ad.vp_from_pair(lane(400), lane(880), H)
    assert vp == pytest.approx(VP, abs=1.0)
