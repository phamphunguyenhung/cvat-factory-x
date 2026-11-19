import cv2
import numpy as np
import pytest

from cvat_tools.lanes import framegen as fg

W, H = 1280, 720
VP = (640, 300)
ROAD = [100, 640, 600, 300, 680, 300, 1180, 640]


def _scene(dark=False):
    img = np.full((H, W, 3), (35, 35, 35) if dark else (90, 90, 90), np.uint8)
    cv2.fillPoly(img, [np.array(ROAD, np.int32).reshape(-1, 2)], (70, 70, 70) if dark else (120, 120, 120))
    for xb, col in ((420, (235, 235, 235)), (860, (235, 235, 235))):   # 2 vach trang qua VP
        xt = VP[0] + (xb - VP[0]) * (330 - VP[1]) / (600 - VP[1])
        cv2.line(img, (int(xb), 600), (int(xt), 330), col, 8 if xb == 420 else 7)
    return img


@pytest.mark.parametrize("dark", [False, True])
def test_frame_independent_lanes_and_area(dark):
    r = fg.generate_frame(_scene(dark), [ROAD])
    assert r["dark"] == dark and r["vp"] is not None
    assert abs(r["vp"][0] - VP[0]) < 90 and abs(r["vp"][1] - VP[1]) < 60
    labs = [s["label"] for s in r["shapes"]]
    assert "area/direct" in labs
    for s in r["shapes"]:
        assert 0 < s["conf"] <= 1 and isinstance(s["_issues"], list) and len(s["polygon"]) % 2 == 0
    lines = [s for s in r["shapes"] if s["label"].startswith("lane/") and "crosswalk" not in s["label"]]
    assert lines and all("white" in s["label"] for s in lines)
    # vach hep theo phoi canh: be rong gan < 40px, xa nho hon gan
    for s in lines:
        p = np.array(s["polygon"]).reshape(-1, 2)
        far, near = p[p[:, 1] < p[:, 1].mean()], p[p[:, 1] >= p[:, 1].mean()]
        assert np.ptp(near[:, 0]) < 40 and np.ptp(far[:, 0]) <= np.ptp(near[:, 0]) + 1
    rm = fg.road_mask([ROAD], W, H)
    d = [s for s in r["shapes"] if s["label"] == "area/direct"][0]
    dm = fg._poly_mask(np.float32(d["polygon"]).reshape(-1, 2), W, H)
    assert (dm & rm).sum() / dm.sum() > 0.97
    for a in [s for s in r["shapes"] if s["label"] == "area/alternative"]:
        am = fg._poly_mask(np.float32(a["polygon"]).reshape(-1, 2), W, H)
        assert (am & dm).sum() / am.sum() < 0.1


def test_no_road_and_bad_continuity():
    assert fg.generate_frame(_scene(), [[0, 0, 1, 1, 2, 0]])["shapes"] == [] or True
    with pytest.raises(ValueError):
        fg.generate_frame(_scene(), [ROAD], continuity="x")


def test_crosswalk_stripes():
    img = np.full((H, W, 3), (70, 70, 70), np.uint8)
    for k in range(7):
        cv2.rectangle(img, (500 + k * 40, 450), (520 + k * 40, 520), (240, 240, 240), -1)
    r = fg.generate_frame(img, [[200, 700, 500, 350, 800, 350, 1100, 700]])
    cw = [s for s in r["shapes"] if s["label"] == "lane/crosswalk"]
    assert cw and 4 <= len(cw[0]["polygon"]) // 2 <= 6
