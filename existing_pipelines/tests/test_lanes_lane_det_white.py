import cv2
import numpy as np

from cvat_tools.lanes.detectors.white import detect_double_white, detect_single_white

W, H = 1280, 720
ROAD = [200, 300, 1000, 300, 1280, 700, 0, 700]


def _scene(lines, night=False):
    img = np.full((H, W, 3), 25 if night else 90, np.uint8)
    cv2.fillPoly(img, [np.array(ROAD, np.int32).reshape(-1, 2)], (35, 35, 35) if night else (95, 95, 95))
    for (x1, y1, x2, y2, t) in lines:
        cv2.line(img, (x1, y1), (x2, y2), (240, 240, 240), t)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _x_at(pl, y):
    return pl[0] + (y - pl[1]) * (pl[2] - pl[0]) / (pl[3] - pl[1])


def test_single_white_found():
    r = detect_single_white(_scene([(640, 320, 400, 600, 8)]), ROAD)
    assert len(r) == 1 and r[0]["label"] == "lane/single white"
    assert abs(_x_at(r[0]["polyline"], 450) - (640 - 240 * 130 / 280)) < 8
    assert 0 < r[0]["conf"] <= 1 and r[0]["id"] and len(r[0]["polygon"]) >= 8


def test_broken_line_is_merged_into_one():
    # one straight line cut into 3 dashes by gaps
    segs = [(640, 320, 600, 380, 8), (586, 400, 546, 460, 8), (532, 480, 492, 540, 8)]
    r = detect_single_white(_scene(segs), ROAD)
    assert len(r) == 1
    p = r[0]["polyline"]
    assert p[1] < 340 and p[3] > 520  # spans all dashes


def test_double_white_pair():
    img = _scene([(640, 320, 400, 600, 6), (662, 320, 422, 600, 6)])
    d = detect_double_white(img, ROAD)
    assert len(d) == 1 and d[0]["label"] == "lane/double white"
    assert detect_single_white(img, ROAD) == []


def test_far_apart_lines_are_not_double():
    img = _scene([(500, 320, 250, 650, 7), (780, 320, 1000, 650, 7)])
    assert detect_double_white(img, ROAD) == []
    assert len(detect_single_white(img, ROAD)) == 2


def test_outside_road_ignored_and_short_blob_dropped():
    img = _scene([(640, 100, 640, 250, 8), (600, 500, 585, 520, 8)])  # above road; tiny dash
    assert detect_single_white(img, ROAD) == []


def test_night_line():
    r = detect_single_white(_scene([(640, 320, 400, 600, 8)], night=True), ROAD)
    assert len(r) == 1
