"""Synthetic checks for the dedicated yellow / curb lane detectors (no dataset needed)."""

import cv2
import numpy as np

from cvat_tools.lanes.detectors.curb import detect_curb
from cvat_tools.lanes.detectors.yellow import detect_double_yellow, detect_single_yellow

W, H = 1280, 720
ROAD = [[100, 700], [560, 300], [720, 300], [1180, 700]]
YELLOW = (0, 190, 240)  # BGR


def _scene(bg=(85, 85, 85)):
    img = np.full((H, W, 3), (40, 40, 40), np.uint8)
    cv2.fillPoly(img, [np.array(ROAD, np.int32)], bg)
    return img


def _rail(img, p, q, w=7, color=YELLOW):
    cv2.line(img, p, q, color, w)


def test_two_close_rails_give_one_double_and_no_single():
    img = _scene()
    _rail(img, (360, 600), (600, 330))
    _rail(img, (385, 600), (622, 330))
    dbl = detect_double_yellow(img, ROAD)
    assert len(dbl) == 1 and dbl[0]["label"] == "lane/double yellow"
    assert len(dbl[0]["polyline"]) >= 2 and len(dbl[0]["polygon"]) >= 3 and 0 < dbl[0]["conf"] <= 1
    assert detect_single_yellow(img, ROAD) == []


def test_single_rail_gives_single_yellow():
    img = _scene()
    _rail(img, (400, 620), (620, 320))
    sgl = detect_single_yellow(img, ROAD)
    assert len(sgl) == 1 and sgl[0]["label"] == "lane/single yellow"
    assert detect_double_yellow(img, ROAD) == []


def test_fat_band_is_not_split_into_a_double():
    img = _scene()
    _rail(img, (400, 620), (620, 320), w=26)
    assert detect_double_yellow(img, ROAD) == []


def test_paint_outside_road_is_ignored():
    img = _scene()
    _rail(img, (40, 650), (200, 330))   # on the pavement, left of the road polygon
    assert detect_single_yellow(img, ROAD) == [] and detect_double_yellow(img, ROAD) == []


def test_sodium_tinted_road_without_paint_gives_nothing():
    # whole road is orange-yellow from street lamps; plus a round lamp reflection and a horizontal streak
    img = _scene(bg=(50, 105, 130))
    cv2.circle(img, (640, 520), 38, (120, 210, 250), -1)
    cv2.line(img, (300, 650), (900, 640), (110, 190, 235), 9)   # flat: stop-line-like reflection
    assert detect_single_yellow(img, ROAD) == []
    assert detect_double_yellow(img, ROAD) == []


def test_real_yellow_line_survives_sodium_tint():
    img = _scene(bg=(50, 105, 130))
    _rail(img, (390, 620), (610, 320), w=8, color=(60, 175, 240))
    _rail(img, (415, 620), (632, 320), w=8, color=(60, 175, 240))
    assert len(detect_double_yellow(img, ROAD)) == 1


def test_orange_cone_hue_is_not_yellow_paint():
    img = _scene()
    _rail(img, (400, 620), (620, 320), w=12, color=(20, 110, 235))   # orange
    assert detect_single_yellow(img, ROAD) == []


def test_no_road_returns_empty():
    img = _scene()
    assert detect_single_yellow(img, None) == [] and detect_double_yellow(img, []) == []


def _sidewalk_scene():
    img = np.full((H, W, 3), (95, 95, 95), np.uint8)
    road = [[100, 700], [560, 300], [720, 300], [1000, 700]]
    side = [[720, 300], [1280, 300], [1280, 700], [1000, 700]]
    cv2.fillPoly(img, [np.array(side, np.int32)], (170, 170, 170))   # lighter pavement
    return img, road, side


def test_curb_found_on_road_sidewalk_edge():
    img, road, side = _sidewalk_scene()
    out = detect_curb(img, road, side)
    assert len(out) == 1 and out[0]["label"] == "lane/road curb"
    (x1, y1), (x2, y2) = out[0]["polyline"][0], out[0]["polyline"][-1]
    assert abs(x2 - x1) > 100 and min(x1, x2) > 700 and min(y1, y2) <= 305
    assert len(out[0]["polygon"]) >= 4


def test_curb_without_sidewalk_uses_light_dark_step():
    img, road, _ = _sidewalk_scene()
    assert len(detect_curb(img, road)) == 1


def test_curb_empty_when_other_side_is_same_surface():
    img = np.full((H, W, 3), (95, 95, 95), np.uint8)          # road and surroundings identical
    road = [[100, 700], [560, 300], [720, 300], [1000, 700]]
    assert detect_curb(img, road) == []
    side_far = [[0, 0], [100, 0], [100, 100], [0, 100]]       # sidewalk nowhere near the road edge
    assert detect_curb(img, road, side_far) == []


def test_curb_ignores_flat_top_edge_and_image_border():
    img = np.full((H, W, 3), (95, 95, 95), np.uint8)
    road = [[0, 700], [0, 300], [1279, 300], [1279, 700]]     # box hugging the border with a flat top
    side = [[0, 0], [1279, 0], [1279, 300], [0, 300]]
    assert detect_curb(img, road, side) == []
