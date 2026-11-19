import cv2
import numpy as np

from cvat_tools.lanes.detectors.crosswalk import LABEL, detect_crosswalk

ROAD = [0, 300, 1280, 300, 1280, 720, 0, 720]


def _zebra(dark=False):
    img = np.full((720, 1280, 3), 70, np.uint8)
    rng = np.random.default_rng(0)
    img = (img + rng.integers(-5, 5, img.shape)).clip(0, 255).astype(np.uint8)
    for k in range(8):  # vertical ticks side by side, 40 px period
        x = 400 + 40 * k
        cv2.rectangle(img, (x, 450), (x + 20, 500), (235, 235, 235), -1)
    return (img * (0.25 if dark else 1.0)).astype(np.uint8)


def test_blank_road_no_detection():
    assert detect_crosswalk(np.full((720, 1280, 3), 80, np.uint8), ROAD) == []


def test_zebra_day_and_dark_found_inside_box():
    for dark in (False, True):
        res = detect_crosswalk(_zebra(dark), ROAD)
        assert res and res[0]["label"] == LABEL and 0 < res[0]["conf"] <= 1
        poly = np.array(res[0]["polygon"]).reshape(-1, 2)
        assert 4 <= len(poly) <= 6
        assert poly[:, 0].min() > 350 and poly[:, 0].max() < 760 and poly[:, 1].min() > 420 and poly[:, 1].max() < 530


def test_outside_road_ignored():
    assert detect_crosswalk(_zebra(), [0, 600, 1280, 600, 1280, 720, 0, 720]) == []
