import cv2
import numpy as np

from cvat_tools.lanes.detectors import REGISTRY, detect_all  # noqa: E402

W, H = 1280, 720
ROAD = [100, 700, 560, 300, 720, 300, 1180, 700]


def _img():
    img = np.full((H, W, 3), (40, 40, 40), np.uint8)
    cv2.fillPoly(img, [np.array(ROAD, np.int32).reshape(-1, 2)], (85, 85, 85))
    cv2.line(img, (400, 620), (620, 320), (0, 190, 240), 7)
    return img


def test_registry_labels():
    assert set(REGISTRY) == {"lane/crosswalk", "lane/single white", "lane/double white",
                             "lane/single yellow", "lane/double yellow", "lane/road curb"}


def test_detect_all_flat_polygons_and_filters():
    res = detect_all(_img(), ROAD, labels=["lane/single yellow"])
    assert res and all(r["label"] == "lane/single yellow" for r in res)
    for r in res:
        assert all(isinstance(v, float) for v in r["polygon"]) and len(r["polygon"]) % 2 == 0 and 0 < r["conf"] <= 1
    assert detect_all(_img(), ROAD, labels=["lane/single yellow"], min_conf=1.01) == []
    assert detect_all(_img(), []) == []


def test_detect_all_flat_with_curb_and_multi_road():
    res = detect_all(_img(), [ROAD, ROAD[:]], sidewalk_poly=[[0, 0, 90, 0, 90, 700, 0, 700]])
    for r in res:
        assert not isinstance(r["polygon"][0], list)
