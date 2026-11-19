import numpy as np

from cvat_tools.pose import person17 as P

# người đứng thẳng giả lập (px): torso ~100
GOOD = np.array([[100, 20], [105, 15], [95, 15], [110, 20], [90, 20], [130, 60], [70, 60], [150, 100], [50, 100], [160, 140], [40, 140],
                 [120, 160], [80, 160], [125, 220], [75, 220], [125, 280], [75, 280]], float)


def _person(xy, conf=0.9, score=0.9, box=(0, 0, 200, 300)):
    return {"bbox": list(box), "score": score, "xy": np.asarray(xy, float), "conf": np.full(17, conf) if np.isscalar(conf) else np.asarray(conf)}


def test_point_state_rules():
    assert P.point_state(0.0, 0.0, 0.9, 640, 480) == "outside"          # model không trả điểm
    assert P.point_state(-5, 10, 0.9, 640, 480) == "outside"
    assert P.point_state(300, 479, 0.1, 640, 480) == "outside"          # conf thấp sát biên dưới
    assert P.point_state(300, 400, 0.1, 640, 480, limb=True) == "outside"    # tay/chân conf < 0.3 trong ảnh: không đoán nổi -> v0
    assert P.point_state(300, 400, 0.1, 640, 480) == "occluded"         # mặt/vai/hông conf thấp: vẫn Q, đặt vị trí ước lượng, không v0
    assert P.point_state(300, 400, 0.35, 640, 480, limb=True) == "occluded"  # 0.3-0.5: che nhưng ước lượng được -> Q
    assert P.point_state(300, 200, 0.4, 640, 480) == "occluded"
    assert P.point_state(300, 200, 0.9, 640, 480) == "visible"


def test_enforce_image_lr_follows_image_side_not_anatomy():
    conf = np.full(17, 0.9)
    xy, c, st = P.enforce_image_lr(GOOD, conf)
    assert st == "ok" and np.allclose(xy, GOOD)                          # số chẵn (chỉ số lẻ) đã ở phải ảnh
    flipped = GOOD.copy()
    for a in range(1, 17, 2):                                            # model quay lưng: trái/phải giải phẫu ngược ảnh
        flipped[[a, a + 1]] = flipped[[a + 1, a]]
    xy, c, st = P.enforce_image_lr(flipped, conf)
    assert st == "swapped" and np.allclose(xy, GOOD)
    flat = GOOD.copy()
    flat[:, 0] = 100.0                                                    # người nằm ngang/xoay: hai bên cùng x -> không chắc
    assert P.enforce_image_lr(flat, conf)[2] == "uncertain"


def test_plausibility_accepts_normal_rejects_stretched_and_missing():
    conf = np.full(17, 0.9)
    assert P.plausibility_pose(GOOD, conf) == []
    bad = GOOD.copy()
    bad[9] = [900, 900]                                                  # cổ tay bay xa
    assert any("cẳng tay" in r for r in P.plausibility_pose(bad, conf))
    low = conf.copy()
    low[[5, 6]] = 0.1                                                    # mất cả hai vai
    assert P.plausibility_pose(GOOD, low) == ["thiếu vai hoặc hông tin cậy"]


def test_pick_driver_prefers_big_confident_and_drops_tiny():
    small = _person(GOOD, box=(0, 0, 20, 30))
    big = _person(GOOD, box=(0, 0, 200, 300), score=0.6)
    assert P.pick_driver([small, big]) is big
    assert P.pick_driver([small]) is None


class RotBackend:
    """Chỉ đoán tốt khi ảnh đã xoay 90° (đánh dấu ở (150,30) ảnh gốc -> (x'=30, y'=49) sau np.rot90 một lần)."""
    def __init__(self, q0=0.2):
        self.q0, self.calls = q0, 0

    def poses(self, img):
        self.calls += 1
        ys, xs = np.nonzero(img[:, :, 0])
        mx, my = int(xs[0]), int(ys[0])
        good = img.shape[0] == 200 and mx == 30 and my == 49
        xy = np.tile([mx, my], (17, 1)).astype(float)
        return [{"bbox": [0, 0, 80, 100], "score": 0.9, "xy": xy, "conf": np.full(17, 0.95 if good else self.q0)}]


def _marker_img():
    img = np.zeros((100, 200, 3), np.uint8)       # h=100, w=200
    img[30, 150] = 255
    return img


def test_detect_rotates_when_upright_pass_is_bad_and_unrotates_coordinates():
    det = P.detect(RotBackend(), _marker_img())
    assert det["rot"] == 90
    assert np.allclose(det["xy"], [[150, 30]] * 17)


def test_detect_keeps_rot0_when_quality_is_fine():
    be = RotBackend(q0=0.9)
    det = P.detect(be, _marker_img())
    assert det["rot"] == 0 and be.calls == 1                              # không thử xoay khi 0° đã tốt


def test_frame_people_maps_sublabel_names_and_clips():
    det = {"xy": GOOD.copy(), "conf": np.full(17, 0.9), "score": 0.8, "rot": 0, "quality": 0.8}
    det["xy"][16] = [500, 500]                                            # ngoài ảnh 300x400
    ppl = P.frame_people(det, [str(i) for i in range(1, 18)], 300, 400)
    pts = ppl[0]["points"]
    assert ppl[0]["label"] == "person_vf17" and sorted(pts, key=int) == [str(i) for i in range(1, 18)]
    assert pts["17"]["state"] == "outside" and pts["17"]["x"] == 300.0 and pts["17"]["y"] == 400.0
    assert P.frame_people(None, [], 10, 10) == []


def test_merge_consensus_averages_agreement_and_flags_disagreement():
    xy_y = np.array([[100.0, 100.0], [300.0, 300.0]])
    xy_v = np.array([[104.0, 100.0], [500.0, 300.0]])
    xy, conf, agree = P.merge_consensus(xy_y, [0.9, 0.8], xy_v, [0.7, 0.9], scale=100.0)
    assert agree.tolist() == [True, False]
    assert np.allclose(xy[0], [102.0, 100.0]) and abs(conf[0] - 0.8) < 1e-9
    assert np.allclose(xy[1], [500.0, 300.0])           # bất đồng: lấy điểm conf cao hơn (ViTPose 0.9)
    assert conf[1] <= P.DISAGREE_CONF < P.Q_CONF         # và bị hạ xuống mức Q
