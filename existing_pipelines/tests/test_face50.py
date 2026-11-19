import numpy as np
import pytest

from cvat_tools.pose import face50


def _face68(scale=100.0):
    """Mặt thẳng giả lập iBUG-68: đủ để kiểm chỉ số và hình học, không cần model."""
    p = np.zeros((68, 2))
    cx = 150.0
    for i, x in enumerate(np.linspace(-1, 1, 5)):                      # mày trái-ảnh 17-21, phải-ảnh 22-26
        p[17 + i] = (cx - 0.9 * scale + 0.35 * scale * (x + 1) / 2 * 2, 100 - 8 * (1 - x * x))
        p[22 + i] = (cx + 0.2 * scale + 0.35 * scale * (x + 1) / 2 * 2, 100 - 8 * (1 - x * x))
    for i in range(4):
        p[27 + i] = (cx, 110 + 12 * i)
    def eye(c, first, outer_left):
        xs = [-1, -0.5, 0.5, 1, 0.5, -0.5]
        ys = [0, -0.4, -0.4, 0, 0.4, 0.4]
        for k, i in enumerate(range(first, first + 6)):
            sgn = 1 if outer_left else 1
            p[i] = (c + 18 * xs[k] * sgn, 125 + 8 * ys[k])
    eye(cx - 0.55 * scale, 36, True)
    eye(cx + 0.55 * scale, 42, True)
    for k in range(12):
        a = np.pi + k * 2 * np.pi / 12
        p[48 + k] = (cx + 28 * np.cos(a), 190 - 12 * np.sin(a))
    for k in range(8):
        a = np.pi + k * 2 * np.pi / 8
        p[60 + k] = (cx + 20 * np.cos(a), 190 - 5 * np.sin(a))
    return p


def test_map_shape_and_groups():
    p = _face68()
    q = face50.map68_to_50(p)
    assert q.shape == (50, 2)
    assert np.allclose(q[0:10], p[17:27])
    assert np.allclose(q[10], p[27]) and np.allclose(q[13], p[27] + face50.NOSE_BASE_T * (p[30] - p[27]))   # 13 = chân sống mũi, không phải chóp
    assert np.allclose(np.diff(q[10:14], axis=0), np.diff(q[10:14], axis=0)[0])                           # chia đều ba đoạn
    assert np.allclose(q[30:42], p[48:60]) and np.allclose(q[42:50], p[60:68])
    assert np.allclose(q[14], p[36]) and np.allclose(q[18], p[39])   # khóe mắt giữ nguyên
    assert np.allclose(q[22], p[42]) and np.allclose(q[26], p[45])


def test_lid_points_between_corners_and_collapsed_lid_ok():
    p = _face68()
    q = face50.map68_to_50(p)
    assert q[14, 0] < q[15, 0] < q[16, 0] < q[17, 0] < q[18, 0]       # mí trên đi từ khóe ngoài vào trong
    p[37] = p[38] = p[36]                                              # mí khép / điểm trùng không được làm hỏng spline
    face50.map68_to_50(p)


def test_lid_points_at_quarter_half_threequarter_of_eye_width():
    q = face50.map68_to_50(_face68())
    for a, b, first in ((14, 18, 15), (22, 26, 23)):
        ax = q[b] - q[a]
        frac = [float((q[first + k] - q[a]) @ ax / (ax @ ax)) for k in range(3)]
        assert np.allclose(frac, [0.25, 0.5, 0.75], atol=0.01)


def test_rotated_face_is_flagged_for_mentor():
    det = {"p68": _face68(), "score68": np.full(68, 0.9), "face_score": 1.0, "rot": 90}
    assert all(p["flags"] == ["UNCERTAIN_CLASS_ROT90"] for p in face50.frame_people(det, 400, 300))
    det["rot"] = 0
    assert all(p["flags"] == [] for p in face50.frame_people(det, 400, 300))


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        face50.map68_to_50(np.zeros((51, 2)))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unrotate_roundtrip(k):
    h, w = 40, 70
    img = np.zeros((h, w), np.uint8)
    img[12, 55] = 255                                                  # (x=55, y=12)
    shapes = [img.shape]
    r = img
    for _ in range(k):
        r = np.rot90(r)
        shapes.append(r.shape)
    ys, xs = np.nonzero(r)
    assert face50.unrotate(float(xs[0]), float(ys[0]), shapes, k) == (55.0, 12.0)


def test_state_for_outside_and_low_score():
    assert face50.state_for(-1, 5, 100, 100) == "outside"
    assert face50.state_for(5, 101, 100, 100) == "outside"
    assert face50.state_for(5, 5, 100, 100, score=0.1) == "occluded"
    assert face50.state_for(5, 5, 100, 100, score=0.9) == "visible"


def test_frame_people_has_seven_labels_fifty_points():
    det = {"p68": _face68(), "score68": np.full(68, 0.9), "face_score": 1.0, "rot": 0}
    ppl = face50.frame_people(det, 400, 300)
    assert [p["label"] for p in ppl] == list(face50.LABEL_ORDER)
    assert sorted(int(k) for p in ppl for k in p["points"]) == list(range(50))
    assert face50.frame_people(None, 400, 300) == []


def test_plausibility_accepts_upright_rejects_upside_down():
    q = face50.map68_to_50(_face68())
    assert face50.plausibility(q) == []
    flipped = q.copy()
    flipped[:, 1] = 2 * q[14:22, 1].mean() - q[:, 1]                    # lật dọc quanh đường mắt: miệng lên trên mắt
    assert face50.plausibility(flipped)
