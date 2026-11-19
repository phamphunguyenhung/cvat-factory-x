"""Landmark mặt thật (FAN 2D, iBUG-68) -> bộ 50 điểm của Week 2.

Bộ 50 điểm = iBUG-68 bỏ cằm (0–16) và đáy mũi (31–35), mắt nâng từ 6 lên 8 điểm:
  mày trái-ảnh 0–4 = iBUG 17–21 | mày phải-ảnh 5–9 = 22–26 (iBUG nằm giữa sợi lông, BTC đòi bờ trên: duyệt tay)
  sống mũi 10–13 = đoạn iBUG 27 -> 85% tới iBUG 30 (BTC: 13 là chân sống mũi, không phải chóp), chia ba đoạn
  mắt trái-ảnh 14–21: 14=36 (khóe ngoài), 15–17 mí trên, 18=39 (khóe trong), 19–21 mí dưới (trong -> ngoài)
  mắt phải-ảnh 22–29: 22=42 (khóe trong), 23–25 mí trên, 26=45 (khóe ngoài), 27–29 mí dưới (ngoài -> trong)
  môi ngoài 30–41 = iBUG 48–59 | môi trong 42–49 = iBUG 60–67
Mí trên/dưới của iBUG có 2 điểm giữa hai khóe; scheme cần 3 nên nội suy spline tại 1/4, 1/2, 3/4 chiều rộng mắt (BTC).
`trai` = bên TRÁI ẢNH khi mặt thẳng đứng; ảnh xoay được thử 4 hướng rồi quay ngược toạ độ.
"""
from __future__ import annotations

import numpy as np

LABEL_ORDER = ("longmaytrai", "longmayphai", "songmui", "mattrai", "matphai", "moingoai", "moitrong")
LABEL_RANGE = {"longmaytrai": (0, 5), "longmayphai": (5, 10), "songmui": (10, 14), "mattrai": (14, 22),
               "matphai": (22, 30), "moingoai": (30, 42), "moitrong": (42, 50)}
# (corner_a, upper1, upper2, corner_b, lower1, lower2) theo thứ tự đi vòng của iBUG
EYE_L = (36, 37, 38, 39, 40, 41)   # mắt trái-ảnh: outer, upper, upper, inner, lower, lower
EYE_R = (42, 43, 44, 45, 46, 47)   # mắt phải-ảnh: inner, upper, upper, outer, lower, lower


NOSE_BASE_T = 0.85      # điểm 13 = chân sống mũi (BTC): trên lỗ mũi, chưa chạm chóp mũi (iBUG 30). 0.85 là ước lượng, chờ mẫu mentor


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    """n điểm bên trong (không gồm hai đầu) trên spline qua poly, tại k/(n+1) CHIỀU RỘNG mắt (chiếu lên trục khóe a->b) như BTC
    (1/4, 1/2, 3/4). Khóe trùng nhau (không có trục) thì rơi về cách đều theo dây cung."""
    from scipy.interpolate import CubicSpline
    d = np.r_[0.0, np.cumsum(np.maximum(np.linalg.norm(np.diff(poly, axis=0), axis=1), 1e-3))]   # mí khép: điểm trùng vẫn tăng nghiêm ngặt
    cs = CubicSpline(d, poly, bc_type="natural")
    axis = poly[-1] - poly[0]
    L2 = float(axis @ axis)
    if L2 < 1e-6:
        return cs(np.linspace(0, d[-1], n + 2)[1:-1])
    t = np.linspace(0, d[-1], 400)
    proj = np.maximum.accumulate(((cs(t) - poly[0]) @ axis) / L2) + np.arange(400) * 1e-9
    return cs(np.interp(np.arange(1, n + 1) / (n + 1), proj, t))


def map68_to_50(p: np.ndarray) -> np.ndarray:
    """p: (68,2) iBUG -> (50,2) theo thứ tự scheme 0..49."""
    p = np.asarray(p, float)
    if p.shape != (68, 2):
        raise ValueError(f"cần (68,2), nhận {p.shape}")
    out = np.zeros((50, 2))
    out[0:10] = p[17:27]
    out[10:14] = p[27] + np.outer(np.array([0.0, 1 / 3, 2 / 3, 1.0]) * NOSE_BASE_T, p[30] - p[27])      # đỉnh sống mũi -> chân sống mũi, chia ba đoạn
    a, u1, u2, b, l1, l2 = (p[i] for i in EYE_L)           # outer -> inner trên; inner -> outer dưới
    out[14] = a
    out[15:18] = _resample(np.array([a, u1, u2, b]), 3)
    out[18] = b
    out[19:22] = _resample(np.array([b, l1, l2, a]), 3)
    a, u1, u2, b, l1, l2 = (p[i] for i in EYE_R)           # inner -> outer trên; outer -> inner dưới
    out[22] = a
    out[23:26] = _resample(np.array([a, u1, u2, b]), 3)
    out[26] = b
    out[27:30] = _resample(np.array([b, l1, l2, a]), 3)
    out[30:42] = p[48:60]
    out[42:50] = p[60:68]
    return out


def unrotate(x: float, y: float, shapes: list, k: int):
    """Điểm trên ảnh đã np.rot90(k lần, ngược kim đồng hồ) -> toạ độ ảnh gốc. shapes[i] = (h, w) trước bước i."""
    for step in range(k, 0, -1):
        w_prev = shapes[step - 1][1]
        x, y = (w_prev - 1 - y), x
    return x, y


class FaceLandmarker:
    """Bọc face_alignment.FaceAlignment (nạp lười, CPU, không torch.compile)."""

    def __init__(self, min_face_score: float = 0.8):
        self._fa = None
        self.min_face_score = min_face_score

    def _load(self):
        if self._fa is None:
            import face_alignment
            self._fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False,
                                                    compile=False, face_detector="sfd")
        return self._fa

    def _run(self, img_rgb: np.ndarray):
        lms, scores, boxes = self._load().get_landmarks_from_image(img_rgb, return_bboxes=True, return_landmark_score=True)
        if lms is None or len(lms) == 0:
            return None
        best = max(range(len(lms)), key=lambda i: boxes[i][4] * (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        return np.asarray(lms[best]), np.asarray(scores[best]), float(boxes[best][4]), boxes[best][:4]

    def landmarks68(self, img_bgr: np.ndarray, try_rotations: bool = False):
        """Trả dict {p68 (68,2) trên ảnh gốc, score68, face_score, rot} hoặc None. Thử 0°; kém/không thấy thì thử 90/180/270."""
        best, best_key = None, -1.0
        shapes = [img_bgr.shape[:2]]
        for k in (0, 1, 2, 3):
            if k and not (try_rotations or best is None or best["face_score"] < self.min_face_score):
                break
            rot = np.ascontiguousarray(np.rot90(img_bgr, k))
            if k:
                shapes.append(rot.shape[:2])
            res = self._run(np.ascontiguousarray(rot[:, :, ::-1]))
            if res is None:
                continue
            lm, sc, fs, _ = res
            key = fs * float(np.mean(sc))
            if key > best_key:
                pts = np.array([unrotate(x, y, shapes, k) for x, y in lm])
                best, best_key = {"p68": pts, "score68": sc, "face_score": fs, "rot": k * 90}, key
        return best


def state_for(x: float, y: float, w: int, h: int, score: float | None = None, low: float = 0.3) -> str:
    if x < 0 or y < 0 or x > w or y > h:
        return "outside"
    return "occluded" if (score is not None and score < low) else "visible"


def frame_people(det: dict | None, w: int, h: int, low: float = 0.3) -> list:
    """7 skeleton (đúng LABEL_ORDER) từ kết quả landmarks68; det None -> []."""
    if det is None:
        return []
    p50 = map68_to_50(det["p68"])
    sc68 = np.asarray(det["score68"])
    # điểm nội suy lấy score nhỏ nhất của hai iBUG lân cận (bảo thủ)
    sc50 = np.r_[sc68[17:27], sc68[27:31],
                 [sc68[36], *[min(sc68[36], sc68[37], sc68[38], sc68[39])] * 3, sc68[39],
                  *[min(sc68[39], sc68[40], sc68[41], sc68[36])] * 3],
                 [sc68[42], *[min(sc68[42], sc68[43], sc68[44], sc68[45])] * 3, sc68[45],
                  *[min(sc68[45], sc68[46], sc68[47], sc68[42])] * 3],
                 sc68[48:60], sc68[60:68]]
    people = []
    for lab in LABEL_ORDER:
        a, b = LABEL_RANGE[lab]
        pts = {}
        for i in range(a, b):
            x, y = float(p50[i, 0]), float(p50[i, 1])
            pts[str(i)] = {"x": float(min(max(x, 0.0), w)), "y": float(min(max(y, 0.0), h)),
                           "state": state_for(x, y, w, h, float(sc50[i]), low)}
        people.append({"label": lab, "points": pts, "anchor": f"fan2d(rot{det['rot']},face{det['face_score']:.2f})",
                       "flags": [f"UNCERTAIN_CLASS_ROT{det['rot']}"] if det["rot"] else []})      # mặt xoay: trai/phai theo ảnh chưa chốt (guideline mục 8)
    return people


def plausibility(p50: np.ndarray) -> list:
    """Danh sách lý do nghi sai hình học (rỗng = hợp lý). Chuẩn hóa theo khoảng cách hai mắt và trục mắt, nên chịu được ảnh xoay.
    Ngưỡng lấy từ nháp 29 frame của job 1975 (frame tốt: mouth_down 0.6–2.4, brow_up 0.15–0.8, nose3 0.2–1.2, eye 0.3–0.7)."""
    P = np.asarray(p50, float)
    e1, e2 = P[14:22].mean(0), P[22:30].mean(0)
    v = e2 - e1
    iod = float(np.linalg.norm(v))
    if iod < 3.0:
        return ["iod<3px"]
    down = np.array([-v[1], v[0]]) / iod
    em = (e1 + e2) / 2
    pr = lambda q: float((q - em) @ down / iod)       # noqa: E731
    size = lambda a: float(np.hypot(np.ptp(a[:, 0]), np.ptp(a[:, 1])) / iod)   # noqa: E731
    bad = []
    if not 0.4 <= pr(P[30:42].mean(0)) <= 2.8:
        bad.append("miệng không nằm dưới mắt")
    if not 0.0 <= -pr(P[0:10].mean(0)) <= 1.0:
        bad.append("mày không nằm trên mắt")
    if not 0.0 <= pr(P[13]) <= 1.4:
        bad.append("sống mũi sai hướng")
    if size(P[14:22]) > 1.0 or size(P[22:30]) > 1.0:
        bad.append("mắt quá to so với khoảng cách hai mắt")
    if not 0.3 <= float(np.linalg.norm(P[30] - P[36])) / iod <= 1.6:
        bad.append("độ rộng miệng bất thường")
    return bad
