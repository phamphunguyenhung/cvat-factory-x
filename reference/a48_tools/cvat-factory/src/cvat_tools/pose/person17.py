"""Nháp pose 17 điểm (COCO, đúng thứ tự `person_vf17`) cho tài xế trong cabin.

Backend: `yolo` (ultralytics pose, nhanh) | `vitpose` (transformers ViTPose top-down, hộp người lấy từ YOLO) | `auto` (cả hai, bất đồng -> Q).
Quanh backend: xoay 4 hướng (ảnh fisheye người nằm ngang -> đưa về thẳng đứng rồi mới đoán, để trái/phải đúng theo NGƯỜI),
chọn tài xế (score x diện tích), cờ Q/O theo score + biên ảnh, kiểm tra hình học để loại frame vô lý (vẽ tay, không đoán bừa).
Mọi kết quả chỉ là NHÁP (`source=auto`).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from cvat_tools.core import paths
from cvat_tools.pose.face50 import unrotate

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SPM_SRC = paths.REPO / "tools" / "skeleton-pose-manager" / "src"
VITPOSE_ID = "usyd-community/vitpose-base-simple"
MIN_PERSON_PX = 60
Q_CONF = 0.5          # conf < 0.5 -> occluded (Q)
V0_CONF = 0.3         # tay/chân (LIMB_IDX) conf < 0.3 -> không đoán nổi -> v0 = outside (O), kể cả khi điểm nằm trong ảnh
LIMB_IDX = {7, 8, 9, 10, 13, 14, 15, 16}      # khuỷu, cổ tay, gối, cổ chân (0-based COCO). Mặt/vai/hông không v0 vì che
BORDER_PX = 8


def _yolo_model(name: str):
    from ultralytics import YOLO
    p = paths.REPO / "tools" / "skeleton-pose-manager" / "models" / name
    return YOLO(str(p if p.exists() else name))


class YoloPose:
    def __init__(self, model: str = "yolo11l-pose.pt"):
        self.name, self._m = model, None

    def _model(self):
        if self._m is None:
            self._m = _yolo_model(self.name)
        return self._m

    def poses(self, img_bgr: np.ndarray) -> list:
        r = self._model().predict(img_bgr, imgsz=960, conf=0.2, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []
        out = []
        for b, s, xy, kc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()):
            out.append({"bbox": [float(v) for v in b], "score": float(s), "xy": np.asarray(xy, float), "conf": np.asarray(kc, float)})
        return out


AGREE_FRAC = 0.35      # lệch YOLO-ViTPose < 0.35 x thân thì coi là đồng thuận
DISAGREE_CONF = 0.4    # điểm bất đồng bị hạ conf xuống mức Q


def merge_consensus(xy_y, c_y, xy_v, c_v, scale: float):
    """Hợp nhất hai model: đồng thuận -> trung bình tọa độ; bất đồng -> lấy điểm conf cao hơn nhưng hạ conf (-> Q, bạn kiểm tay)."""
    xy_y, xy_v = np.asarray(xy_y, float), np.asarray(xy_v, float)
    c_y, c_v = np.asarray(c_y, float), np.asarray(c_v, float)
    d = np.linalg.norm(xy_y - xy_v, axis=1)
    agree = d < AGREE_FRAC * scale
    xy = np.where(agree[:, None], (xy_y + xy_v) / 2, np.where((c_v >= c_y)[:, None], xy_v, xy_y))
    conf = np.where(agree, (c_y + c_v) / 2, np.minimum(np.minimum(c_y, c_v), DISAGREE_CONF))
    return xy, conf, agree


class VitPose:
    """ViTPose top-down: hộp người từ YOLO (cùng model pose), điểm + score từ ViTPose."""

    def __init__(self, hf_id: str = VITPOSE_ID, yolo: str = "yolo11l-pose.pt", consensus: bool = False):
        self.hf_id, self.yolo, self._m, self._p, self.consensus = hf_id, YoloPose(yolo), None, None, consensus

    def _load(self):
        if self._m is None:
            from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
            self._p = VitPoseImageProcessor.from_pretrained(self.hf_id)
            self._m = VitPoseForPoseEstimation.from_pretrained(self.hf_id).eval()
        return self._m, self._p

    def poses(self, img_bgr: np.ndarray) -> list:
        import torch
        from PIL import Image
        people = self.yolo.poses(img_bgr)
        if not people:
            return []
        m, proc = self._load()
        boxes = np.array([[b["bbox"][0], b["bbox"][1], b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1]] for b in people], float)
        im = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
        with torch.no_grad():
            out = m(**proc(im, boxes=[boxes], return_tensors="pt"))
        res = proc.post_process_pose_estimation(out, boxes=[boxes])[0]
        for p, r in zip(people, res):
            p["xy_yolo"], p["conf_yolo"] = p["xy"], p["conf"]
            p["xy"], p["conf"] = r["keypoints"].cpu().numpy().astype(float), r["scores"].cpu().numpy().astype(float)
            if self.consensus:
                b = p["bbox"]
                scale = max(0.5 * float(np.hypot(b[2] - b[0], b[3] - b[1])), 40.0)      # ~ độ dài thân
                p["xy"], p["conf"], _ = merge_consensus(p["xy_yolo"], p["conf_yolo"], p["xy"], p["conf"], scale)
        return people


def make_backend(name: str, yolo: str = "yolo11l-pose.pt"):
    if name == "yolo":
        return YoloPose(yolo)
    if name == "vitpose":
        return VitPose(yolo=yolo)
    if name == "auto":      # YOLO + ViTPose: đồng thuận -> trung bình; bất đồng -> điểm đó bị đánh Q
        return VitPose(yolo=yolo, consensus=True)
    raise ValueError(f"backend không biết: {name} (yolo|vitpose|auto)")


def pick_driver(people: list) -> dict | None:
    """Tài xế = người có score x diện tích hộp lớn nhất, bỏ hộp cao dưới MIN_PERSON_PX."""
    ok = [p for p in people if (p["bbox"][3] - p["bbox"][1]) >= MIN_PERSON_PX and p["score"] >= 0.25]
    return max(ok, key=lambda p: p["score"] * (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1])) if ok else None


def torso_quality(p: dict) -> float:
    """Điểm tin cậy thân trên (vai, khuỷu, hông) x score hộp: dùng chọn hướng xoay."""
    return float(np.mean(p["conf"][5:13])) * p["score"]


def detect(backend, img_bgr: np.ndarray, try_rotations: bool = False, min_quality: float = 0.6, rot_bias: float = 0.08) -> dict | None:
    """Thử 0°; thân trên kém (< min_quality) hoặc không thấy người thì thử cả 90/180/270 và chọn hướng tốt nhất.
    Hướng khác 0° phải hơn ít nhất rot_bias mới được chọn (tránh xoay oan). Trả {xy (17,2) ảnh gốc, conf, score, rot, bbox, quality} hoặc None."""
    best, best_key = None, -1.0
    shapes = [img_bgr.shape[:2]]
    for k in (0, 1, 2, 3):
        if k and not (try_rotations or best is None or best["quality"] < min_quality):
            break
        rot = np.ascontiguousarray(np.rot90(img_bgr, k))
        if k:
            shapes.append(rot.shape[:2])
        p = pick_driver(backend.poses(rot))
        if p is None:
            continue
        q = torso_quality(p)
        key = q - (rot_bias if k else 0.0)
        if key > best_key:
            xy = np.array([unrotate(x, y, shapes, k) if (x > 0.5 or y > 0.5) else (0.0, 0.0) for x, y in p["xy"]])
            best, best_key = {"xy": xy, "conf": p["conf"], "score": p["score"], "rot": k * 90, "bbox": p["bbox"], "quality": q}, key
    return best


def point_state(x: float, y: float, conf: float, w: int, h: int, limb: bool = False) -> str:
    """limb=True (khuỷu, cổ tay, gối, cổ chân): conf < V0_CONF -> v0. Đầu/mặt, vai, hông: không bao giờ v0 vì che, chỉ V hoặc Q;
    chỉ O khi thật sự ra ngoài/sát biên ảnh."""
    if (x <= 0.5 and y <= 0.5) or x < 0 or y < 0 or x > w or y > h:      # (0,0): model không trả điểm này
        return "outside"
    if limb and conf < V0_CONF:      # tay/chân che hẳn, không ước lượng được (báo trải đùi, ghế che gối...) -> v0
        return "outside"
    near_border = x < BORDER_PX or y < BORDER_PX or x > w - BORDER_PX or y > h - BORDER_PX
    if conf < V0_CONF and near_border:
        return "outside"
    return "occluded" if conf < Q_CONF else "visible"


def plausibility_pose(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> list:
    """Lý do nghi sai hình học (rỗng = hợp lý). Chỉ xét đoạn có cả hai đầu conf >= min_conf, chuẩn hóa theo thân (vai-hông)."""
    def pt(i):
        return xy[i] if conf[i] >= min_conf else None
    ls, rs, lh, rh = pt(5), pt(6), pt(11), pt(12)
    sh = [p for p in (ls, rs) if p is not None]
    hp = [p for p in (lh, rh) if p is not None]
    if not sh or not hp:
        return ["thiếu vai hoặc hông tin cậy"]
    torso = float(np.linalg.norm(np.mean(sh, 0) - np.mean(hp, 0)))
    if torso < 15:
        return ["thân quá ngắn (<15px)"]
    bad = []
    seg = lambda a, b: float(np.linalg.norm(xy[a] - xy[b])) if conf[a] >= min_conf and conf[b] >= min_conf else None   # noqa: E731
    for name, (a, b) in {"cánh tay trên T": (5, 7), "cánh tay trên P": (6, 8), "cẳng tay T": (7, 9), "cẳng tay P": (8, 10),
                         "đùi T": (11, 13), "đùi P": (12, 14), "cẳng chân T": (13, 15), "cẳng chân P": (14, 16)}.items():
        d = seg(a, b)
        if d is not None and d > 2.6 * torso:
            bad.append(f"{name} dài bất thường")
    if ls is not None and rs is not None and np.linalg.norm(ls - rs) > 2.6 * torso:
        bad.append("hai vai quá xa")
    nose = pt(0)
    if nose is not None and float(np.linalg.norm(nose - np.mean(sh, 0))) > 1.8 * torso:
        bad.append("đầu quá xa vai")
    return bad


LR_PAIRS = [(1, 2), (3, 4), (5, 6), (11, 12)]      # mắt, tai, vai, hông (0-based): chỉ cặp ở sát trục thân mới đủ tin để chọn bên
LR_MIN_SEP = 0.12                                  # tách trung bình theo x < 0.12 x thân -> không chắc (người nằm ngang...)


def enforce_image_lr(xy: np.ndarray, conf: np.ndarray) -> tuple:
    """Luật BTC: số chẵn (2,4,..16 = chỉ số 0-based lẻ) = R luôn nằm PHẢI ảnh, số lẻ = L nằm TRÁI ảnh, bất kể người quay lưng/xoay 90°.
    Model gán trái/phải theo giải phẫu nên khi người quay lưng hoặc ảnh xoay sẽ ngược: biểu quyết theo cặp mắt/tai/vai/hông (trọng số conf),
    ngược thì đổi chỗ CẢ 8 cặp (giữ chuỗi vai-khuỷu-cổ tay liên tục). Trả (xy, conf, "ok"|"swapped"|"uncertain")."""
    xy, conf = np.array(xy, float), np.array(conf, float)
    score = sep = wsum = 0.0
    for r, l in LR_PAIRS:
        w = min(conf[r], conf[l])
        score += w * (xy[r][0] - xy[l][0])
        sep += w * abs(xy[r][0] - xy[l][0])
        wsum += w
    sh = [xy[i] for i in (5, 6) if conf[i] >= 0.3]
    hp = [xy[i] for i in (11, 12) if conf[i] >= 0.3]
    torso = float(np.linalg.norm(np.mean(sh, 0) - np.mean(hp, 0))) if sh and hp else 0.0
    if wsum <= 0 or torso < 15 or sep / wsum < LR_MIN_SEP * torso:
        return xy, conf, "uncertain"
    if score >= 0:
        return xy, conf, "ok"
    for a in range(1, 17, 2):          # 1/2, 3/4, ... 15/16
        xy[[a, a + 1]], conf[[a, a + 1]] = xy[[a + 1, a]], conf[[a + 1, a]]
    return xy, conf, "swapped"


def frame_people(det: dict | None, sub_names: list, w: int, h: int) -> list:
    """1 skeleton `person_vf17`; điểm theo tên sublabel của CVAT ('1'..'17')."""
    if det is None:
        return []
    det = {**det}
    det["xy"], det["conf"], lr = enforce_image_lr(det["xy"], det["conf"])
    pts = {}
    for i, name in enumerate(sub_names):
        x, y = float(det["xy"][i][0]), float(det["xy"][i][1])
        pts[name] = {"x": float(min(max(x, 0.0), w)), "y": float(min(max(y, 0.0), h)), "state": point_state(x, y, float(det["conf"][i]), w, h, limb=i in LIMB_IDX),
                     "conf": float(det["conf"][i])}
    return [{"label": "person_vf17", "points": pts, "score": det["score"], "flags": {"swapped": ["LR_SWAPPED_THEO_ANH"], "uncertain": ["UNCERTAIN_CLASS_LR"]}.get(lr, []),
             "anchor": f"rot{det['rot']},box{det['score']:.2f},q{det['quality']:.2f}"}]


def agreement(a: dict, b: dict) -> float:
    """Lệch trung bình (px) giữa hai nháp cùng frame (theo tên điểm), dùng so sánh backend khi chưa có gold."""
    d = [math.hypot(a[n]["x"] - b[n]["x"], a[n]["y"] - b[n]["y"]) for n in a if n in b]
    return float(np.mean(d)) if d else float("nan")


if __name__ == "__main__":
    sys.exit("module thư viện: dùng `python -m cvat_tools pose draft --job <id> --backend yolo|vitpose`")
