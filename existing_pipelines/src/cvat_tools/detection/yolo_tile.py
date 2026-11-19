"""YOLO theo tile (PORT tu tools/cvat-seg-copilot/scripts/bbox_draft.py: detect() + bo loc add(); khong sua repo do).

Moi lan goi model deu qua INPUT CONTRACT (segmentation.inputspec): prepare_input(crop, model_id) -> canvas letterbox
dung spec, model chay tren canvas, hop duoc anh xa nguoc ve toa do anh goc bang restore_boxes. Model chua khai bao spec -> RAISE.

  dets = detect_tiled(make_predictor(YOLO(path)), img_bgr)       # [(label_job, [x1,y1,x2,y2], conf)]
  dets = [d for d in dets if keep_detection(d, W, H)]

`predict(canvas_bgr, conf, imgsz_hw) -> [(ten_coco, xyxy_tren_canvas, conf)]` la diem tiem de test bang model gia.
Khong bo hop chi vi trung hop tay: nguoi dung muon thay du dau ra cua may (co trung thi gan co o buoc bao cao).
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from cvat_tools.segmentation.inputspec import get_spec, prepare_input, restore_boxes

YOLO2JOB = {"person": "pedestrian", "car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
            "bicycle": "bicycle", "traffic light": "traffic light", "train": "train"}
TILE_W, TILE_H = 800, 480          # nhu bbox_draft.py
EDGE = 6                           # hop cham mep tile (khong phai mep anh) bi cat do -> bo
NMS_IOU = 0.5
NIGHT_GRAY = 70.0                  # nhu bbox_draft.py (do sang TB < 70 -> dung conf dem)

Det = Tuple[str, List[float], float]
Predict = Callable[[np.ndarray, float, Tuple[int, int]], Iterable[Tuple[str, Sequence[float], float]]]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def make_predictor(yolo, device: str = "cpu") -> Predict:
    """Boc model ultralytics. Canvas da la letterbox (boi 32) nen imgsz=(h,w) khong doi hinh."""
    def predict(canvas, conf, imgsz_hw):
        r = yolo.predict(canvas, conf=conf, imgsz=list(imgsz_hw), device=device, verbose=False)[0]
        return [(r.names[int(k)], b.tolist(), float(c)) for k, b, c in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)]
    return predict


def _run(predict: Predict, crop: np.ndarray, model_id: str, conf: float, policy: dict, tile_index=None):
    """Mot lan suy luan tren `crop` (BGR) qua inputspec -> [(ten, xyxy toa do crop, conf)]."""
    canvas, meta = prepare_input(crop, model_id, policy=policy)
    out = []
    for name, b, c in predict(canvas, conf, canvas.shape[:2]):
        if name in YOLO2JOB:
            out.append((name, restore_boxes(np.asarray(b, np.float32), meta)[0].tolist(), float(c)))
    return out


def detect_tiled(predict: Predict, img: np.ndarray, *, conf: float = 0.25, night_conf: float = 0.2,
                 model_id: str = "yolo26l", full_long_side: int = 1280, tile: Tuple[int, int] = (TILE_W, TILE_H),
                 tile_long_side: int = 800, tiles: bool = True) -> List[Det]:
    """YOLO toan anh + luoi tile chong nhau (vat nho/xa), gop bang NMS theo label. Tra [(label_job, xyxy, conf)]."""
    get_spec(model_id)  # khong co spec -> UnknownModelSpec, khong chay
    H, W = img.shape[:2]
    gray = float(np.asarray(img, np.float32).mean())
    cf = night_conf if gray < NIGHT_GRAY else conf
    raw: List[Det] = [(YOLO2JOB[n], b, c) for n, b, c in _run(predict, img, model_id, cf, {"long_side": full_long_side})]
    if tiles:
        tw, th = min(tile[0], W), min(tile[1], H)
        for oy in sorted({0, (H - th) // 2, H - th}):
            for ox in sorted({0, (W - tw) // 2, W - tw}):
                crop = np.ascontiguousarray(img[oy:oy + th, ox:ox + tw])
                cw, ch = crop.shape[1], crop.shape[0]
                for n, b, c in _run(predict, crop, model_id, cf, {"long_side": tile_long_side}):
                    x1, y1, x2, y2 = b
                    if (ox > 0 and x1 < EDGE) or (oy > 0 and y1 < EDGE) or (ox + cw < W and x2 > cw - EDGE) or (oy + ch < H and y2 > ch - EDGE):
                        continue  # hop bi tile cat do
                    raw.append((YOLO2JOB[n], [x1 + ox, y1 + oy, x2 + ox, y2 + oy], c))
    raw.sort(key=lambda d: -d[2])
    keep: List[Det] = []
    for d in raw:  # NMS cung label
        if any(k[0] == d[0] and iou(k[1], d[1]) > NMS_IOU for k in keep):
            continue
        keep.append(d)
    return keep


def derive_riders(dets: Sequence[Det]) -> List[Det]:
    """pedestrian co chan nam trong hop xe may/xe dap -> rider (luat hinh hoc cua bbox_draft)."""
    bikes = [d for d in dets if d[0] in ("motorcycle", "bicycle")]
    out = []
    for lab, b, c in dets:
        if lab == "pedestrian":
            cx, foot = (b[0] + b[2]) / 2, b[3]
            if any(bk[1][0] <= cx <= bk[1][2] and bk[1][1] <= foot <= bk[1][3] + 10 for bk in bikes):
                lab = "rider"
        out.append((lab, b, c))
    return out


def keep_detection(d: Det, W: int, H: int) -> bool:
    """Bo loc cua bbox_draft.add() (khong gom 'trung hop tay'): tam o day anh = phan chieu capo; hop be/khong lo."""
    label, box, conf = d
    bw, bh = box[2] - box[0], box[3] - box[1]
    if (box[1] + box[3]) / 2 > 0.88 * H and label != "traffic light":
        return False
    if bw < 10 or bh < 10 or (bw * bh > 0.2 * W * H and conf < 0.6):
        return False
    return True
