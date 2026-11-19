"""Giao dien thong nhat cho "bang chung" phan vung nen (stuff) + son vach tu nhieu model, CPU-only.

    from cvat_tools.segmentation.evidence import get_evidence
    ev = get_evidence(img_bgr, models=["b2", "b5", "m2f", "lane"])
    ev["b5"]["road"]  # uint8 HxW, 0/255

Model: b2 (SegFormer-B2 ONNX, dung lai adapter cvat_seg.scene), b5 (nvidia SegFormer-B5 Cityscapes),
m2f (Mask2Former Swin-L Cityscapes), lane (SegFormer-Lane-Marking, nhan "lane_paint").
KHONG sua tools/cvat-seg-copilot, chi import qua sys.path.

LOI TIEN XU LY da tim ra o lan do truoc (scripts/scene_model_eval.py dung AutoImageProcessor mac dinh):
  * M2F: preprocessor_config.json dat size = 384x384 -> anh 1280x720 bi ep vuong 384x384 (mat chi tiet, sai ti le)
    roi moi upsample len. Day la nguyen nhan chinh khien M2F thap. Sua: resize giu ti le, canh dai = long_side
    (mac dinh 1024), boi so cua 32, chuan hoa ImageNet, semantic = softmax(class)[:, :-1] x sigmoid(mask) roi
    UPSAMPLE ve HxW truoc argmax.
  * B5: preprocessor size = 1024 -> ep vuong 1024x1024 (1280x720 bi keo doc 1.8x), khac phan phoi luc train
    (crop giu ti le). Sua: resize giu ti le canh dai 1024 (boi 32), ImageNet mean/std, upsample logits (bilinear,
    align_corners=False) ve HxW roi moi argmax.
  * Anh xa lop: KHONG hardcode, doc id2label tu config roi doi chieu voi ten Cityscapes (labels_from_id2label);
    ca 3 model deu dung id2label Cityscapes chuan (0 road ... 10 sky), nen anh xa lop khong phai nguyen nhan.
  * Canh bao M2F "thieu swin.layernorm": VO HAI. SwinBackbone (transformers.models.swin) dat
    _keys_to_ignore_on_load_missing = ["swin.layernorm.*"]; feature di qua hidden_states_norms (co trong checkpoint),
    layernorm cuoi khong nam tren duong feature.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
_COPILOT_SRC = REPO / "tools" / "cvat-seg-copilot" / "src"

STUFF = ("road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky")
LANE_LABEL = "lane_paint"
ALL_LABELS = STUFF + (LANE_LABEL,)
CITYSCAPES_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
)
HF_REPOS = {
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    "m2f": "facebook/mask2former-swin-large-cityscapes-semantic",
}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def labels_from_id2label(id2label: Mapping, wanted: Iterable[str] = STUFF) -> Dict[str, int]:
    """{label: class id} tu config.id2label (khoa co the la str hoac int). Bao loi neu thieu ten / lech chuan Cityscapes."""
    by_name = {str(v).lower(): int(k) for k, v in id2label.items()}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise KeyError(f"id2label thieu {missing}")
    for i, n in enumerate(CITYSCAPES_NAMES):
        if by_name.get(n) != i:
            raise ValueError(f"id2label lech chuan Cityscapes: {n!r} -> {by_name.get(n)} (mong doi {i})")
    return {w: by_name[w] for w in wanted}


def preprocess(img_bgr: np.ndarray, long_side: int = 1024, multiple: int = 32) -> np.ndarray:
    """BGR HxWx3 -> float32 (1,3,h,w): resize GIU TI LE canh dai = long_side (boi so cua `multiple`), RGB, ImageNet mean/std."""
    H, W = img_bgr.shape[:2]
    s = long_side / max(H, W)
    w = max(multiple, int(round(W * s / multiple)) * multiple)
    h = max(multiple, int(round(H * s / multiple)) * multiple)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    x = cv2.cvtColor(cv2.resize(img_bgr, (w, h), interpolation=interp), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None])


def class_map_to_masks(cm: np.ndarray, ids: Mapping[str, int]) -> Dict[str, np.ndarray]:
    return {n: (cm == i).astype(np.uint8) * 255 for n, i in ids.items()}


def _upsample_argmax(scores: "np.ndarray", H: int, W: int) -> np.ndarray:
    """(C,h,w) -> uint8 (H,W) argmax sau khi upsample bilinear tung kenh ve do phan giai goc."""
    up = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in scores])
    return up.argmax(0).astype(np.uint8)


class _B2:
    def __init__(self, long_side: int = 1024):
        if str(_COPILOT_SRC) not in sys.path:
            sys.path.insert(0, str(_COPILOT_SRC))
        from cvat_seg.scene import CITYSCAPES, SceneParser  # adapter co san

        self.parser = SceneParser(long_side=long_side)
        self.ids = labels_from_id2label(dict(enumerate(CITYSCAPES_NAMES)))
        assert all(CITYSCAPES[k] == self.ids[k] for k in CITYSCAPES if k in self.ids)

    def class_map(self, img):
        return self.parser.class_map(img)


class _B5:
    def __init__(self, long_side: int = 1024):
        import torch
        from transformers import SegformerForSemanticSegmentation

        self.torch, self.long_side = torch, long_side
        self.model = SegformerForSemanticSegmentation.from_pretrained(HF_REPOS["b5"]).eval()
        self.ids = labels_from_id2label(self.model.config.id2label)

    def class_map(self, img):
        H, W = img.shape[:2]
        x = self.torch.from_numpy(preprocess(img, self.long_side))
        with self.torch.inference_mode():
            lg = self.model(pixel_values=x).logits  # (1,19,h/4,w/4)
            lg = self.torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
        return lg.argmax(1)[0].numpy().astype(np.uint8)


class _M2F:
    def __init__(self, long_side: int = 1024):
        import torch
        from transformers import Mask2FormerForUniversalSegmentation

        self.torch, self.long_side = torch, long_side
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(HF_REPOS["m2f"]).eval()
        self.ids = labels_from_id2label(self.model.config.id2label)

    def class_map(self, img):
        torch = self.torch
        H, W = img.shape[:2]
        x = torch.from_numpy(preprocess(img, self.long_side))
        with torch.inference_mode():
            o = self.model(pixel_values=x)
            cls = o.class_queries_logits.softmax(-1)[..., :-1]  # (1,Q,19) bo lop "no object"
            msk = o.masks_queries_logits.sigmoid()  # (1,Q,h/4,w/4)
            sem = torch.einsum("bqc,bqhw->bchw", cls, msk)
            sem = torch.nn.functional.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)
        return sem.argmax(1)[0].numpy().astype(np.uint8)


class _Lane:
    def __init__(self, short_side: int = 224, threshold: float = 0.5):
        if str(_COPILOT_SRC) not in sys.path:
            sys.path.insert(0, str(_COPILOT_SRC))
        from cvat_seg.scene import PaintNet

        self.net = PaintNet(short_side=short_side, threshold=threshold)

    def masks(self, img):
        return {LANE_LABEL: self.net.mask(img)}


_BACKENDS: Dict[str, Callable] = {"b2": _B2, "b5": _B5, "m2f": _M2F, "lane": _Lane}
_LOADED: Dict[str, object] = {}


def parse_spec(spec: str) -> tuple:
    """'b5' -> ('b5', {}); 'b5@1536' -> ('b5', {'long_side': 1536}); 'lane@448' -> ('lane', {'short_side': 448})."""
    name, _, size = spec.partition("@")
    if name not in _BACKENDS:
        raise KeyError(f"model {name!r} khong co; co: {sorted(_BACKENDS)}")
    if not size:
        return name, {}
    return name, {("short_side" if name == "lane" else "long_side"): int(size)}


def load_model(spec: str, **kw):
    name, kw0 = parse_spec(spec)
    if spec not in _LOADED:
        _LOADED[spec] = _BACKENDS[name](**{**kw0, **kw})
    return _LOADED[spec]


def free_model(spec: Optional[str] = None) -> None:
    for k in [spec] if spec else list(_LOADED):
        _LOADED.pop(k, None)
    import gc

    gc.collect()


def model_evidence(name: str, img_bgr: np.ndarray, labels: Iterable[str] = ALL_LABELS) -> Dict[str, np.ndarray]:
    m = load_model(name)
    labels = list(labels)
    if hasattr(m, "masks"):
        out = m.masks(img_bgr)
    else:
        out = class_map_to_masks(m.class_map(img_bgr), {k: v for k, v in m.ids.items() if k in labels})
    return {k: v for k, v in out.items() if k in labels}


def get_evidence(img_bgr: np.ndarray, models: Iterable[str] = ("b2",), labels: Iterable[str] = ALL_LABELS,
                 timings: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """{model: {label: uint8 mask 0/255 (HxW)}}. `timings` (neu co) nhan s/frame tung model (ke ca lan nap dau)."""
    res: Dict[str, Dict[str, np.ndarray]] = {}
    for n in models:
        t0 = time.time()
        res[n] = model_evidence(n, img_bgr, labels)
        if timings is not None:
            timings[n] = time.time() - t0
    return res


def ensemble(evidence: Mapping[str, Mapping[str, np.ndarray]], best: Mapping[str, str]) -> Dict[str, np.ndarray]:
    """Chon mask cua model tot nhat cho tung label: best = {label: model}."""
    return {lab: evidence[m][lab] for lab, m in best.items() if m in evidence and lab in evidence[m]}
