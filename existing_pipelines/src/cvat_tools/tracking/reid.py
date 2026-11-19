"""ReID CHON LOC: chi tinh embedding o su kien ID nghi van; embedding cache theo (frame, detection id, model hash).

ReID chi la BANG CHUNG (do giong cosine) dua cho nguoi duyet; KHONG tu gan lai ID sau khi xuat hien lai khong chac chan.
Anh xa P1-P5 do nguoi duyet luu rieng (data/review/*.json), khong nam trong tracker id.

Embedder:
- OSNetEmbedder (osnet_x0_25): can file trong so (onnx/pt) + bo chay; thieu -> BLOCKED (khong cai lan man torchreid/boxmot).
- HsvEmbedder: histogram mau HSV nua than tren, ban BASELINE yeu (khong phai OSNet), dung kiem thu duong ong va lam bang chung phu.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from cvat_tools.core.cache import Cache, file_hash


class Blocked(RuntimeError):
    pass


class HsvEmbedder:
    name = "hsv_hist_baseline"
    model_hash = "hsv-v1"

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        h, w = crop_bgr.shape[:2]
        torso = crop_bgr[int(0.15 * h):int(0.6 * h), int(0.1 * w):int(0.9 * w)]
        if torso.size == 0:
            torso = crop_bgr
        hsv = cv2.cvtColor(cv2.resize(torso, (32, 64)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        n = np.linalg.norm(hist)
        return (hist / n).astype(np.float32) if n else hist.astype(np.float32)


class OSNetEmbedder:
    name = "osnet_x0_25"

    def __init__(self, weights: str | Path | None = None):
        p = Path(weights) if weights else None
        if p is None or not p.exists():
            raise Blocked("BLOCKED: thieu trong so OSNet x0.25 (dat models/checkpoints/osnet_x0_25.onnx). "
                          "Khong tu cai torchreid/boxmot; tai trong so co kiem tra hash roi chay lai.")
        import onnxruntime as ort
        self.model_hash = file_hash(p)
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        x = cv2.resize(crop_bgr, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        v = self.sess.run(None, {self.inp: x})[0].ravel()
        return v / (np.linalg.norm(v) + 1e-9)


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embed_box(emb, source, frame: int, box, det_id: str, cache: Cache | None = None) -> np.ndarray:
    """Embedding cache theo (frame, det_id, model hash). det_id on dinh, vd 'trackname#tid'."""
    key = hashlib.sha256(f"{frame}|{det_id}|{emb.model_hash}|{[round(v) for v in box]}".encode()).hexdigest()[:20]
    if cache:
        got = cache.get(key)
        if got:
            return np.array(got["value"], dtype=np.float32)
    img = source.image(frame)
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    crop = img[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    v = emb.embed(crop)
    if cache:
        cache.put(key, v.tolist())
    return v


def evidence_for_events(tracks: dict, events: list[dict], emb, source, cache: Cache | None = None) -> list[dict]:
    """events: [{kind:'break'|'overlap'|'handoff', frame_a, track_a, frame_b, track_b}] -> them 'reid_cosine' + 'verdict'.
    verdict chi la goi y 'giong'/'khac'/'khong ro'; KHONG sua track. Chi chay tren cac su kien (khong moi frame)."""
    out = []
    for e in events:
        ta, tb = tracks.get(e["track_a"], {}), tracks.get(e["track_b"], {})
        if e["frame_a"] not in ta or e["frame_b"] not in tb:
            out.append({**e, "reid_cosine": None, "verdict": "khong ro"})
            continue
        va = embed_box(emb, source, e["frame_a"], ta[e["frame_a"]], f"{e['track_a']}", cache)
        vb = embed_box(emb, source, e["frame_b"], tb[e["frame_b"]], f"{e['track_b']}", cache)
        c = cosine(va, vb)
        out.append({**e, "reid_cosine": round(c, 3), "embedder": emb.name,
                    "verdict": "giong" if c > 0.9 else "khac" if c < 0.6 else "khong ro", "auto_relabel": False})
    return out
