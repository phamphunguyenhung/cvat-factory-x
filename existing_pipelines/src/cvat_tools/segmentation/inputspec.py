"""INPUT CONTRACT: moi model khai bao do phan giai/ti le huan luyen (config/model_inputs.yaml); frame duoc resize/chuan hoa
dung roi anh xa NGUOC ve toa do anh goc.

    x, meta = prepare_input(frame_bgr, "m2f", policy={"long_side": 768, "enhance": "clahe"})
    out = model(x)                      # (C,h,w) hoac (N,C,h,w) neu tile
    full = restore(out, meta)           # (C,H,W) hoac (H,W) o do phan giai goc
    boxes = restore_boxes(xyxy, meta)   # det: toa do hop ve anh goc

Model chua khai bao -> UnknownModelSpec (khong chay). Tang sang chi ap dung cho frame toi (gray mean < dark_thresh),
chi doi gia tri diem anh, khong doi hinh hoc -> khong anh huong anh xa nguoc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parents[3] / "config" / "model_inputs.yaml"
ENHANCE = ("none", "gamma", "clahe", "gamma_clahe")
_REQUIRED = ("kind", "train", "norm", "multiple", "resize", "policy")


class UnknownModelSpec(KeyError):
    """Model khong co (hoac thieu) input spec -> tu choi chay."""


def load_specs(path: Optional[Path] = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG).read_text(encoding="utf8"))


def get_spec(model_id: str, specs: Optional[dict] = None) -> dict:
    specs = specs or load_specs()
    m = (specs.get("models") or {}).get(model_id)
    if not m:
        raise UnknownModelSpec(f"model {model_id!r} chua khai bao input spec trong model_inputs.yaml; co: {sorted(specs.get('models', {}))}")
    miss = [k for k in _REQUIRED if k not in m]
    if miss:
        raise UnknownModelSpec(f"spec {model_id!r} thieu {miss}")
    if m["kind"] != "det" and m["kind"] != "seg_instance" and (not m["norm"] or m["norm"].get("mean") is None):
        raise UnknownModelSpec(f"spec {model_id!r} thieu mean/std")
    return m


def _round_mult(v: float, mult: int) -> int:
    return max(mult, int(round(v / mult)) * mult)


def mean_gray(img_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())


def enhance_dark(img_bgr: np.ndarray, mode: str, dark_thresh: float, params: dict) -> Tuple[np.ndarray, dict]:
    """Tang sang CHI khi mean gray < dark_thresh. -> (anh, thong tin da ap dung)."""
    if mode not in ENHANCE:
        raise ValueError(f"enhance {mode!r} khong hop le: {ENHANCE}")
    mg = mean_gray(img_bgr)
    info = {"mode": mode, "mean_gray": round(mg, 1), "applied": False}
    if mode == "none" or mg >= dark_thresh:
        return img_bgr, info
    out = img_bgr
    if mode in ("gamma", "gamma_clahe"):
        g = float(np.clip(np.log(params["gamma_target"] / 255.0) / np.log(max(mg, 1.0) / 255.0), params["gamma_min"], 1.0))
        lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
        out = cv2.LUT(out, lut)
        info["gamma"] = round(g, 3)
    if mode in ("clahe", "gamma_clahe"):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        c = cv2.createCLAHE(clipLimit=params["clahe_clip"], tileGridSize=(params["clahe_grid"],) * 2)
        lab[..., 0] = c.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    info["applied"] = True
    return out, info


def _starts(total: int, tile: int, overlap: float) -> list:
    if tile >= total:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    n = int(np.ceil((total - tile) / step)) + 1
    return [int(round(v)) for v in np.linspace(0, total - tile, n)]


def prepare_input(frame_bgr: np.ndarray, model_id: str, policy: Optional[dict] = None, specs: Optional[dict] = None):
    """-> (x, meta). Seg: float32 (1,3,h,w) (tile: (N,3,th,tw)) chuan hoa. Det/yolo: uint8 BGR letterbox (tile: list).
    policy ghi de spec.policy: long_side | short_side | tile={size,overlap} | enhance."""
    specs = specs or load_specs()
    sp = get_spec(model_id, specs)
    dflt = specs.get("defaults", {})
    pol = {**sp["policy"], **(policy or {})}
    H, W = frame_bgr.shape[:2]
    img, einfo = enhance_dark(frame_bgr, pol.get("enhance", "none"), dflt.get("dark_thresh", 80), dflt.get("enhance_params", {}))
    mult, mode = int(sp["multiple"]), sp["resize"]
    if mode == "keep_ratio_short":
        s = pol["short_side"] / min(H, W)
        h, w = _round_mult(H * s, mult), _round_mult(W * s, mult)
        pad = (0, 0, 0, 0)
    else:
        s = pol["long_side"] / max(H, W)
        if mode == "letterbox":
            w0, h0 = int(round(W * s)), int(round(H * s))
            h, w = int(np.ceil(h0 / mult) * mult), int(np.ceil(w0 / mult) * mult)
            top, left = (h - h0) // 2, (w - w0) // 2
            pad = (top, h - h0 - top, left, w - w0 - left)
        else:
            h, w = _round_mult(H * s, mult), _round_mult(W * s, mult)
            pad = (0, 0, 0, 0)
    interp = cv2.INTER_AREA if (w < W or h < H) else cv2.INTER_LINEAR
    if mode == "letterbox":
        rs = cv2.resize(img, (w - pad[2] - pad[3], h - pad[0] - pad[1]), interpolation=interp)
        canvas = cv2.copyMakeBorder(rs, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=(sp.get("pad_value", 114),) * 3)
        sx, sy = rs.shape[1] / W, rs.shape[0] / H
    else:
        canvas = cv2.resize(img, (w, h), interpolation=interp)
        sx, sy = w / W, h / H
    meta: Dict[str, Any] = {"model_id": model_id, "orig_hw": (H, W), "in_hw": (h, w), "pad": pad, "scale": (sx, sy),
                            "mode": mode, "enhance": einfo, "policy": pol, "tiles": None}
    tiles = pol.get("tile")
    boxes = [(0, 0, w, h)]
    if tiles:
        th, tw = min(_round_mult(tiles["size"], mult), h), min(_round_mult(tiles["size"], mult), w)
        ys, xs = _starts(h, th, tiles["overlap"]), _starts(w, tw, tiles["overlap"])
        boxes = [(x, y, x + tw, y + th) for y in ys for x in xs]
        meta["tiles"] = boxes
    crops = [canvas[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    if sp["kind"] in ("det", "seg_instance"):
        return (crops if tiles else canvas), meta
    nm = sp["norm"]
    mean, std = np.array(nm["mean"], np.float32), np.array(nm["std"], np.float32)
    arr = [np.ascontiguousarray(((cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)) for c in crops]
    return np.stack(arr), meta


def _resize_map(m: np.ndarray, size_wh, nearest: bool) -> np.ndarray:
    """(C,h,w) hoac (h,w) -> resize cv2 tung kenh (cv2 gioi han 512 kenh/lan)."""
    it = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if m.ndim == 2:
        return cv2.resize(m, size_wh, interpolation=it)
    return np.stack([cv2.resize(c, size_wh, interpolation=it) for c in m])


def restore(output: np.ndarray, meta: dict, nearest: Optional[bool] = None) -> np.ndarray:
    """Output model (C,h,w)|(h,w) [tile: (N,C,h,w)|(N,h,w)], moi tile co the o do phan giai thap hon input (logit h/4)
    -> (C,H,W)|(H,W) goc. Mask int/bool dung nearest, float dung bilinear."""
    H, W = meta["orig_hw"]
    h, w = meta["in_hw"]
    nearest = (output.dtype.kind in "biu") if nearest is None else nearest
    out = output.astype(np.float32) if (not nearest and output.dtype.kind != "f") else output
    if meta["tiles"]:
        squeeze2d = out.ndim == 3
        if squeeze2d:
            out = out[:, None]
        N, C, oh, ow = out.shape
        x0, y0, x1, y1 = meta["tiles"][0]
        fy, fx = oh / (y1 - y0), ow / (x1 - x0)
        ch, cw = int(round(h * fy)), int(round(w * fx))
        acc, wsum = np.zeros((C, ch, cw), np.float32), np.zeros((ch, cw), np.float32)
        wy = np.minimum(np.arange(oh) + 1, oh - np.arange(oh)).astype(np.float32)
        wx = np.minimum(np.arange(ow) + 1, ow - np.arange(ow)).astype(np.float32)
        win = np.minimum(wy[:, None], wx[None, :]) / max(1.0, min(oh, ow) / 2)
        for t, (a, b, _, _) in zip(out, meta["tiles"]):
            ya, xa = int(round(b * fy)), int(round(a * fx))
            acc[:, ya:ya + oh, xa:xa + ow] += t.astype(np.float32) * win
            wsum[ya:ya + oh, xa:xa + ow] += win
        out = acc / np.maximum(wsum, 1e-6)
        if squeeze2d:
            out = out[0]
        if nearest:
            out = np.rint(out)
    oh, ow = out.shape[-2:]
    top, bot, left, right = meta["pad"]
    if top or bot or left or right:  # letterbox: bo vung pad (quy theo ti le output/input)
        fy, fx = oh / h, ow / w
        out = out[..., int(round(top * fy)):oh - int(round(bot * fy)), int(round(left * fx)):ow - int(round(right * fx))]
    res = _resize_map(out, (W, H), nearest)
    return res.astype(output.dtype) if nearest else res


def restore_boxes(xyxy: np.ndarray, meta: dict, tile_index: Optional[int] = None) -> np.ndarray:
    """Hop (n,4) xyxy theo toa do input (hoac toa do trong tile `tile_index`) -> toa do anh goc, cat vao khung."""
    b = np.asarray(xyxy, np.float32).reshape(-1, 4).copy()
    H, W = meta["orig_hw"]
    if tile_index is not None and meta["tiles"]:
        x0, y0, _, _ = meta["tiles"][tile_index]
        b[:, [0, 2]] += x0
        b[:, [1, 3]] += y0
    top, _, left, _ = meta["pad"]
    sx, sy = meta["scale"]
    b[:, [0, 2]] = (b[:, [0, 2]] - left) / sx
    b[:, [1, 3]] = (b[:, [1, 3]] - top) / sy
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    return b


def verify_against_hf(model_id: str, hub: Optional[Path] = None) -> dict:
    """Doi chieu mean/std/size_divisor trong spec voi preprocessor_config.json trong HF cache (khong doan). {} neu khong co cache."""
    import json

    sp = get_spec(model_id)
    ref = str(sp.get("ref", "")).split(" ")[0].replace("/", "--")
    hub = hub or Path.home() / ".cache" / "huggingface" / "hub"
    files = list(hub.glob(f"models--{ref}/snapshots/*/preprocessor_config.json"))
    if not files:
        return {}
    pc = json.load(open(files[0]))
    out = {}
    if "image_mean" in pc and sp["norm"].get("mean"):
        out["mean_ok"] = bool(np.allclose(pc["image_mean"], sp["norm"]["mean"], atol=2e-3))
        out["std_ok"] = bool(np.allclose(pc["image_std"], sp["norm"]["std"], atol=2e-3))
    if "size_divisor" in pc:
        out["divisor_ok"] = pc["size_divisor"] == sp["multiple"]
    return out
