"""Test input contract: restore(prepare(x)) khop toa do <=1px tren mask tong hop, tu choi model thieu spec. Khong nap model nang."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvat_tools.segmentation import inputspec as I  # noqa: E402

H, W = 720, 1280
RECT = (301, 203, 937, 611)  # x0,y0,x1,y1 (x1,y1 exclusive)


def _frame():
    f = np.zeros((H, W, 3), np.uint8) + 20
    f[RECT[1]:RECT[3], RECT[0]:RECT[2]] = 235
    return f


def _gt():
    m = np.zeros((H, W), bool)
    m[RECT[1]:RECT[3], RECT[0]:RECT[2]] = True
    return m


def _edges(m):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _check(mask_restored):
    e, g = _edges(mask_restored > 0.5), _edges(_gt())
    assert max(abs(a - b) for a, b in zip(e, g)) <= 1, (e, g)


def _fake_output(x, scale=1.0):
    """Gia lap model: kenh 0 cua tensor da chuan hoa -> nguong thanh mask (xac suat 0/1), tuy chon ha do phan giai (logit h/4)."""
    import cv2

    ch = (x[:, 0] > 0.0).astype(np.float32)  # (N,h,w), anh nen 20 -> am, hop 235 -> duong
    if scale != 1.0:
        ch = np.stack([cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for c in ch])
    return ch


@pytest.mark.parametrize("model", ["b2", "b5", "m2f"])
def test_keep_ratio_roundtrip(model):
    x, meta = I.prepare_input(_frame(), model)
    assert x.shape[0] == 1 and x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0
    assert max(x.shape[2:]) == 1024  # giu ti le, canh dai = long_side
    _check(I.restore(_fake_output(x)[0], meta))


def test_logit_quarter_res_roundtrip():
    x, meta = I.prepare_input(_frame(), "b5", policy={"long_side": 1024})
    _check(I.restore(_fake_output(x, 1.0)[0], meta))  # do phan giai input
    # logit h/4 mat chinh xac 4 diem input -> cho phep ~ 4/scale px; chi kiem tra kich thuoc + vung
    out = I.restore(_fake_output(x, 0.25)[0], meta)
    assert out.shape == (H, W)
    e, g = _edges(out > 0.5), _edges(_gt())
    assert max(abs(a - b) for a, b in zip(e, g)) <= 6


def test_lane_short_side():
    x, meta = I.prepare_input(_frame(), "lane")
    assert min(x.shape[2:]) == 224 and x.shape[3] % 32 == 0
    out = I.restore(_fake_output(x)[0], meta)
    e, g = _edges(out > 0.5), _edges(_gt())
    assert max(abs(a - b) for a, b in zip(e, g)) <= 6  # 224px -> 3.2x phong dai: sai so luong tu hoa


def test_tile_roundtrip_multichannel():
    pol = {"long_side": 1280, "tile": {"size": 512, "overlap": 0.25}}
    x, meta = I.prepare_input(_frame(), "b2", policy=pol)
    assert x.shape[0] > 1 and meta["tiles"]
    out = _fake_output(x)  # (N,h,w)
    _check(I.restore(out, meta))
    out2 = np.stack([out, 1 - out], 1)  # (N,2,h,w)
    r = I.restore(out2, meta)
    assert r.shape == (2, H, W)
    _check(r[0])


def test_yolo_letterbox_boxes_roundtrip():
    x, meta = I.prepare_input(_frame(), "yolo26s")
    h, w = x.shape[:2]
    assert h % 32 == 0 and w % 32 == 0 and max(h, w) == 640
    m = (x[..., 0] > 100)  # hop sang trong anh letterbox (pad=114 tren kenh xam khong bi nham: hop=235, nen 20)
    ys, xs = np.where(x[..., 0] > 200)
    box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
    r = I.restore_boxes(box, meta)[0]
    assert np.abs(r - np.array(RECT)).max() <= 2.0
    assert m.shape == (h, w)


def test_yolo_tile_boxes():
    pol = {"long_side": 1280, "tile": {"size": 640, "overlap": 0.2}}
    tiles, meta = I.prepare_input(_frame(), "yolo26n", policy=pol)
    assert isinstance(tiles, list) and len(tiles) == len(meta["tiles"]) > 1
    x0, y0, _, _ = meta["tiles"][-1]
    r = I.restore_boxes(np.array([[10, 20, 50, 60]], np.float32), meta, tile_index=len(tiles) - 1)[0]
    sx, sy = meta["scale"]
    top, _, left, _ = meta["pad"]  # toa do tile nam tren canvas letterbox -> tru pad
    assert np.allclose(r, [(10 + x0 - left) / sx, (20 + y0 - top) / sy, (50 + x0 - left) / sx, (60 + y0 - top) / sy], atol=1e-3)


def test_enhance_only_dark():
    dark = np.full((H, W, 3), 30, np.uint8)
    bright = np.full((H, W, 3), 150, np.uint8)
    for mode in ("gamma", "clahe", "gamma_clahe"):
        d, info = I.enhance_dark(dark, mode, 80, I.load_specs()["defaults"]["enhance_params"])
        assert info["applied"] and I.mean_gray(d) > 30
        b, info = I.enhance_dark(bright, mode, 80, I.load_specs()["defaults"]["enhance_params"])
        assert not info["applied"] and b is bright


def test_rejects_unknown_and_incomplete_spec():
    with pytest.raises(I.UnknownModelSpec):
        I.prepare_input(_frame(), "model_la")
    specs = I.load_specs()
    del specs["models"]["b5"]["norm"]
    with pytest.raises(I.UnknownModelSpec):
        I.prepare_input(_frame(), "b5", specs=specs)
    specs = I.load_specs()
    specs["models"]["b2"]["norm"]["mean"] = None
    with pytest.raises(I.UnknownModelSpec):
        I.get_spec("b2", specs)


def test_spec_matches_hf_cache_if_present():
    for m in ("b5", "m2f"):
        r = I.verify_against_hf(m)
        if not r:
            pytest.skip("khong co HF cache")
        assert all(r.values()), (m, r)


def test_all_declared_models_have_spec_fields():
    for m in I.load_specs()["models"]:
        sp = I.get_spec(m)
        assert sp["train"] and sp["multiple"] % 8 == 0
