"""Test evidence.py bang anh tong hop nho: anh xa id2label, tien xu ly, hinh dang mask, ensemble. Khong nap model nang."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvat_tools.segmentation import evidence as E  # noqa: E402

STD_ID2LABEL = {str(i): n for i, n in enumerate(E.CITYSCAPES_NAMES)}


def test_labels_from_id2label_standard():
    ids = E.labels_from_id2label(STD_ID2LABEL)
    assert ids["road"] == 0 and ids["sidewalk"] == 1 and ids["building"] == 2 and ids["wall"] == 3
    assert ids["fence"] == 4 and ids["pole"] == 5 and ids["vegetation"] == 8 and ids["terrain"] == 9 and ids["sky"] == 10
    assert set(ids) == set(E.STUFF)


def test_labels_from_id2label_int_keys_and_rejects_shift():
    assert E.labels_from_id2label({int(k): v for k, v in STD_ID2LABEL.items()})["sky"] == 10
    shifted = dict(STD_ID2LABEL)
    shifted["0"], shifted["1"] = "sidewalk", "road"
    with pytest.raises(ValueError):
        E.labels_from_id2label(shifted)
    with pytest.raises(KeyError):
        E.labels_from_id2label({"0": "road"})


def test_hf_configs_match_standard_if_cached():
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    cfgs = list(hub.glob("models--*cityscapes*/snapshots/*/config.json"))
    if not cfgs:
        pytest.skip("khong co HF cache")
    for c in cfgs:
        assert E.labels_from_id2label(json.load(open(c))["id2label"])["sky"] == 10


def test_preprocess_keeps_aspect_and_normalizes():
    img = np.full((720, 1280, 3), 255, np.uint8)  # trang -> (1-mean)/std
    x = E.preprocess(img, long_side=1024)
    assert x.shape == (1, 3, 576, 1024) and x.dtype == np.float32  # giu ti le 16:9, boi so 32 (khong ep vuong)
    np.testing.assert_allclose(x[0, :, 0, 0], (1 - E._MEAN) / E._STD, rtol=1e-5)
    # BGR -> RGB: kenh R (index 0) cua anh thuan do phai lon nhat
    red_bgr = np.zeros((64, 64, 3), np.uint8)
    red_bgr[..., 2] = 255
    y = E.preprocess(red_bgr, long_side=64)
    assert y[0, 0].mean() > y[0, 1].mean() and y[0, 0].mean() > y[0, 2].mean()


def test_class_map_to_masks_shape_and_values():
    cm = np.zeros((40, 60), np.uint8)
    cm[:10] = 10
    cm[10:20] = 2
    ids = E.labels_from_id2label(STD_ID2LABEL)
    m = E.class_map_to_masks(cm, ids)
    assert set(m) == set(E.STUFF) and all(v.shape == (40, 60) and v.dtype == np.uint8 for v in m.values())
    assert (m["sky"][:10] == 255).all() and m["sky"][10:].sum() == 0
    assert (m["building"][10:20] == 255).all() and (m["road"][20:] == 255).all()


def test_upsample_argmax_returns_original_resolution():
    scores = np.zeros((19, 10, 16), np.float32)
    scores[10, :4] = 1.0  # sky o 4 hang tren
    scores[0, 4:] = 1.0
    out = E._upsample_argmax(scores, 80, 128)
    assert out.shape == (80, 128) and out[0, 0] == 10 and out[-1, -1] == 0


def test_parse_spec_and_unknown_model():
    assert E.parse_spec("b5") == ("b5", {})
    assert E.parse_spec("m2f@768") == ("m2f", {"long_side": 768})
    assert E.parse_spec("lane@448") == ("lane", {"short_side": 448})
    with pytest.raises(KeyError):
        E.parse_spec("nope")


def test_ensemble_and_get_evidence_with_fake_backend(monkeypatch):
    class Fake:
        ids = E.labels_from_id2label(STD_ID2LABEL)

        def __init__(self, v):
            self.v = v

        def class_map(self, img):
            return np.full(img.shape[:2], self.v, np.uint8)

    monkeypatch.setitem(E._BACKENDS, "fa", lambda **kw: Fake(0))
    monkeypatch.setitem(E._BACKENDS, "fb", lambda **kw: Fake(10))
    img = np.zeros((48, 64, 3), np.uint8)
    t = {}
    ev = E.get_evidence(img, models=["fa", "fb"], labels=["road", "sky"], timings=t)
    assert set(ev) == {"fa", "fb"} and set(ev["fa"]) == {"road", "sky"} and set(t) == {"fa", "fb"}
    assert ev["fa"]["road"].shape == (48, 64) and ev["fa"]["road"].all() and not ev["fa"]["sky"].any()
    best = E.ensemble(ev, {"road": "fa", "sky": "fb"})
    assert best["road"].all() and best["sky"].all()
    E.free_model()


def test_b2_real_model_small_image_shapes():
    if not E._COPILOT_SRC.exists() or not (E._COPILOT_SRC.parent / "models" / "segformer_b2_cityscapes.onnx").exists():
        pytest.skip("khong co B2 ONNX")
    img = np.random.default_rng(0).integers(0, 255, (96, 160, 3), np.uint8)
    ev = E.get_evidence(img, models=["b2"], labels=["road", "sky"])
    assert ev["b2"]["road"].shape == (96, 160) and set(np.unique(ev["b2"]["road"])) <= {0, 255}
    E.free_model()
