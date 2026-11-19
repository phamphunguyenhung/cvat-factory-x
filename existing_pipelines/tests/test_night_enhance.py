"""Unit tests for adaptive CPU night enhancement."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvat_tools.detection.night_enhance import (
    enhance_night_bgr,
    get_detection_variants,
    is_night_frame,
)

FRAMES_DIR = Path(r"D:\06-Learning\AIA-VINGROUP\learning\tools\cvat-seg-copilot\frames")


def _make_bgr(h: int, w: int, val: int) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_is_night_frame_thresholds() -> None:
    dark = _make_bgr(64, 64, 25)
    bright = _make_bgr(64, 64, 120)

    is_n, luma_d = is_night_frame(dark)
    assert is_n is True
    assert luma_d == pytest.approx(25.0, abs=1.0)

    is_b, luma_b = is_night_frame(bright)
    assert is_b is False
    assert luma_b == pytest.approx(120.0, abs=1.0)

    assert is_night_frame(dark, luma_thresh=20.0)[0] is False
    assert is_night_frame(np.full((32, 32), 30, dtype=np.uint8))[0] is True


def test_is_night_frame_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        is_night_frame(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        is_night_frame(np.zeros((10, 10, 4), dtype=np.uint8))


def test_enhance_modes_and_immutability() -> None:
    dark = _make_bgr(120, 160, 22)
    # Add a bright headlight patch and subtle texture so CLAHE has non-flat histogram.
    dark[10:30, 10:30] = 250
    dark[50:90, 40:120] = 45
    snapshot = dark.copy()

    assert enhance_night_bgr(dark, mode="none") is dark

    for mode in ("clahe", "gamma_clahe", "auto"):
        out = enhance_night_bgr(dark, mode=mode)
        assert out.shape == dark.shape
        assert out.dtype == np.uint8
        assert np.array_equal(dark, snapshot)  # input never mutated
        # Dark region should be lifted
        assert float(out[50:90, 40:120].mean()) > float(dark[50:90, 40:120].mean())
        # Headlight patch must stay bright without wrapping/darkening
        assert float(out[10:30, 10:30].mean()) >= 240.0


def test_enhance_auto_noop_on_bright_frame() -> None:
    bright = _make_bgr(64, 64, 140)
    out = enhance_night_bgr(bright, mode="auto")
    assert out is bright


def test_enhance_invalid_mode_or_shape() -> None:
    with pytest.raises(ValueError):
        enhance_night_bgr(_make_bgr(32, 32, 20), mode="invalid_mode")
    with pytest.raises(ValueError):
        enhance_night_bgr(np.zeros((32, 32), dtype=np.uint8), mode="auto")


def test_get_detection_variants() -> None:
    bright = _make_bgr(64, 64, 130)
    v_bright = get_detection_variants(bright)
    assert len(v_bright) == 1
    assert v_bright[0][0] == "orig"
    assert v_bright[0][1] is bright

    dark = _make_bgr(64, 64, 25)
    dark[20:40, 20:40] = 50
    v_dark = get_detection_variants(dark)
    assert len(v_dark) == 2
    assert v_dark[0][0] == "orig"
    assert v_dark[1][0] == "night_clahe"
    assert v_dark[1][1].shape == dark.shape
    assert v_dark[1][1].dtype == np.uint8


@pytest.mark.skipif(not FRAMES_DIR.exists(), reason="Job 285 frames directory not present")
def test_job285_night_day_split() -> None:
    import cv2

    night_count = 0
    day_count = 0
    for f in range(60, 80):
        img = cv2.imread(str(FRAMES_DIR / f"j285_f{f}.jpg"))
        assert img is not None
        is_n, _ = is_night_frame(img)
        variants = get_detection_variants(img)
        if is_n:
            night_count += 1
            assert len(variants) == 2
        else:
            day_count += 1
            assert len(variants) == 1

    assert night_count == 13
    assert day_count == 7
    for f in (60, 67, 68, 75):
        img = cv2.imread(str(FRAMES_DIR / f"j285_f{f}.jpg"))
        assert is_night_frame(img)[0] is True
