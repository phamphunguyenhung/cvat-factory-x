"""Fast CPU-only adaptive night and low-light enhancement for vehicle/person detection.

Designed for multi-pass detection: bright frames run a single pass on ``orig``,
while dark/night frames run ``orig`` plus an adaptive shadow-lifted ``night_clahe``
variant that recovers dark vehicles without blowing out headlights or bright sky.
"""
from __future__ import annotations

import cv2
import numpy as np

ENHANCE_MODES = ("none", "clahe", "gamma_clahe", "auto")
DEFAULT_LUMA_THRESH = 70.0


def _compute_luma_mean(img_bgr: np.ndarray) -> float:
    if img_bgr.ndim == 2:
        return float(img_bgr.mean())
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    raise ValueError(f"Expected BGR (H,W,3) or grayscale (H,W) image, got shape {img_bgr.shape}")


def is_night_frame(img_bgr: np.ndarray, luma_thresh: float = DEFAULT_LUMA_THRESH) -> tuple[bool, float]:
    """Return ``(is_night, mean_luma)`` using BT.601 grayscale luminance."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty")
    mean_luma = _compute_luma_mean(img_bgr)
    return (mean_luma < luma_thresh, mean_luma)


def _apply_lab_clahe(img_bgr: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _apply_gamma_lut(img_bgr: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    idx = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((idx ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)


def _enhance_auto_night(img_bgr: np.ndarray, mean_luma: float) -> np.ndarray:
    """Highlight-preserving adaptive shadow lift + mild CLAHE blended with original."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y_ch = ycrcb[:, :, 0]

    # Darker frames get stronger shadow lift; bright highlights (>215) stay anchored.
    gamma = float(np.clip(0.38 + 0.42 * (mean_luma / DEFAULT_LUMA_THRESH), 0.38, 0.80))
    idx = np.arange(256, dtype=np.float32)
    lifted = ((idx / 255.0) ** gamma) * 255.0
    shadow_weight = np.clip((215.0 - idx) / 145.0, 0.0, 1.0)
    lut = np.clip(shadow_weight * lifted + (1.0 - shadow_weight) * idx, 0, 255).astype(np.uint8)

    y_lifted = cv2.LUT(y_ch, lut)
    clip = 2.6 if mean_luma < 48.0 else 1.8
    y_clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(y_lifted)
    # Prevent CLAHE from blowing out headlights or bright signage.
    ycrcb[:, :, 0] = np.where(y_ch > 205, y_ch, y_clahe)
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    orig_weight = 0.15 if mean_luma < 48.0 else 0.30
    return cv2.addWeighted(img_bgr, orig_weight, enhanced, 1.0 - orig_weight, 0)


def enhance_night_bgr(img_bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Enhance a BGR image for low-light detection without mutating the input.

    Modes:
      - ``"none"``: return ``img_bgr`` unchanged.
      - ``"clahe"``: LAB-space CLAHE on L channel.
      - ``"gamma_clahe"``: global gamma lift (0.55) followed by LAB CLAHE.
      - ``"auto"``: no-op on bright frames; highlight-preserving shadow lift + CLAHE
        blended with original on night frames.
    """
    if mode not in ENHANCE_MODES:
        raise ValueError(f"Unsupported enhance mode {mode!r}; expected one of {ENHANCE_MODES}")
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR (H,W,3) image, got shape {img_bgr.shape}")

    if mode == "none":
        return img_bgr
    if mode == "clahe":
        return _apply_lab_clahe(img_bgr, clip_limit=2.5, tile_size=8)
    if mode == "gamma_clahe":
        return _apply_lab_clahe(_apply_gamma_lut(img_bgr, gamma=0.55), clip_limit=2.5, tile_size=8)

    is_night, mean_luma = is_night_frame(img_bgr)
    if not is_night:
        return img_bgr
    return _enhance_auto_night(img_bgr, mean_luma)


def get_detection_variants(img_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return ``[("orig", img_bgr)]`` for bright frames or ``[("orig", ...), ("night_clahe", ...)]`` for night frames."""
    is_night, _ = is_night_frame(img_bgr)
    if not is_night:
        return [("orig", img_bgr)]
    return [("orig", img_bgr), ("night_clahe", enhance_night_bgr(img_bgr, mode="auto"))]
