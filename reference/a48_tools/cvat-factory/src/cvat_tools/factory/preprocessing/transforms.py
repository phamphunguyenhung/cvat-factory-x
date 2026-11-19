"""Deterministic, parameterized image transforms for adaptive preprocessing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TransformStep:
    """Single deterministic preprocessing step with parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformStep":
        return cls(name=str(data["name"]), params=dict(data.get("params") or {}))


def apply_clahe(
    img_bgr: np.ndarray,
    *,
    clip_limit: float = 2.5,
    tile_size: int = 8,
) -> np.ndarray:
    """Apply LAB-space CLAHE to the luminance channel only."""
    if img_bgr.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
        return clahe.apply(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma(
    img_bgr: np.ndarray,
    *,
    gamma: float = 0.55,
    preserve_highlights: bool = True,
    highlight_thresh: int = 215,
) -> np.ndarray:
    """Apply gamma LUT (< 1.0 brightens shadows) with optional highlight preservation."""
    g = float(np.clip(gamma, 0.25, 3.0))
    idx = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((idx ** g) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.LUT(img_bgr, lut)
    if preserve_highlights and img_bgr.ndim == 3:
        y_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        mask = y_orig > highlight_thresh
        if np.any(mask):
            out = out.copy()
            out[mask] = img_bgr[mask]
    return out


def apply_clahe_gamma(
    img_bgr: np.ndarray,
    *,
    gamma: float | None = None,
    target_luma: float = 115.0,
    dark_thresh: float = 90.0,
    clip_limit: float = 2.5,
    tile_size: int = 8,
    preserve_highlights: bool = True,
) -> np.ndarray:
    """Generalized Face50 CLAHE+Gamma enhancement: adaptive shadow lift when dark + LAB CLAHE.

    Unlike legacy single-image scripts, gamma is computed adaptively from image luma
    (or explicitly provided) and never hardcodes any frame/job filename.
    """
    work = img_bgr
    gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    mean_l = float(np.mean(gray))

    if gamma is not None:
        eff_gamma = float(gamma)
    elif mean_l < dark_thresh:
        # Adaptive gamma mapping mean_l -> target_luma, bounded to avoid noise blowup
        eff_gamma = float(
            np.clip(
                np.log(max(target_luma, 1.0) / 255.0) / np.log(max(mean_l, 2.0) / 255.0),
                0.38,
                0.85,
            )
        )
    else:
        eff_gamma = 1.0

    if abs(eff_gamma - 1.0) > 1e-3:
        work = apply_gamma(work, gamma=eff_gamma, preserve_highlights=preserve_highlights)
    return apply_clahe(work, clip_limit=clip_limit, tile_size=tile_size)


def apply_brightness_contrast(
    img_bgr: np.ndarray,
    *,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    max_alpha: float = 2.2,
) -> np.ndarray:
    """Percentile-based contrast & brightness normalization on luminance."""
    if img_bgr.ndim == 2:
        lo = float(np.percentile(img_bgr, low_pct))
        hi = float(np.percentile(img_bgr, high_pct))
        if hi - lo < 5.0:
            return img_bgr
        alpha = min(max_alpha, 240.0 / (hi - lo))
        beta = -lo * alpha + 8.0
        return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    lo = float(np.percentile(y, low_pct))
    hi = float(np.percentile(y, high_pct))
    if hi - lo < 5.0:
        return img_bgr
    alpha = min(max_alpha, 240.0 / (hi - lo))
    beta = -lo * alpha + 8.0
    ycrcb[:, :, 0] = cv2.convertScaleAbs(y, alpha=alpha, beta=beta)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_denoise(
    img_bgr: np.ndarray,
    *,
    d: int = 5,
    sigma_color: float = 35.0,
    sigma_space: float = 35.0,
) -> np.ndarray:
    """Edge-preserving fast bilateral denoise."""
    return cv2.bilateralFilter(img_bgr, d=int(d), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


def apply_white_balance(
    img_bgr: np.ndarray,
    *,
    strength: float = 0.75,
) -> np.ndarray:
    """Gray-world LAB a/b channel centering with configurable blend strength."""
    if img_bgr.ndim != 3:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_mean = float(np.mean(lab[:, :, 1]))
    b_mean = float(np.mean(lab[:, :, 2]))
    s = float(np.clip(strength, 0.0, 1.0))
    # Luminance-weighted shift avoids blowing out pure white/black
    lum_weight = lab[:, :, 0] / 255.0
    lab[:, :, 1] = np.clip(lab[:, :, 1] - (a_mean - 128.0) * s * lum_weight, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] - (b_mean - 128.0) * s * lum_weight, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_orientation(img_bgr: np.ndarray, *, angle_deg: int = 0) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees clockwise."""
    angle = int(angle_deg) % 360
    if angle == 0:
        return img_bgr
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported orientation angle_deg={angle_deg}; expected 0, 90, 180, or 270.")


def apply_step(img_bgr: np.ndarray, step: TransformStep) -> np.ndarray:
    """Dispatch a TransformStep by name."""
    name = step.name
    p = step.params
    if name == "baseline":
        return img_bgr
    if name == "clahe":
        return apply_clahe(img_bgr, **p)
    if name == "gamma":
        return apply_gamma(img_bgr, **p)
    if name in ("clahe_gamma", "gamma_clahe"):
        return apply_clahe_gamma(img_bgr, **p)
    if name == "brightness_contrast":
        return apply_brightness_contrast(img_bgr, **p)
    if name == "denoise":
        return apply_denoise(img_bgr, **p)
    if name == "white_balance":
        return apply_white_balance(img_bgr, **p)
    if name == "orientation":
        return apply_orientation(img_bgr, **p)
    raise ValueError(f"Unknown preprocessing transform: {name!r}")
