"""Image quality statistics and deterministic scoring for adaptive preprocessing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageStats:
    """Quantitative image statistics for low-light, contrast, sharpness, noise, and color cast."""

    width: int
    height: int
    mean_luma: float
    median_luma: float
    std_luma: float
    p5_luma: float
    p95_luma: float
    dynamic_range: float
    dark_pixel_ratio: float
    highlight_clip_ratio: float
    laplacian_var: float
    noise_estimate: float
    color_cast_score: float
    is_low_light: bool
    is_low_contrast: bool
    has_color_cast: bool
    is_noisy: bool
    is_blurry: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageStats":
        return cls(**data)


def compute_image_stats(
    img_bgr: np.ndarray,
    *,
    low_light_thresh: float = 85.0,
    low_contrast_std_thresh: float = 28.0,
    low_contrast_range_thresh: float = 90.0,
    color_cast_thresh: float = 14.0,
    noise_thresh: float = 10.0,
    blur_thresh: float = 45.0,
) -> ImageStats:
    """Compute fast, deterministic image quality statistics on a BGR uint8 image."""
    if img_bgr is None or img_bgr.ndim < 2 or img_bgr.size == 0:
        raise ValueError("img_bgr must be a non-empty image array.")

    h, w = img_bgr.shape[:2]
    # Downsample large images for fast bounded-latency stats (preserving global distribution)
    max_dim = max(h, w)
    if max_dim > 640:
        scale = 640.0 / max_dim
        work = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        work = img_bgr

    if work.ndim == 2:
        gray = work
        cast_score = 0.0
    else:
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        a_shift = float(np.mean(lab[:, :, 1])) - 128.0
        b_shift = float(np.mean(lab[:, :, 2])) - 128.0
        cast_score = float(np.hypot(a_shift, b_shift))

    gray_f = gray.astype(np.float32)
    mean_luma = float(np.mean(gray_f))
    median_luma = float(np.median(gray_f))
    std_luma = float(np.std(gray_f))
    p5 = float(np.percentile(gray_f, 5))
    p95 = float(np.percentile(gray_f, 95))
    dyn_range = max(0.0, p95 - p5)
    dark_ratio = float(np.mean(gray < 40))
    clip_ratio = float(np.mean(gray > 245))

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    # Fast noise estimate via median absolute deviation of high-pass residual
    blurred = cv2.GaussianBlur(gray_f, (3, 3), 0)
    residual = np.abs(gray_f - blurred)
    noise_est = float(np.median(residual) * 1.4826)

    is_low_light = mean_luma < low_light_thresh or p95 < 130.0
    is_low_contrast = std_luma < low_contrast_std_thresh or dyn_range < low_contrast_range_thresh
    has_color_cast = cast_score > color_cast_thresh
    is_noisy = noise_est > noise_thresh
    is_blurry = lap_var < blur_thresh

    return ImageStats(
        width=w,
        height=h,
        mean_luma=round(mean_luma, 2),
        median_luma=round(median_luma, 2),
        std_luma=round(std_luma, 2),
        p5_luma=round(p5, 2),
        p95_luma=round(p95, 2),
        dynamic_range=round(dyn_range, 2),
        dark_pixel_ratio=round(dark_ratio, 4),
        highlight_clip_ratio=round(clip_ratio, 4),
        laplacian_var=round(lap_var, 2),
        noise_estimate=round(noise_est, 2),
        color_cast_score=round(cast_score, 2),
        is_low_light=is_low_light,
        is_low_contrast=is_low_contrast,
        has_color_cast=has_color_cast,
        is_noisy=is_noisy,
        is_blurry=is_blurry,
    )


def score_image_quality(stats: ImageStats) -> float:
    """Score image suitability in [0.0, 1.0] balancing exposure, contrast, sharpness, and artifacts."""
    # Exposure score peaks when mean_luma is in [95, 160]
    if 95.0 <= stats.mean_luma <= 160.0:
        exposure = 1.0
    elif stats.mean_luma < 95.0:
        exposure = max(0.0, stats.mean_luma / 95.0)
    else:
        exposure = max(0.0, 1.0 - (stats.mean_luma - 160.0) / 95.0)

    # Dynamic range & contrast
    range_score = min(1.0, stats.dynamic_range / 180.0)
    contrast_score = min(1.0, stats.std_luma / 55.0)

    # Sharpness (capped so noise spikes don't dominate)
    sharp_score = min(1.0, stats.laplacian_var / 300.0)

    # Penalties for clipping, crushed shadows, noise amplification, and strong color cast
    clip_pen = min(0.35, stats.highlight_clip_ratio * 1.5)
    dark_pen = min(0.30, max(0.0, stats.dark_pixel_ratio - 0.15) * 0.6)
    noise_pen = min(0.25, max(0.0, stats.noise_estimate - 8.0) / 30.0)
    cast_pen = min(0.15, max(0.0, stats.color_cast_score - 12.0) / 60.0)

    raw = 0.30 * exposure + 0.25 * range_score + 0.25 * contrast_score + 0.20 * sharp_score
    total = max(0.0, min(1.0, raw - clip_pen - dark_pen - noise_pen - cast_pen))
    return round(total, 4)
