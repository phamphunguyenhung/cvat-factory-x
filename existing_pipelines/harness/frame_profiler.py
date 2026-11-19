"""Frame profiling and chronological continuity validator."""
from __future__ import annotations

import statistics
from typing import Sequence

import cv2
import numpy as np

from harness.contracts import ContinuityReport, FrameProfile


def profile_frame(
    img_bgr: np.ndarray,
    frame_id: int,
    *,
    night_luma_thresh: float = 70.0,
    continuity_group: str = "independent",
) -> FrameProfile:
    """Measure frame luminance, glare, contrast, and sharpness for adaptive routing."""
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {img_bgr.shape}")
    h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    contrast_std = float(gray.std())
    glare_ratio = float(np.count_nonzero(gray >= 245)) / float(max(1, gray.size))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    blur_score = float(lap.var())
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    high_sat_luma = float(np.count_nonzero((gray >= 230) & (hsv[:, :, 1] < 40))) / float(max(1, gray.size))
    return FrameProfile(
        frame_id=int(frame_id),
        width=w,
        height=h,
        mean_luma=round(mean_luma, 3),
        is_night=bool(mean_luma < night_luma_thresh),
        contrast_std=round(contrast_std, 3),
        glare_ratio=round(glare_ratio, 5),
        blur_score=round(blur_score, 3),
        continuity_group=continuity_group,
        metadata={"white_lamp_glare_ratio": round(high_sat_luma, 5)},
    )


def _hs_hist(img_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(img_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def assess_continuity(
    frames: Sequence[tuple[int, np.ndarray]],
    *,
    continuity_hint: str = "auto",
    min_hist_corr: float = 0.45,
) -> ContinuityReport:
    """Validate whether frames form a continuous temporal video sequence.

    Rejects tracking whenever ruleset specifies `continuity=none` OR adjacent
    appearance correlation drops below threshold (like Job 285 dashcam snapshots).
    """
    ordered = sorted(frames, key=lambda x: x[0])
    n = len(ordered)
    if continuity_hint.strip().lower() == "none":
        return ContinuityReport(
            is_continuous=False,
            reason="ruleset continuity=none (explicit non-sequential job)",
            frame_count=n,
        )
    if n < 2:
        return ContinuityReport(
            is_continuous=False,
            reason="fewer than 2 frames; temporal tracking meaningless",
            frame_count=n,
        )

    hists = [(fid, _hs_hist(img), float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())) for fid, img in ordered]
    pair_corrs: list[dict[str, float | int]] = []
    corrs: list[float] = []
    for (fa, ha, la), (fb, hb, lb) in zip(hists, hists[1:]):
        c = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
        corrs.append(c)
        pair_corrs.append({"from": fa, "to": fb, "hist_corr": round(c, 4), "dluma": round(lb - la, 2)})

    med_corr = float(statistics.median(corrs))
    sharp_breaks = sum(1 for row in pair_corrs if row["hist_corr"] < 0.25 or abs(float(row["dluma"])) > 35.0)
    is_cont = (med_corr >= min_hist_corr) and (sharp_breaks <= max(1, n // 10))
    reason = (
        f"chronological continuity verified (median_hist_corr={med_corr:.3f}, breaks={sharp_breaks}/{len(corrs)})"
        if is_cont
        else f"frames are visually non-continuous (median_hist_corr={med_corr:.3f}, breaks={sharp_breaks}/{len(corrs)})"
    )
    return ContinuityReport(
        is_continuous=is_cont,
        reason=reason,
        median_hist_corr=round(med_corr, 4),
        frame_count=n,
        per_pair_corr=pair_corrs,
    )
