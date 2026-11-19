"""Budget-bounded Adaptive Image Preprocessing with strict inference vs display separation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cvat_tools.factory.preprocessing.recipe import PreprocessRecipe, Purpose
from cvat_tools.factory.preprocessing.stats import (
    ImageStats,
    compute_image_stats,
    score_image_quality,
)
from cvat_tools.factory.preprocessing.transforms import TransformStep, apply_step


@dataclass(frozen=True)
class PreprocessBudget:
    """Compute and candidate budget for adaptive preprocessing."""

    max_candidates: int = 5
    max_time_ms: float = 150.0
    min_improvement: float = 0.03
    fast_path_bypass: bool = True


@dataclass(frozen=True)
class CandidateEvaluation:
    """Comparison record for one candidate transform pipeline vs baseline."""

    name: str
    steps: tuple[TransformStep, ...]
    quality_score: float
    delta_vs_baseline: float
    elapsed_ms: float
    stats: ImageStats


@dataclass
class DualPreprocessOutput:
    """Strictly separated inference-input image and display-enhanced image with recipes."""

    inference_image: np.ndarray
    inference_recipe: PreprocessRecipe
    display_image: np.ndarray
    display_recipe: PreprocessRecipe
    evaluations: list[CandidateEvaluation] = field(default_factory=list)


class AdaptivePreprocessor:
    """Evaluates baseline vs candidate transforms within a strict time/candidate budget."""

    def __init__(self, budget: PreprocessBudget | None = None) -> None:
        self.budget = budget or PreprocessBudget()

    def _build_candidates(
        self,
        stats: ImageStats,
        *,
        purpose: Purpose,
        allow_orientation: bool = False,
        orientation_angles: tuple[int, ...] = (90, 180, 270),
    ) -> list[tuple[str, tuple[TransformStep, ...]]]:
        candidates: list[tuple[str, tuple[TransformStep, ...]]] = []

        if stats.is_low_light:
            # Reuse Face50 CLAHE+Gamma insight (adaptive gamma + LAB CLAHE, never hardcoded to a filename)
            target_luma = 125.0 if purpose == "display" else 110.0
            clip = 2.8 if purpose == "display" else 2.5
            candidates.append((
                "clahe_gamma",
                (TransformStep("clahe_gamma", {"target_luma": target_luma, "clip_limit": clip}),),
            ))
            candidates.append((
                "gamma",
                (TransformStep("gamma", {"gamma": 0.55 if purpose == "display" else 0.65, "preserve_highlights": True}),),
            ))
            candidates.append((
                "clahe",
                (TransformStep("clahe", {"clip_limit": clip, "tile_size": 8}),),
            ))

        if stats.is_low_contrast:
            candidates.append((
                "clahe",
                (TransformStep("clahe", {"clip_limit": 2.5, "tile_size": 8}),),
            ))
            candidates.append((
                "brightness_contrast",
                (TransformStep("brightness_contrast", {"low_pct": 2.0, "high_pct": 98.0}),),
            ))

        if stats.has_color_cast:
            candidates.append((
                "white_balance",
                (TransformStep("white_balance", {"strength": 0.75}),),
            ))
            if stats.is_low_light or stats.is_low_contrast:
                candidates.append((
                    "wb_clahe_gamma",
                    (
                        TransformStep("white_balance", {"strength": 0.7}),
                        TransformStep("clahe_gamma", {"target_luma": 115.0, "clip_limit": 2.5}),
                    ),
                ))

        if stats.is_noisy and (stats.is_low_light or stats.is_low_contrast):
            candidates.append((
                "denoise_clahe_gamma",
                (
                    TransformStep("denoise", {"d": 5, "sigma_color": 30.0, "sigma_space": 30.0}),
                    TransformStep("clahe_gamma", {"target_luma": 110.0, "clip_limit": 2.2}),
                ),
            ))
        elif stats.is_noisy:
            candidates.append((
                "denoise",
                (TransformStep("denoise", {"d": 5, "sigma_color": 30.0, "sigma_space": 30.0}),),
            ))

        if allow_orientation:
            for ang in orientation_angles:
                if ang % 360 != 0:
                    candidates.append((
                        f"orient_{ang}",
                        (TransformStep("orientation", {"angle_deg": ang}),),
                    ))

        # Deduplicate by candidate name while preserving order
        seen: set[str] = set()
        unique: list[tuple[str, tuple[TransformStep, ...]]] = []
        for name, steps in candidates:
            if name not in seen:
                seen.add(name)
                unique.append((name, steps))
        return unique

    def run(
        self,
        img_bgr: np.ndarray,
        *,
        purpose: Purpose = "inference",
        allow_orientation: bool = False,
    ) -> tuple[np.ndarray, PreprocessRecipe, list[CandidateEvaluation]]:
        """Run budget-bounded adaptive preprocessing for either 'inference' or 'display'."""
        t0 = time.perf_counter()
        base_stats = compute_image_stats(img_bgr)
        base_quality = score_image_quality(base_stats)

        has_issue = (
            base_stats.is_low_light
            or base_stats.is_low_contrast
            or base_stats.has_color_cast
            or base_stats.is_noisy
        )

        # Fast Path First: healthy images skip candidate search
        if self.budget.fast_path_bypass and not has_issue and not allow_orientation:
            elapsed = (time.perf_counter() - t0) * 1000.0
            recipe = PreprocessRecipe(
                purpose=purpose,
                selected_candidate="baseline",
                steps=(),
                quality_before=base_quality,
                quality_after=base_quality,
                stats_before=base_stats,
                stats_after=base_stats,
                budget_used_ms=round(elapsed, 2),
                geometry_preserved=True,
                metadata={"fast_path": True},
            )
            return img_bgr.copy(), recipe, []

        candidates = self._build_candidates(
            base_stats,
            purpose=purpose,
            allow_orientation=allow_orientation and purpose == "inference",
        )

        best_name = "baseline"
        best_steps: tuple[TransformStep, ...] = ()
        best_img = img_bgr
        best_stats = base_stats
        best_quality = base_quality
        evaluations: list[CandidateEvaluation] = []

        for idx, (cand_name, steps) in enumerate(candidates):
            if idx >= self.budget.max_candidates:
                break
            if (time.perf_counter() - t0) * 1000.0 >= self.budget.max_time_ms:
                break

            tc0 = time.perf_counter()
            cand_img = img_bgr
            for st in steps:
                cand_img = apply_step(cand_img, st)
            cand_stats = compute_image_stats(cand_img)
            cand_quality = score_image_quality(cand_stats)
            cand_ms = (time.perf_counter() - tc0) * 1000.0
            delta = round(cand_quality - base_quality, 4)

            evaluations.append(
                CandidateEvaluation(
                    name=cand_name,
                    steps=steps,
                    quality_score=cand_quality,
                    delta_vs_baseline=delta,
                    elapsed_ms=round(cand_ms, 2),
                    stats=cand_stats,
                )
            )

            if delta >= self.budget.min_improvement and cand_quality > best_quality:
                best_name = cand_name
                best_steps = steps
                best_img = cand_img
                best_stats = cand_stats
                best_quality = cand_quality

        total_ms = (time.perf_counter() - t0) * 1000.0
        geom_preserved = not any(s.name == "orientation" and int(s.params.get("angle_deg", 0)) % 360 != 0 for s in best_steps)

        recipe = PreprocessRecipe(
            purpose=purpose,
            selected_candidate=best_name,
            steps=best_steps,
            quality_before=base_quality,
            quality_after=best_quality,
            stats_before=base_stats,
            stats_after=best_stats,
            budget_used_ms=round(total_ms, 2),
            geometry_preserved=geom_preserved,
            metadata={"fast_path": False, "candidates_evaluated": len(evaluations)},
        )
        return best_img.copy(), recipe, evaluations


def prepare_dual_images(
    img_bgr: np.ndarray,
    *,
    inference_budget: PreprocessBudget | None = None,
    display_budget: PreprocessBudget | None = None,
    allow_orientation_for_inference: bool = False,
) -> DualPreprocessOutput:
    """Produce strictly separated inference-input and display-enhanced images + recipes."""
    inf_proc = AdaptivePreprocessor(inference_budget or PreprocessBudget())
    disp_proc = AdaptivePreprocessor(
        display_budget or PreprocessBudget(max_candidates=3, max_time_ms=100.0, min_improvement=0.01)
    )

    inf_img, inf_recipe, inf_evals = inf_proc.run(
        img_bgr,
        purpose="inference",
        allow_orientation=allow_orientation_for_inference,
    )
    disp_img, disp_recipe, disp_evals = disp_proc.run(
        img_bgr,
        purpose="display",
        allow_orientation=False,
    )

    return DualPreprocessOutput(
        inference_image=inf_img,
        inference_recipe=inf_recipe,
        display_image=disp_img,
        display_recipe=disp_recipe,
        evaluations=inf_evals + disp_evals,
    )
