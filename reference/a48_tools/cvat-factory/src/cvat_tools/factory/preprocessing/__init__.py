"""Phase 3 Budget-Bounded Adaptive Image Preprocessing (Inference vs Display Separated)."""
from cvat_tools.factory.preprocessing.adaptive import (
    AdaptivePreprocessor,
    CandidateEvaluation,
    DualPreprocessOutput,
    PreprocessBudget,
    prepare_dual_images,
)
from cvat_tools.factory.preprocessing.recipe import (
    PreprocessRecipe,
    Purpose,
    RECIPE_VERSION,
    apply_recipe,
    load_recipe,
    save_recipe,
)
from cvat_tools.factory.preprocessing.stats import (
    ImageStats,
    compute_image_stats,
    score_image_quality,
)
from cvat_tools.factory.preprocessing.transforms import (
    TransformStep,
    apply_brightness_contrast,
    apply_clahe,
    apply_clahe_gamma,
    apply_denoise,
    apply_gamma,
    apply_orientation,
    apply_step,
    apply_white_balance,
)

__all__ = [
    "AdaptivePreprocessor",
    "CandidateEvaluation",
    "DualPreprocessOutput",
    "ImageStats",
    "PreprocessBudget",
    "PreprocessRecipe",
    "Purpose",
    "RECIPE_VERSION",
    "TransformStep",
    "apply_brightness_contrast",
    "apply_clahe",
    "apply_clahe_gamma",
    "apply_denoise",
    "apply_gamma",
    "apply_orientation",
    "apply_recipe",
    "apply_step",
    "apply_white_balance",
    "compute_image_stats",
    "load_recipe",
    "prepare_dual_images",
    "save_recipe",
    "score_image_quality",
]
