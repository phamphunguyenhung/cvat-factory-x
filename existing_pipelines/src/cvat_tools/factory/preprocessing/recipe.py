"""Serializable preprocessing recipes with strict inference vs display separation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cvat_tools.factory.preprocessing.stats import ImageStats
from cvat_tools.factory.preprocessing.transforms import TransformStep, apply_step

RECIPE_VERSION = "1.0"
Purpose = Literal["inference", "display"]


@dataclass(frozen=True)
class PreprocessRecipe:
    """Deterministic, auditable preprocessing recipe with explicit purpose."""

    purpose: Purpose
    selected_candidate: str
    steps: tuple[TransformStep, ...] = ()
    quality_before: float = 0.0
    quality_after: float = 0.0
    stats_before: ImageStats | None = None
    stats_after: ImageStats | None = None
    budget_used_ms: float = 0.0
    geometry_preserved: bool = True
    version: str = RECIPE_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "purpose": self.purpose,
            "selected_candidate": self.selected_candidate,
            "steps": [s.to_dict() for s in self.steps],
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "stats_before": self.stats_before.to_dict() if self.stats_before else None,
            "stats_after": self.stats_after.to_dict() if self.stats_after else None,
            "budget_used_ms": self.budget_used_ms,
            "geometry_preserved": self.geometry_preserved,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessRecipe":
        purpose = data.get("purpose", "inference")
        if purpose not in ("inference", "display"):
            raise ValueError(f"Invalid recipe purpose: {purpose!r}")
        return cls(
            version=str(data.get("version", RECIPE_VERSION)),
            purpose=purpose,
            selected_candidate=str(data.get("selected_candidate", "baseline")),
            steps=tuple(TransformStep.from_dict(s) for s in (data.get("steps") or ())),
            quality_before=float(data.get("quality_before", 0.0)),
            quality_after=float(data.get("quality_after", 0.0)),
            stats_before=ImageStats.from_dict(data["stats_before"]) if data.get("stats_before") else None,
            stats_after=ImageStats.from_dict(data["stats_after"]) if data.get("stats_after") else None,
            budget_used_ms=float(data.get("budget_used_ms", 0.0)),
            geometry_preserved=bool(data.get("geometry_preserved", True)),
            metadata=dict(data.get("metadata") or {}),
        )


def save_recipe(recipe: PreprocessRecipe, path: str | Path) -> Path:
    """Persist a PreprocessRecipe as UTF-8 JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recipe.to_dict(), indent=2), encoding="utf-8")
    return p


def load_recipe(path: str | Path) -> PreprocessRecipe:
    """Load a PreprocessRecipe from JSON."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return PreprocessRecipe.from_dict(data)


def apply_recipe(
    img_bgr: np.ndarray,
    recipe: PreprocessRecipe,
    *,
    expected_purpose: Purpose | None = None,
) -> np.ndarray:
    """Replay a PreprocessRecipe deterministically.

    If ``expected_purpose`` is set, raises ValueError if the recipe's purpose does not match,
    preventing display-enhanced recipes from leaking into inference inputs.
    """
    if expected_purpose is not None and recipe.purpose != expected_purpose:
        raise ValueError(
            f"Purpose mismatch: expected {expected_purpose!r} recipe, got {recipe.purpose!r}."
        )
    out = img_bgr
    for step in recipe.steps:
        out = apply_step(out, step)
    return out
