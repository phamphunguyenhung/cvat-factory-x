"""Task-aware label router classifying CVAT schema labels and routing models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cvat_tools.core import paths
from harness.contracts import (
    FrameProfile,
    FrameRoutePlan,
    LabelCategory,
    RouteDecision,
)

DEFAULT_Z_ORDER: dict[str, int] = {
    "sky": -9,
    "building": -8,
    "wall": -8,
    "fence": -8,
    "vegetation": -8,
    "terrain": -8,
    "sidewalk": -7,
    "pole": -2,
    "road": -1,
}

THING_LABELS = (
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "traffic light",
    "traffic sign",
)

STUFF_LABELS = (
    "road",
    "sidewalk",
    "sky",
    "building",
    "wall",
    "fence",
    "vegetation",
    "terrain",
)

STRUCTURE_LABELS = (
    "lane/crosswalk",
    "lane/single white",
    "lane/double white",
    "lane/single yellow",
    "lane/double yellow",
    "lane/road curb",
    "pole",
)

DERIVED_AREA_LABELS = ("area/direct", "area/alternative")


def load_job_ruleset(job_id: int | None = None) -> dict[str, Any]:
    """Load merged default + job-specific ruleset without mutating repo config."""
    cfg_dir = paths.CONFIG / "rulesets"
    base_file = cfg_dir / "default.yaml"
    cfg = yaml.safe_load(base_file.read_text(encoding="utf-8")) if base_file.exists() else {}
    if job_id is not None:
        job_file = cfg_dir / "jobs" / f"{job_id}.yaml"
        if job_file.exists():
            override = yaml.safe_load(job_file.read_text(encoding="utf-8")) or {}
            cfg = _deep_merge(cfg or {}, override)
    return cfg or {}


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class LabelRouter:
    """Routes each schema label to specialized Thing / Stuff / Structure / Derived tools."""

    def __init__(self, job_id: int | None = 285, ruleset: dict[str, Any] | None = None):
        self.job_id = job_id
        self.ruleset = ruleset if ruleset is not None else load_job_ruleset(job_id)

    @property
    def continuity_hint(self) -> str:
        return str(self.ruleset.get("continuity", "auto"))

    def classify_label(self, label: str) -> LabelCategory:
        if label in THING_LABELS:
            return LabelCategory.THING
        if label in STUFF_LABELS:
            return LabelCategory.STUFF
        if label in DERIVED_AREA_LABELS:
            return LabelCategory.DERIVED_AREA
        if label in STRUCTURE_LABELS or label.startswith("lane/"):
            return LabelCategory.STRUCTURE
        raise ValueError(f"Unknown label in V3 router: {label!r}")

    def export_shape_for(self, label: str) -> str:
        cat = self.classify_label(label)
        if cat == LabelCategory.THING:
            return "rectangle"
        if label.startswith("lane/") and label != "lane/crosswalk":
            lane_geom = str(self.ruleset.get("lane_shape_type", "")).lower()
            if lane_geom in ("polyline", "polygon"):
                return lane_geom
            # Job 285 uses polygon lane ribbons; Job 987 uses polyline.
            return "polyline" if self.job_id == 987 else "polygon"
        return "polygon"

    def z_order_for(self, label: str) -> int:
        zmap = self.ruleset.get("z_order", {})
        if isinstance(zmap, dict) and label in zmap:
            return int(zmap[label])
        return int(DEFAULT_Z_ORDER.get(label, 0))

    def route_frame(
        self,
        profile: FrameProfile,
        *,
        mode: str = "balanced",
        tracking_enabled: bool = False,
        run_stuff: bool = True,
        run_structure: bool = True,
    ) -> FrameRoutePlan:
        """Select optimal model/fallback routing per label for a given frame profile."""
        if mode not in ("fast", "balanced", "high_recall", "challenger"):
            raise ValueError(f"Unsupported routing mode: {mode!r}")

        is_night = profile.is_night
        det_primary = "yolo26s_ov_640" if mode == "fast" else "yolo26l_ov_1280"
        det_fallbacks = ("yolo26s_ov_960", "yolo26l_pt") if mode != "fast" else ("yolo26n_ov_640",)
        stuff_primary = "mask2former_swin_l" if mode == "challenger" else "segformer_b2"
        stuff_fallbacks = ("segformer_b2",) if mode == "challenger" else ("segformer_b5",)

        decisions: dict[str, RouteDecision] = {}
        for lab in THING_LABELS:
            use_sam = (mode in ("balanced", "high_recall", "challenger")) and lab in (
                "car",
                "truck",
                "bus",
                "pedestrian",
                "motorcycle",
                "bicycle",
            )
            accept_conf = 0.16 if (is_night and lab in ("car", "truck", "traffic light")) else 0.24
            if lab == "traffic light" and not is_night:
                accept_conf = 0.20
            if lab == "traffic sign":
                accept_conf = 0.24 if is_night else 0.25
            decisions[lab] = RouteDecision(
                label=lab,
                category=LabelCategory.THING,
                export_shape_type="rectangle",
                z_order=self.z_order_for(lab),
                primary_model=det_primary,
                fallback_models=det_fallbacks,
                run_sam_on_overlap=use_sam,
                use_night_enhance=is_night,
                accept_conf=accept_conf,
                review_conf_floor=0.07 if is_night else 0.12,
                notes="night enhanced + glare filtering" if is_night else "day horizon + edge crops",
            )

        for lab in STUFF_LABELS:
            decisions[lab] = RouteDecision(
                label=lab,
                category=LabelCategory.STUFF,
                export_shape_type="polygon",
                z_order=self.z_order_for(lab),
                primary_model=stuff_primary,
                fallback_models=stuff_fallbacks,
                use_night_enhance=False,  # SegFormer Cityscapes calibrated on raw BGR
                notes="probability RLE first; validate before polygon simplification",
            )

        for lab in STRUCTURE_LABELS:
            decisions[lab] = RouteDecision(
                label=lab,
                category=LabelCategory.STRUCTURE,
                export_shape_type=self.export_shape_for(lab),
                z_order=self.z_order_for(lab),
                primary_model="classical_cv" if lab != "pole" else "segformer_b2+pole_rule",
                fallback_models=("segformer_lane",),
                notes="glare/streak rejection required at night" if is_night else "perspective + continuity checks",
            )

        for lab in DERIVED_AREA_LABELS:
            decisions[lab] = RouteDecision(
                label=lab,
                category=LabelCategory.DERIVED_AREA,
                export_shape_type="polygon",
                z_order=self.z_order_for(lab),
                primary_model="layer_composer",
                notes="derived from validated road mask - foreground obstacles",
            )

        return FrameRoutePlan(
            frame_id=profile.frame_id,
            mode=mode,
            is_night=is_night,
            tracking_enabled=bool(tracking_enabled),
            run_stuff=bool(run_stuff),
            run_structure=bool(run_structure),
            decisions=decisions,
        )
