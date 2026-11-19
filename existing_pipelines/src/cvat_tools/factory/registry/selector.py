"""Configurable Auto Model Selection Policy based on CVAT schema, stats, hardware, and availability."""
from __future__ import annotations

from cvat_tools.factory.registry.models import (
    CVATLabelDef,
    ModelRecord,
    ModelSelectionContext,
    SelectionCandidate,
    SelectionDecision,
)
from cvat_tools.factory.registry.store import ModelRegistryStore

KNOWN_GEOMETRIES = frozenset({
    "bbox",
    "rectangle",
    "polygon",
    "mask",
    "skeleton",
    "points",
    "polyline",
    "track",
    "tag",
})

_GEOM_ALIASES = {
    "rectangle": "bbox",
    "bbox": "bbox",
    "box": "bbox",
    "polygon": "polygon",
    "mask": "mask",
    "skeleton": "skeleton",
    "points": "points",
    "polyline": "polyline",
    "track": "track",
    "tag": "tag",
}


def _normalize_geom(g: str) -> str:
    return _GEOM_ALIASES.get(g.strip().lower(), g.strip().lower())


def _normalize_label(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


class ModelSelector:
    """Selects the best model for a CVAT job using schema, stats, hardware, and policy weights."""

    def __init__(self, store: ModelRegistryStore) -> None:
        self.store = store

    def select(self, context: ModelSelectionContext) -> SelectionDecision:
        # 1. Validate schema presence and known geometry types (never guess on unknown schema)
        if not context.labels:
            return SelectionDecision(
                status="inspect_required",
                selected_model_id=None,
                reason="CVAT label schema is empty; inspect task schema before selecting a model.",
            )

        norm_geoms = {_normalize_geom(lbl.geometry_type) for lbl in context.labels}
        unknown_geoms = {g for g in norm_geoms if g not in KNOWN_GEOMETRIES}
        if unknown_geoms:
            return SelectionDecision(
                status="inspect_required",
                selected_model_id=None,
                reason=f"Unknown geometry type(s) in schema: {sorted(unknown_geoms)}. Manual inspection required.",
            )

        # 2. Handle explicit user override
        if context.user_override_model_id:
            return self._handle_override(context, norm_geoms)

        # 3. Score all models in registry
        models = self.store.list_models(only_available=not context.policy.allow_unavailable_models)
        if not models:
            return SelectionDecision(
                status="needs_review",
                selected_model_id=None,
                reason="No available models found in registry.",
            )

        candidates: list[SelectionCandidate] = []
        rejected_reasons: list[str] = []

        for model in models:
            cand, reject = self._score_model(model, context, norm_geoms)
            if reject is not None:
                rejected_reasons.append(f"{model.model_id}: {reject}")
            elif cand is not None:
                candidates.append(cand)

        if not candidates:
            return SelectionDecision(
                status="needs_review",
                selected_model_id=None,
                reason="No registered model satisfies schema, hardware, and policy constraints.",
                warnings=tuple(rejected_reasons[:5]),
            )

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        best = candidates[0]

        # If best schema match is too weak (< 0.25), stop at inspect_required rather than guessing
        if best.schema_score < 0.25:
            return SelectionDecision(
                status="inspect_required",
                selected_model_id=None,
                reason=f"Schema match too low (best={best.model_id}, schema_score={best.schema_score:.2f}); manual review required.",
                candidates=tuple(candidates[:5]),
            )

        return SelectionDecision(
            status="selected",
            selected_model_id=best.model_id,
            reason="; ".join(best.reasons) if best.reasons else "Best policy score.",
            candidates=tuple(candidates[:5]),
            warnings=best.warnings,
        )

    def _handle_override(
        self,
        context: ModelSelectionContext,
        norm_geoms: set[str],
    ) -> SelectionDecision:
        override_id = context.user_override_model_id or ""
        model = self.store.get(override_id)
        if model is None:
            return SelectionDecision(
                status="needs_review",
                selected_model_id=None,
                reason=f"User override model '{override_id}' is not registered.",
                override_applied=False,
            )

        cand, reject = self._score_model(model, context, norm_geoms, ignore_policy_filters=True)
        warnings: list[str] = []
        if reject:
            warnings.append(reject)
        if cand and cand.schema_score < 0.25:
            warnings.append(f"Override model '{override_id}' has low schema compatibility ({cand.schema_score:.2f}).")
        if not model.is_available and not context.policy.allow_unavailable_models:
            warnings.append(f"Override model '{override_id}' weights are not locally available.")

        if warnings and not context.force_override:
            return SelectionDecision(
                status="needs_review",
                selected_model_id=None,
                reason=f"User override '{override_id}' triggered compatibility warnings; pass force_override=True to proceed.",
                candidates=(cand,) if cand else (),
                override_applied=False,
                warnings=tuple(warnings),
            )

        return SelectionDecision(
            status="overridden",
            selected_model_id=model.model_id,
            reason=f"User override applied: {model.model_id}",
            candidates=(cand,) if cand else (),
            override_applied=True,
            warnings=tuple(warnings),
        )

    def _score_model(
        self,
        model: ModelRecord,
        context: ModelSelectionContext,
        norm_geoms: set[str],
        *,
        ignore_policy_filters: bool = False,
    ) -> tuple[SelectionCandidate | None, str | None]:
        policy = context.policy
        hw = context.hardware
        ds = context.dataset_stats
        reasons: list[str] = []
        warnings: list[str] = []

        # Hardware & availability gates
        ram_limit = min(hw.ram_mb, policy.max_ram_mb)
        if model.ram_mb > ram_limit:
            if not ignore_policy_filters:
                return None, f"requires {model.ram_mb}MB RAM > budget {ram_limit}MB"
            warnings.append(f"RAM {model.ram_mb}MB exceeds budget {ram_limit}MB")

        if not model.cpu_supported and not hw.gpu_available:
            if not ignore_policy_filters:
                return None, "requires GPU but no GPU available"
            warnings.append("requires GPU")

        if policy.require_verified_benchmark and model.benchmark_status != "supported":
            if not ignore_policy_filters:
                return None, f"benchmark_status={model.benchmark_status} (verified required)"
            warnings.append(f"benchmark_status={model.benchmark_status}")

        if model.benchmark_status == "blocked":
            if not ignore_policy_filters:
                return None, "benchmark_status=blocked"
            warnings.append("benchmark_status=blocked")

        # 1. Schema compatibility score (geometry + skeleton + class overlap)
        model_geoms = {_normalize_geom(g) for g in model.expected_label_schema.geometry_types}
        if not model_geoms:
            geom_overlap = 0.0
        else:
            matched_geoms = norm_geoms & model_geoms
            geom_overlap = len(matched_geoms) / max(1, len(norm_geoms))

        if geom_overlap == 0.0 and not ignore_policy_filters:
            return None, f"geometry mismatch (need {sorted(norm_geoms)}, model has {sorted(model_geoms)})"

        # Skeleton topology check
        skeleton_labels = [lbl for lbl in context.labels if _normalize_geom(lbl.geometry_type) == "skeleton"]
        skel_score = 1.0
        if skeleton_labels:
            model_kps = {_normalize_label(k) for k in model.expected_label_schema.skeleton_keypoints}
            for sl in skeleton_labels:
                req_kps = {_normalize_label(k) for k in sl.keypoint_names}
                if req_kps and model_kps:
                    if len(req_kps) != len(model_kps):
                        if not ignore_policy_filters:
                            return None, f"skeleton keypoint count mismatch ({len(req_kps)} vs {len(model_kps)})"
                        skel_score = min(skel_score, 0.2)
                        warnings.append(f"skeleton keypoint count mismatch ({len(req_kps)} vs {len(model_kps)})")
                    else:
                        overlap = len(req_kps & model_kps) / len(req_kps)
                        skel_score = min(skel_score, overlap)
                elif req_kps and not model_kps:
                    # Model has no declared skeleton keypoints; cannot blindly match custom skeleton
                    skel_score = min(skel_score, 0.2)
                    warnings.append("model lacks skeleton_keypoints schema")

        # Class name overlap
        req_classes = {_normalize_label(lbl.name) for lbl in context.labels}
        model_classes = {_normalize_label(c) for c in model.expected_label_schema.classes}
        if model_classes:
            class_overlap = len(req_classes & model_classes) / max(1, len(req_classes))
        else:
            # Promptable / generic models (e.g. SAM2) or undeclared class list get partial credit
            class_overlap = 0.5 if "prompt-segment" in model.supported_tasks else 0.3

        schema_score = round(0.45 * geom_overlap + 0.35 * class_overlap + 0.20 * skel_score, 4)
        reasons.append(f"schema={schema_score:.2f} (geom={geom_overlap:.2f}, cls={class_overlap:.2f})")

        # 2. Quality score (benchmark status + input resolution fit for small objects)
        bench_bonus = {
            "supported": 1.0,
            "unverified": 0.5,
            "degraded": 0.25,
            "blocked": 0.0,
        }.get(model.benchmark_status, 0.4)

        res_fit = 0.7
        if ds.small_object_ratio >= 0.3:
            # Higher input resolution helps small objects
            max_in = max(model.input_size) if model.input_size else 640
            res_fit = min(1.0, max_in / 1024.0)
        quality_score = round(0.7 * bench_bonus + 0.3 * res_fit, 4)

        # 3. Speed score (CPU-friendly frameworks & lightweight models score higher)
        fw_speed = {
            "openvino": 1.0,
            "onnx": 0.9,
            "tflite": 0.9,
            "mediapipe": 0.85,
            "ultralytics": 0.75,
            "pytorch": 0.65,
            "huggingface": 0.55,
        }.get(model.framework.lower(), 0.5)
        light_bonus = 0.35 if not model.heavyweight else 0.1
        speed_score = round(min(1.0, 0.65 * fw_speed + light_bonus), 4)

        # 4. Memory score (lower RAM usage relative to budget scores higher)
        memory_score = round(max(0.0, 1.0 - (model.ram_mb / max(1, ram_limit))), 4)

        # Weighted combination
        w_sum = max(1e-6, policy.weight_quality + policy.weight_speed + policy.weight_memory)
        policy_score = (
            policy.weight_quality * quality_score
            + policy.weight_speed * speed_score
            + policy.weight_memory * memory_score
        ) / w_sum

        total_score = round(0.5 * schema_score + 0.5 * policy_score, 4)
        reasons.append(
            f"quality={quality_score:.2f}, speed={speed_score:.2f}, mem={memory_score:.2f}"
        )

        return (
            SelectionCandidate(
                model_id=model.model_id,
                total_score=total_score,
                schema_score=schema_score,
                quality_score=quality_score,
                speed_score=speed_score,
                memory_score=memory_score,
                reasons=tuple(reasons),
                warnings=tuple(warnings),
            ),
            None,
        )
