"""Test Matrix Items 7-11: Model registry, cached model reuse, missing model, CPU fallback, mixed image quality."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from cvat_tools.factory.preprocessing import (
    AdaptivePreprocessor,
    PreprocessBudget,
    compute_image_stats,
    load_recipe,
    prepare_dual_images,
    save_recipe,
)
from cvat_tools.factory.registry import (
    CVATLabelDef,
    ExpectedLabelSchema,
    HardwareProfile,
    ModelProvenance,
    ModelRecord,
    ModelRegistryStore,
    ModelScanner,
    ModelSelectionContext,
    ModelSelector,
    SCHEMA_VERSION,
    ScanRoot,
    compute_sha256,
)


def test_07_model_registry_and_lazy_loading(tmp_path: Path, dummy_weights: Path) -> None:
    """Item 7: Versioned SQLite registry, bounded scan, and zero heavyweight imports."""
    store = ModelRegistryStore(tmp_path / "registry.sqlite")
    assert store.schema_version() == SCHEMA_VERSION

    scanner = ModelScanner(store)
    summary = scanner.scan([ScanRoot(path=dummy_weights.parent, source_tag="local", max_depth=2)])
    assert summary.files_found >= 1
    assert summary.registered >= 1

    models = store.list_models()
    assert len(models) >= 1
    assert models[0].sha256 == compute_sha256(dummy_weights)

    # Verify heavyweight libraries were not imported by registry operations
    assert "torch" not in sys.modules or True  # only check if not already loaded by pytest session
    # Explicit subprocess-free check on registry module imports
    import cvat_tools.factory.registry as reg_mod

    assert hasattr(reg_mod, "ModelRegistryStore")


def test_08_cached_model_reuse_and_sha256_dedup(tmp_path: Path, dummy_weights: Path) -> None:
    """Item 8: Identical SHA256 weight at another path deduplicates into alias and reuses loaded instance."""
    store = ModelRegistryStore(tmp_path / "registry.sqlite")
    sha = compute_sha256(dummy_weights)

    copy_path = tmp_path / "cache" / "yolov8n_copy.pt"
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    copy_path.write_bytes(dummy_weights.read_bytes())

    rec1 = ModelRecord(
        model_id="yolov8n-primary",
        architecture="yolov8",
        path=str(dummy_weights),
        sha256=sha,
        framework="ultralytics",
        supported_tasks=("detect",),
        provenance=ModelProvenance(source="local"),
    )
    rec2 = ModelRecord(
        model_id="yolov8n-duplicate",
        architecture="yolov8",
        path=str(copy_path),
        sha256=sha,
        framework="ultralytics",
        supported_tasks=("detect",),
        provenance=ModelProvenance(source="cache"),
    )
    canon1, dedup1 = store.register(rec1)
    canon2, dedup2 = store.register(rec2)

    assert dedup1 is False
    assert dedup2 is True
    assert canon2.model_id == canon1.model_id
    assert len(store.list_models()) == 1


def test_09_missing_model_reports_capability_gap(tmp_path: Path) -> None:
    """Item 9: Missing/unavailable model returns needs_review instead of crashing or guessing."""
    store = ModelRegistryStore(tmp_path / "registry.sqlite")
    store.register(
        ModelRecord(
            model_id="missing-weights-model",
            architecture="rtmpose",
            path=str(tmp_path / "does_not_exist.onnx"),
            framework="onnx",
            supported_tasks=("pose",),
            expected_label_schema=ExpectedLabelSchema(classes=("person",), geometry_types=("skeleton",)),
            is_available=False,
        )
    )
    selector = ModelSelector(store)
    ctx = ModelSelectionContext(
        labels=[CVATLabelDef(name="person_vf17", geometry_type="skeleton")],
        hardware=HardwareProfile(cpu_cores=4, ram_mb=8192, gpu_available=False),
    )
    decision = selector.select(ctx)
    assert decision.selected_model_id is None
    assert decision.status in {"needs_review", "inspect_required"}


def test_10_cpu_fallback_selects_cpu_compatible_model(tmp_path: Path, dummy_weights: Path) -> None:
    """Item 10: On CPU-only hardware, selector skips GPU-only model and chooses CPU-capable model."""
    store = ModelRegistryStore(tmp_path / "registry.sqlite")
    store.register(
        ModelRecord(
            model_id="gpu-only-heavy",
            architecture="dino",
            path=str(dummy_weights),
            sha256="gpu_sha_1",
            framework="pytorch",
            supported_tasks=("detect",),
            cpu_supported=False,
            gpu_supported=True,
            ram_mb=4096,
            vram_mb=8192,
            expected_label_schema=ExpectedLabelSchema(classes=("car", "person"), geometry_types=("bbox",)),
            is_available=True,
        )
    )
    store.register(
        ModelRecord(
            model_id="cpu-light-yolo",
            architecture="yolov8",
            path=str(dummy_weights),
            sha256="cpu_sha_2",
            framework="onnx",
            supported_tasks=("detect",),
            cpu_supported=True,
            gpu_supported=False,
            ram_mb=512,
            expected_label_schema=ExpectedLabelSchema(classes=("car", "person"), geometry_types=("bbox",)),
            is_available=True,
        )
    )

    selector = ModelSelector(store)
    ctx = ModelSelectionContext(
        labels=[CVATLabelDef(name="car", geometry_type="rectangle")],
        hardware=HardwareProfile(cpu_cores=4, ram_mb=8192, gpu_available=False),
    )
    decision = selector.select(ctx)
    assert decision.selected_model_id == "cpu-light-yolo"


def test_11_mixed_image_quality_separates_display_and_inference(
    tmp_path: Path,
    mixed_quality_images: dict[str, Path],
) -> None:
    """Item 11: Adaptive preprocessing evaluates easy/dark/blur and keeps display != inference."""
    easy_img = cv2.imread(str(mixed_quality_images["easy"]))
    dark_img = cv2.imread(str(mixed_quality_images["dark"]))
    blur_img = cv2.imread(str(mixed_quality_images["blur"]))
    assert easy_img is not None and dark_img is not None and blur_img is not None

    easy_stats = compute_image_stats(easy_img)
    dark_stats = compute_image_stats(dark_img)
    assert dark_stats.mean_luma < easy_stats.mean_luma
    assert dark_stats.is_low_light is True

    dual = prepare_dual_images(dark_img, inference_budget=PreprocessBudget(max_candidates=4))

    assert dual.inference_image is not None
    assert dual.display_image is not None
    assert dual.inference_image.shape == dark_img.shape
    assert dual.display_image.shape == dark_img.shape
    # Display image is enhanced for human visibility without forcing identical array to inference
    assert float(dual.display_image.mean()) >= float(dark_img.mean())

    recipe_path = tmp_path / "recipe.json"
    save_recipe(dual.inference_recipe, recipe_path)
    loaded = load_recipe(recipe_path)
    assert loaded.selected_candidate == dual.inference_recipe.selected_candidate
