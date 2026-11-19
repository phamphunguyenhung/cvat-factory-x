"""CPU-first memory-guarded ModelRegistry wrapper for V3 Harness.

Guarantees lazy loading, strict single-heavyweight residency via explicit
unload()+gc.collect() before loading another heavyweight model, and RSS tracking.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import psutil

from cvat_tools.core import paths
from cvat_tools.core.model_registry import ModelRegistry, ModelSpec

COPILOT_ROOT = paths.REPO.parent / "tools" / "cvat-seg-copilot"


def current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)


def available_ram_mb() -> float:
    return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)


def _find_openvino_dir(prefix: str, imgsz: int) -> Path | None:
    ov_dir = paths.MODELS / "openvino"
    if not ov_dir.exists():
        return None
    matches = sorted(p for p in ov_dir.iterdir() if p.is_dir() and p.name.startswith(prefix) and f"_{imgsz}_" in p.name and "fp16" not in p.name)
    if not matches:
        matches = sorted(p for p in ov_dir.iterdir() if p.is_dir() and p.name.startswith(prefix) and f"_{imgsz}_" in p.name)
    return matches[0] if matches else None


def _unload_generic(instance: Any) -> None:
    """Release heavy references inside YOLO / SAM / ONNX / HF wrappers."""
    for attr in ("session", "sess", "model", "predictor", "ov_model"):
        if hasattr(instance, attr):
            try:
                setattr(instance, attr, None)
            except Exception:
                pass
    gc.collect()


class HarnessModelRegistry:
    """Single-heavyweight CPU model manager with automatic eviction and peak RSS tracking."""

    def __init__(self, min_free_ram_mb: float = 900.0):
        self.core = ModelRegistry.from_config()
        self.min_free_ram_mb = float(min_free_ram_mb)
        self.peak_rss_mb: float = current_rss_mb()
        self._register_v3_specs()

    def _touch_peak(self) -> float:
        rss = current_rss_mb()
        if rss > self.peak_rss_mb:
            self.peak_rss_mb = rss
        return rss

    def _register_spec(self, spec: ModelSpec, loader, unloader=_unload_generic) -> None:
        self.core._specs[spec.model_id] = spec
        self.core.register_loader(spec.model_id, loader, unloader)

    def _register_v3_specs(self) -> None:
        # 1. OpenVINO detectors (preferred fast CPU execution)
        ov_defs = [
            ("yolo26l_ov_1280", "yolo26l_", 1280, 1400, True),
            ("yolo26s_ov_960", "yolo26s_", 960, 600, False),
            ("yolo26s_ov_640", "yolo26s_", 640, 450, False),
            ("yolo26n_ov_640", "yolo26n_", 640, 250, False),
        ]
        for mid, prefix, sz, mem_mb, heavy in ov_defs:
            ov_path = _find_openvino_dir(prefix, sz)
            if ov_path is not None:
                self._register_spec(
                    ModelSpec(
                        model_id=mid,
                        task="detection",
                        version="26",
                        backend="openvino",
                        checkpoint=str(ov_path),
                        memory_mb=mem_mb,
                        imgsz=sz,
                        heavyweight=heavy,
                    ),
                    lambda spec, _sz=sz, _p=ov_path: self._load_ultralytics_yolo(_p, imgsz=_sz, task="detect"),
                )

        # 2. PyTorch fallback detectors
        for mid, pt_name, sz, heavy in [
            ("yolo26l_pt", "yolo26l.pt", 1280, True),
            ("yolo26s_pt", "yolo26s.pt", 960, False),
        ]:
            pt_path = paths.CHECKPOINTS / pt_name
            if pt_path.exists():
                self._register_spec(
                    ModelSpec(
                        model_id=mid,
                        task="detection",
                        version="26",
                        backend="ultralytics_pt",
                        checkpoint=str(pt_path),
                        memory_mb=1200 if heavy else 600,
                        imgsz=sz,
                        heavyweight=heavy,
                    ),
                    lambda spec, _sz=sz, _p=pt_path: self._load_ultralytics_yolo(_p, imgsz=_sz, task="detect"),
                )

        # 3. YOLO26s-seg instance segmentation
        yolo_seg = COPILOT_ROOT / "yolo26s-seg.pt"
        if yolo_seg.exists():
            self._register_spec(
                ModelSpec(
                    model_id="yolo26s_seg",
                    task="instance_seg",
                    version="26s-seg",
                    backend="ultralytics_pt",
                    checkpoint=str(yolo_seg),
                    memory_mb=950,
                    imgsz=960,
                    heavyweight=True,
                ),
                lambda spec, _p=yolo_seg: self._load_ultralytics_yolo(_p, imgsz=960, task="segment"),
            )

        # 4. Crop SAM models (MobileSAM / SAM 2.1 base)
        for mid, fname, mem_mb, heavy in [
            ("mobile_sam", "mobile_sam.pt", 550, False),
            ("sam2_1_b", "sam2.1_b.pt", 1300, True),
        ]:
            p = COPILOT_ROOT / fname
            if p.exists():
                self._register_spec(
                    ModelSpec(
                        model_id=mid,
                        task="prompted_mask",
                        version=fname,
                        backend="ultralytics_sam",
                        checkpoint=str(p),
                        memory_mb=mem_mb,
                        heavyweight=heavy,
                    ),
                    lambda spec, _p=p: self._load_ultralytics_sam(_p),
                )

        # 5. Semantic Stuff baseline (SegFormer-B2 ONNX)
        b2_path = COPILOT_ROOT / "models" / "segformer_b2_cityscapes.onnx"
        if b2_path.exists():
            self._register_spec(
                ModelSpec(
                    model_id="segformer_b2",
                    task="segmentation",
                    version="b2-cityscapes",
                    backend="onnxruntime",
                    checkpoint=str(b2_path),
                    memory_mb=1200,
                    heavyweight=True,
                ),
                lambda spec, _p=b2_path: self._load_onnx_seg(_p),
            )

        # 6. Controlled Challengers (local HuggingFace cache ONLY, never auto-download)
        self._register_spec(
            ModelSpec(
                model_id="mask2former_swin_l",
                task="segmentation",
                version="swin-large-cityscapes",
                backend="transformers",
                checkpoint="facebook/mask2former-swin-large-cityscapes-semantic",
                memory_mb=2200,
                heavyweight=True,
            ),
            lambda spec: self._load_hf_seg(spec.checkpoint, kind="mask2former"),
        )
        self._register_spec(
            ModelSpec(
                model_id="segformer_lane",
                task="lane_segmentation",
                version="8bits-lane",
                backend="transformers",
                checkpoint="8bits-ai/SegFormer-Lane-Marking",
                memory_mb=850,
                heavyweight=True,
            ),
            lambda spec: self._load_hf_seg(spec.checkpoint, kind="segformer"),
        )

    @staticmethod
    def _load_ultralytics_yolo(weights: Path, *, imgsz: int, task: str) -> dict[str, Any]:
        from ultralytics import YOLO

        model = YOLO(str(weights), task=task)
        return {"model": model, "imgsz": imgsz, "path": str(weights)}

    @staticmethod
    def _load_ultralytics_sam(weights: Path) -> dict[str, Any]:
        from ultralytics import SAM

        model = SAM(str(weights))
        return {"model": model, "path": str(weights)}

    @staticmethod
    def _load_onnx_seg(onnx_path: Path):
        from cvat_tools.segmentation.onnx_seg import OnnxSeg

        return OnnxSeg(model_id="b2", onnx_path=onnx_path)

    @staticmethod
    def _load_hf_seg(repo_id: str, *, kind: str) -> dict[str, Any]:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, Mask2FormerForUniversalSegmentation

        proc = AutoImageProcessor.from_pretrained(repo_id, local_files_only=True)
        cls = Mask2FormerForUniversalSegmentation if kind == "mask2former" else AutoModelForSemanticSegmentation
        mod = cls.from_pretrained(repo_id, local_files_only=True)
        mod.eval()
        return {"processor": proc, "model": mod, "kind": kind, "repo_id": repo_id}

    def available_models(self) -> list[str]:
        return sorted(self.core._specs.keys())

    def has_model(self, model_id: str) -> bool:
        return model_id in self.core._specs and model_id in self.core._loaders

    def load_only(self, model_id: str) -> Any:
        """Evict any loaded heavyweight models, verify free RAM, and lazy-load `model_id`."""
        self._touch_peak()
        spec = self.core._specs.get(model_id)
        if spec is None:
            raise KeyError(f"Unknown model_id {model_id!r}; available={self.available_models()}")
        if model_id in self.core._loaded:
            return self.core._loaded[model_id]

        # Enforce sequential single-heavyweight policy in tight CPU RAM
        for loaded_id in list(self.core._loaded.keys()):
            if loaded_id != model_id:
                self.core.unload(loaded_id)
        gc.collect()

        free_mb = available_ram_mb()
        if spec.heavyweight and free_mb < self.min_free_ram_mb:
            raise MemoryError(
                f"Refusing to load heavyweight model {model_id!r}: free RAM {free_mb:.0f} MB < {self.min_free_ram_mb:.0f} MB"
            )
        inst = self.core.load(model_id, allow_multi_heavy=False)
        self._touch_peak()
        return inst

    def unload_all(self) -> list[str]:
        unloaded = self.core.unload_all()
        gc.collect()
        self._touch_peak()
        return unloaded

    def status(self) -> dict[str, Any]:
        self._touch_peak()
        return {
            **self.core.status(),
            "current_rss_mb": round(current_rss_mb(), 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "available_ram_mb": round(available_ram_mb(), 1),
        }
