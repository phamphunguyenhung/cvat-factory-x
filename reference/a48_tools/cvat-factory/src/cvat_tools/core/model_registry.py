"""Lightweight, lazy-loading ModelRegistry with single-heavyweight-model enforcement (16GB RAM safe)."""
from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cvat_tools.core.config import load_models


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    task: str
    version: str
    backend: str
    checkpoint: str
    memory_mb: int = 512
    device: str = "cpu"
    imgsz: int = 640
    classes: tuple[str, ...] = ()
    license: str = "unknown"
    provenance: str = ""
    heavyweight: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        d["classes"] = list(self.classes)
        return d


class ModelRegistry:
    """Tracks declared models, lazy-loads on demand, and keeps at most one heavyweight model in RAM."""

    def __init__(self, specs: dict[str, ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = dict(specs or {})
        self._loaded: dict[str, Any] = {}
        self._loaders: dict[str, Callable[[ModelSpec], Any]] = {}
        self._unloaders: dict[str, Callable[[Any], None]] = {}
        self._active_heavy: str | None = None

    @classmethod
    def from_config(cls) -> "ModelRegistry":
        raw = load_models() or {}
        specs: dict[str, ModelSpec] = {}
        for mid, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            inf = cfg.get("inference") or {}
            backends = cfg.get("backends") or {}
            default_backend = next(iter(backends), cfg.get("backend", "pytorch"))
            specs[mid] = ModelSpec(
                model_id=mid,
                task=str(cfg.get("task", "detect")),
                version=str(cfg.get("version", mid)),
                backend=str(default_backend),
                checkpoint=str(cfg.get("checkpoint", "")),
                memory_mb=int(cfg.get("memory_mb", 512)),
                device=str(cfg.get("device", "cpu")),
                imgsz=int(inf.get("imgsz", 640)),
                classes=tuple(cfg.get("classes") or ()),
                license=str(cfg.get("license", "unknown")),
                provenance=str(cfg.get("source", "")),
                heavyweight=bool(cfg.get("heavyweight", int(cfg.get("memory_mb", 512)) >= 800)),
                raw=cfg,
            )
        return cls(specs)

    def register_loader(
        self,
        model_id: str,
        loader: Callable[[ModelSpec], Any],
        unloader: Callable[[Any], None] | None = None,
    ) -> None:
        self._loaders[model_id] = loader
        if unloader is not None:
            self._unloaders[model_id] = unloader

    def list(self) -> list[ModelSpec]:
        return [self._specs[k] for k in sorted(self._specs)]

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._specs:
            raise KeyError(f"Khong tim thay model '{model_id}'. Co: {sorted(self._specs)}")
        return self._specs[model_id]

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def load(self, model_id: str, *, allow_multi_heavy: bool = False) -> Any:
        spec = self.get(model_id)
        if model_id in self._loaded:
            return self._loaded[model_id]
        if spec.heavyweight and not allow_multi_heavy and self._active_heavy and self._active_heavy != model_id:
            self.unload(self._active_heavy)
        if model_id not in self._loaders:
            raise NotImplementedError(f"Chua dang ky loader cho model '{model_id}'.")
        obj = self._loaders[model_id](spec)
        self._loaded[model_id] = obj
        if spec.heavyweight:
            self._active_heavy = model_id
        return obj

    def unload(self, model_id: str) -> bool:
        obj = self._loaded.pop(model_id, None)
        if obj is None:
            return False
        unloader = self._unloaders.get(model_id)
        if unloader is not None:
            unloader(obj)
        if self._active_heavy == model_id:
            self._active_heavy = None
        gc.collect()
        return True

    def unload_all(self) -> int:
        ids = list(self._loaded)
        for mid in ids:
            self.unload(mid)
        return len(ids)

    def status(self) -> dict[str, Any]:
        return {
            "declared": len(self._specs),
            "loaded": sorted(self._loaded),
            "active_heavy": self._active_heavy,
        }
