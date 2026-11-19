"""Unit tests for lazy ModelRegistry and single-heavyweight-model enforcement."""
from __future__ import annotations

from cvat_tools.core.model_registry import ModelRegistry, ModelSpec


def test_model_registry_from_config_does_not_load_weights():
    reg = ModelRegistry.from_config()
    ids = {m.model_id for m in reg.list()}
    assert {"yolo26n", "yolo26s", "yolo26l", "segformer-b2-cityscapes", "segformer-lane-marking", "sam2.1-tiny"}.issubset(ids)
    assert reg.status()["loaded"] == []


def test_single_heavyweight_model_unloads_previous():
    unloaded = []
    reg = ModelRegistry({
        "heavy_a": ModelSpec("heavy_a", "segment", "1", "onnx", "a.onnx", memory_mb=1200, heavyweight=True),
        "heavy_b": ModelSpec("heavy_b", "lane-segment", "1", "pytorch", "b.pt", memory_mb=900, heavyweight=True),
        "light_c": ModelSpec("light_c", "detect", "1", "onnx", "c.onnx", memory_mb=128, heavyweight=False),
    })
    reg.register_loader("heavy_a", lambda s: {"name": s.model_id}, unloader=lambda o: unloaded.append(o["name"]))
    reg.register_loader("heavy_b", lambda s: {"name": s.model_id}, unloader=lambda o: unloaded.append(o["name"]))
    reg.register_loader("light_c", lambda s: {"name": s.model_id}, unloader=lambda o: unloaded.append(o["name"]))

    reg.load("heavy_a")
    assert reg.is_loaded("heavy_a")
    reg.load("light_c")
    assert reg.is_loaded("heavy_a") and reg.is_loaded("light_c")
    reg.load("heavy_b")
    assert not reg.is_loaded("heavy_a") and reg.is_loaded("heavy_b") and reg.is_loaded("light_c")
    assert unloaded == ["heavy_a"]
    assert reg.unload_all() == 2
    assert reg.status()["loaded"] == []
