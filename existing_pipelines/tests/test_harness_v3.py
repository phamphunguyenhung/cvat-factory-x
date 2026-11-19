"""Unit and integration tests for Object-First Panoptic Harness V3."""
from __future__ import annotations

import numpy as np
import pytest

from cvat_tools.core.client import WriteNotAuthorized
from harness.contracts import (
    ElementStatus,
    InstanceCandidate,
    LabelCategory,
    StuffRegion,
    bbox_to_rle,
    decode_rle,
    encode_rle,
    rle_iou,
)
from harness.cvat_adapter import CvatAdapter
from harness.frame_profiler import assess_continuity, profile_frame
from harness.label_router import LabelRouter
from harness.layer_composer import LayerComposer
from harness.model_registry import HarnessModelRegistry
from harness.temporal_engine import TemporalAssociationEngine
from harness.thing_engine import ThingConfig, ThingEngine


def test_rle_roundtrip_and_iou():
    m = np.zeros((32, 48), dtype=np.uint8)
    m[4:12, 8:20] = 1
    rle = encode_rle(m)
    dec = decode_rle(rle)
    assert np.array_equal(m, dec)
    assert rle.area == 8 * 12
    assert rle_iou(rle, rle) == pytest.approx(1.0)


def test_frame_profiler_and_continuity_gate():
    dark = np.full((120, 160, 3), 25, dtype=np.uint8)
    bright = np.full((120, 160, 3), 140, dtype=np.uint8)
    p_dark = profile_frame(dark, 60)
    p_bright = profile_frame(bright, 61)
    assert p_dark.is_night is True
    assert p_bright.is_night is False

    rep_none = assess_continuity([(60, dark), (61, bright)], continuity_hint="none")
    assert rep_none.is_continuous is False
    assert "continuity=none" in rep_none.reason

    # Visually continuous sequence should pass when continuity_hint != none
    seq = [(i, np.clip(bright + i, 0, 255).astype(np.uint8)) for i in range(5)]
    rep_seq = assess_continuity(seq, continuity_hint="auto")
    assert rep_seq.is_continuous is True


def test_label_router_schema_and_z_order():
    router = LabelRouter(job_id=285)
    assert router.classify_label("car") == LabelCategory.THING
    assert router.classify_label("road") == LabelCategory.STUFF
    assert router.classify_label("lane/crosswalk") == LabelCategory.STRUCTURE
    assert router.classify_label("area/direct") == LabelCategory.DERIVED_AREA
    assert router.z_order_for("sky") == -9
    assert router.z_order_for("road") == -1
    assert router.export_shape_for("car") == "rectangle"


def test_model_registry_single_heavyweight_eviction(monkeypatch):
    reg = HarnessModelRegistry(min_free_ram_mb=1.0)
    unloaded_calls: list[str] = []

    from cvat_tools.core.model_registry import ModelSpec

    for name in ("heavy_a", "heavy_b"):
        reg._register_spec(
            ModelSpec(
                model_id=name,
                task="test",
                version="1",
                backend="mock",
                checkpoint="mock.bin",
                heavyweight=True,
            ),
            lambda spec, _n=name: {"name": _n},
            unloader=lambda inst, _n=name: unloaded_calls.append(_n),
        )

    reg.load_only("heavy_a")
    assert "heavy_a" in reg.core._loaded
    reg.load_only("heavy_b")
    assert "heavy_b" in reg.core._loaded
    assert "heavy_a" not in reg.core._loaded
    assert "heavy_a" in unloaded_calls


def test_thing_cross_class_nms_and_never_merge_touching_cars():
    reg = HarnessModelRegistry()
    eng = ThingEngine(reg, ThingConfig())
    c1 = InstanceCandidate("c1", 71, "car", [10.0, 20.0, 110.0, 120.0], 0.90, "ov")
    c_dup_truck = InstanceCandidate("c2", 71, "truck", [12.0, 21.0, 108.0, 118.0], 0.55, "ov")
    c_touching_car = InstanceCandidate("c3", 71, "car", [110.0, 20.0, 210.0, 120.0], 0.85, "ov")
    cands = [c1, c_dup_truck, c_touching_car]
    eng._suppress_cross_class_vehicle_duplicates(cands)
    assert c1.status == ElementStatus.ACCEPTED
    assert c_dup_truck.status == ElementStatus.SUPPRESSED
    assert c_touching_car.status == ElementStatus.ACCEPTED


def test_layer_composer_preserves_raw_and_computes_occlusion():
    prof = profile_frame(np.full((100, 100, 3), 120, dtype=np.uint8), 71)
    front_car = InstanceCandidate(
        "front", 71, "car", [20.0, 40.0, 60.0, 90.0], 0.95, "ov", mask_rle=bbox_to_rle([20, 40, 60, 90], 100, 100)
    )
    back_car = InstanceCandidate(
        "back", 71, "car", [20.0, 30.0, 60.0, 70.0], 0.85, "ov", mask_rle=bbox_to_rle([20, 30, 60, 70], 100, 100)
    )
    road_mask = np.zeros((100, 100), dtype=np.uint8)
    road_mask[45:95, 10:90] = 1
    road = StuffRegion(
        "road0",
        71,
        "road",
        encode_rle(road_mask),
        0.9,
        0.95,
        0.05,
        1,
        simplified_polygons=[[10.0, 45.0, 90.0, 45.0, 90.0, 95.0, 10.0, 95.0]],
    )
    scene = LayerComposer().compose(job_id=285, profile=prof, things=[front_car, back_car], stuffs=[road], structures=[])
    by_id = {e.element_id: e for e in scene.elements}
    assert by_id["front"].occlusion_ratio == pytest.approx(0.0)
    assert by_id["back"].occlusion_ratio > 0.5
    assert by_id["back"].occluded is True
    # Road raw mask stays intact even though cars sit in front
    assert by_id["road0"].raw_mask_rle.area == int(road_mask.sum())
    assert by_id["road0"].visible_mask_rle.area < by_id["road0"].raw_mask_rle.area


def test_temporal_engine_disabled_when_non_continuous():
    eng = TemporalAssociationEngine()
    dark = np.full((60, 80, 3), 15, dtype=np.uint8)
    bright = np.full((60, 80, 3), 160, dtype=np.uint8)
    rep, tracks = eng.associate([(60, dark), (61, bright)], {}, continuity_hint="none")
    assert rep.is_continuous is False
    assert tracks == []


def test_cvat_adapter_blocks_unconfirmed_remote_write():
    with pytest.raises(WriteNotAuthorized):
        CvatAdapter.push_remote_guarded(285, [], write_enabled=False, confirm_yes=False)
    with pytest.raises(WriteNotAuthorized):
        CvatAdapter.push_remote_guarded(285, [], write_enabled=True, confirm_yes=False)
