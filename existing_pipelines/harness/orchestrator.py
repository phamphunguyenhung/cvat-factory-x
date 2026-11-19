"""Harness Orchestrator V3: task-aware sequential CPU-first execution."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from harness.contracts import ComposedScene, FrameProfile, InstanceCandidate, StructureGeometry, StuffRegion
from harness.frame_profiler import profile_frame
from harness.label_router import LabelRouter
from harness.layer_composer import LayerComposer
from harness.model_registry import HarnessModelRegistry
from harness.review_queue import build_review_queue
from harness.structure_engine import StructureConfig, StructureEngine
from harness.stuff_engine import StuffConfig, StuffEngine
from harness.temporal_engine import TemporalAssociationEngine
from harness.thing_engine import ThingConfig, ThingEngine


class HarnessOrchestrator:
    """Coordinates FrameProfiler -> LabelRouter -> Thing/Stuff/Structure -> Temporal -> Composer."""

    def __init__(
        self,
        *,
        job_id: int = 285,
        mode: str = "balanced",
        thing_config: ThingConfig | None = None,
        stuff_config: StuffConfig | None = None,
        structure_config: StructureConfig | None = None,
        registry: HarnessModelRegistry | None = None,
    ):
        self.job_id = job_id
        self.mode = mode
        self.registry = registry or HarnessModelRegistry()
        self.router = LabelRouter(job_id=job_id)
        self.thing_engine = ThingEngine(self.registry, thing_config or self._default_thing_cfg(mode))
        self.stuff_engine = StuffEngine(self.registry, stuff_config or StuffConfig())
        self.structure_engine = StructureEngine(
            structure_config or StructureConfig(lane_shape_type=self.router.export_shape_for("lane/single white"))
        )
        self.temporal_engine = TemporalAssociationEngine()
        self.composer = LayerComposer()

    @staticmethod
    def _default_thing_cfg(mode: str) -> ThingConfig:
        if mode == "fast":
            return ThingConfig(
                detector_model="yolo26s_ov_960",
                mask_mode="none",
                use_horizon_tiles=True,
                use_edge_crops=False,
                use_night_enhance=False,
            )
        if mode == "high_recall":
            return ThingConfig(
                detector_model="yolo26l_ov_1280",
                mask_mode="sam_crop",
                use_horizon_tiles=True,
                use_edge_crops=True,
                use_night_enhance=True,
            )
        return ThingConfig(
            detector_model="yolo26l_ov_1280",
            mask_mode="none",
            use_horizon_tiles=True,
            use_edge_crops=True,
            use_night_enhance=True,
        )

    def run_batch(
        self,
        frames_bgr: dict[int, np.ndarray],
        *,
        run_things: bool = True,
        run_stuff: bool = True,
        run_structure: bool = True,
        enable_tracking: bool = True,
    ) -> list[ComposedScene]:
        ordered_ids = sorted(frames_bgr.keys())
        profiles: dict[int, FrameProfile] = {
            fid: profile_frame(frames_bgr[fid], fid, continuity_group=self.router.continuity_hint)
            for fid in ordered_ids
        }
        timings: dict[int, dict[str, float]] = {fid: {} for fid in ordered_ids}

        # Stage 1: Things (load detector ONCE -> run batch -> unload)
        things_by_f: dict[int, list[InstanceCandidate]] = {fid: [] for fid in ordered_ids}
        if run_things:
            det_handle = self.thing_engine._build_loaded_detector(self.thing_engine.config.detector_model)
            for fid in ordered_ids:
                t0 = time.perf_counter()
                things_by_f[fid] = self.thing_engine.detect_frame(
                    frames_bgr[fid], profiles[fid], detector=det_handle
                )
                timings[fid]["thing_detect"] = round(time.perf_counter() - t0, 4)
            self.registry.unload_all()
            self.thing_engine.refine_masks_batch(frames_bgr, things_by_f)

        # Stage 2: Stuff (load SegFormer ONCE -> run batch -> unload)
        stuffs_by_f: dict[int, list[StuffRegion]] = {fid: [] for fid in ordered_ids}
        sem_maps_by_f: dict[int, np.ndarray] = {}
        if run_stuff:
            for fid in ordered_ids:
                t0 = time.perf_counter()
                probs, meta = self.stuff_engine.infer_probs(frames_bgr[fid])
                sem_maps_by_f[fid] = np.argmax(probs, axis=0).astype(np.uint8)
                stuffs_by_f[fid] = self.stuff_engine.segment_frame(
                    frames_bgr[fid], profiles[fid], probs_and_meta=(probs, meta)
                )
                timings[fid]["stuff"] = round(time.perf_counter() - t0, 4)
            self.registry.unload_all()

        # Stage 3: Structure (pure CV + sem_map poles)
        structs_by_f: dict[int, list[StructureGeometry]] = {fid: [] for fid in ordered_ids}
        if run_structure:
            for fid in ordered_ids:
                t0 = time.perf_counter()
                road_polys = [
                    p
                    for sr in stuffs_by_f[fid]
                    if sr.label == "road"
                    for p in sr.simplified_polygons
                ]
                structs_by_f[fid] = self.structure_engine.extract_frame(
                    frames_bgr[fid],
                    profiles[fid],
                    road_polygons=road_polys,
                    things=things_by_f[fid],
                    sem_class_map=sem_maps_by_f.get(fid),
                )
                timings[fid]["structure"] = round(time.perf_counter() - t0, 4)

        # Stage 4: Conditional Temporal Association (gated by continuity)
        if enable_tracking and run_things and len(ordered_ids) >= 2:
            seq = [(fid, frames_bgr[fid]) for fid in ordered_ids]
            self.temporal_engine.associate(seq, things_by_f, continuity_hint=self.router.continuity_hint)

        # Stage 5: Panoptic Layer Composition & Review Queue
        scenes: list[ComposedScene] = []
        peak_mb = self.registry.peak_rss_mb
        for fid in ordered_ids:
            t0 = time.perf_counter()
            sc = self.composer.compose(
                job_id=self.job_id,
                profile=profiles[fid],
                things=things_by_f[fid],
                stuffs=stuffs_by_f[fid],
                structures=structs_by_f[fid],
                derive_areas=run_stuff,
            )
            timings[fid]["compose"] = round(time.perf_counter() - t0, 4)
            sc.timings_s = timings[fid]
            sc.peak_rss_mb = round(peak_mb, 1)
            build_review_queue(sc)
            scenes.append(sc)
        return scenes
