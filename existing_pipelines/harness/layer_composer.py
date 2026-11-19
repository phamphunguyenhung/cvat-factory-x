"""Panoptic Layer Composer: separates raw_mask vs visible_mask, enforces Z-order,
computes occlusion ratios, and derives drivable areas without leaking outside road.
"""
from __future__ import annotations

import cv2
import numpy as np

from harness.contracts import (
    ComposedScene,
    ElementStatus,
    FrameProfile,
    InstanceCandidate,
    LabelCategory,
    PanopticElement,
    RleMask,
    StructureGeometry,
    StuffRegion,
    decode_rle,
    encode_rle,
)
from harness.label_router import DEFAULT_Z_ORDER


def _poly_to_mask(points: list[float], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if len(points) >= 6:
        pts = np.round(np.array(points, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def _largest_poly(mask: np.ndarray, eps_px: float = 2.0, min_area: float = 180.0) -> list[float]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return []
    approx = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
    return [float(v) for v in approx.reshape(-1)] if len(approx) >= 3 else []


class LayerComposer:
    """Deterministic front-to-back panoptic visibility & occlusion composer."""

    def compose(
        self,
        *,
        job_id: int,
        profile: FrameProfile,
        things: list[InstanceCandidate],
        stuffs: list[StuffRegion],
        structures: list[StructureGeometry],
        derive_areas: bool = True,
    ) -> ComposedScene:
        h, w = profile.height, profile.width
        owner = np.full((h, w), -1, dtype=np.int32)
        elements: list[PanopticElement] = []

        # 1. Things (foreground priority: smaller/lower-y-bottom or higher-y2 closest to camera first)
        active_things = [t for t in things if t.status != ElementStatus.SUPPRESSED]
        active_things.sort(key=lambda t: (-t.bbox_xyxy[3], -t.area))

        for t in active_things:
            raw_m = decode_rle(t.mask_rle) if t.mask_rle is not None else np.zeros((h, w), dtype=np.uint8)
            if raw_m.shape != (h, w):
                raw_m = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = [int(round(v)) for v in t.bbox_xyxy]
                raw_m[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1

            raw_area = int(np.count_nonzero(raw_m))
            vis_m = (raw_m > 0) & (owner < 0)
            vis_area = int(np.count_nonzero(vis_m))
            occ_ratio = 1.0 - (vis_area / float(raw_area)) if raw_area > 0 else 0.0
            eid = t.candidate_id
            idx = len(elements)
            owner[vis_m] = idx
            elements.append(
                PanopticElement(
                    element_id=eid,
                    frame_id=profile.frame_id,
                    label=t.label,
                    category=LabelCategory.THING,
                    export_shape_type="rectangle",
                    points=list(t.bbox_xyxy),
                    z_order=0,
                    raw_mask_rle=t.mask_rle or encode_rle(raw_m),
                    visible_mask_rle=encode_rle(vis_m.astype(np.uint8)),
                    instance_id=t.candidate_id,
                    track_id=t.track_id,
                    occluded=bool(t.occluded or occ_ratio >= 0.15),
                    truncated=t.truncated,
                    occlusion_ratio=round(occ_ratio, 4),
                    conf=t.conf,
                    uncertainty=t.uncertainty,
                    status=t.status,
                    review_flags=[t.provenance["review_reason"]] if t.provenance.get("review_reason") else [],
                    provenance=dict(t.provenance),
                )
            )

        # 2. Structures (lanes, crosswalk, curb, poles)
        for st in structures:
            if st.status == ElementStatus.SUPPRESSED:
                continue
            raw_m = (
                decode_rle(st.mask_rle)
                if st.mask_rle is not None
                else (_poly_to_mask(st.points, h, w) if st.geometry_type == "polygon" else np.zeros((h, w), dtype=np.uint8))
            )
            vis_m = (raw_m > 0) & (owner < 0)
            idx = len(elements)
            owner[vis_m] = idx
            z = DEFAULT_Z_ORDER.get(st.label, 0)
            elements.append(
                PanopticElement(
                    element_id=st.structure_id,
                    frame_id=profile.frame_id,
                    label=st.label,
                    category=LabelCategory.STRUCTURE,
                    export_shape_type=st.geometry_type,
                    points=list(st.points),
                    z_order=z,
                    raw_mask_rle=encode_rle(raw_m),
                    visible_mask_rle=encode_rle(vis_m.astype(np.uint8)),
                    conf=st.conf,
                    status=st.status,
                    provenance=dict(st.provenance),
                )
            )

        # 3. Stuff layers ordered front-to-back: road (-1) -> sidewalk (-7) -> mid (-8) -> sky (-9)
        ordered_stuffs = sorted(stuffs, key=lambda s: -DEFAULT_Z_ORDER.get(s.label, -8))
        road_union = np.zeros((h, w), dtype=np.uint8)
        for sf in ordered_stuffs:
            if sf.status == ElementStatus.SUPPRESSED or not sf.simplified_polygons:
                continue
            raw_m = decode_rle(sf.raw_mask_rle)
            if sf.label == "road":
                road_union |= (raw_m > 0).astype(np.uint8)
            vis_m = (raw_m > 0) & (owner < 0)
            raw_area = int(np.count_nonzero(raw_m))
            vis_area = int(np.count_nonzero(vis_m))
            occ_ratio = 1.0 - (vis_area / float(raw_area)) if raw_area > 0 else 0.0
            idx = len(elements)
            owner[vis_m] = idx
            elements.append(
                PanopticElement(
                    element_id=sf.region_id,
                    frame_id=profile.frame_id,
                    label=sf.label,
                    category=LabelCategory.STUFF,
                    export_shape_type="polygon",
                    points=list(sf.simplified_polygons[0]),
                    z_order=DEFAULT_Z_ORDER.get(sf.label, -8),
                    raw_mask_rle=sf.raw_mask_rle,
                    visible_mask_rle=encode_rle(vis_m.astype(np.uint8)),
                    occlusion_ratio=round(occ_ratio, 4),
                    conf=sf.prob_mean,
                    uncertainty=sf.boundary_uncertainty,
                    status=sf.status,
                    provenance=dict(sf.provenance),
                )
            )

        # 4. Derived drivable areas strictly bounded inside validated road minus foreground obstacles
        if derive_areas and np.any(road_union):
            fg_block = np.zeros((h, w), dtype=np.uint8)
            for el in elements:
                if el.category == LabelCategory.THING and el.raw_mask_rle is not None:
                    fg_block |= decode_rle(el.raw_mask_rle)
            drivable = (road_union > 0) & (fg_block == 0)
            # Ego corridor heuristic centered at lower-center of road
            xs_grid = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
            ys_grid = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
            half_w = 0.14 * w + 0.22 * w * np.clip((ys_grid - 0.4 * h) / max(1.0, 0.6 * h), 0.0, 1.0)
            ego_corridor = np.abs(xs_grid - 0.5 * w) <= half_w
            direct_mask = (drivable & ego_corridor).astype(np.uint8)
            direct_poly = _largest_poly(direct_mask, eps_px=2.5, min_area=0.008 * h * w)
            if direct_poly:
                elements.append(
                    PanopticElement(
                        element_id=f"f{profile.frame_id}_area_direct",
                        frame_id=profile.frame_id,
                        label="area/direct",
                        category=LabelCategory.DERIVED_AREA,
                        export_shape_type="polygon",
                        points=direct_poly,
                        z_order=0,
                        raw_mask_rle=encode_rle(direct_mask),
                        visible_mask_rle=encode_rle(direct_mask),
                        conf=0.75 if not profile.is_night else 0.55,
                        status=ElementStatus.ACCEPTED if not profile.is_night else ElementStatus.REVIEW,
                        provenance={"derived_from": "road-fg_obstacles"},
                    )
                )

        return ComposedScene(job_id=job_id, frame_id=profile.frame_id, profile=profile, elements=elements)
