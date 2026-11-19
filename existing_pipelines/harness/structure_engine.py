"""Structure Engine: thin geometric markings (lanes, crosswalks, curbs, poles)
with glare-streak rejection, schema-aware geometry, and pole precision gating.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cvat_tools.lanes import framegen as LF
from cvat_tools.segmentation import pole_rule as PR
from cvat_tools.segmentation.onnx_seg import CITYSCAPES
from harness.contracts import ElementStatus, FrameProfile, InstanceCandidate, StructureGeometry, encode_rle


@dataclass
class StructureConfig:
    pole_max_width: float = 28.0
    pole_min_aspect: float = 3.8
    pole_min_height: float = 35.0
    max_poles_per_frame: int = 4
    reject_night_glare_lanes: bool = True
    lane_shape_type: str = "polygon"  # "polygon" | "polyline"


def _is_glare_streak(img_bgr: np.ndarray, points: list[float]) -> bool:
    """Reject bright headlight/wet-road glare streaks mistaken for lane paint."""
    if len(points) < 6:
        return False
    h, w = img_bgr.shape[:2]
    pts = np.round(np.array(points, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    if not np.any(mask):
        return True
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2][mask > 0]
    s = hsv[:, :, 1][mask > 0]
    # Pure saturated white/blooming glare streak without crisp paint edges
    return float((v >= 242).mean()) > 0.65 and float(s.mean()) < 22.0


class StructureEngine:
    """Specialized geometric branch for lanes, crosswalks, curbs, and poles."""

    def __init__(self, config: StructureConfig | None = None):
        self.config = config or StructureConfig()

    def extract_frame(
        self,
        img_bgr: np.ndarray,
        profile: FrameProfile,
        *,
        road_polygons: list[list[float]],
        things: list[InstanceCandidate],
        sem_class_map: np.ndarray | None = None,
    ) -> list[StructureGeometry]:
        h, w = img_bgr.shape[:2]
        out: list[StructureGeometry] = []
        thing_boxes = [t.bbox_xyxy for t in things if t.status != ElementStatus.SUPPRESSED]

        # 1. Lanes / crosswalks / curbs strictly constrained to validated road polygons
        if road_polygons:
            road_shapes = [{"label": "road", "type": "polygon", "points": p} for p in road_polygons]
            try:
                lane_res = LF.generate_frame(img_bgr, road_shapes, thing_boxes=thing_boxes)
                for idx, s in enumerate(lane_res.get("shapes", [])):
                    pts = [float(v) for v in s.get("points", [])]
                    if len(pts) < 6:
                        continue
                    lab = str(s.get("label", "lane/single white"))
                    if profile.is_night and self.config.reject_night_glare_lanes and _is_glare_streak(img_bgr, pts):
                        continue
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(m, [np.round(np.array(pts, dtype=np.float32).reshape(-1, 2)).astype(np.int32)], 1)
                    out.append(
                        StructureGeometry(
                            structure_id=f"f{profile.frame_id}_st_{idx}",
                            frame_id=profile.frame_id,
                            label=lab,
                            geometry_type=self.config.lane_shape_type if lab != "lane/crosswalk" else "polygon",
                            points=pts,
                            mask_rle=encode_rle(m),
                            conf=float(s.get("_conf", 0.65)),
                            continuity_score=0.8,
                            glare_risk=0.4 if profile.is_night else 0.05,
                            source_model="lanes_framegen_v3",
                            status=ElementStatus.ACCEPTED if not profile.is_night else ElementStatus.REVIEW,
                            provenance={"road_constrained": True},
                        )
                    )
            except Exception:
                pass

        # 2. Poles: strict aspect + height + anti-vegetation + top-N gating to prevent 9-FP explosion
        if sem_class_map is not None and sem_class_map.shape == (h, w):
            pole_mask = sem_class_map == CITYSCAPES["pole"]
            veg_mask = sem_class_map == CITYSCAPES["vegetation"]
            raw_poles = PR.extract_poles(
                pole_mask,
                veg_mask,
                thing_boxes,
                {
                    "max_width": self.config.pole_max_width,
                    "min_aspect": self.config.pole_min_aspect,
                    "min_height": self.config.pole_min_height,
                },
            )
            # Keep tallest / cleanest poles up to max_poles_per_frame
            raw_poles.sort(key=lambda r: float(r.get("height", 0.0)), reverse=True)
            for idx, r in enumerate(raw_poles[: self.config.max_poles_per_frame]):
                pts = [float(v) for v in r["polygon"]]
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [np.round(np.array(pts, dtype=np.float32).reshape(-1, 2)).astype(np.int32)], 1)
                out.append(
                    StructureGeometry(
                        structure_id=f"f{profile.frame_id}_pole_{idx}",
                        frame_id=profile.frame_id,
                        label="pole",
                        geometry_type="polygon",
                        points=pts,
                        mask_rle=encode_rle(m),
                        conf=0.55,
                        source_model="segformer_b2+pole_rule_v3",
                        status=ElementStatus.ACCEPTED if idx == 0 else ElementStatus.REVIEW,
                        provenance={"height_px": r.get("height")},
                    )
                )

        return out
