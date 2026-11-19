"""P0 Thing Engine: OpenVINO-first multi-scale detection, night glare filtering,
cross-class NMS, and conditional instance-mask separation (YOLO-seg / crop SAM).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cvat_tools.detection import all_objects as AO
from cvat_tools.detection.night_enhance import enhance_night_bgr, is_night_frame
from harness.contracts import (
    ElementStatus,
    FrameProfile,
    InstanceCandidate,
    RleMask,
    bbox_to_rle,
    encode_rle,
)
from harness.model_registry import HarnessModelRegistry

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "train", "motorcycle", "bicycle"})


@dataclass
class ThingConfig:
    detector_model: str = "yolo26l_ov_1280"
    mask_mode: str = "none"  # "none" | "yolo_seg" | "sam_crop" | "auto"
    sam_model: str = "mobile_sam"  # "mobile_sam" | "sam2_1_b"
    use_horizon_tiles: bool = True
    use_edge_crops: bool = True
    use_night_enhance: bool = True
    conf_day: float = 0.22
    conf_night: float = 0.14
    review_conf_floor: float = 0.08
    cross_class_nms_iou: float = 0.72
    filter_streetlamp_glare: bool = True
    ego_hood_y_ratio: float = 0.875


def _box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _is_achromatic_streetlamp_glare(img_bgr: np.ndarray, box: list[float], conf: float) -> bool:
    """Detect pure-white circular streetlamp halos misclassified as traffic lights at night."""
    if conf >= 0.45:
        return False
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(w, int(round(box[2])))
    y2 = min(h, int(round(box[3])))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return False
    crop = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    bright = v >= 225
    if not np.any(bright):
        return False
    bright_ratio = float(np.count_nonzero(bright)) / float(v.size)
    mean_sat_bright = float(s[bright].mean())
    aspect = float(x2 - x1) / float(max(1, y2 - y1))
    # Pure white round/square glare blob with negligible emitter saturation
    return bright_ratio >= 0.15 and mean_sat_bright < 38.0 and 0.65 <= aspect <= 1.55


def _is_ego_hood_artifact(box: list[float], height: int, label: str, y_ratio: float) -> bool:
    if label not in VEHICLE_CLASSES:
        return False
    yc = 0.5 * (box[1] + box[3])
    return yc > y_ratio * height and box[1] > 0.78 * height


def _is_truncated(box: list[float], width: int, height: int, margin: float = 4.0) -> bool:
    return (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= (width - margin)
        or box[3] >= (height - margin)
    )


class ThingEngine:
    """Object-First Thing detector & conditional instance mask refiner."""

    def __init__(self, registry: HarnessModelRegistry, config: ThingConfig | None = None):
        self.registry = registry
        self.config = config or ThingConfig()

    def _build_loaded_detector(self, model_id: str) -> AO.LoadedDetector:
        inst = self.registry.load_only(model_id)
        spec = self.registry.core._specs[model_id]
        return AO.LoadedDetector(
            _model=inst["model"],
            backend=spec.backend,
            model_name=model_id,
            weights_path=inst["path"],
            imgsz=int(inst.get("imgsz", spec.imgsz)),
        )

    def detect_frame(
        self,
        img_bgr: np.ndarray,
        profile: FrameProfile,
        *,
        detector: AO.LoadedDetector | None = None,
        accept_conf_override: float | None = None,
    ) -> list[InstanceCandidate]:
        """Run multi-scale detection on a single frame without touching Stuff models."""
        det_handle = detector or self._build_loaded_detector(self.config.detector_model)
        is_night = profile.is_night
        floor_conf = self.config.review_conf_floor
        accept_conf = (
            accept_conf_override
            if accept_conf_override is not None
            else (self.config.conf_night if is_night else self.config.conf_day)
        )

        # Primary multi-scale detection at review_conf_floor so borderline objects are preserved for review
        raw_dets = AO.detect_objects_multiscale(
            img_bgr,
            det_handle,
            is_night=is_night,
            conf_day=floor_conf,
            conf_night=floor_conf,
            use_horizon_tiles=self.config.use_horizon_tiles,
            use_edge_crops=self.config.use_edge_crops,
        )

        # Optional night CLAHE pass if frame is dark and primary missed dim parked cars
        if is_night and self.config.use_night_enhance:
            enh = enhance_night_bgr(img_bgr, mode="clahe")
            enh_dets = AO.detect_objects_multiscale(
                enh,
                det_handle,
                is_night=True,
                conf_day=floor_conf,
                conf_night=floor_conf,
                use_horizon_tiles=False,
                use_edge_crops=self.config.use_edge_crops,
            )
            for d in enh_dets:
                d.source_model = f"{d.source_model}+clahe"
            raw_dets = AO._class_aware_nms([*raw_dets, *enh_dets], iou_threshold=0.55)

        h, w = img_bgr.shape[:2]
        candidates: list[InstanceCandidate] = []
        for idx, d in enumerate(raw_dets):
            box = [float(v) for v in d.box]
            if _is_ego_hood_artifact(box, h, d.label, self.config.ego_hood_y_ratio):
                continue

            status = ElementStatus.ACCEPTED if d.conf >= accept_conf else ElementStatus.REVIEW
            review_reason = None if status == ElementStatus.ACCEPTED else "low_confidence"

            if (
                is_night
                and self.config.filter_streetlamp_glare
                and d.label == "traffic light"
                and _is_achromatic_streetlamp_glare(img_bgr, box, d.conf)
            ):
                status = ElementStatus.SUPPRESSED
                review_reason = "achromatic_streetlamp_glare"

            trunc = bool(d.truncated) or _is_truncated(box, w, h)
            candidates.append(
                InstanceCandidate(
                    candidate_id=f"f{profile.frame_id}_t{idx:03d}",
                    frame_id=profile.frame_id,
                    label=d.label,
                    bbox_xyxy=box,
                    conf=round(float(d.conf), 4),
                    source_model=d.source_model or self.config.detector_model,
                    mask_rle=None,
                    mask_source="none",
                    occluded=bool(d.occluded),
                    truncated=trunc,
                    uncertainty=round(max(0.0, 1.0 - float(d.conf)), 4),
                    status=status,
                    provenance={
                        "review_reason": review_reason,
                        "is_night": is_night,
                        "detector": self.config.detector_model,
                    },
                )
            )

        self._suppress_cross_class_vehicle_duplicates(candidates)
        return candidates

    def _suppress_cross_class_vehicle_duplicates(self, candidates: list[InstanceCandidate]) -> None:
        """Prevent emitting both `car` and `truck`/`bus` for the same physical vehicle."""
        veh = [c for c in candidates if c.label in VEHICLE_CLASSES and c.status != ElementStatus.SUPPRESSED]
        veh.sort(key=lambda c: c.conf, reverse=True)
        for i, hi in enumerate(veh):
            if hi.status == ElementStatus.SUPPRESSED:
                continue
            for lo in veh[i + 1 :]:
                if lo.status == ElementStatus.SUPPRESSED:
                    continue
                if hi.label == lo.label:
                    continue
                if _box_iou(hi.bbox_xyxy, lo.bbox_xyxy) >= self.config.cross_class_nms_iou:
                    lo.status = ElementStatus.SUPPRESSED
                    lo.duplicate_group = hi.candidate_id
                    lo.provenance["review_reason"] = f"cross_class_dup_of_{hi.label}"

    def refine_masks_batch(
        self,
        frames_bgr: dict[int, np.ndarray],
        candidates_by_frame: dict[int, list[InstanceCandidate]],
    ) -> None:
        """Attach instance masks sequentially after detector unload (never merge touching cars)."""
        mode = self.config.mask_mode
        if mode == "none":
            for fid, cands in candidates_by_frame.items():
                h, w = frames_bgr[fid].shape[:2]
                for c in cands:
                    if c.mask_rle is None:
                        c.mask_rle = bbox_to_rle(c.bbox_xyxy, h, w)
                        c.mask_source = "bbox_fallback"
            return

        if mode == "yolo_seg" and self.registry.has_model("yolo26s_seg"):
            seg_inst = self.registry.load_only("yolo26s_seg")
            seg_model = seg_inst["model"]
            for fid, cands in candidates_by_frame.items():
                img = frames_bgr[fid]
                h, w = img.shape[:2]
                res = seg_model.predict(img, imgsz=960, conf=0.12, verbose=False)[0]
                seg_masks: list[tuple[list[float], np.ndarray]] = []
                if res.masks is not None and res.boxes is not None:
                    mdata = res.masks.data.cpu().numpy()
                    bdata = res.boxes.xyxy.cpu().numpy()
                    for b, m in zip(bdata, mdata):
                        if m.shape != (h, w):
                            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        seg_masks.append((b.tolist(), (m > 0.5).astype(np.uint8)))
                for c in cands:
                    best_iou, best_m = 0.0, None
                    for sb, sm in seg_masks:
                        iou = _box_iou(c.bbox_xyxy, sb)
                        if iou > best_iou:
                            best_iou, best_m = iou, sm
                    if best_m is not None and best_iou >= 0.45:
                        c.mask_rle = encode_rle(best_m)
                        c.mask_source = "yolo_seg"
                    else:
                        c.mask_rle = bbox_to_rle(c.bbox_xyxy, h, w)
                        c.mask_source = "bbox_fallback"
            self.registry.unload_all()
            return

        # Crop-based SAM refinement ONLY for overlapping/ambiguous instances
        sam_id = self.config.sam_model if self.registry.has_model(self.config.sam_model) else "mobile_sam"
        if mode in ("sam_crop", "auto") and self.registry.has_model(sam_id):
            sam_inst = self.registry.load_only(sam_id)
            sam_model = sam_inst["model"]
            for fid, cands in candidates_by_frame.items():
                img = frames_bgr[fid]
                h, w = img.shape[:2]
                active = [c for c in cands if c.status != ElementStatus.SUPPRESSED]
                for idx, c in enumerate(active):
                    # Check if overlaps another active instance
                    neighbors = [
                        other for j, other in enumerate(active)
                        if j != idx and _box_iou(c.bbox_xyxy, other.bbox_xyxy) >= 0.12
                    ]
                    if not neighbors:
                        c.mask_rle = bbox_to_rle(c.bbox_xyxy, h, w)
                        c.mask_source = "bbox_fallback"
                        continue
                    # Crop with context margin and negative neighbor center prompts
                    bw = c.width
                    bh = c.height
                    cx1 = max(0, int(c.bbox_xyxy[0] - 0.35 * bw))
                    cy1 = max(0, int(c.bbox_xyxy[1] - 0.35 * bh))
                    cx2 = min(w, int(c.bbox_xyxy[2] + 0.35 * bw))
                    cy2 = min(h, int(c.bbox_xyxy[3] + 0.35 * bh))
                    crop = img[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        c.mask_rle = bbox_to_rle(c.bbox_xyxy, h, w)
                        c.mask_source = "bbox_fallback"
                        continue
                    rel_box = [
                        c.bbox_xyxy[0] - cx1,
                        c.bbox_xyxy[1] - cy1,
                        c.bbox_xyxy[2] - cx1,
                        c.bbox_xyxy[3] - cy1,
                    ]
                    neg_pts = []
                    for nb in neighbors[:3]:
                        nx = 0.5 * (nb.bbox_xyxy[0] + nb.bbox_xyxy[2]) - cx1
                        ny = 0.5 * (nb.bbox_xyxy[1] + nb.bbox_xyxy[3]) - cy1
                        if 0 <= nx < (cx2 - cx1) and 0 <= ny < (cy2 - cy1):
                            neg_pts.append([nx, ny])
                    try:
                        if neg_pts:
                            res = sam_model.predict(
                                crop,
                                bboxes=[rel_box],
                                points=neg_pts,
                                labels=[0] * len(neg_pts),
                                verbose=False,
                            )[0]
                        else:
                            res = sam_model.predict(crop, bboxes=[rel_box], verbose=False)[0]
                        if res.masks is not None and len(res.masks.data) > 0:
                            cmask = (res.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
                            full = np.zeros((h, w), dtype=np.uint8)
                            full[cy1:cy2, cx1:cx2] = cmask[: cy2 - cy1, : cx2 - cx1]
                            c.mask_rle = encode_rle(full)
                            c.mask_source = "sam_crop"
                            continue
                    except Exception:
                        pass
                    c.mask_rle = bbox_to_rle(c.bbox_xyxy, h, w)
                    c.mask_source = "bbox_fallback"
            self.registry.unload_all()
