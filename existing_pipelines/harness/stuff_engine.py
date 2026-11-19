"""P1 Stuff Engine: probability-first semantic masks, component preservation,
edge-aware smoothing, RLE internal representation, and clean polygon export.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cvat_tools.segmentation.onnx_seg import CITYSCAPES
from harness.contracts import ElementStatus, FrameProfile, StuffRegion, encode_rle
from harness.model_registry import HarnessModelRegistry

STUFF_SPECS: dict[str, tuple[float, int, float]] = {
    # label: (min_area_frac, max_components, poly_eps_scale)
    "road": (0.012, 2, 0.0028),
    "sidewalk": (0.006, 4, 0.0028),
    "building": (0.008, 8, 0.0032),
    "wall": (0.003, 4, 0.0030),
    "fence": (0.002, 4, 0.0030),
    "vegetation": (0.004, 8, 0.0022),
    "terrain": (0.004, 4, 0.0025),
    "sky": (0.010, 4, 0.0025),
}


@dataclass
class StuffConfig:
    model_id: str = "segformer_b2"
    min_prob_thresh: float = 0.35
    night_road_prob_thresh: float = 0.25
    smooth_probs: bool = True
    ego_hood_y_ratio: float = 0.89


def _softmax_chw(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(1e-6, np.sum(e, axis=0, keepdims=True))


def _mask_to_polygons(mask: np.ndarray, eps_px: float, min_area_px: float) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area_px:
            continue
        approx = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([float(v) for v in approx.reshape(-1)])
    return polys


class StuffEngine:
    """Probability-first semantic segmentation engine for continuous background classes."""

    def __init__(self, registry: HarnessModelRegistry, config: StuffConfig | None = None):
        self.registry = registry
        self.config = config or StuffConfig()

    def infer_probs(self, img_bgr: np.ndarray, *, model_id: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Run semantic segmenter via ModelRegistry and return (C, H, W) softmax probabilities."""
        mid = model_id or self.config.model_id
        inst = self.registry.load_only(mid)
        h, w = img_bgr.shape[:2]

        if hasattr(inst, "logits"):
            # OnnxSeg (SegFormer-B2)
            logits, meta = inst.logits(img_bgr)
            probs = _softmax_chw(logits)
            if probs.shape[1:] != (h, w):
                probs = np.stack(
                    [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs],
                    axis=0,
                )
            return probs, {"model": mid, **meta}

        # HuggingFace transformers fallback (Mask2Former / SegFormer-B5)
        import torch

        proc, mod = inst["processor"], inst["model"]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = proc(images=rgb, return_tensors="pt")
        with torch.no_grad():
            out = mod(**inputs)
        if inst.get("kind") == "mask2former":
            sem = proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0].cpu().numpy()
            # Convert hard semantic map to sharp one-hot probability proxy (19 Cityscapes classes)
            num_cls = max(19, int(sem.max()) + 1)
            probs = np.zeros((num_cls, h, w), dtype=np.float32)
            for c in range(min(num_cls, 19)):
                probs[c] = (sem == c).astype(np.float32)
            return probs, {"model": mid, "kind": "mask2former"}

        logits = torch.nn.functional.interpolate(
            out.logits, size=(h, w), mode="bilinear", align_corners=False
        )[0].cpu().numpy()
        return _softmax_chw(logits), {"model": mid, "kind": "hf_segformer"}

    def segment_frame(
        self,
        img_bgr: np.ndarray,
        profile: FrameProfile,
        *,
        probs_and_meta: tuple[np.ndarray, dict[str, Any]] | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> list[StuffRegion]:
        """Extract StuffRegions from probability maps without merging separate components."""
        probs, meta = probs_and_meta if probs_and_meta is not None else self.infer_probs(img_bgr)
        h, w = img_bgr.shape[:2]
        img_area = float(h * w)

        # Compute pixel-wise margin (top1 - top2) for boundary uncertainty estimation
        top2 = np.partition(probs, -2, axis=0)[-2:]
        margin_map = np.clip(top2[1] - top2[0], 0.0, 1.0)
        argmax_map = np.argmax(probs, axis=0)

        target_labels = labels or tuple(STUFF_SPECS.keys())
        regions: list[StuffRegion] = []
        hood_y = int(round(self.config.ego_hood_y_ratio * h))

        for lab in target_labels:
            if lab not in CITYSCAPES or lab not in STUFF_SPECS:
                continue
            cid = CITYSCAPES[lab]
            if cid >= probs.shape[0]:
                continue
            min_frac, max_comps, eps_scale = STUFF_SPECS[lab]
            min_area_px = min_frac * img_area
            p_map = probs[cid].copy()

            if self.config.smooth_probs:
                p_map = cv2.GaussianBlur(p_map, (5, 5), sigmaX=0.9)

            # Exclude ego car hood for road/sidewalk
            if lab in ("road", "sidewalk"):
                p_map[hood_y:, :] = 0.0

            thresh = (
                self.config.night_road_prob_thresh
                if (profile.is_night and lab == "road")
                else self.config.min_prob_thresh
            )
            # Combine probability threshold + argmax agreement (allows soft recovery near boundaries)
            binary = ((p_map >= thresh) & ((argmax_map == cid) | (p_map >= thresh + 0.12))).astype(np.uint8)

            # Light 3x3 open removes speck noise WITHOUT merging separate buildings
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            n_comp, comp_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            valid_ids = [
                i
                for i in sorted(range(1, n_comp), key=lambda idx: -stats[idx, cv2.CC_STAT_AREA])
                if stats[i, cv2.CC_STAT_AREA] >= min_area_px
            ][:max_comps]

            eps_px = max(1.5, eps_scale * float(min(h, w)))
            for comp_idx, cid_comp in enumerate(valid_ids):
                comp_mask = (comp_map == cid_comp).astype(np.uint8)
                polys = _mask_to_polygons(comp_mask, eps_px=eps_px, min_area_px=min_area_px)
                if not polys:
                    continue
                comp_probs = p_map[comp_mask > 0]
                prob_mean = float(comp_probs.mean())
                prob_p90 = float(np.percentile(comp_probs, 90))
                b_unc = float(1.0 - margin_map[comp_mask > 0].mean())
                status = ElementStatus.REVIEW if prob_mean < 0.48 else ElementStatus.ACCEPTED
                regions.append(
                    StuffRegion(
                        region_id=f"f{profile.frame_id}_s_{lab}_{comp_idx}",
                        frame_id=profile.frame_id,
                        label=lab,
                        raw_mask_rle=encode_rle(comp_mask),
                        prob_mean=round(prob_mean, 4),
                        prob_p90=round(prob_p90, 4),
                        boundary_uncertainty=round(b_unc, 4),
                        component_count=1,
                        simplified_polygons=polys,
                        source_model=str(meta.get("model", self.config.model_id)),
                        status=status,
                        provenance={"area_px": int(comp_mask.sum()), "poly_count": len(polys)},
                    )
                )

        return regions
