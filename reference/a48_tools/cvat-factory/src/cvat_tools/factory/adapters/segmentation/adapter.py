"""Unified SegmentationAdapter wrapping cvat_tools.segmentation.{onnx_seg,sam_refine,geomrefine,layers,stuff_things} lazily."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from cvat_tools.factory.adapters.segmentation.rules import (
    canonical_label,
    compute_mask_readback_iou,
    layer_z_order,
    mask_to_flat_polygon,
    validate_frame_shared_boundaries,
    validate_segmentation_proposal,
)
from cvat_tools.factory.core.adapter import AnnotationAdapter
from cvat_tools.factory.core.models import (
    AdapterContext,
    EvidencePacket,
    ImageInput,
    ModelSelection,
    PreprocessedBatch,
    Proposal,
    Provenance,
    QualityVerdict,
    RawPrediction,
    RevisionStage,
    TaskKind,
    TaskSchema,
    ValidationReport,
)


def _hash_image_input(img: ImageInput) -> str:
    if img.image_hash:
        return img.image_hash
    if img.image_bytes is not None:
        return hashlib.sha256(img.image_bytes).hexdigest()
    if img.image_path:
        p = Path(img.image_path)
        if p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(f"frame:{img.frame_id}".encode("utf-8")).hexdigest()


def _load_bgr(item: dict[str, Any]) -> Any | None:
    if item.get("image_array") is not None:
        return item["image_array"]
    img_bytes = item.get("image_bytes")
    img_path = item.get("image_path")
    if img_bytes is None and not img_path:
        return None
    import cv2
    import numpy as np

    if img_bytes is not None:
        return cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    p = Path(str(img_path))
    return cv2.imread(str(p), cv2.IMREAD_COLOR) if p.is_file() else None


class SegmentationAdapter(AnnotationAdapter):
    """Unified semantic/instance/mask/polygon Segmentation adapter."""

    adapter_name = "segmentation"

    def supports(self, task_schema: TaskSchema) -> bool:
        if task_schema.task_kind == TaskKind.SEGMENTATION:
            return True
        if task_schema.task_kind in (TaskKind.SKELETON, TaskKind.DETECTION, TaskKind.TRACKING):
            return False
        if any(t in task_schema.annotation_types for t in ("polygon", "mask")):
            return True
        return any(lbl.type in ("polygon", "mask") for lbl in task_schema.labels)

    def select_model(self, context: AdapterContext) -> ModelSelection:
        if context.user_model_override:
            return ModelSelection(
                model_id=context.user_model_override,
                supported=True,
                reason="User model override selected for segmentation",
                capabilities={"predict": True, "validate": True, "refine": True, "mask_readback": True},
            )

        for m in context.available_models:
            task = str(m.get("supported_task", "")).lower()
            name = str(m.get("name") or m.get("model_id") or "")
            path_str = str(m.get("path", ""))
            if any(k in task for k in ("seg", "mask", "polygon")) or any(
                k in name.lower() for k in ("segformer", "mask2former", "sam", "-seg")
            ):
                if not path_str or Path(path_str).exists():
                    return ModelSelection(
                        model_id=str(m.get("model_id") or name),
                        supported=True,
                        reason="Matched available segmentation model from registry",
                        capabilities={"predict": True, "validate": True, "refine": True, "mask_readback": True},
                    )

        # Check local ONNX / SAM weights without loading them
        from cvat_tools.core import paths

        onnx_b2 = paths.COPILOT / "models" / "segformer_b2_cityscapes.onnx"
        if onnx_b2.is_file():
            return ModelSelection(
                model_id="segformer-b2-onnx",
                supported=True,
                reason="Found local SegFormer-B2 ONNX model",
                capabilities={"predict": True, "validate": True, "refine": True, "mask_readback": True},
            )

        return ModelSelection(
            model_id=None,
            supported=False,
            reason="No local segmentation model weights found (e.g. segformer_b2_cityscapes.onnx missing); validation & refinement remain available",
            capabilities={"predict": False, "validate": True, "refine": True, "mask_readback": True},
            fallback_available=False,
        )

    def preprocess(
        self,
        images: Sequence[ImageInput],
        context: AdapterContext | None = None,
    ) -> PreprocessedBatch:
        cfg = dict(context.config) if context is not None else {}
        items: list[dict[str, Any]] = []
        display_paths: dict[int, str] = {}
        for img in images:
            if img.image_path:
                display_paths[img.frame_id] = img.image_path
            items.append(
                {
                    "frame_id": img.frame_id,
                    "image_path": img.image_path,
                    "image_bytes": img.image_bytes,
                    "image_hash": _hash_image_input(img),
                    "width": img.width,
                    "height": img.height,
                    "metadata": dict(img.metadata),
                }
            )
        return PreprocessedBatch(
            items=items,
            preprocess_params={
                "max_points": int(cfg.get("max_points", 40)),
                "simplify_eps": float(cfg.get("simplify_eps", 1.0)),
                "predictor": cfg.get("predictor"),
            },
            display_paths=display_paths,
        )

    def predict(
        self,
        batch: PreprocessedBatch,
        model_selection: ModelSelection | None = None,
    ) -> list[RawPrediction]:
        if model_selection is not None and not model_selection.supported:
            return []

        predictor = batch.preprocess_params.get("predictor")
        model_id = (model_selection.model_id if model_selection else None) or "segformer-b2-onnx"

        if predictor is None:
            try:
                from cvat_tools.segmentation.onnx_seg import CITYSCAPES, OnnxSeg
                seg = OnnxSeg("b2")
                id_to_name = {v: k for k, v in CITYSCAPES.items()}
                predictor = ("onnx_seg", seg, id_to_name)
            except Exception:
                return []

        predictions: list[RawPrediction] = []
        for item in batch.items:
            bgr = _load_bgr(item)
            if bgr is None and not callable(predictor):
                continue
            h, w = (bgr.shape[:2] if bgr is not None else (int(item.get("height") or 0), int(item.get("width") or 0)))

            if callable(predictor):
                raw_items = predictor(bgr, item)
                for pred_item in raw_items:
                    predictions.append(
                        RawPrediction(
                            frame_id=int(item["frame_id"]),
                            label=str(pred_item["label"]),
                            data={
                                "polygon": pred_item.get("polygon"),
                                "mask": pred_item.get("mask"),
                                "image_width": w,
                                "image_height": h,
                            },
                            confidence=float(pred_item.get("confidence", 0.8)),
                            model_id=model_id,
                            image_hash=str(item.get("image_hash", "")),
                            preprocess_params=dict(batch.preprocess_params),
                        )
                    )
            else:
                _, seg, id_to_name = predictor
                cm, info = seg.class_map(bgr)
                import numpy as np
                for cid in np.unique(cm):
                    cname = id_to_name.get(int(cid))
                    if cname is None:
                        continue
                    mask = (cm == int(cid)).astype(np.uint8)
                    predictions.append(
                        RawPrediction(
                            frame_id=int(item["frame_id"]),
                            label=cname,
                            data={"mask": mask, "image_width": w, "image_height": h},
                            confidence=0.80,
                            model_id=model_id,
                            image_hash=str(item.get("image_hash", "")),
                            preprocess_params=info,
                        )
                    )
        return predictions

    def convert_to_cvat(
        self,
        predictions: Sequence[RawPrediction],
        task_schema: TaskSchema,
        run_id: str = "",
    ) -> list[Proposal]:
        from cvat_tools.segmentation.geomrefine import kind_for_label, refine as geom_refine

        proposals: list[Proposal] = []
        for pred in predictions:
            canon = canonical_label(pred.label)
            lbl = task_schema.label_by_name(canon) or task_schema.label_by_name(pred.label)
            final_label = lbl.name if lbl is not None else canon
            label_id = lbl.id if lbl is not None else None

            poly = pred.data.get("polygon")
            mask = pred.data.get("mask")
            decisions: list[dict[str, Any]] = []
            readback_iou: float | None = None

            if poly is None and mask is not None:
                poly = mask_to_flat_polygon(mask)
                decisions.append({"rule": "mask_to_polygon"})
            if poly is None or len(poly) < 6:
                continue

            kind = kind_for_label(final_label)
            refined_poly = geom_refine(poly, kind=kind)
            if refined_poly and len(refined_poly) >= 6:
                poly = refined_poly
                decisions.append({"rule": "geomrefine_simplify", "kind": kind})

            w = pred.data.get("image_width") or task_schema.image_width
            h = pred.data.get("image_height") or task_schema.image_height
            if mask is not None:
                readback_iou = compute_mask_readback_iou(poly, mask, width=w, height=h)
                decisions.append({"rule": "mask_readback_iou", "iou": readback_iou})

            z_ord = layer_z_order(final_label)
            proposals.append(
                Proposal(
                    job_id=task_schema.job_id or 0,
                    frame_id=pred.frame_id,
                    label=final_label,
                    shape_type="polygon",
                    geometry={"points": [round(float(v), 2) for v in poly], "label_id": label_id},
                    stage=RevisionStage.MODEL_PROPOSAL,
                    confidence=pred.confidence,
                    z_order=z_ord,
                    provenance=Provenance(
                        model_id=pred.model_id,
                        image_hash=pred.image_hash,
                        preprocess_params=dict(pred.preprocess_params),
                        confidence=pred.confidence,
                        rule_decisions=decisions,
                        run_id=run_id,
                        adapter_name=self.adapter_name,
                    ),
                    metadata={
                        "image_width": w,
                        "image_height": h,
                        "mask_readback_iou": readback_iou,
                    },
                )
            )
        return proposals

    def validate(
        self,
        proposals: Sequence[Proposal],
        context: dict[str, Any] | None = None,
    ) -> list[ValidationReport]:
        ctx = context or {}
        task_schema: TaskSchema | None = ctx.get("task_schema")
        img_w = ctx.get("image_width") or (task_schema.image_width if task_schema else None)
        img_h = ctx.get("image_height") or (task_schema.image_height if task_schema else None)
        min_area = float(ctx.get("min_area", 4.0))
        min_conf = float(ctx.get("min_conf", 0.25))

        reports_map: dict[str, ValidationReport] = {}
        ordered_reports: list[ValidationReport] = []
        for p in proposals:
            rep = validate_segmentation_proposal(
                p,
                task_schema=task_schema,
                image_width=img_w or p.metadata.get("image_width"),
                image_height=img_h or p.metadata.get("image_height"),
                min_area=min_area,
                min_conf=min_conf,
            )
            reports_map[p.proposal_id] = rep
            ordered_reports.append(rep)

        if ctx.get("check_shared_boundaries", True):
            validate_frame_shared_boundaries(proposals, reports_map)
        return ordered_reports

    def refine(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Proposal]:
        from cvat_tools.segmentation.geomrefine import kind_for_label, refine as geom_refine, valid_polygon

        ctx = context or {}
        task_schema: TaskSchema | None = ctx.get("task_schema")
        img_w = ctx.get("image_width") or (task_schema.image_width if task_schema else None)
        img_h = ctx.get("image_height") or (task_schema.image_height if task_schema else None)
        report_by_id = {r.proposal_id: r for r in (validation_reports or ())}

        out: list[Proposal] = []
        for p in proposals:
            rep = report_by_id.get(p.proposal_id)
            if rep is not None and rep.verdict != QualityVerdict.REFINE:
                out.append(p)
                continue

            pts = list(p.geometry.get("points") or [])
            if len(pts) < 6:
                out.append(p)
                continue

            poly_obj = valid_polygon(pts)
            if poly_obj is None:
                out.append(p)
                continue

            w = img_w or p.metadata.get("image_width")
            h = img_h or p.metadata.get("image_height")
            if w is not None and h is not None:
                from shapely.geometry import box as sbox
                clipped = poly_obj.intersection(sbox(0, 0, float(w), float(h)))
                poly_obj = valid_polygon(clipped) or poly_obj

            flat = [round(float(v), 2) for xy in list(poly_obj.exterior.coords)[:-1] for v in xy]
            refined_flat = geom_refine(flat, kind=kind_for_label(p.label))
            if refined_flat and len(refined_flat) >= 6:
                flat = refined_flat

            new_z = layer_z_order(p.label)
            new_prov = Provenance(
                model_id=p.provenance.model_id,
                image_hash=p.provenance.image_hash,
                preprocess_params=dict(p.provenance.preprocess_params),
                confidence=p.confidence,
                rule_decisions=[*p.provenance.rule_decisions, {"rule": "bounded_seg_refine", "z_order": new_z}],
                run_id=p.provenance.run_id,
                adapter_name=self.adapter_name,
            )
            out.append(
                Proposal(
                    job_id=p.job_id,
                    frame_id=p.frame_id,
                    label=p.label,
                    shape_type=p.shape_type,
                    geometry={**p.geometry, "points": flat},
                    provenance=new_prov,
                    stage=RevisionStage.REFINED_PROPOSAL,
                    confidence=p.confidence,
                    cvat_shape_id=p.cvat_shape_id,
                    parent_proposal_id=p.proposal_id,
                    z_order=new_z,
                    occluded=p.occluded,
                    outside=p.outside,
                    attributes=list(p.attributes),
                    metadata=dict(p.metadata),
                )
            )
        return out

    def render_evidence(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        output_dir: str | Path | None = None,
    ) -> list[EvidencePacket]:
        report_by_id = {r.proposal_id: r for r in (validation_reports or ())}
        by_frame: dict[int, list[Proposal]] = {}
        for p in proposals:
            rep = report_by_id.get(p.proposal_id)
            if rep is None or rep.verdict != QualityVerdict.PASS:
                by_frame.setdefault(p.frame_id, []).append(p)

        packets: list[EvidencePacket] = []
        for frame_id, frame_props in sorted(by_frame.items()):
            issues = []
            regions: list[str] = []
            verdict = QualityVerdict.REFINE
            for p in frame_props:
                rep = report_by_id.get(p.proposal_id)
                if rep:
                    issues.extend(rep.issues)
                    regions.extend(i.region for i in rep.issues if i.region)
                    if rep.verdict == QualityVerdict.REVIEW:
                        verdict = QualityVerdict.REVIEW
            priority = float(len(issues) * 10 + (20 if verdict == QualityVerdict.REVIEW else 5))
            packets.append(
                EvidencePacket(
                    job_id=frame_props[0].job_id,
                    frame_id=frame_id,
                    proposal_ids=[p.proposal_id for p in frame_props],
                    verdict=verdict,
                    priority_score=priority,
                    affected_regions=sorted(set(regions)),
                    summary=f"Frame {frame_id}: {len(frame_props)} segmentation proposals flagged ({len(issues)} issues)",
                    issues=issues,
                )
            )
        packets.sort(key=lambda pkt: pkt.priority_score, reverse=True)
        return packets[:5] if output_dir is not None else packets
