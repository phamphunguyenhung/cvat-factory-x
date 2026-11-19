"""Unified DetectionAdapter wrapping cvat_tools.detection.all_objects & yolo_tile lazily."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from cvat_tools.factory.adapters.detection.rules import (
    COCO_TO_CVAT_LABELS,
    calibrate_confidence,
    compute_iou,
    detect_duplicates_and_missing,
    map_raw_label,
    normalize_label_key,
    validate_bbox_proposal,
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
    """Lazily decode BGR numpy array from ImageInput item without top-level cv2/numpy import."""
    if item.get("image_array") is not None:
        return item["image_array"]
    img_bytes = item.get("image_bytes")
    img_path = item.get("image_path")
    if img_bytes is None and not img_path:
        return None
    import cv2
    import numpy as np

    if img_bytes is not None:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    p = Path(str(img_path))
    if not p.is_file():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


class DetectionAdapter(AnnotationAdapter):
    """Unified BBox / Object Detection adapter for CVAT tasks."""

    adapter_name = "detection"

    def supports(self, task_schema: TaskSchema) -> bool:
        if task_schema.task_kind == TaskKind.DETECTION:
            return True
        if task_schema.task_kind in (TaskKind.SKELETON, TaskKind.SEGMENTATION, TaskKind.TRACKING):
            return False
        if "rectangle" in task_schema.annotation_types:
            return True
        return any(lbl.type == "rectangle" for lbl in task_schema.labels)

    def select_model(self, context: AdapterContext) -> ModelSelection:
        if context.user_model_override:
            return ModelSelection(
                model_id=context.user_model_override,
                supported=True,
                reason="User model override selected for detection",
                capabilities={"predict": True, "validate": True, "refine": True, "tiling": True},
            )

        # Check registry-provided available_models first
        for m in context.available_models:
            task = str(m.get("supported_task", "")).lower()
            name = str(m.get("name") or m.get("model_id") or "")
            path_str = str(m.get("path", ""))
            if task in ("detection", "detect", "bbox", "object_detection") or "yolo" in name.lower():
                if not path_str or Path(path_str).exists():
                    return ModelSelection(
                        model_id=str(m.get("model_id") or name),
                        supported=True,
                        reason="Matched available detection model from registry",
                        capabilities={"predict": True, "validate": True, "refine": True, "tiling": True},
                    )

        # Probe local repo OpenVINO/checkpoint directories without loading weights
        repo_root = Path(__file__).resolve().parents[5]
        ov_dir = repo_root / "models" / "openvino"
        ckpt_dir = repo_root / "models" / "checkpoints"
        for model_name in ("yolo26l", "yolo26s", "yolo26n"):
            if ov_dir.is_dir() and any(
                p.is_dir() and p.name.startswith(f"{model_name}_") and p.name.endswith("_openvino_model")
                for p in ov_dir.iterdir()
            ):
                return ModelSelection(
                    model_id=f"{model_name}-openvino",
                    supported=True,
                    reason=f"Found OpenVINO detection weights for {model_name}",
                    capabilities={"predict": True, "validate": True, "refine": True, "tiling": True},
                )
            if (ckpt_dir / f"{model_name}.pt").is_file():
                return ModelSelection(
                    model_id=f"{model_name}-pytorch",
                    supported=True,
                    reason=f"Found PyTorch checkpoint for {model_name}",
                    capabilities={"predict": True, "validate": True, "refine": True, "tiling": True},
                )

        return ModelSelection(
            model_id=None,
            supported=False,
            reason="No local detection model weights found; capability discovery and validation remain available",
            capabilities={"predict": False, "validate": True, "refine": True, "tiling": False},
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
            img_hash = _hash_image_input(img)
            if img.image_path:
                display_paths[img.frame_id] = img.image_path

            is_night = bool(img.metadata.get("is_night", False))
            mean_luma = img.metadata.get("mean_luma")
            if mean_luma is not None and float(mean_luma) < 70.0:
                is_night = True

            items.append(
                {
                    "frame_id": img.frame_id,
                    "image_path": img.image_path,
                    "image_bytes": img.image_bytes,
                    "image_hash": img_hash,
                    "width": img.width,
                    "height": img.height,
                    "is_night": is_night,
                    "mean_luma": mean_luma,
                    "metadata": dict(img.metadata),
                }
            )

        return PreprocessedBatch(
            items=items,
            preprocess_params={
                "use_horizon_tiles": bool(cfg.get("use_horizon_tiles", True)),
                "use_edge_crops": bool(cfg.get("use_edge_crops", True)),
                "conf_day": float(cfg.get("conf_day", 0.22)),
                "conf_night": float(cfg.get("conf_night", 0.09)),
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
        model_id = (model_selection.model_id if model_selection else None) or "yolo26s-openvino"

        detector: Any = predictor
        if detector is None:
            from cvat_tools.detection.all_objects import load_detector

            base_name = model_id.split("-")[0] if "-" in model_id else model_id
            backend = "openvino" if "openvino" in model_id else ("pytorch" if "pytorch" in model_id else "auto")
            try:
                detector = load_detector(backend=backend, model_name=base_name)
            except FileNotFoundError:
                return []

        from cvat_tools.detection.all_objects import detect_objects_multiscale

        use_horizon_tiles = bool(batch.preprocess_params.get("use_horizon_tiles", True))
        use_edge_crops = bool(batch.preprocess_params.get("use_edge_crops", True))
        conf_day = float(batch.preprocess_params.get("conf_day", 0.22))
        conf_night = float(batch.preprocess_params.get("conf_night", 0.09))

        predictions: list[RawPrediction] = []
        for item in batch.items:
            bgr = _load_bgr(item)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            is_night = bool(item.get("is_night", False))
            if item.get("mean_luma") is None:
                luma = float(bgr.mean())
                is_night = luma < 70.0
            dets = detect_objects_multiscale(
                bgr,
                detector,
                is_night=is_night,
                conf_day=conf_day,
                conf_night=conf_night,
                use_horizon_tiles=use_horizon_tiles,
                use_edge_crops=use_edge_crops,
            )
            for det in dets:
                bw = float(det.box[2] - det.box[0])
                bh = float(det.box[3] - det.box[1])
                is_small = (bw * bh) < 0.005 * (w * h)
                predictions.append(
                    RawPrediction(
                        frame_id=int(item["frame_id"]),
                        label=det.label,
                        data={
                            "box": [round(float(v), 2) for v in det.box],
                            "truncated": bool(det.truncated),
                            "occluded": bool(det.occluded),
                            "is_night": is_night,
                            "is_small": is_small,
                            "from_tile": bool(use_horizon_tiles and is_small),
                            "image_width": w,
                            "image_height": h,
                        },
                        confidence=float(det.conf),
                        model_id=model_id,
                        image_hash=str(item.get("image_hash", "")),
                        preprocess_params={
                            "is_night": is_night,
                            "use_horizon_tiles": use_horizon_tiles,
                            "use_edge_crops": use_edge_crops,
                        },
                    )
                )
        return predictions

    def convert_to_cvat(
        self,
        predictions: Sequence[RawPrediction],
        task_schema: TaskSchema,
        run_id: str = "",
    ) -> list[Proposal]:
        proposals: list[Proposal] = []
        for pred in predictions:
            mapped_label, label_id = map_raw_label(pred.label, task_schema)
            box = [round(float(v), 2) for v in (pred.data.get("box") or [])]
            raw_conf = float(pred.confidence) if pred.confidence is not None else 0.0
            cal_conf, decisions = calibrate_confidence(
                raw_conf,
                mapped_label,
                is_night=bool(pred.data.get("is_night", False)),
                is_small=bool(pred.data.get("is_small", False)),
                from_tile=bool(pred.data.get("from_tile", False)),
            )
            if mapped_label != pred.label:
                decisions.append({"rule": "label_taxonomy_map", "from": pred.label, "to": mapped_label})

            proposals.append(
                Proposal(
                    job_id=task_schema.job_id or 0,
                    frame_id=pred.frame_id,
                    label=mapped_label,
                    shape_type="rectangle",
                    geometry={
                        "points": box,
                        "label_id": label_id,
                    },
                    stage=RevisionStage.MODEL_PROPOSAL,
                    confidence=cal_conf,
                    occluded=bool(pred.data.get("occluded", False)),
                    attributes=[],
                    provenance=Provenance(
                        model_id=pred.model_id,
                        image_hash=pred.image_hash,
                        preprocess_params=dict(pred.preprocess_params),
                        confidence=cal_conf,
                        rule_decisions=decisions,
                        run_id=run_id,
                        adapter_name=self.adapter_name,
                    ),
                    metadata={
                        "truncated": bool(pred.data.get("truncated", False)),
                        "image_width": pred.data.get("image_width"),
                        "image_height": pred.data.get("image_height"),
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
        min_box_px = float(ctx.get("min_box_px", 6.0))
        min_conf = float(ctx.get("min_conf", 0.20))
        dup_iou = float(ctx.get("dup_iou", 0.70))
        reference_boxes = ctx.get("reference_boxes")

        reports_map: dict[str, ValidationReport] = {}
        ordered_reports: list[ValidationReport] = []
        for p in proposals:
            pw = img_w or p.metadata.get("image_width")
            ph = img_h or p.metadata.get("image_height")
            rep = validate_bbox_proposal(
                p,
                task_schema=task_schema,
                image_width=pw,
                image_height=ph,
                min_box_px=min_box_px,
                min_conf=min_conf,
            )
            reports_map[p.proposal_id] = rep
            ordered_reports.append(rep)

        detect_duplicates_and_missing(
            proposals,
            reports_map,
            dup_iou=dup_iou,
            reference_boxes=reference_boxes,
        )
        return ordered_reports

    def refine(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Proposal]:
        ctx = context or {}
        task_schema: TaskSchema | None = ctx.get("task_schema")
        img_w = ctx.get("image_width") or (task_schema.image_width if task_schema else None)
        img_h = ctx.get("image_height") or (task_schema.image_height if task_schema else None)
        report_by_id = {r.proposal_id: r for r in (validation_reports or ())}

        refined_out: list[Proposal] = []
        kept_boxes_by_frame: dict[int, list[tuple[str, list[float]]]] = {}

        for p in proposals:
            rep = report_by_id.get(p.proposal_id)
            if rep is not None and rep.verdict == QualityVerdict.PASS:
                # Never alter PASS proposals
                refined_out.append(p)
                pts = p.geometry.get("points") or []
                if len(pts) == 4:
                    kept_boxes_by_frame.setdefault(p.frame_id, []).append((p.label, pts))
                continue

            if rep is not None and rep.verdict != QualityVerdict.REFINE:
                refined_out.append(p)
                continue

            pts = list(p.geometry.get("points") or [])
            if len(pts) != 4:
                refined_out.append(p)
                continue

            x1, y1, x2, y2 = (float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3]))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            w = img_w or p.metadata.get("image_width")
            h = img_h or p.metadata.get("image_height")
            if w is not None and h is not None:
                x1 = max(0.0, min(float(w), x1))
                x2 = max(0.0, min(float(w), x2))
                y1 = max(0.0, min(float(h), y1))
                y2 = max(0.0, min(float(h), y2))

            new_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            # Drop exact/near duplicate during REFINE if higher-priority box already kept
            issue_codes = {i.code for i in rep.issues} if rep else set()
            if "BBOX_DUPLICATE_OVERLAP" in issue_codes:
                peers = kept_boxes_by_frame.get(p.frame_id, [])
                if any(lbl == p.label and compute_iou(b, new_box) >= 0.70 for lbl, b in peers):
                    continue

            kept_boxes_by_frame.setdefault(p.frame_id, []).append((p.label, new_box))
            new_prov = Provenance(
                model_id=p.provenance.model_id,
                image_hash=p.provenance.image_hash,
                preprocess_params=dict(p.provenance.preprocess_params),
                confidence=p.confidence,
                rule_decisions=[*p.provenance.rule_decisions, {"rule": "bounded_bbox_refine", "clipped": True}],
                run_id=p.provenance.run_id,
                adapter_name=self.adapter_name,
            )
            refined_out.append(
                Proposal(
                    job_id=p.job_id,
                    frame_id=p.frame_id,
                    label=p.label,
                    shape_type=p.shape_type,
                    geometry={**p.geometry, "points": new_box},
                    provenance=new_prov,
                    stage=RevisionStage.REFINED_PROPOSAL,
                    confidence=p.confidence,
                    cvat_shape_id=p.cvat_shape_id,
                    track_id=p.track_id,
                    parent_proposal_id=p.proposal_id,
                    verdict=None,
                    attributes=list(p.attributes),
                    z_order=p.z_order,
                    occluded=p.occluded,
                    outside=p.outside,
                    metadata=dict(p.metadata),
                )
            )
        return refined_out

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
                    summary=f"Frame {frame_id}: {len(frame_props)} detection proposals flagged ({len(issues)} issues)",
                    issues=issues,
                )
            )
        packets.sort(key=lambda pkt: pkt.priority_score, reverse=True)
        return packets[:5] if output_dir is not None else packets
