"""Unified TrackingAdapter wrapping cvat_tools.tracking.{engine,geometry,keyframes,reid} lazily."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from cvat_tools.factory.adapters.tracking.rules import (
    _extract_track_frames,
    detect_fragmentation_and_id_switches,
    validate_track_proposal,
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


class TrackingAdapter(AnnotationAdapter):
    """Unified multi-frame object Tracking adapter for CVAT video/sequence jobs."""

    adapter_name = "tracking"

    def supports(self, task_schema: TaskSchema) -> bool:
        if task_schema.task_kind == TaskKind.TRACKING:
            return True
        if task_schema.task_kind in (TaskKind.SKELETON, TaskKind.DETECTION, TaskKind.SEGMENTATION):
            return False
        if "track" in task_schema.annotation_types:
            return True
        return bool(task_schema.metadata.get("is_video", False))

    def select_model(self, context: AdapterContext) -> ModelSelection:
        if context.user_model_override:
            return ModelSelection(
                model_id=context.user_model_override,
                supported=True,
                reason="User model override selected for tracking",
                capabilities={"predict": True, "validate": True, "refine": True, "reid": False},
            )

        for m in context.available_models:
            task = str(m.get("supported_task", "")).lower()
            name = str(m.get("name") or m.get("model_id") or "")
            path_str = str(m.get("path", ""))
            if task in ("tracking", "track", "detection", "detect") or "yolo" in name.lower():
                if not path_str or Path(path_str).exists():
                    return ModelSelection(
                        model_id=f"{m.get('model_id') or name}+bytetrack",
                        supported=True,
                        reason="Matched detector+tracker for video tracking",
                        capabilities={"predict": True, "validate": True, "refine": True, "reid": False},
                    )

        repo_root = Path(__file__).resolve().parents[5]
        ckpt_dir = repo_root / "models" / "checkpoints"
        for model_name in ("yolo26l", "yolo26s", "yolo26n"):
            if (ckpt_dir / f"{model_name}.pt").is_file():
                return ModelSelection(
                    model_id=f"{model_name}+bytetrack",
                    supported=True,
                    reason=f"Found {model_name}.pt for ByteTrack/BoT-SORT tracking",
                    capabilities={"predict": True, "validate": True, "refine": True, "reid": False},
                )

        return ModelSelection(
            model_id=None,
            supported=False,
            reason="No local tracking/detection weights found; cached-detection tracking and track validation remain available",
            capabilities={"predict": False, "validate": True, "refine": True, "reid": False},
            fallback_available=False,
        )

    def preprocess(
        self,
        images: Sequence[ImageInput],
        context: AdapterContext | None = None,
    ) -> PreprocessedBatch:
        cfg = dict(context.config) if context is not None else {}
        ordered = sorted(images, key=lambda im: im.frame_id)
        items: list[dict[str, Any]] = []
        display_paths: dict[int, str] = {}
        for img in ordered:
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
                "tracker": str(cfg.get("tracker", "bytetrack")),
                "reid": bool(cfg.get("reid", False)),
                "interp_tol": float(cfg.get("interp_tol", 4.0)),
                "max_gap": int(cfg.get("max_gap", 2)),
                "cached_detections": cfg.get("cached_detections"),
                "stop_frame": cfg.get("stop_frame"),
                "label": str(cfg.get("default_label", "pedestrian")),
            },
            display_paths=display_paths,
        )

    def predict(
        self,
        batch: PreprocessedBatch,
        model_selection: ModelSelection | None = None,
    ) -> list[RawPrediction]:
        cached_dets = batch.preprocess_params.get("cached_detections")
        tracker_name = str(batch.preprocess_params.get("tracker", "bytetrack"))
        reid = bool(batch.preprocess_params.get("reid", False))
        default_label = str(batch.preprocess_params.get("label", "pedestrian"))
        model_id = (model_selection.model_id if model_selection else None) or f"cached+{tracker_name}"

        if cached_dets is not None:
            from cvat_tools.tracking.engine import run_on_cache

            class _BatchSource:
                def __init__(self, items: list[dict[str, Any]]):
                    self._by_f = {int(it["frame_id"]): it for it in items}

                def image(self, fid: int):
                    import cv2
                    import numpy as np
                    it = self._by_f.get(fid, {})
                    if it.get("image_path") and Path(it["image_path"]).is_file():
                        return cv2.imread(str(it["image_path"]), cv2.IMREAD_COLOR)
                    if it.get("image_bytes") is not None:
                        return cv2.imdecode(np.frombuffer(it["image_bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    h = int(it.get("height") or 100)
                    w = int(it.get("width") or 100)
                    return np.zeros((h, w, 3), dtype=np.uint8)

            res = run_on_cache(cached_dets, _BatchSource(batch.items), tracker_name, reid=reid)
            predictions: list[RawPrediction] = []
            for tid, frames_map in sorted(res["tracks"].items()):
                if not frames_map:
                    continue
                first_f = min(frames_map)
                predictions.append(
                    RawPrediction(
                        frame_id=int(first_f),
                        label=default_label,
                        data={
                            "track_id": int(tid),
                            "frames": {int(f): [round(float(v), 2) for v in box] for f, box in sorted(frames_map.items())},
                            "scores": {int(f): round(float(s), 4) for f, s in res["scores"].get(tid, {}).items()},
                            "stop_frame": batch.preprocess_params.get("stop_frame") or (max(it["frame_id"] for it in batch.items) if batch.items else max(frames_map)),
                        },
                        confidence=(
                            sum(res["scores"].get(tid, {}).values()) / max(1, len(res["scores"].get(tid, {})))
                            if res["scores"].get(tid)
                            else None
                        ),
                        model_id=model_id,
                        image_hash="",
                        preprocess_params={"tracker": tracker_name, "reid": reid},
                    )
                )
            return predictions

        if model_selection is not None and not model_selection.supported:
            return []

        return []

    def convert_to_cvat(
        self,
        predictions: Sequence[RawPrediction],
        task_schema: TaskSchema,
        run_id: str = "",
    ) -> list[Proposal]:
        from cvat_tools.tracking.geometry import fill_gaps, to_cvat_track
        from cvat_tools.tracking.keyframes import must_keep_frames, optimal_keyframes

        proposals: list[Proposal] = []
        for pred in predictions:
            tid = pred.data.get("track_id")
            frames_raw = pred.data.get("frames") or {}
            if tid is None or not frames_raw:
                continue

            track = {int(f): [float(v) for v in box] for f, box in frames_raw.items()}
            max_gap = int(pred.preprocess_params.get("max_gap", 2))
            tol = float(pred.preprocess_params.get("interp_tol", 4.0))
            stop_frame = int(pred.data.get("stop_frame") or max(track))

            filled = fill_gaps(track, max_gap=max_gap)
            why = must_keep_frames(filled)
            hard = {f for f, rs in why.items() if any(r.startswith(("vao_khung", "ra_khung", "bien_khoang")) for r in rs)}
            kfs = optimal_keyframes(filled, tol=tol, must_keep=sorted(hard))

            lbl = task_schema.label_by_name(pred.label)
            label_id = lbl.id if lbl is not None else 0
            cvat_payload = to_cvat_track(label_id=label_id, track=filled, keyframes=kfs, stop_frame=stop_frame)

            proposals.append(
                Proposal(
                    job_id=task_schema.job_id or 0,
                    frame_id=min(filled),
                    label=lbl.name if lbl is not None else pred.label,
                    shape_type="track",
                    track_id=int(tid),
                    geometry={
                        "label_id": label_id,
                        "frames": {str(f): [round(v, 2) for v in box] for f, box in sorted(filled.items())},
                        "keyframes": kfs,
                        "shapes": cvat_payload["shapes"],
                        "stop_frame": stop_frame,
                    },
                    stage=RevisionStage.MODEL_PROPOSAL,
                    confidence=pred.confidence,
                    provenance=Provenance(
                        model_id=pred.model_id,
                        image_hash=pred.image_hash,
                        preprocess_params=dict(pred.preprocess_params),
                        confidence=pred.confidence,
                        rule_decisions=[
                            {"rule": "fill_gaps", "max_gap": max_gap, "frames_after": len(filled)},
                            {"rule": "optimal_keyframes", "tol": tol, "keyframes": len(kfs)},
                        ],
                        run_id=run_id,
                        adapter_name=self.adapter_name,
                    ),
                    metadata={"stop_frame": stop_frame},
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
        interp_tol = float(ctx.get("interp_tol", 4.0))
        max_jump_ratio = float(ctx.get("max_jump_ratio", 2.5))

        reports_map: dict[str, ValidationReport] = {}
        ordered_reports: list[ValidationReport] = []
        for p in proposals:
            rep = validate_track_proposal(
                p,
                task_schema=task_schema,
                interp_tol=interp_tol,
                max_jump_ratio=max_jump_ratio,
            )
            reports_map[p.proposal_id] = rep
            ordered_reports.append(rep)

        detect_fragmentation_and_id_switches(proposals, reports_map)
        return ordered_reports

    def refine(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Proposal]:
        """Bounded local keyframe refinement for REFINE tracks; never alters track_id or PASS tracks."""
        from cvat_tools.tracking.geometry import to_cvat_track
        from cvat_tools.tracking.keyframes import optimal_keyframes

        ctx = context or {}
        tol = float(ctx.get("interp_tol", 4.0))
        report_by_id = {r.proposal_id: r for r in (validation_reports or ())}

        out: list[Proposal] = []
        for p in proposals:
            rep = report_by_id.get(p.proposal_id)
            if rep is not None and rep.verdict != QualityVerdict.REFINE:
                out.append(p)
                continue

            track, _ = _extract_track_frames(p)
            if not track:
                out.append(p)
                continue

            kfs = optimal_keyframes(track, tol=tol)
            label_id = int(p.geometry.get("label_id") or 0)
            stop_frame = int(p.geometry.get("stop_frame") or max(track))
            cvat_payload = to_cvat_track(label_id=label_id, track=track, keyframes=kfs, stop_frame=stop_frame)

            new_prov = Provenance(
                model_id=p.provenance.model_id,
                image_hash=p.provenance.image_hash,
                preprocess_params=dict(p.provenance.preprocess_params),
                confidence=p.confidence,
                rule_decisions=[*p.provenance.rule_decisions, {"rule": "bounded_keyframe_refine", "keyframes": len(kfs)}],
                run_id=p.provenance.run_id,
                adapter_name=self.adapter_name,
            )
            out.append(
                Proposal(
                    job_id=p.job_id,
                    frame_id=p.frame_id,
                    label=p.label,
                    shape_type=p.shape_type,
                    track_id=p.track_id,
                    geometry={
                        **p.geometry,
                        "keyframes": kfs,
                        "shapes": cvat_payload["shapes"],
                    },
                    provenance=new_prov,
                    stage=RevisionStage.REFINED_PROPOSAL,
                    confidence=p.confidence,
                    cvat_shape_id=p.cvat_shape_id,
                    parent_proposal_id=p.proposal_id,
                    z_order=p.z_order,
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
        packets: list[EvidencePacket] = []
        for p in proposals:
            rep = report_by_id.get(p.proposal_id)
            if rep is not None and rep.verdict == QualityVerdict.PASS:
                continue
            issues = list(rep.issues) if rep else []
            regions = sorted({i.region for i in issues if i.region})
            pts = sorted({pt for i in issues for pt in i.affected_points})
            verdict = rep.verdict if rep else QualityVerdict.REVIEW
            priority = float(len(issues) * 10 + (20 if verdict == QualityVerdict.REVIEW else 5))
            packets.append(
                EvidencePacket(
                    job_id=p.job_id,
                    frame_id=p.frame_id,
                    proposal_ids=[p.proposal_id],
                    verdict=verdict,
                    priority_score=priority,
                    affected_regions=regions,
                    affected_points=pts,
                    summary=f"Track {p.track_id} ({p.label}): {len(issues)} temporal/keyframe issue(s)",
                    issues=issues,
                )
            )
        packets.sort(key=lambda pkt: pkt.priority_score, reverse=True)
        return packets[:5] if output_dir is not None else packets
