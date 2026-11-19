"""Unified SkeletonAdapter wrapping face50_engine, autolabel/*, and pose17_* lazily.

Never copy-pastes engine code, never imports heavyweight ML libraries at module import time,
never hardcodes landmark counts for custom schemas, and never mixes Pose17 & Face50 mappings.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from cvat_tools.factory.adapters.skeleton.schemas import (
    SkeletonProfile,
    build_skeleton_geometry,
    detect_skeleton_profile,
    is_skeleton_task,
    normalize_keypoint_record,
    sublabel_lookup_from_task,
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
    SkeletonSchema,
    TaskSchema,
    ValidationIssue,
    ValidationReport,
    utc_now_iso,
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


def _proposal_to_cvat_shapes(
    proposal: Proposal,
    task_schema: TaskSchema | None = None,
) -> list[dict[str, Any]]:
    """Convert a Skeleton Proposal into CVAT-compatible skeleton shape dict(s)."""
    geom = proposal.geometry
    elements = geom.get("elements")
    if not elements and "keypoints" in geom:
        elements = [
            {
                "type": "points",
                "label_id": kp.get("label_id"),
                "points": list(kp["points"]),
                "occluded": bool(kp.get("occluded", False)),
                "outside": bool(kp.get("outside", False)),
            }
            for kp in geom["keypoints"]
        ]
    label_id = geom.get("root_label_id")
    if label_id is None and task_schema is not None:
        lbl = task_schema.label_by_name(proposal.label)
        if lbl is not None:
            label_id = lbl.id
    return [
        {
            "id": proposal.cvat_shape_id,
            "frame": proposal.frame_id,
            "type": "skeleton",
            "label_id": label_id if label_id is not None else 0,
            "occluded": proposal.occluded,
            "outside": proposal.outside,
            "z_order": proposal.z_order,
            "attributes": list(proposal.attributes),
            "elements": list(elements or []),
        }
    ]


class SkeletonAdapter(AnnotationAdapter):
    """Unified AnnotationAdapter for Pose17, Face50, and arbitrary custom skeleton schemas."""

    adapter_name = "skeleton"

    def supports(self, task_schema: TaskSchema) -> bool:
        return is_skeleton_task(task_schema)

    def select_model(self, context: AdapterContext) -> ModelSelection:
        profile, schemas = detect_skeleton_profile(context.task_schema)
        if context.user_model_override:
            return ModelSelection(
                model_id=context.user_model_override,
                supported=True,
                reason=f"User override selected for {profile.value}",
                capabilities={"predict": True, "validate": True, "refine": profile == SkeletonProfile.FACE50},
            )

        if profile == SkeletonProfile.FACE50:
            for m in context.available_models:
                if "face" in str(m.get("supported_task", "")).lower() or "fan" in str(m.get("name", "")).lower():
                    return ModelSelection(
                        model_id=str(m.get("model_id") or m.get("name")),
                        supported=True,
                        reason="Matched Face50 landmark model from registry",
                        capabilities={"predict": True, "validate": True, "refine": True},
                    )
            return ModelSelection(
                model_id="face50-fan-2d",
                supported=True,
                reason="Face50 profile detected; uses face50_engine + autolabel QC/refinement",
                capabilities={"predict": True, "validate": True, "refine": True},
                fallback_available=True,
            )

        if profile == SkeletonProfile.POSE17:
            for m in context.available_models:
                if "pose" in str(m.get("supported_task", "")).lower() or "vitpose" in str(m.get("name", "")).lower():
                    return ModelSelection(
                        model_id=str(m.get("model_id") or m.get("name")),
                        supported=True,
                        reason="Matched Pose17 model from registry",
                        capabilities={"predict": True, "validate": True, "refine": False},
                    )
            return ModelSelection(
                model_id="pose17-vitpose-yolo",
                supported=True,
                reason="Pose17 profile detected; uses person17 + pose17_validate",
                capabilities={"predict": True, "validate": True, "refine": False},
                fallback_available=True,
            )

        # Custom skeleton schema: never fake model support if no matching model exists
        expected_counts = {name: s.landmark_count for name, s in schemas.items()}
        for m in context.available_models:
            if m.get("supported_task") == "skeleton" and m.get("landmark_count") in expected_counts.values():
                return ModelSelection(
                    model_id=str(m.get("model_id") or m.get("name")),
                    supported=True,
                    reason="Matched custom skeleton model from registry",
                    capabilities={"predict": True, "validate": True, "refine": False},
                )
        return ModelSelection(
            model_id=None,
            supported=True,
            reason=f"Custom skeleton schema ({expected_counts}); schema/geometry validation active without pretrained model",
            capabilities={"predict": False, "validate": True, "refine": False},
            fallback_available=False,
        )

    def preprocess(
        self,
        images: Sequence[ImageInput],
        context: AdapterContext | None = None,
    ) -> PreprocessedBatch:
        """Compute image statistics lazily; keep display-enhanced images separate from inference inputs."""
        items: list[dict[str, Any]] = []
        display_paths: dict[int, str] = {}
        dark_frames: list[int] = []

        for img in images:
            img_hash = _hash_image_input(img)
            stats: dict[str, Any] = {}
            if img.image_path and Path(img.image_path).is_file():
                try:
                    import cv2
                    from cvat_tools.autolabel.qc import compute_image_stats

                    bgr = cv2.imread(str(img.image_path))
                    if bgr is not None:
                        stats = compute_image_stats(bgr)
                        if stats.get("is_dark"):
                            dark_frames.append(img.frame_id)
                except Exception:
                    stats = {}
            items.append(
                {
                    "frame_id": img.frame_id,
                    "image_path": img.image_path,
                    "image_hash": img_hash,
                    "width": img.width,
                    "height": img.height,
                    "stats": stats,
                    "metadata": dict(img.metadata),
                }
            )
            if img.image_path:
                display_paths[img.frame_id] = img.image_path

        return PreprocessedBatch(
            items=items,
            preprocess_params={
                "adapter": self.adapter_name,
                "dark_frames": dark_frames,
                "inference_input": "raw_original",
                "display_enhancement": "lazy_on_review",
            },
            display_paths=display_paths,
        )

    def predict(
        self,
        batch: PreprocessedBatch,
        model_selection: ModelSelection | None = None,
    ) -> list[RawPrediction]:
        """Run lazy skeleton prediction; never fakes inference if model/deps unavailable."""
        if model_selection is not None and not model_selection.capabilities.get("predict", True):
            return []

        model_id = model_selection.model_id if model_selection is not None else "unknown"
        predictions: list[RawPrediction] = []

        if model_id and "pose17" in model_id:
            from cvat_tools.pose import person17
            import cv2

            backend_name = "vitpose" if "vitpose" in model_id else "yolo"
            backend = person17.make_backend(backend_name)
            for item in batch.items:
                img_path = item.get("image_path")
                if not img_path or not Path(img_path).is_file():
                    continue
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                det = person17.detect(backend, bgr)
                if det is None:
                    continue
                h, w = bgr.shape[:2]
                xy = det["xy"]
                conf = det["conf"]
                kps = []
                for idx in range(len(xy)):
                    st = person17.point_state(float(xy[idx][0]), float(xy[idx][1]), float(conf[idx]), w, h)
                    kps.append(
                        {
                            "id": idx,
                            "points": [float(xy[idx][0]), float(xy[idx][1])],
                            "confidence": float(conf[idx]),
                            "occluded": st == "Q",
                            "outside": st == "O",
                            "visibility": st,
                        }
                    )
                predictions.append(
                    RawPrediction(
                        frame_id=int(item["frame_id"]),
                        label="person",
                        data={"keypoints": kps, "profile": SkeletonProfile.POSE17.value},
                        confidence=float(person17.torso_quality(det)),
                        model_id=model_id,
                        image_hash=str(item.get("image_hash", "")),
                        preprocess_params=dict(batch.preprocess_params),
                    )
                )
            return predictions

        # For Face50 / custom where live prediction is driven by engine/job snapshot
        return predictions

    def convert_to_cvat(
        self,
        predictions: Sequence[RawPrediction],
        task_schema: TaskSchema,
        run_id: str = "",
    ) -> list[Proposal]:
        profile, schemas = detect_skeleton_profile(task_schema)
        proposals: list[Proposal] = []

        for pred in predictions:
            schema = schemas.get(pred.label)
            if schema is None and schemas:
                schema = next(iter(schemas.values()))
            if schema is None:
                schema = SkeletonSchema(
                    schema_name=pred.label,
                    schema_version=task_schema.schema_version,
                    keypoints=(),
                )

            raw_kps = pred.data.get("keypoints") or []
            normalized_kps: list[dict[str, Any]] = []
            for idx, kp in enumerate(raw_kps):
                kp_id = int(kp.get("id", idx))
                spec = schema.keypoint_by_id(kp_id)
                if spec is None and idx < len(schema.keypoints):
                    spec = schema.keypoints[idx]
                if spec is not None:
                    pts = kp.get("points") or [kp.get("x", 0.0), kp.get("y", 0.0)]
                    normalized_kps.append(
                        normalize_keypoint_record(
                            spec,
                            float(pts[0]),
                            float(pts[1]),
                            occluded=bool(kp.get("occluded", False)),
                            outside=bool(kp.get("outside", False)),
                            confidence=kp.get("confidence"),
                            visibility=kp.get("visibility"),
                            sublabel_id=kp.get("label_id"),
                        )
                    )
                else:
                    normalized_kps.append(dict(kp))

            geometry = build_skeleton_geometry(normalized_kps, schema, profile=profile)
            if schema.root_label_id is not None:
                geometry["root_label_id"] = schema.root_label_id

            prov = Provenance(
                model_id=pred.model_id,
                image_hash=pred.image_hash,
                preprocess_params=dict(pred.preprocess_params),
                confidence=pred.confidence,
                timestamp=utc_now_iso(),
                rule_decisions=[],
                run_id=run_id,
                adapter_name=self.adapter_name,
                schema_version=schema.schema_version,
            )
            proposals.append(
                Proposal(
                    job_id=int(task_schema.job_id or 0),
                    frame_id=int(pred.frame_id),
                    label=pred.label,
                    shape_type="skeleton",
                    geometry=geometry,
                    provenance=prov,
                    stage=RevisionStage.MODEL_PROPOSAL,
                    confidence=pred.confidence,
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
        width = int(ctx.get("width") or (task_schema.image_width if task_schema and task_schema.image_width else 1920))
        height = int(ctx.get("height") or (task_schema.image_height if task_schema and task_schema.image_height else 1080))
        frame_images: dict[int, Path] = ctx.get("frame_images") or {}

        profile = SkeletonProfile.CUSTOM
        schemas: dict[str, SkeletonSchema] = {}
        lbl_id_to_name: dict[int, str] = {}
        sub_id_to_kp: dict[int, int] = {}
        if task_schema is not None:
            profile, schemas = detect_skeleton_profile(task_schema)
            lbl_id_to_name, sub_id_to_kp = sublabel_lookup_from_task(task_schema)

        reports: list[ValidationReport] = []
        for prop in proposals:
            prop_profile = prop.geometry.get("profile")
            eff_profile = SkeletonProfile(prop_profile) if prop_profile else profile

            if eff_profile == SkeletonProfile.FACE50 and lbl_id_to_name and sub_id_to_kp:
                reports.append(
                    self._validate_face50_proposal(
                        prop,
                        task_schema=task_schema,
                        sub_id_to_kp=sub_id_to_kp,
                        lbl_id_to_name=lbl_id_to_name,
                        width=width,
                        height=height,
                        img_path=frame_images.get(prop.frame_id),
                    )
                )
            elif eff_profile == SkeletonProfile.POSE17:
                reports.append(
                    self._validate_pose17_proposal(
                        prop,
                        task_schema=task_schema,
                        width=width,
                        height=height,
                        bone_medians=ctx.get("bone_medians"),
                    )
                )
            else:
                reports.append(
                    self._validate_custom_proposal(
                        prop,
                        schema=schemas.get(prop.label),
                        width=width,
                        height=height,
                    )
                )
        return reports

    def _validate_face50_proposal(
        self,
        prop: Proposal,
        *,
        task_schema: TaskSchema | None,
        sub_id_to_kp: dict[int, int],
        lbl_id_to_name: dict[int, str],
        width: int,
        height: int,
        img_path: Path | None,
    ) -> ValidationReport:
        from cvat_tools.autolabel.qc import evaluate_frame_qc

        shapes = prop.metadata.get("cvat_shapes") or _proposal_to_cvat_shapes(prop, task_schema)
        qc = evaluate_frame_qc(
            prop.frame_id,
            shapes,
            sub_id_to_kp,
            lbl_id_to_name,
            width,
            height,
            img_path=img_path,
        )
        issues = [
            ValidationIssue(
                code=str(f.get("code", "FACE50_ISSUE")),
                severity=str(f.get("severity", "warning")),
                message=str(f.get("detail", "")),
                details=dict(f),
            )
            for f in qc.get("findings") or []
        ]
        regions = qc.get("regions") or {}
        if any(v == "REVIEW" for v in regions.values()) or qc.get("error_count", 0) > 0:
            verdict = QualityVerdict.REVIEW
        elif any(v == "REFINE" for v in regions.values()):
            verdict = QualityVerdict.REFINE
        else:
            verdict = QualityVerdict.PASS

        prop.verdict = verdict
        prop.issues = issues
        return ValidationReport(
            proposal_id=prop.proposal_id,
            verdict=verdict,
            geometry_valid=qc.get("error_count", 0) == 0,
            image_alignment_good=not any(v == "REFINE" for v in regions.values()),
            schema_valid=not any(i.code in {"LABEL_SET_MISMATCH", "KP_COUNT_MISMATCH"} for i in issues),
            confidence_sufficient=(prop.confidence is None or prop.confidence >= 0.3),
            no_regression=True,
            issues=issues,
            metrics={"regions": regions, "image_stats": qc.get("image_stats", {})},
        )

    def _validate_pose17_proposal(
        self,
        prop: Proposal,
        *,
        task_schema: TaskSchema | None,
        width: int,
        height: int,
        bone_medians: dict[str, float] | None,
    ) -> ValidationReport:
        from cvat_tools.pose.pose17_validate import validate_frame

        kps = prop.geometry.get("keypoints") or []
        skel_dict = {
            "id": prop.cvat_shape_id or 0,
            "label": prop.label,
            "occluded": prop.occluded,
            "outside": prop.outside,
            "keypoints": [
                {
                    "id": kp.get("id", idx),
                    "name": kp.get("name", str(idx)),
                    "x": float(kp["points"][0]),
                    "y": float(kp["points"][1]),
                    "occluded": bool(kp.get("occluded", False)),
                    "outside": bool(kp.get("outside", False)),
                }
                for idx, kp in enumerate(kps)
            ],
        }
        findings = validate_frame(
            prop.frame_id,
            [skel_dict],
            width,
            height,
            bone_medians=bone_medians,
        )
        issues = [
            ValidationIssue(
                code=f.code,
                severity=f.severity,
                message=f.message,
                affected_points=list(f.kp_indices),
                rule_id=f.rule_id,
                details=dict(f.details),
            )
            for f in findings
        ]
        has_error = any(i.severity == "error" for i in issues)
        verdict = QualityVerdict.REVIEW if has_error else QualityVerdict.PASS
        prop.verdict = verdict
        prop.issues = issues
        return ValidationReport(
            proposal_id=prop.proposal_id,
            verdict=verdict,
            geometry_valid=not has_error,
            image_alignment_good=not has_error,
            schema_valid=not any(i.code == "KP_COUNT_MISMATCH" for i in issues),
            confidence_sufficient=(prop.confidence is None or prop.confidence >= 0.3),
            no_regression=True,
            issues=issues,
        )

    def _validate_custom_proposal(
        self,
        prop: Proposal,
        *,
        schema: SkeletonSchema | None,
        width: int,
        height: int,
    ) -> ValidationReport:
        """Dynamic schema & geometry validation for arbitrary skeleton schemas (no hardcoded counts)."""
        issues: list[ValidationIssue] = []
        kps = prop.geometry.get("keypoints") or []

        if schema is not None and schema.landmark_count > 0 and len(kps) != schema.landmark_count:
            issues.append(
                ValidationIssue(
                    code="KP_COUNT_MISMATCH",
                    severity="error",
                    message=f"Expected {schema.landmark_count} keypoints, got {len(kps)}",
                )
            )

        if not prop.geometry.get("coordinate_system"):
            issues.append(
                ValidationIssue(
                    code="MISSING_COORDINATE_SYSTEM",
                    severity="error",
                    message="Skeleton geometry missing coordinate_system",
                )
            )

        seen_ids: set[int] = set()
        for idx, kp in enumerate(kps):
            kp_id = int(kp.get("id", idx))
            seen_ids.add(kp_id)
            occ = bool(kp.get("occluded", False))
            out = bool(kp.get("outside", False))
            if occ and out:
                issues.append(
                    ValidationIssue(
                        code="Q_AND_O_CONFLICT",
                        severity="error",
                        message=f"Keypoint {kp_id} has both occluded=True and outside=True",
                        affected_points=[kp_id],
                    )
                )
            pts = kp.get("points") or [0.0, 0.0]
            px, py = float(pts[0]), float(pts[1])
            if not out and not (0.0 <= px <= width and 0.0 <= py <= height):
                issues.append(
                    ValidationIssue(
                        code="COORD_OOB",
                        severity="error",
                        message=f"Keypoint {kp_id} ({px:.1f}, {py:.1f}) out of bounds {width}x{height}",
                        affected_points=[kp_id],
                    )
                )

        # Verify parent-child references when declared
        for kp in kps:
            pid = kp.get("parent_id")
            if pid is not None and int(pid) not in seen_ids:
                issues.append(
                    ValidationIssue(
                        code="ORPHAN_PARENT_ID",
                        severity="error",
                        message=f"Keypoint {kp.get('id')} references missing parent_id {pid}",
                        affected_points=[int(kp.get("id", 0))],
                    )
                )

        has_error = any(i.severity == "error" for i in issues)
        verdict = QualityVerdict.REVIEW if has_error else QualityVerdict.PASS
        prop.verdict = verdict
        prop.issues = issues
        return ValidationReport(
            proposal_id=prop.proposal_id,
            verdict=verdict,
            geometry_valid=not has_error,
            image_alignment_good=not has_error,
            schema_valid=not any(i.code in {"KP_COUNT_MISMATCH", "ORPHAN_PARENT_ID"} for i in issues),
            confidence_sufficient=(prop.confidence is None or prop.confidence >= 0.3),
            no_regression=True,
            issues=issues,
        )

    def refine(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Proposal]:
        """Bounded local refinement for Face50 REFINE regions via autolabel.refine; preserves PASS."""
        ctx = context or {}
        task_schema: TaskSchema | None = ctx.get("task_schema")
        frame_images: dict[int, Path] = ctx.get("frame_images") or {}
        max_attempts = int(ctx.get("max_attempts", 2))
        width = int(ctx.get("width") or (task_schema.image_width if task_schema and task_schema.image_width else 1920))
        height = int(ctx.get("height") or (task_schema.image_height if task_schema and task_schema.image_height else 1080))

        report_by_id = {r.proposal_id: r for r in (validation_reports or [])}
        refined_list: list[Proposal] = []

        lbl_id_to_name: dict[int, str] = {}
        sub_id_to_kp: dict[int, int] = {}
        if task_schema is not None:
            lbl_id_to_name, sub_id_to_kp = sublabel_lookup_from_task(task_schema)

        for prop in proposals:
            rep = report_by_id.get(prop.proposal_id)
            verdict = rep.verdict if rep is not None else prop.verdict
            if verdict != QualityVerdict.REFINE:
                refined_list.append(prop)
                continue

            if prop.geometry.get("profile") == SkeletonProfile.FACE50.value and lbl_id_to_name and sub_id_to_kp:
                from cvat_tools.autolabel.refine import selective_refine_frame

                shapes = prop.metadata.get("cvat_shapes") or _proposal_to_cvat_shapes(prop, task_schema)
                res = selective_refine_frame(
                    prop.frame_id,
                    shapes,
                    sub_id_to_kp,
                    lbl_id_to_name,
                    width,
                    height,
                    img_path=frame_images.get(prop.frame_id),
                    max_attempts=max_attempts,
                )
                updated = Proposal.from_dict(prop.to_dict())
                updated.stage = RevisionStage.REFINED_PROPOSAL
                updated.parent_proposal_id = prop.proposal_id
                updated.metadata["cvat_shapes"] = shapes
                updated.metadata["refine_result"] = {
                    "status": res.get("status"),
                    "changed_kps": res.get("changed_kps", []),
                    "attempts": res.get("attempts", 0),
                }
                updated.provenance.rule_decisions.append(
                    {"action": "selective_refine_frame", "changed_kps": res.get("changed_kps", [])}
                )
                refined_list.append(updated)
            else:
                refined_list.append(prop)

        return refined_list

    def render_evidence(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        output_dir: str | Path | None = None,
    ) -> list[EvidencePacket]:
        report_by_id = {r.proposal_id: r for r in (validation_reports or [])}
        packets: list[EvidencePacket] = []

        for prop in proposals:
            rep = report_by_id.get(prop.proposal_id)
            verdict = rep.verdict if rep is not None else (prop.verdict or QualityVerdict.REVIEW)
            issues = rep.issues if rep is not None else prop.issues
            affected_pts = sorted({pt for iss in issues for pt in iss.affected_points})
            affected_regs = sorted({iss.region for iss in issues if iss.region})
            priority = 100.0 if verdict == QualityVerdict.REVIEW else (50.0 if verdict == QualityVerdict.REFINE else 0.0)
            priority += len(issues) * 5.0

            packets.append(
                EvidencePacket(
                    job_id=prop.job_id,
                    frame_id=prop.frame_id,
                    proposal_ids=[prop.proposal_id],
                    verdict=verdict,
                    priority_score=priority,
                    affected_regions=affected_regs,
                    affected_points=affected_pts,
                    image_paths={},
                    summary=f"{prop.label} frame={prop.frame_id} verdict={verdict.value} issues={len(issues)}",
                    issues=list(issues),
                    metadata={"profile": prop.geometry.get("profile"), "output_dir": str(output_dir) if output_dir else ""},
                )
            )
        return packets
