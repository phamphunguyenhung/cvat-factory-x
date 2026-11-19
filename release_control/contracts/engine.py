"""Contract loading and validation engine for CVAT Factory X."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import uuid

import yaml

from release_control.models import (
    AnnotationBox,
    ApprovalRecord,
    DatasetManifest,
    FaultType,
    ReleaseStatus,
    Severity,
    ValidationIssue,
    ValidationReport,
)


class ContractEngine:
    """Loads ODCS schema, taxonomy, and release policy contracts and validates manifests."""

    def __init__(self, contracts_dir: Path | str | None = None) -> None:
        self.contracts_dir = (
            Path(contracts_dir)
            if contracts_dir is not None
            else Path(__file__).resolve().parent
        )
        self.schema: dict[str, Any] = self._load_yaml("schema.yaml")
        self.taxonomy: dict[str, Any] = self._load_yaml("taxonomy.yaml")
        self.release_policy: dict[str, Any] = self._load_yaml("release_policy.yaml")

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self.contracts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Contract file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Contract file {path} must deserialize to a mapping")
        return data

    def supported_profiles(self) -> list[str]:
        return list(self.schema.get("manifest", {}).get("supported_profiles", []))

    def get_taxonomy(self, profile: str) -> dict[int, str]:
        profiles = self.taxonomy.get("profiles", {})
        if profile not in profiles:
            raise KeyError(f"Unknown taxonomy profile: {profile}")
        raw_classes = profiles[profile].get("classes", {})
        return {int(k): str(v) for k, v in raw_classes.items()}

    def is_valid_class(self, profile: str, class_id: int) -> bool:
        try:
            classes = self.get_taxonomy(profile)
        except KeyError:
            return False
        return class_id in classes

    def get_roundtrip_tolerance(self) -> dict[str, float]:
        tol = self.release_policy.get("geometry", {}).get("roundtrip_tolerance", {})
        return {
            "iou_min": float(tol.get("iou_min", 0.99)),
            "bbox_l1_max": float(tol.get("bbox_l1_max", 0.005)),
        }

    def mandatory_approval_roles(self) -> list[str]:
        items = self.release_policy.get("governance", {}).get("mandatory_approvals", [])
        return [str(item["role"]) for item in items if item.get("required", True)]

    def published_is_not_accepted(self) -> bool:
        return bool(
            self.release_policy.get("governance", {}).get(
                "published_is_not_accepted", True
            )
        )

    def can_transition(self, current: ReleaseStatus, target: ReleaseStatus) -> bool:
        transitions = self.release_policy.get("governance", {}).get(
            "allowed_transitions", {}
        )
        allowed = transitions.get(current.value, [])
        return target.value in allowed

    def check_approvals(self, approvals: list[ApprovalRecord]) -> tuple[bool, list[str]]:
        """Verify all mandatory approval roles have an APPROVED decision."""
        approved_roles = {
            a.role for a in approvals if a.decision == "APPROVED"
        }
        missing = [
            role for role in self.mandatory_approval_roles() if role not in approved_roles
        ]
        return (len(missing) == 0, missing)

    def validate_box(
        self, box: AnnotationBox, profile: str, image_id: str
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not self.is_valid_class(profile, box.class_id):
            issues.append(
                ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    code="INVALID_CLASS_ID",
                    fault_type=FaultType.T3_INVALID_CLASS,
                    severity=Severity.CRITICAL,
                    message=f"Class ID {box.class_id} is not valid for profile '{profile}'",
                    image_id=image_id,
                    details={"class_id": box.class_id, "profile": profile},
                )
            )

        if box.bbox_format == "yolo_norm":
            geom = self.release_policy.get("geometry", {})
            cmin = float(geom.get("normalized_coord_min", 0.0))
            cmax = float(geom.get("normalized_coord_max", 1.0))
            min_w = float(geom.get("min_box_width", 0.0001))
            min_h = float(geom.get("min_box_height", 0.0001))

            x1 = box.x_center - box.width / 2.0
            y1 = box.y_center - box.height / 2.0
            x2 = box.x_center + box.width / 2.0
            y2 = box.y_center + box.height / 2.0

            out_of_bounds = (
                box.width < min_w
                or box.height < min_h
                or x1 < cmin - 1e-6
                or y1 < cmin - 1e-6
                or x2 > cmax + 1e-6
                or y2 > cmax + 1e-6
            )
            if out_of_bounds:
                issues.append(
                    ValidationIssue(
                        issue_id=str(uuid.uuid4()),
                        code="BBOX_OUT_OF_BOUNDS",
                        fault_type=FaultType.T4_BBOX_OUT_OF_BOUNDS,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Bounding box out of normalized bounds [{cmin}, {cmax}]: "
                            f"xc={box.x_center:.4f}, yc={box.y_center:.4f}, "
                            f"w={box.width:.4f}, h={box.height:.4f}"
                        ),
                        image_id=image_id,
                        details={
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": box.width,
                            "height": box.height,
                        },
                    )
                )
        return issues

    def validate_manifest(
        self,
        manifest: DatasetManifest,
        snapshot_id: str | None = None,
        release_id: str | None = None,
    ) -> ValidationReport:
        """Validate a DatasetManifest against schema, taxonomy, and release policy."""
        issues: list[ValidationIssue] = []

        if manifest.profile not in self.supported_profiles():
            issues.append(
                ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    code="UNSUPPORTED_PROFILE",
                    fault_type=FaultType.T8_GUIDELINE_MISMATCH,
                    severity=Severity.CRITICAL,
                    message=f"Unsupported dataset profile '{manifest.profile}'",
                    details={"supported_profiles": self.supported_profiles()},
                )
            )

        allowed_exts = set(
            self.schema.get("structure", {})
            .get("image_record", {})
            .get("allowed_extensions", [".jpg", ".jpeg", ".png", ".webp"])
        )
        scene_splits: dict[str, set[str]] = defaultdict(set)
        image_ids: set[str] = set()

        for img in manifest.images:
            image_ids.add(img.image_id)
            ext = Path(img.file_name).suffix.lower()
            if ext not in allowed_exts:
                issues.append(
                    ValidationIssue(
                        issue_id=str(uuid.uuid4()),
                        code="INVALID_IMAGE_EXTENSION",
                        severity=Severity.WARNING,
                        message=f"Unexpected image extension '{ext}' for {img.file_name}",
                        image_id=img.image_id,
                        file_path=img.relative_path,
                    )
                )
            if img.scene_id and img.split:
                scene_splits[img.scene_id].add(img.split)

        if self.release_policy.get("validation", {}).get("detect_scene_leak", True):
            for scene_id, splits in scene_splits.items():
                if len(splits) > 1:
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            code="SCENE_LEAK_ACROSS_SPLITS",
                            fault_type=FaultType.T7_SCENE_LEAK,
                            severity=Severity.CRITICAL,
                            message=f"Scene '{scene_id}' appears in multiple splits: {sorted(splits)}",
                            details={"scene_id": scene_id, "splits": sorted(splits)},
                        )
                    )

        for img_id, boxes in manifest.annotations.items():
            if img_id not in image_ids:
                issues.append(
                    ValidationIssue(
                        issue_id=str(uuid.uuid4()),
                        code="ORPHAN_ANNOTATION",
                        fault_type=FaultType.T1_MISSING_IMAGE,
                        severity=Severity.CRITICAL,
                        message=f"Annotations exist for missing image_id '{img_id}'",
                        image_id=img_id,
                    )
                )
            for box in boxes:
                issues.extend(self.validate_box(box, manifest.profile, img_id))

        critical_count = sum(1 for i in issues if i.severity == Severity.CRITICAL)
        warning_count = sum(1 for i in issues if i.severity == Severity.WARNING)
        info_count = sum(1 for i in issues if i.severity == Severity.INFO)

        max_critical = int(
            self.release_policy.get("validation", {}).get("max_critical_issues", 0)
        )
        passed = critical_count <= max_critical

        return ValidationReport(
            report_id=str(uuid.uuid4()),
            release_id=release_id,
            snapshot_id=snapshot_id,
            passed=passed,
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            issues=issues,
            metrics={
                "image_count": len(manifest.images),
                "annotated_image_count": len(manifest.annotations),
                "box_count": sum(len(b) for b in manifest.annotations.values()),
            },
        )
