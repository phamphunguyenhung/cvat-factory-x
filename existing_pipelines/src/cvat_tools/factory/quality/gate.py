"""Three-level Quality Gate (PASS / REFINE / REVIEW) with 6-dimension separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domains import (
    validate_bbox,
    validate_face50,
    validate_segmentation,
    validate_tracking,
)
from .models import QualityDimensions, QualityIssue, QualityLevel, QualityReport


@dataclass
class QualityPolicy:
    """Configurable policy thresholds for the Quality Gate."""

    min_confidence: float = 0.50
    max_refinement_attempts: int = 3
    require_human_approval: bool = False
    fail_on_warn_alignment: bool = False


class QualityGate:
    """Evaluates annotations across 6 distinct dimensions into PASS / REFINE / REVIEW."""

    def __init__(self, policy: QualityPolicy | None = None) -> None:
        self.policy = policy or QualityPolicy()

    def evaluate(
        self,
        issues: list[QualityIssue],
        *,
        confidence: float | None = None,
        schema_valid: bool = True,
        image_alignment_good: bool = True,
        no_regression: bool = True,
        human_approved: bool = False,
        missing_evidence: bool = False,
        model_disagreement: bool = False,
        attempts_by_region: dict[str, int] | None = None,
        frame_id: int | str | None = None,
        label: str = "",
    ) -> QualityReport:
        """Compute QualityDimensions and decide PASS / REFINE / REVIEW."""
        attempts_by_region = attempts_by_region or {}

        geom_ok = True
        align_ok = bool(image_alignment_good)
        schema_ok = bool(schema_valid)
        conf_ok = True if confidence is None else (float(confidence) >= self.policy.min_confidence)
        reg_ok = bool(no_regression)

        for iss in issues:
            is_err = iss.severity == "error" or (
                self.policy.fail_on_warn_alignment and iss.dimension == "image_alignment_good"
            )
            if not is_err and iss.severity != "error":
                continue
            if iss.dimension == "geometry_valid":
                geom_ok = False
            elif iss.dimension == "image_alignment_good":
                align_ok = False
            elif iss.dimension == "schema_valid":
                schema_ok = False
            elif iss.dimension == "confidence_sufficient":
                conf_ok = False
            elif iss.dimension == "no_regression":
                reg_ok = False

        dims = QualityDimensions(
            geometry_valid=geom_ok,
            image_alignment_good=align_ok,
            schema_valid=schema_ok,
            confidence_sufficient=conf_ok,
            no_regression=reg_ok,
            human_approved=bool(human_approved),
        )

        # Compute normalized quality score in [0, 1]
        penalty = sum(0.25 if i.severity == "error" else 0.08 if i.severity == "warn" else 0.0 for i in issues)
        if not conf_ok:
            penalty += 0.25
        if not align_ok and not any(i.dimension == "image_alignment_good" for i in issues):
            penalty += 0.25
        score = max(0.0, min(1.0, 1.0 - penalty))

        error_issues = [i for i in issues if i.severity == "error"]
        repairable_regions = sorted({i.region for i in issues if i.repairable and i.severity in {"error", "warn"}})

        # Decision logic
        if (
            missing_evidence
            or model_disagreement
            or not schema_ok
            or not reg_ok
            or not conf_ok
            or any(not i.repairable for i in error_issues)
        ):
            level = QualityLevel.REVIEW
        elif dims.all_automated_pass and not error_issues:
            if self.policy.require_human_approval and not dims.human_approved:
                level = QualityLevel.REVIEW
            else:
                level = QualityLevel.PASS
        elif repairable_regions:
            exhausted = any(
                attempts_by_region.get(reg, 0) >= self.policy.max_refinement_attempts
                for reg in repairable_regions
            )
            level = QualityLevel.REVIEW if exhausted else QualityLevel.REFINE
        else:
            level = QualityLevel.REVIEW

        failing = dims.failing_dimensions(include_human=self.policy.require_human_approval)
        if level == QualityLevel.PASS:
            summary = "PASS: all required quality dimensions satisfied."
        elif level == QualityLevel.REFINE:
            summary = f"REFINE: {len(repairable_regions)} repairable region(s) ({', '.join(repairable_regions)})."
        else:
            reasons = []
            if missing_evidence:
                reasons.append("missing_evidence")
            if model_disagreement:
                reasons.append("model_disagreement")
            reasons.extend(failing)
            summary = f"REVIEW: {', '.join(dict.fromkeys(reasons)) or 'manual inspection required'}."

        return QualityReport(
            level=level,
            dimensions=dims,
            issues=list(issues),
            score=score,
            frame_id=frame_id,
            label=label,
            summary=summary,
            repairable_regions=repairable_regions,
            needs_human_review=(level == QualityLevel.REVIEW),
        )

    def evaluate_domain(
        self,
        domain: str,
        payload: Any,
        *,
        confidence: float | None = None,
        image: Any = None,
        frame_wh: tuple[int, int] | None = None,
        frame_id: int | str | None = None,
        label: str = "",
        attempts_by_region: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> QualityReport:
        """Run domain-specific validation and evaluate through the Quality Gate."""
        d = domain.lower().strip()
        if d == "face50":
            issues = validate_face50(payload, image=image, **kwargs)
        elif d == "segmentation":
            issues = validate_segmentation(payload, frame_wh=frame_wh, **kwargs)
        elif d == "bbox":
            issues = validate_bbox(payload, frame_wh=frame_wh, **kwargs)
        elif d == "tracking":
            issues = validate_tracking(payload, **kwargs)
        else:
            raise ValueError(f"Unsupported quality domain: {domain}")

        return self.evaluate(
            issues,
            confidence=confidence,
            frame_id=frame_id,
            label=label,
            attempts_by_region=attempts_by_region,
        )
