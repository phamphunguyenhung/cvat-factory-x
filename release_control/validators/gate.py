"""Validation gate: orchestrates structural, geometry, semantic, and privacy validators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .geometry import validate_geometry
from .models import ValidationIssue
from .privacy import validate_privacy
from .semantic import validate_semantic
from .structural import validate_manifest_schema, validate_structural

FAULT_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation report returned by ValidationEngine."""

    dataset_id: str
    passed: bool
    requires_human_review: bool
    recommended_status: str
    counts_by_fault_type: dict[str, int]
    severity_counts: dict[str, int]
    issues: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return asdict(self)


def _build_report(dataset_id: str, issues: list[ValidationIssue]) -> ValidationReport:
    counts_by_fault_type: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}
    counts_by_fault_type["SCHEMA"] = 0
    counts_by_fault_type["PRIVACY"] = 0

    severity_counts: dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}

    for issue in issues:
        counts_by_fault_type[issue.fault_type] = counts_by_fault_type.get(issue.fault_type, 0) + 1
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    passed = severity_counts["CRITICAL"] == 0
    requires_human_review = any(issue.requires_human_review for issue in issues)
    recommended_status = "VALIDATED" if (passed and not requires_human_review) else "HOLD"

    return ValidationReport(
        dataset_id=dataset_id,
        passed=passed,
        requires_human_review=requires_human_review,
        recommended_status=recommended_status,
        counts_by_fault_type=counts_by_fault_type,
        severity_counts=severity_counts,
        issues=[issue.to_dict() for issue in issues],
    )


class ValidationEngine:
    """Deterministic validation gate for CVAT Factory X datasets."""

    @classmethod
    def validate_dataset_report(
        cls,
        dataset_dir: str | Path,
        contract: dict[str, Any] | None = None,
        reference_dir: str | Path | None = None,
        qc_evidence: dict[str, Any] | None = None,
    ) -> ValidationReport:
        """Run all validators against a dataset directory and return a ValidationReport."""
        dataset_root = Path(dataset_dir)
        manifest_path = dataset_root / "manifest.json"

        if not manifest_path.is_file():
            return _build_report(
                dataset_id="unknown",
                issues=[
                    ValidationIssue(
                        code="SCHEMA_ERROR",
                        fault_type="SCHEMA",
                        severity="CRITICAL",
                        message=f"Manifest file not found at '{manifest_path}'.",
                        file_path="manifest.json",
                    )
                ],
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _build_report(
                dataset_id="unknown",
                issues=[
                    ValidationIssue(
                        code="SCHEMA_ERROR",
                        fault_type="SCHEMA",
                        severity="CRITICAL",
                        message=f"Failed to parse manifest.json: {exc}",
                        file_path="manifest.json",
                    )
                ],
            )

        dataset_id = (
            manifest.get("dataset_id", "unknown")
            if isinstance(manifest, dict) and isinstance(manifest.get("dataset_id"), str)
            else "unknown"
        )

        schema_issues, schema_ok = validate_manifest_schema(manifest)
        if not schema_ok:
            return _build_report(dataset_id=dataset_id, issues=schema_issues)

        issues: list[ValidationIssue] = []
        issues.extend(validate_structural(dataset_root, manifest, contract))
        issues.extend(validate_geometry(dataset_root, manifest, contract))
        issues.extend(
            validate_semantic(
                dataset_root,
                manifest,
                contract=contract,
                reference_dir=reference_dir,
                qc_evidence=qc_evidence,
            )
        )
        issues.extend(validate_privacy(manifest, contract))

        return _build_report(dataset_id=dataset_id, issues=issues)

    @classmethod
    def validate_dataset(
        cls,
        dataset_dir: str | Path,
        contract: dict[str, Any] | None = None,
        reference_dir: str | Path | None = None,
        qc_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all validators against a dataset directory and return a structured dict report."""
        return cls.validate_dataset_report(
            dataset_dir=dataset_dir,
            contract=contract,
            reference_dir=reference_dir,
            qc_evidence=qc_evidence,
        ).to_dict()
