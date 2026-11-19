"""Data models for the Phase 4 Quality Gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class QualityLevel(str, Enum):
    """Three-level Quality Gate outcome."""

    PASS = "PASS"
    REFINE = "REFINE"
    REVIEW = "REVIEW"


@dataclass
class QualityDimensions:
    """Explicitly separated quality dimensions.

    Geometry PASS alone does not imply image alignment, schema validity,
    confidence sufficiency, regression freedom, or human approval.
    """

    geometry_valid: bool = True
    image_alignment_good: bool = True
    schema_valid: bool = True
    confidence_sufficient: bool = True
    no_regression: bool = True
    human_approved: bool = False

    @property
    def all_automated_pass(self) -> bool:
        """Return True iff all five automated dimensions pass."""
        return (
            self.geometry_valid
            and self.image_alignment_good
            and self.schema_valid
            and self.confidence_sufficient
            and self.no_regression
        )

    def failing_dimensions(self, include_human: bool = False) -> list[str]:
        """Return names of dimensions that are currently False."""
        names = []
        if not self.geometry_valid:
            names.append("geometry_valid")
        if not self.image_alignment_good:
            names.append("image_alignment_good")
        if not self.schema_valid:
            names.append("schema_valid")
        if not self.confidence_sufficient:
            names.append("confidence_sufficient")
        if not self.no_regression:
            names.append("no_regression")
        if include_human and not self.human_approved:
            names.append("human_approved")
        return names

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class QualityIssue:
    """Single structured defect detected by a domain validator or gate."""

    code: str
    domain: str
    severity: str  # "error" | "warn" | "info"
    region: str
    message: str
    affected_points: list[int] = field(default_factory=list)
    affected_shape_ids: list[str] = field(default_factory=list)
    repairable: bool = True
    dimension: str = "geometry_valid"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    """Complete Quality Gate evaluation report for a frame or annotation."""

    level: QualityLevel
    dimensions: QualityDimensions
    issues: list[QualityIssue] = field(default_factory=list)
    score: float = 1.0
    frame_id: int | str | None = None
    label: str = ""
    summary: str = ""
    repairable_regions: list[str] = field(default_factory=list)
    needs_human_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "dimensions": self.dimensions.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "score": round(self.score, 4),
            "frame_id": self.frame_id,
            "label": self.label,
            "summary": self.summary,
            "repairable_regions": list(self.repairable_regions),
            "needs_human_review": self.needs_human_review,
        }
