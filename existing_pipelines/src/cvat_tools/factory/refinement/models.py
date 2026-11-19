"""Data models for bounded local refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..quality.models import QualityReport


@dataclass
class RefinementAttempt:
    """Record of a single bounded local refinement attempt on one region."""

    region: str
    attempt: int
    accepted: bool
    reason: str
    score_before: float
    score_after: float
    affected_points: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RefinementResult:
    """Outcome of bounded local refinement."""

    status: str  # "pass" | "refined" | "unchanged" | "exhausted" | "escalate_review"
    payload: Any
    report: QualityReport
    attempts_by_region: dict[str, int] = field(default_factory=dict)
    history: list[RefinementAttempt] = field(default_factory=list)
    patched_regions: list[str] = field(default_factory=list)
    escalate_to_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "report": self.report.to_dict(),
            "attempts_by_region": dict(self.attempts_by_region),
            "history": [h.to_dict() for h in self.history],
            "patched_regions": list(self.patched_regions),
            "escalate_to_review": self.escalate_to_review,
        }
