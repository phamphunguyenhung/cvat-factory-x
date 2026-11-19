"""Data models for Phase 4 Automated VS Code Review."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VSCodeOpenResult:
    """Result of attempting to open review evidence in VS Code."""

    opened: bool
    command: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewProposal:
    """Input proposal for a single frame/region requiring human review."""

    case_id: str
    frame_id: int | str
    label: str
    priority: float
    image: Any  # numpy BGR array or path to image file
    before_shapes: Any
    after_shapes: Any
    affected_points: list[int] = field(default_factory=list)
    focus_box: tuple[float, float, float, float] | None = None
    issues: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


@dataclass
class ReviewCase:
    """Rendered review case with evidence image paths and user decision."""

    case_id: str
    frame_id: int | str
    label: str
    priority: float
    affected_points: list[int]
    reason: str
    issues: list[dict[str, Any]]
    before_path: str
    after_path: str
    overlay_path: str
    crop_zoom_path: str
    original_crop_path: str
    enhanced_crop_path: str
    decision: str = "pending"  # "pending" | "approve" | "reject" | "edit"
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewBatch:
    """A bounded batch (max 5 highest-priority cases) ready for VS Code review."""

    batch_id: str
    job_id: int | str | None
    cases: list[ReviewCase]
    html_path: str
    decisions_path: str
    vscode_open: VSCodeOpenResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "job_id": self.job_id,
            "cases": [c.to_dict() for c in self.cases],
            "html_path": self.html_path,
            "decisions_path": self.decisions_path,
            "vscode_open": self.vscode_open.to_dict() if self.vscode_open else None,
        }
