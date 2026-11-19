"""Domain dataclasses for CVAT job/task discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabelInfo:
    id: int
    name: str
    type: str = "any"
    attributes: tuple[dict[str, Any], ...] = ()
    sublabels: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class CoverageSummary:
    annotated_frames: int
    total_frames: int
    coverage_ratio: float
    shapes_count: int = 0
    tracks_count: int = 0
    per_label_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalRunStatus:
    job_id: int
    run_id: str | None = None
    status: str = "none"  # none | in_progress | paused | completed | failed
    stage: str | None = None
    resumable: bool = False
    updated_at: str | None = None
    shapes_count: int = 0
    source_path: str | None = None


@dataclass(frozen=True)
class DiscoveredJob:
    job_id: int
    task_id: int
    task_name: str
    status: str
    stage: str
    assignee: str | None
    frame_count: int
    frame_wh: tuple[int, int] | None
    annotation_type: str
    labels: tuple[LabelInfo, ...]
    coverage: CoverageSummary
    suitable_model: str | None
    last_local_run_status: LocalRunStatus

    @property
    def last_run_status(self) -> str:
        return self.last_local_run_status.status


@dataclass(frozen=True)
class DiscoveryResult:
    host: str
    org: str
    username: str | None
    jobs: tuple[DiscoveredJob, ...]
    recommended_resume_job_id: int | None = None

    def __iter__(self):
        return iter(self.jobs)

    def __len__(self) -> int:
        return len(self.jobs)

    def __bool__(self) -> bool:
        return bool(self.jobs)
