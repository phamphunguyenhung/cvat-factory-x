"""Task and job discovery, schema/model inference, and resume-first selection."""
from __future__ import annotations

from cvat_tools.factory.discovery.jobs import (
    discover_job,
    discover_jobs,
    infer_annotation_type,
    inspect_job,
    inspect_local_run_status,
    rank_jobs_for_resume,
    recommend_suitable_model,
    resolve_default_job,
    select_job,
)
from cvat_tools.factory.discovery.models import (
    CoverageSummary,
    DiscoveredJob,
    DiscoveryResult,
    LabelInfo,
    LocalRunStatus,
)

__all__ = [
    "CoverageSummary",
    "DiscoveredJob",
    "DiscoveryResult",
    "LabelInfo",
    "LocalRunStatus",
    "discover_job",
    "discover_jobs",
    "infer_annotation_type",
    "inspect_job",
    "inspect_local_run_status",
    "rank_jobs_for_resume",
    "recommend_suitable_model",
    "resolve_default_job",
    "select_job",
]
