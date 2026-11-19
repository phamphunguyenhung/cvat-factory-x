"""Structured JSON + concise summary reporting for CVAT Factory."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvat_tools.core import paths


@dataclass
class FactoryRunReport:
    """Unified structured report for sync and orchestration runs."""
    schema: str = "cvat_factory.report.v1"
    run_id: str = ""
    job_id: int = 0
    mode: str = "dry_run"  # "dry_run" | "live" | "orchestration"
    status: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    counts: dict[str, int] = field(default_factory=dict)
    verified: bool = False
    quality_summary: dict[str, Any] = field(default_factory=dict)
    budget_usage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str | None] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "counts": dict(self.counts),
            "verified": self.verified,
            "quality_summary": dict(self.quality_summary),
            "budget_usage": dict(self.budget_usage),
            "artifacts": dict(self.artifacts),
            "errors": list(self.errors),
            "next_actions": list(self.next_actions),
        }


def build_sync_report(
    sync_result: Any,
    orchestration_result: Any | None = None,
) -> FactoryRunReport:
    """Build a FactoryRunReport from a SyncResult (and optional OrchestrationResult)."""
    plan = getattr(sync_result, "plan", None)
    counts = {
        "n_before": getattr(sync_result, "n_before", 0),
        "n_after": getattr(sync_result, "n_after", 0),
        "planned_creates": len(plan.creates) if plan else 0,
        "planned_updates": len(plan.updates) if plan else 0,
        "planned_deletes": len(plan.deletes) if plan else 0,
        "created": len(getattr(sync_result, "created_ids", [])),
        "updated": len(getattr(sync_result, "updated_ids", [])),
        "deleted": len(getattr(sync_result, "deleted_ids", [])),
        "skipped": len(getattr(sync_result, "skipped", [])),
    }

    skip_reasons: dict[str, int] = {}
    for item in getattr(sync_result, "skipped", []) or []:
        r = getattr(item, "reason", "unknown")
        skip_reasons[r] = skip_reasons.get(r, 0) + 1

    budget_usage = (
        orchestration_result.budget_usage.to_dict()
        if orchestration_result is not None
        else {"agent_calls": 0, "iterations": 0, "tokens_used": 0, "runtime_seconds": 0.0}
    )

    next_actions: list[str] = []
    if getattr(sync_result, "dry_run", True) and plan and plan.total_mutations > 0:
        next_actions.append(f"Run with --yes to apply {plan.total_mutations} planned mutation(s).")
    if not getattr(sync_result, "verified", True):
        next_actions.append("Inspect verification_errors and consider conflict-aware rollback.")

    return FactoryRunReport(
        run_id=sync_result.run_id,
        job_id=sync_result.job_id,
        mode="dry_run" if sync_result.dry_run else "live",
        status=sync_result.status,
        counts=counts,
        verified=bool(sync_result.verified),
        quality_summary={"skip_reasons": skip_reasons},
        budget_usage=budget_usage,
        artifacts={
            "backup_path": getattr(sync_result, "backup_path", None),
            "mutation_log_path": getattr(sync_result, "mutation_log_path", None),
        },
        errors=list(getattr(sync_result, "verification_errors", []) or []),
        next_actions=next_actions,
    )


def build_orchestration_report(orchestration_result: Any) -> FactoryRunReport:
    """Build a FactoryRunReport from an OrchestrationResult."""
    packet_paths = [
        r.packet_path for r in (orchestration_result.responses or []) if getattr(r, "packet_path", None)
    ]
    return FactoryRunReport(
        run_id=orchestration_result.run_id,
        job_id=orchestration_result.job_id,
        mode="orchestration",
        status="happy_path" if orchestration_result.happy_path else "escalated",
        counts={"escalated": orchestration_result.escalated_count},
        verified=True,
        budget_usage=orchestration_result.budget_usage.to_dict(),
        artifacts={"review_packet": packet_paths[0] if packet_paths else None},
        errors=list(orchestration_result.errors or []),
    )


def format_concise_summary(report: FactoryRunReport) -> str:
    """Format a concise human-readable Markdown/text summary."""
    c = report.counts
    lines = [
        f"Job {report.job_id} | Run `{report.run_id}` | Mode: `{report.mode}` | Status: `{report.status}` | Verified: `{report.verified}`",
    ]
    if report.mode in ("dry_run", "live"):
        lines.append(
            f"Counts: before={c.get('n_before', 0)}, after={c.get('n_after', 0)}, "
            f"created={c.get('created', 0)} (planned {c.get('planned_creates', 0)}), "
            f"updated={c.get('updated', 0)} (planned {c.get('planned_updates', 0)}), "
            f"deleted={c.get('deleted', 0)}, skipped={c.get('skipped', 0)}"
        )
    else:
        lines.append(f"Escalated: {c.get('escalated', 0)}")

    bu = report.budget_usage
    if bu:
        lines.append(
            f"Agent Budget: calls={bu.get('agent_calls', 0)}, tokens={bu.get('tokens_used', 0)}, "
            f"runtime={bu.get('runtime_seconds', 0)}s"
        )
    if report.errors:
        lines.append("Errors: " + "; ".join(report.errors))
    if report.next_actions:
        lines.append("Next: " + "; ".join(report.next_actions))
    return "\n".join(lines)


def write_report(
    report: FactoryRunReport,
    output_dir: Path | None = None,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Write structured JSON report and concise Markdown summary to disk."""
    out_dir = Path(output_dir) if output_dir else (paths.DATA / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = stem or f"report_job{report.job_id}_{report.run_id}"
    json_path = out_dir / f"{name}.json"
    summary_path = out_dir / f"{name}.md"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(format_concise_summary(report) + "\n", encoding="utf-8")
    return json_path, summary_path
