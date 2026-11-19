"""Automated Review orchestrator: max 5 priority cases, render evidence, HTML dashboard, VS Code open."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Sequence

from .dashboard import render_review_dashboard, save_decisions
from .models import ReviewBatch, ReviewCase, ReviewProposal
from .renderer import render_case_evidence
from .vscode import open_in_vscode


class AutomatedReviewManager:
    """Builds a bounded (max 5) review batch with visual evidence and opens it in VS Code."""

    MAX_CASES_PER_BATCH = 5

    def __init__(self, max_cases: int = MAX_CASES_PER_BATCH) -> None:
        if not 1 <= max_cases <= self.MAX_CASES_PER_BATCH:
            raise ValueError(f"max_cases must be in [1, {self.MAX_CASES_PER_BATCH}]")
        self.max_cases = max_cases

    def create_batch(
        self,
        proposals: Sequence[ReviewProposal],
        *,
        out_dir: str | Path,
        job_id: int | str | None = None,
        batch_id: str | None = None,
        auto_open_vscode: bool = True,
        vscode_runner: Callable | None = None,
    ) -> ReviewBatch:
        """Select top-N highest-priority cases, render evidence, write HTML dashboard, and open VS Code."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        bid = batch_id or uuid.uuid4().hex[:8]

        ranked = sorted(proposals, key=lambda p: float(p.priority), reverse=True)[: self.max_cases]
        cases: list[ReviewCase] = []
        evidence_dir = out_path / "evidence"

        for prop in ranked:
            paths = render_case_evidence(
                image=prop.image,
                before_shapes=prop.before_shapes,
                after_shapes=prop.after_shapes,
                out_dir=evidence_dir,
                case_id=prop.case_id,
                affected_points=prop.affected_points,
                focus_box=prop.focus_box,
            )
            cases.append(
                ReviewCase(
                    case_id=prop.case_id,
                    frame_id=prop.frame_id,
                    label=prop.label,
                    priority=float(prop.priority),
                    affected_points=list(prop.affected_points),
                    reason=prop.reason,
                    issues=list(prop.issues),
                    **paths,
                )
            )

        html_path = out_path / "review_dashboard.html"
        decisions_path = out_path / "decisions.json"
        batch = ReviewBatch(
            batch_id=bid,
            job_id=job_id,
            cases=cases,
            html_path=str(html_path),
            decisions_path=str(decisions_path),
        )
        render_review_dashboard(batch, html_path)
        save_decisions(decisions_path, cases)

        if auto_open_vscode:
            targets: list[Path] = [html_path]
            if cases:
                targets.append(Path(cases[0].overlay_path))
                targets.append(Path(cases[0].crop_zoom_path))
            batch.vscode_open = open_in_vscode(targets, reuse_window=True, runner=vscode_runner)

        return batch


def open_review(
    job_id: int | str,
    run_id: str | None = None,
    max_cases: int = 5,
    auto_open: bool = True,
    *,
    proposals: Sequence[ReviewProposal] | None = None,
    out_dir: str | Path | None = None,
    vscode_runner: Callable | None = None,
) -> dict:
    """CLI entrypoint helper for `cvat-factory review --job <id>`.

    If `proposals` is provided, renders a fresh review batch; otherwise opens an existing
    dashboard under `data/review/job_<job_id>` or creates a clean empty-state dashboard.
    """
    base = Path(out_dir) if out_dir is not None else Path("data") / "review" / f"job_{job_id}"
    if run_id:
        base = base / f"run_{run_id}"

    if proposals is not None:
        mgr = AutomatedReviewManager(max_cases=min(5, max(1, int(max_cases))))
        batch = mgr.create_batch(
            proposals,
            out_dir=base,
            job_id=job_id,
            batch_id=run_id,
            auto_open_vscode=auto_open,
            vscode_runner=vscode_runner,
        )
        return batch.to_dict()

    html_path = base / "review_dashboard.html"
    decisions_path = base / "decisions.json"
    if not html_path.is_file():
        batch = ReviewBatch(
            batch_id=run_id or f"job_{job_id}",
            job_id=job_id,
            cases=[],
            html_path=str(html_path),
            decisions_path=str(decisions_path),
        )
        render_review_dashboard(batch, html_path)
        save_decisions(decisions_path, [])

    vscode_res = open_in_vscode([html_path], reuse_window=True, runner=vscode_runner) if auto_open else None
    return {
        "job_id": job_id,
        "run_id": run_id,
        "html_path": str(html_path),
        "decisions_path": str(decisions_path),
        "vscode_open": vscode_res.to_dict() if vscode_res else None,
    }
