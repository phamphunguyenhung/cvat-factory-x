"""Structured experiment tracking for V3 hypothesis-driven iterations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvat_tools.core import paths
from harness.contracts import ExperimentRecord, to_serializable


def default_experiments_dir() -> Path:
    return paths.REPORTS / "experiments"


class ExperimentRegistry:
    """Reads/writes numbered JSON experiment logs under `reports/experiments/`."""

    def __init__(self, exp_dir: Path | None = None):
        self.exp_dir = exp_dir or default_experiments_dir()
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> str:
        existing = sorted(self.exp_dir.glob("exp_*.json"))
        max_num = 0
        for p in existing:
            stem = p.stem
            parts = stem.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                max_num = max(max_num, int(parts[1]))
        return f"exp_{max_num + 1:03d}"

    def record(
        self,
        *,
        slug: str,
        hypothesis: str,
        baseline_id: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        latency_s_per_frame: float,
        peak_rss_mb: float,
        verdict: str,
        notes: str = "",
        experiment_id: str | None = None,
    ) -> tuple[ExperimentRecord, Path]:
        eid = experiment_id or self.next_id()
        rec = ExperimentRecord(
            experiment_id=eid,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            hypothesis=hypothesis,
            baseline_id=baseline_id,
            config=config,
            metrics=metrics,
            latency_s_per_frame=round(float(latency_s_per_frame), 4),
            peak_rss_mb=round(float(peak_rss_mb), 1),
            verdict=verdict,
            notes=notes,
        )
        safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in slug).strip("_") or "run"
        out_path = self.exp_dir / f"{eid}_{safe_slug}.json"
        out_path.write_text(json.dumps(to_serializable(rec), indent=2, ensure_ascii=False), encoding="utf-8")
        return rec, out_path

    def list_experiments(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for p in sorted(self.exp_dir.glob("exp_*.json")):
            try:
                rows.append({"path": str(p), **json.loads(p.read_text(encoding="utf-8"))})
            except Exception:
                continue
        return rows
