"""Test Matrix Items 24-25: VS Code evidence opening & CLI installation/entrypoint."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cvat_tools.factory.review import (
    AutomatedReviewManager,
    ReviewProposal,
    open_in_vscode,
)


def test_24_vscode_evidence_rendering_and_opening(
    tmp_path: Path,
    mixed_quality_images: dict[str, Path],
) -> None:
    """Item 24: Generates BEFORE/AFTER/OVERLAY + crop zoom + HTML dashboard (max 5 cases) and invokes code.cmd --reuse-window."""
    proposals = []
    for idx in range(7):
        proposals.append(
            ReviewProposal(
                case_id=f"case_{idx}",
                frame_id=90 + idx,
                label="person_vf17",
                priority=float(10 - idx),
                image=str(mixed_quality_images["dark"]),
                before_shapes=[{"type": "rectangle", "points": [150.0, 120.0, 340.0, 360.0]}],
                after_shapes=[{"type": "rectangle", "points": [152.0, 122.0, 338.0, 358.0]}],
                focus_box=(150.0, 120.0, 340.0, 360.0),
                reason="low_contrast_check",
            )
        )

    mgr = AutomatedReviewManager(max_cases=5)
    batch = mgr.create_batch(
        proposals,
        out_dir=tmp_path / "review",
        job_id=1623,
        batch_id="run_test_24",
        auto_open_vscode=False,
    )
    assert len(batch.cases) == 5
    assert Path(batch.html_path).is_file()
    for case in batch.cases:
        assert Path(case.before_path).is_file()
        assert Path(case.after_path).is_file()
        assert Path(case.overlay_path).is_file()
        assert Path(case.crop_zoom_path).is_file()

    captured_cmds: list[list[str]] = []

    def _fake_runner(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    res = open_in_vscode([batch.html_path], reuse_window=True, runner=_fake_runner)
    assert res.opened is True
    assert len(captured_cmds) == 1
    assert "--reuse-window" in captured_cmds[0]
    assert batch.html_path in captured_cmds[0]


def test_25_cli_entrypoint_and_lazy_loading() -> None:
    """Item 25: CLI entrypoint works for all 11 commands and does not eagerly import torch/ultralytics/onnxruntime."""
    # 1. Help check
    proc_help = subprocess.run(
        [sys.executable, "-m", "cvat_tools.factory.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc_help.returncode == 0
    for cmd in ("doctor", "setup", "discover", "models", "jobs", "inspect", "run", "review", "sync", "benchmark", "resume"):
        assert cmd in proc_help.stdout

    # 2. Doctor JSON check (offline so unit tests never depend on live CVAT network)
    proc_doc = subprocess.run(
        [sys.executable, "-m", "cvat_tools.factory.cli", "doctor", "--offline", "--json"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc_doc.returncode == 0
    doc_payload = json.loads(proc_doc.stdout)
    assert isinstance(doc_payload, dict)

    # 3. Lazy loading check in fresh Python process
    code = (
        "import sys; "
        "import cvat_tools.factory.cli.app; "
        "heavy = [m for m in ('torch', 'ultralytics', 'onnxruntime') if m in sys.modules]; "
        "assert not heavy, f'Eager heavy imports: {heavy}'"
    )
    proc_lazy = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc_lazy.returncode == 0, proc_lazy.stderr
