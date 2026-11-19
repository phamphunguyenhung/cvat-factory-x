"""Tests for scripts/harness_checkpoint.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "harness_checkpoint.py"


def test_checkpoint_save_and_validate_detects_staleness(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("x = 1\n", encoding="utf-8")

    task_id = "_pytest_ckpt_tmp"
    ckpt_file = ROOT / ".agents" / "state" / f"{task_id}.ckpt.json"
    try:
        res_save = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "save",
                task_id,
                "--mode",
                "FAST",
                "--objective",
                "Test checkpoint",
                "--next-action",
                "Run pytest",
                "--inspected",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Saved checkpoint" in res_save.stdout
        data = json.loads(ckpt_file.read_text(encoding="utf-8"))
        assert data["mode"] == "FAST"
        assert str(target).replace("\\", "/") in data["files_inspected"]

        res_ok = subprocess.run(
            [sys.executable, str(SCRIPT), "validate", str(ckpt_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_ok.returncode == 0
        assert "stale=[]" in res_ok.stdout

        # Modify file on disk -> checkpoint must detect staleness
        target.write_text("x = 2\n", encoding="utf-8")
        res_stale = subprocess.run(
            [sys.executable, str(SCRIPT), "validate", str(ckpt_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_stale.returncode == 1
        assert "sample.py" in res_stale.stdout
    finally:
        if ckpt_file.exists():
            ckpt_file.unlink()
