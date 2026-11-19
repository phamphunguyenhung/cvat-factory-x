"""VS Code launcher for Windows (`code.cmd --reuse-window`) with explicit error reporting."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .models import VSCodeOpenResult


def resolve_vscode_cli() -> str:
    """Resolve `code.cmd` on Windows (or `code` on non-Windows) without shell injection."""
    candidates = ["code.cmd", "code"] if os.name == "nt" else ["code", "code.cmd"]
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        win_paths = [
            Path(local_app) / "Programs" / "Microsoft VS Code" / "bin" / "code.cmd",
            Path(prog_files) / "Microsoft VS Code" / "bin" / "code.cmd",
        ]
        for p in win_paths:
            if p.is_file():
                return str(p)

    return "code.cmd" if os.name == "nt" else "code"


def open_in_vscode(
    targets: Sequence[str | Path],
    *,
    reuse_window: bool = True,
    timeout_sec: float = 15.0,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> VSCodeOpenResult:
    """Open review dashboard and/or evidence images in VS Code using `code.cmd --reuse-window`.

    Never silently swallows errors; always returns the exact command and error details on failure.
    """
    cli = resolve_vscode_cli()
    cmd: list[str] = [cli]
    if reuse_window:
        cmd.append("--reuse-window")
    cmd.extend(str(Path(t)) for t in targets)

    exec_fn = runner or subprocess.run
    try:
        proc = exec_fn(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
        )
        ok = proc.returncode == 0
        err = None if ok else (proc.stderr.strip() or f"VS Code exited with code {proc.returncode}")
        return VSCodeOpenResult(
            opened=ok,
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            error=err,
        )
    except Exception as exc:
        return VSCodeOpenResult(
            opened=False,
            command=cmd,
            returncode=None,
            stdout="",
            stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )
