"""Configuration and path auto-discovery for CVAT Factory X Control Tower."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


def load_dotenv(path: Path | None = None) -> list[str]:
    """Tu-viet parser .env (KEY=VALUE), khong ghi de bien da co, tra ve ten khoa da nap (khong bao gio gia tri)."""
    env_file = path or (Path(__file__).resolve().parent.parent / ".env")
    loaded: list[str] = []
    if not env_file.is_file():
        return loaded
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def read_only() -> bool:
    """RC_READ_ONLY=1 -> chan moi hanh dong ghi len CVAT."""
    return os.environ.get("RC_READ_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


class ReadOnlyError(RuntimeError):
    pass


def guard_cvat_write(action: str) -> None:
    if read_only():
        raise ReadOnlyError(f"RC_READ_ONLY=1: chan hanh dong ghi len CVAT ({action})")


_DEFAULT_DAY10_DATA_SOURCE = Path(
    r"D:\06-Learning\AIA-VINGROUP\learning\day10\labs\K5-L23-Day10-Dataset-Release-Handover\data_source"
)
_DEFAULT_CVAT_DATA_DIR = Path(r"D:\06-Learning\AIA-VINGROUP\learning\cvat\data")


def _discover_project_root() -> Path:
    env_root = os.environ.get("CVAT_RELEASE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def _discover_day10_data_source(project_root: Path) -> Path:
    env_path = os.environ.get("CVAT_RELEASE_DAY10_DATA_SOURCE")
    if env_path:
        return Path(env_path).resolve()
    if _DEFAULT_DAY10_DATA_SOURCE.exists():
        return _DEFAULT_DAY10_DATA_SOURCE
    candidate = (
        project_root.parent
        / "labs"
        / "K5-L23-Day10-Dataset-Release-Handover"
        / "data_source"
    )
    return candidate.resolve()


def _discover_cvat_data_dir(project_root: Path) -> Path:
    env_path = os.environ.get("CVAT_DATA_DIR")
    if env_path:
        return Path(env_path).resolve()
    if _DEFAULT_CVAT_DATA_DIR.exists():
        return _DEFAULT_CVAT_DATA_DIR
    candidate = project_root.parent.parent / "cvat" / "data"
    return candidate.resolve()


class ControlTowerConfig(BaseModel):
    """Runtime configuration and directory layout for the Data Release Control Tower."""

    project_root: Path = Field(default_factory=_discover_project_root)
    day10_data_source: Path
    cvat_data_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    db_path: Path
    contracts_dir: Path

    @classmethod
    def auto_discover(cls, project_root: Path | None = None) -> ControlTowerConfig:
        """Build configuration by auto-discovering workspace, data, and artifact paths."""
        root = (project_root or _discover_project_root()).resolve()
        artifacts_dir = root / "artifacts"
        reports_dir = root / "reports"
        db_path = artifacts_dir / "control_tower.sqlite"
        contracts_dir = root / "release_control" / "contracts"

        return cls(
            project_root=root,
            day10_data_source=_discover_day10_data_source(root),
            cvat_data_dir=_discover_cvat_data_dir(root),
            artifacts_dir=artifacts_dir,
            reports_dir=reports_dir,
            db_path=db_path,
            contracts_dir=contracts_dir,
        )

    @property
    def snapshots_dir(self) -> Path:
        return self.artifacts_dir / "snapshots"

    @property
    def lineage_dir(self) -> Path:
        return self.artifacts_dir / "lineage"

    @property
    def mlruns_dir(self) -> Path:
        return self.artifacts_dir / "mlruns"

    @property
    def working_dir(self) -> Path:
        return self.artifacts_dir / "working"

    def ensure_dirs(self) -> None:
        """Create required artifact and report directories if they do not exist."""
        for directory in (
            self.artifacts_dir,
            self.snapshots_dir,
            self.lineage_dir,
            self.mlruns_dir,
            self.working_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
