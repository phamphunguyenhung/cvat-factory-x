"""Duong dan workspace (khong hardcode task/job/credential)."""
from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]          # .../learning/cvat
REPO = WORKSPACE.parent                                   # .../learning
CONFIG = WORKSPACE / "config"
DATA = WORKSPACE / "data"
CACHE = DATA / "cache"
MODELS = WORKSPACE / "models"
CHECKPOINTS = MODELS / "checkpoints"
PROJECTS = WORKSPACE / "projects"
LEGACY_MTM = REPO / "tools" / "mot-track-manager"

# noi tim .env, theo thu tu (chi doc, khong in gia tri)
COPILOT = REPO / "tools" / "cvat-seg-copilot"
ENV_CANDIDATES = [WORKSPACE / ".env", COPILOT / ".env.local", COPILOT / ".env", LEGACY_MTM / ".env"]


def job_dir(profile: str, job_id: int) -> Path:
    """Du lieu 1 job. Job id 2 server co the trung nhau -> edufun (va profile khac local) tach thu muc rieng.
    local giu dung cho cu data/jobs/<id> (vd 285)."""
    return DATA / "jobs" / (str(job_id) if profile == "local" else f"{profile}/{job_id}")
