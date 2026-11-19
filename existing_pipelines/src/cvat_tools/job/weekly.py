"""Job theo tuan: weeks/<YYYY>-W<ww>.yaml. Chi doc CVAT."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

from cvat_tools.core import paths

WEEKS = paths.WORKSPACE / "weeks"
KINDS = ("seg", "det", "pose", "mot")


def current_week(today: dt.date | None = None) -> str:
    y, w, _ = (today or dt.date.today()).isocalendar()
    return f"{y}-W{w:02d}"


def week_file(week: str) -> Path:
    return WEEKS / f"{week}.yaml"


def list_weeks() -> list[str]:
    return sorted(p.stem for p in WEEKS.glob("????-W??.yaml"))


def new_week(week: str | None = None) -> Path:
    week = week or current_week()
    dst = week_file(week)
    if dst.exists():
        raise ValueError(f"{dst.name} da ton tai, khong ghi de.")
    dst.write_text((WEEKS / "_template.yaml").read_text(encoding="utf8").replace("{week}", week), encoding="utf8")
    return dst


def load_week(week: str) -> dict:
    f = week_file(week)
    if not f.exists():
        raise FileNotFoundError(f"Chua co {f.name}. Chay: python -m cvat_tools week new --week {week}")
    d = yaml.safe_load(f.read_text(encoding="utf8")) or {}
    for j in d.get("jobs", []):
        if j.get("kind") not in KINDS:
            raise ValueError(f"job {j.get('id')}: kind phai thuoc {KINDS}")
        if not isinstance(j.get("id"), int) or j["id"] <= 0:
            raise ValueError(f"job id khong hop le trong {f.name}: {j.get('id')!r} (dien so job giang vien giao)")
    return d


def status(week: str, client=None) -> list[dict]:
    """Moi job: thong tin that tu CVAT (state/stage/task/so frame). Moi job co the co `profile` rieng
    (local = docker localhost:8080, edufun = cvat.edufun.ai); mac dinh theo profile cua tuan.
    Loi tung job khong lam hong ca bang."""
    from cvat_tools.core.client import Cvat
    d = load_week(week)
    clients: dict = {}
    rows = []
    for j in d.get("jobs", []):
        prof = j.get("profile") or d.get("profile", "edufun")
        row = {"id": j["id"], "profile": prof, "kind": j["kind"], "due": j.get("due", ""), "local": j.get("local_status", ""), "note": j.get("note", "")}
        try:
            c = client or clients.setdefault(prof, Cvat.from_env(prof))
            cj = c.job(j["id"])
            d_ = paths.job_dir(prof, j["id"]); d_.mkdir(parents=True, exist_ok=True)
            row.update(dir=str(d_.relative_to(paths.DATA.parent)), state=cj.get("state"), stage=cj.get("stage"), task=cj.get("task_id"), frames=cj.get("stop_frame", 0) - cj.get("start_frame", 0) + 1)
        except Exception as e:  # noqa: BLE001 - bao loi tung job
            row["error"] = str(e)[:120]
        rows.append(row)
    return rows
