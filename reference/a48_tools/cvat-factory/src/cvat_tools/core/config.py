"""Doc config/project/model YAML (khong hardcode task/job)."""
from __future__ import annotations

import yaml

from cvat_tools.core import paths


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf8")) if path.exists() else {}


def list_projects() -> list[dict]:
    out = []
    for d in sorted(paths.PROJECTS.glob("*/project.yaml")):
        out.append({"name": d.parent.name, **(load_yaml(d) or {})})
    return out


def load_project(name: str) -> dict:
    f = paths.PROJECTS / name / "project.yaml"
    if not f.exists():
        raise SystemExit(f"Khong co project '{name}' (co: {[p['name'] for p in list_projects()]})")
    return {"name": name, "dir": str(f.parent), **(load_yaml(f) or {})}


def load_models() -> dict:
    return (load_yaml(paths.CONFIG / "models.yaml") or {}).get("models", {})


def load_profiles() -> dict:
    return (load_yaml(paths.CONFIG / "profiles.yaml") or {}).get("profiles", {})
