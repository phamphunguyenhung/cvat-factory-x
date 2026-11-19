"""Nap quy uoc chung (default.yaml) + ghi de theo job (jobs/<N>.yaml). Thuan YAML, khong goi mang."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from cvat_tools.core import paths

RULESETS = paths.CONFIG / "rulesets"
CONTINUITY = ("none", "video")


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load(p: Path) -> dict:
    return (yaml.safe_load(p.read_text(encoding="utf8")) or {}) if p.exists() else {}


def load_ruleset(job_id: int | None = None, root: Path | None = None) -> dict:
    """default + jobs/<job>.yaml. Them rs['_sources'] va rs['_warnings'] (khong raise)."""
    root = Path(root) if root else RULESETS
    rs = _load(root / "default.yaml")
    src, warns = [str(root / "default.yaml")], []
    jf = root / "jobs" / f"{job_id}.yaml"
    if job_id is not None and jf.exists():
        rs = _merge(rs, _load(jf))
        src.append(str(jf))
    c = rs.get("continuity", "none")
    if c not in CONTINUITY:
        warns.append(f"continuity='{c}' khong hop le -> dung 'none'")
        c = "none"
    rs["continuity"] = c
    if c == "video":
        rs["keyframe_every"] = int(rs.get("keyframe_every") or rs.get("video_keyframe_every") or 10)
    else:
        rs["keyframe_every"] = None
    rs["_sources"], rs["_warnings"] = src, warns
    return rs


def group_of(rs: dict, label: str) -> str | None:
    label = rs.get("aliases", {}).get(label, label)
    for g, labs in rs.get("groups", {}).items():
        if label in labs:
            return g
    return None


def expected_shape(rs: dict, label: str) -> str | None:
    g = group_of(rs, label)
    return rs.get("shape_by_group", {}).get(g) if g else None


def expected_z(rs: dict, label: str) -> int | None:
    label = rs.get("aliases", {}).get(label, label)
    z, g = rs.get("z_order", {}), group_of(rs, label)
    if label in z:
        return z[label]
    if g in z:
        return z[g]
    if g == "curb" or g == "crosswalk":
        return z.get("lane")
    return z.get("other_stuff") if g == "stuff" else None


def all_known_labels(rs: dict) -> list[str]:
    return [l for labs in rs.get("groups", {}).values() for l in labs]
