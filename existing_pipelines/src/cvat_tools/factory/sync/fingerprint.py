"""Deterministic annotation fingerprinting, geometry comparison, and minimal PATCH builders."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _round_pts(points: list[float] | None, decimals: int = 3) -> list[float]:
    if not points:
        return []
    return [round(float(x), decimals) for x in points]


def normalize_element(el: dict[str, Any]) -> dict[str, Any]:
    """Canonical representation of a skeleton element for hashing and comparison."""
    attrs = sorted(
        (str(a.get("spec_id", "")), str(a.get("value", "")))
        for a in (el.get("attributes") or [])
    )
    return {
        "id": el.get("id"),
        "label_id": el.get("label_id"),
        "type": el.get("type", "points"),
        "points": _round_pts(el.get("points")),
        "occluded": bool(el.get("occluded", False)),
        "outside": bool(el.get("outside", False)),
        "z_order": int(el.get("z_order", 0)),
        "attributes": attrs,
    }


def normalize_shape(shape: dict[str, Any], include_id: bool = True) -> dict[str, Any]:
    """Canonical representation of a CVAT shape."""
    attrs = sorted(
        (str(a.get("spec_id", "")), str(a.get("value", "")))
        for a in (shape.get("attributes") or [])
    )
    elements = [normalize_element(e) for e in (shape.get("elements") or [])]
    elements.sort(key=lambda e: (e.get("label_id") or 0, e.get("id") or 0))
    out: dict[str, Any] = {
        "frame": int(shape.get("frame", 0)),
        "label_id": shape.get("label_id"),
        "type": shape.get("type", ""),
        "points": _round_pts(shape.get("points")),
        "occluded": bool(shape.get("occluded", False)),
        "outside": bool(shape.get("outside", False)),
        "z_order": int(shape.get("z_order", 0)),
        "rotation": round(float(shape.get("rotation", 0.0)), 3),
        "group": int(shape.get("group") or 0),
        "attributes": attrs,
        "elements": elements,
    }
    if include_id:
        out["id"] = shape.get("id")
    return out


def compute_shape_fingerprint(shape: dict[str, Any], include_id: bool = True) -> str:
    """Return SHA256 hex digest of a normalized shape."""
    canonical = json.dumps(normalize_shape(shape, include_id=include_id), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_annotations_fingerprint(annotations: dict[str, Any]) -> str:
    """Return SHA256 hex digest of an entire CVAT job annotations payload."""
    shapes = [normalize_shape(s, include_id=True) for s in (annotations.get("shapes") or [])]
    shapes.sort(key=lambda s: (s["frame"], s.get("id") or 0, s.get("label_id") or 0))
    tracks = annotations.get("tracks") or []
    tags = annotations.get("tags") or []
    payload = {
        "version": annotations.get("version", 0),
        "shapes": shapes,
        "tracks_count": len(tracks),
        "tags_count": len(tags),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def vertex_deviation(pts_a: list[float] | None, pts_b: list[float] | None) -> float:
    """Compute max Euclidean vertex distance when point counts match, else chamfer distance."""
    a = pts_a or []
    b = pts_b or []
    if not a and not b:
        return 0.0
    if not a or not b:
        return float("inf")
    pa = [(a[i], a[i + 1]) for i in range(0, len(a) - 1, 2)]
    pb = [(b[i], b[i + 1]) for i in range(0, len(b) - 1, 2)]
    if len(pa) == len(pb):
        return max(math.hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(pa, pb))
    d_ab = max(min(math.hypot(x1 - x2, y1 - y2) for x2, y2 in pb) for x1, y1 in pa)
    d_ba = max(min(math.hypot(x1 - x2, y1 - y2) for x1, y1 in pa) for x2, y2 in pb)
    return max(d_ab, d_ba)


def geometry_matches(shape_a: dict[str, Any], shape_b: dict[str, Any], tol_px: float = 0.05) -> bool:
    """Check if two shapes have equivalent geometry (type, points, and elements)."""
    if shape_a.get("type") != shape_b.get("type"):
        return False
    if int(shape_a.get("frame", -1)) != int(shape_b.get("frame", -2)):
        return False
    if shape_a.get("label_id") != shape_b.get("label_id"):
        return False
    pts_a = shape_a.get("points") or []
    pts_b = shape_b.get("points") or []
    if len(pts_a) != len(pts_b):
        return False
    if pts_a and vertex_deviation(pts_a, pts_b) > tol_px:
        return False
    el_a = shape_a.get("elements") or []
    el_b = shape_b.get("elements") or []
    if len(el_a) != len(el_b):
        return False
    if el_a:
        by_lbl_a = {e.get("label_id"): e for e in el_a}
        by_lbl_b = {e.get("label_id"): e for e in el_b}
        if set(by_lbl_a) != set(by_lbl_b):
            return False
        for lid, ea in by_lbl_a.items():
            eb = by_lbl_b[lid]
            if vertex_deviation(ea.get("points"), eb.get("points")) > tol_px:
                return False
            if bool(ea.get("occluded", False)) != bool(eb.get("occluded", False)):
                return False
            if bool(ea.get("outside", False)) != bool(eb.get("outside", False)):
                return False
    return True


def compute_minimal_patch(
    live_shape: dict[str, Any],
    desired_shape: dict[str, Any],
    preserve_points: bool = True,
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]], list[dict[str, Any]]]:
    """Build a minimal PATCH body for an existing CVAT shape.

    Returns (patch_payload, changed_fields, element_diffs).
    Never drops untouched fields; preserves live points when preserve_points=True.
    Always forces source='manual' on mutated shapes/elements per CVAT conventions.
    """
    changed_fields: dict[str, tuple[Any, Any]] = {}
    element_diffs: list[dict[str, Any]] = []
    patch: dict[str, Any] = {
        "id": live_shape["id"],
        "frame": live_shape["frame"],
        "label_id": desired_shape.get("label_id", live_shape["label_id"]),
        "type": live_shape["type"],
        "source": "manual",
    }

    if not preserve_points and "points" in desired_shape:
        if vertex_deviation(live_shape.get("points"), desired_shape.get("points")) > 1e-3:
            changed_fields["points"] = (live_shape.get("points"), desired_shape["points"])
            patch["points"] = list(desired_shape["points"])
        else:
            patch["points"] = list(live_shape.get("points") or [])
    else:
        patch["points"] = list(live_shape.get("points") or [])

    for key in ("occluded", "outside", "z_order", "rotation", "group", "attributes"):
        if key in desired_shape and desired_shape[key] != live_shape.get(key):
            changed_fields[key] = (live_shape.get(key), desired_shape[key])
            patch[key] = desired_shape[key]
        elif key in live_shape:
            patch[key] = live_shape[key]

    live_elements = live_shape.get("elements") or []
    desired_elements = desired_shape.get("elements") or []
    if live_elements or desired_elements:
        desired_by_id = {e["id"]: e for e in desired_elements if e.get("id") is not None}
        desired_by_lid = {e["label_id"]: e for e in desired_elements if e.get("label_id") is not None}
        patched_elements: list[dict[str, Any]] = []
        for le in live_elements:
            de = desired_by_id.get(le.get("id")) or desired_by_lid.get(le.get("label_id"))
            new_el = dict(le)
            new_el["source"] = "manual"
            if de is not None:
                el_diff: dict[str, Any] = {"id": le.get("id"), "label_id": le.get("label_id"), "changes": {}}
                for flag in ("occluded", "outside"):
                    if flag in de and bool(de[flag]) != bool(le.get(flag, False)):
                        el_diff["changes"][flag] = (bool(le.get(flag, False)), bool(de[flag]))
                        new_el[flag] = bool(de[flag])
                if not preserve_points and "points" in de:
                    if vertex_deviation(le.get("points"), de.get("points")) > 1e-3:
                        el_diff["changes"]["points"] = (le.get("points"), de["points"])
                        new_el["points"] = list(de["points"])
                if el_diff["changes"]:
                    element_diffs.append(el_diff)
            patched_elements.append(new_el)
        patch["elements"] = patched_elements

    return patch, changed_fields, element_diffs
