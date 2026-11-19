"""Stages B, D, E: Selective Region Refinement + Self-Correction Loop (max 3 attempts)."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cvat_tools.autolabel.qc import REGIONS, enhance_if_dark, evaluate_frame_qc
from cvat_tools.pose.face50_engine import (
    _face_rotation_basis,
    _nudge_inner_lip_point,
    _proj,
    _state_str,
)


def _refine_inner_lip(kp_el: dict[int, tuple[dict[str, Any], dict[str, Any]]]) -> list[int]:
    """Selectively untangle moitrong (42..49) and ensure strict containment inside moingoai (30..41)."""
    # Mouth-local basis from outer corners 30 -> 36
    p30 = kp_el[30][1]["points"]
    p36 = kp_el[36][1]["points"]
    dx, dy = float(p36[0]) - float(p30[0]), float(p36[1]) - float(p30[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-3:
        return []
    ux, uy = dx / norm, dy / norm
    vx, vy = -uy, ux

    outer_poly = np.array([[kp_el[k][1]["points"][0], kp_el[k][1]["points"][1]] for k in range(30, 42)], dtype=np.float32)
    mcx = float(np.mean(outer_poly[:, 0]))
    mcy = float(np.mean(outer_poly[:, 1]))

    def to_uv(pt: list[float]) -> tuple[float, float]:
        rx, ry = float(pt[0]) - mcx, float(pt[1]) - mcy
        return rx * ux + ry * uy, rx * vx + ry * vy

    def to_xy(u: float, v: float) -> list[float]:
        return [round(mcx + u * ux + v * vx, 2), round(mcy + u * uy + v * vy, 2)]

    uv = {k: to_uv(kp_el[k][1]["points"]) for k in range(42, 50)}
    old_xy = {k: list(kp_el[k][1]["points"]) for k in range(42, 50)}

    # Ensure 42 is left-most, 46 is right-most along u
    if uv[42][0] > uv[46][0]:
        uv[42], uv[46] = uv[46], uv[42]

    # Sort upper trio (43, 44, 45) by increasing u; lower trio (49, 48, 47) by increasing u
    up_pts = sorted([uv[43], uv[44], uv[45]], key=lambda t: t[0])
    lo_pts = sorted([uv[49], uv[48], uv[47]], key=lambda t: t[0])

    # Prevent vertical crossover between paired upper & lower points
    new_up, new_lo = [], []
    for (uu, uv_y), (lu, lv_y) in zip(up_pts, lo_pts):
        if uv_y > lv_y - 0.6:
            mid = 0.5 * (uv_y + lv_y)
            uv_y = mid - 0.45
            lv_y = mid + 0.45
        new_up.append((uu, uv_y))
        new_lo.append((lu, lv_y))

    # Clamp u of interior points strictly between 42 and 46
    u_min, u_max = uv[42][0] + 0.5, uv[46][0] - 0.5
    if u_min < u_max:
        for idx, (uu, uy_v) in enumerate(new_up):
            new_up[idx] = (min(max(uu, u_min), u_max), uy_v)
        for idx, (lu, ly_v) in enumerate(new_lo):
            new_lo[idx] = (min(max(lu, u_min), u_max), ly_v)

    assigned = {
        42: uv[42],
        43: new_up[0], 44: new_up[1], 45: new_up[2],
        46: uv[46],
        47: new_lo[2], 48: new_lo[1], 49: new_lo[0],
    }

    changed: list[int] = []
    for k, (uk, vk) in assigned.items():
        cand = to_xy(uk, vk)
        # Ensure strict containment inside moingoai (>= 0.5 px)
        if cv2.pointPolygonTest(outer_poly, (cand[0], cand[1]), True) < 0.5:
            nx, ny = _nudge_inner_lip_point((cand[0], cand[1]), (mcx, mcy), outer_poly)
            cand = [nx, ny]
        if abs(cand[0] - old_xy[k][0]) > 0.05 or abs(cand[1] - old_xy[k][1]) > 0.05:
            kp_el[k][1]["points"] = cand
            changed.append(k)
    return changed


def _refine_brow(
    kp_el: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    region_name: str,
    img_bgr: np.ndarray | None,
    luma: float,
) -> list[int]:
    """Selectively fix brow monotonicity and keep brow strictly above upper eyelid in face-local coords."""
    indices = REGIONS[region_name]
    eye_upper_ids = (15, 16, 17) if region_name == "brow_L" else (23, 24, 25)
    pts = {k: {"x": float(kp_el[k][1]["points"][0]), "y": float(kp_el[k][1]["points"][1])} for k in range(50)}
    u_axis, v_axis, _ = _face_rotation_basis(pts)
    origin = (pts[10]["x"], pts[10]["y"])
    ux, uy = u_axis
    vx, vy = v_axis

    def to_xy(u: float, v: float) -> list[float]:
        return [round(origin[0] + u * ux + v * vx, 2), round(origin[1] + u * uy + v * vy, 2)]

    eye_top_v = min(_proj(pts[k], origin, u_axis, v_axis)[1] for k in eye_upper_ids)
    uv_brow = [_proj(pts[k], origin, u_axis, v_axis) for k in indices]

    # Sort u strictly monotonically (0->4 or 5->9) and clamp v <= eye_top_v - 3.2
    us_sorted = sorted(t[0] for t in uv_brow)
    for i in range(1, len(us_sorted)):
        if us_sorted[i] <= us_sorted[i - 1] + 0.35:
            us_sorted[i] = us_sorted[i - 1] + 0.35
    vs = [min(t[1], eye_top_v - 3.2) for t in uv_brow]

    # Optional gradient snap in brow ROI if image provided
    if img_bgr is not None:
        enh = enhance_if_dark(img_bgr, luma)
        gray = cv2.GaussianBlur(cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0.8)
        h, w = gray.shape[:2]
        snapped_vs = []
        for u_val, v_val in zip(us_sorted, vs):
            best_v, best_sc = v_val, -1e9
            for dv in (-2.0, -1.0, 0.0, 1.0, 2.0):
                vc = min(v_val + dv, eye_top_v - 3.2)
                cx, cy = origin[0] + u_val * ux + vc * vx, origin[1] + u_val * uy + vc * vy
                ax, ay = int(round(np.clip(cx - 2.0 * vx, 0, w - 1))), int(round(np.clip(cy - 2.0 * vy, 0, h - 1)))
                bx, by = int(round(np.clip(cx + 2.0 * vx, 0, w - 1))), int(round(np.clip(cy + 2.0 * vy, 0, h - 1)))
                sc = (float(gray[ay, ax]) - float(gray[by, bx])) - 0.1 * float(gray[by, bx])
                if sc > best_sc:
                    best_sc = sc
                    best_v = vc
            snapped_vs.append(best_v)
        vs = snapped_vs

    changed: list[int] = []
    for k, u_val, v_val in zip(indices, us_sorted, vs):
        old = list(kp_el[k][1]["points"])
        cand = to_xy(u_val, v_val)
        if abs(cand[0] - old[0]) > 0.05 or abs(cand[1] - old[1]) > 0.05:
            kp_el[k][1]["points"] = cand
            changed.append(k)
    return changed


def selective_refine_frame(
    frame_idx: int,
    shapes: list[dict[str, Any]],
    sub_id_to_kp: dict[int, int],
    lbl_id_to_name: dict[int, str],
    width: int,
    height: int,
    img_path: Path | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Run selective region refinement with self-correction loop (no regression on PASS regions)."""
    before_qc = evaluate_frame_qc(frame_idx, shapes, sub_id_to_kp, lbl_id_to_name, width, height, img_path)
    refine_regions = [r for r, st in before_qc["regions"].items() if st == "REFINE"]
    review_regions = [r for r, st in before_qc["regions"].items() if st == "REVIEW"]

    if not shapes or not refine_regions:
        return {
            "frame": frame_idx,
            "status": "review" if (review_regions or before_qc["error_count"] > 0) else "pass",
            "before_qc": before_qc,
            "after_qc": before_qc,
            "changed_kpts": [],
            "proposed_shapes": [],
            "before_shapes": [],
            "attempts": 0,
        }

    img_bgr = cv2.imread(str(img_path)) if (img_path is not None and img_path.exists()) else None
    luma = float(before_qc["img_stats"].get("luma", 100.0))

    best_shapes = copy.deepcopy(shapes)
    best_qc = before_qc
    changed_kpts: set[int] = set()
    attempts = 0

    for _ in range(max_attempts):
        attempts += 1
        cand_shapes = copy.deepcopy(best_shapes)
        kp_el: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for sh in cand_shapes:
            for el in sh.get("elements") or []:
                k = sub_id_to_kp.get(int(el["label_id"]))
                if k is not None:
                    kp_el[k] = (sh, el)

        if len(kp_el) != 50:
            break

        step_changed: list[int] = []
        if best_qc["regions"].get("inner_lip") == "REFINE":
            step_changed.extend(_refine_inner_lip(kp_el))
        for brow_r in ("brow_L", "brow_R"):
            if best_qc["regions"].get(brow_r) == "REFINE":
                step_changed.extend(_refine_brow(kp_el, brow_r, img_bgr, luma))

        if not step_changed:
            break

        cand_qc = evaluate_frame_qc(frame_idx, cand_shapes, sub_id_to_kp, lbl_id_to_name, width, height, img_path)

        # Verify zero regression on any PASS region
        regressed = any(
            before_qc["regions"][r] == "PASS" and cand_qc["regions"][r] != "PASS"
            for r in REGIONS
        )
        improved = (cand_qc["error_count"] < best_qc["error_count"]) or (
            cand_qc["error_count"] == best_qc["error_count"] and cand_qc["warning_count"] < best_qc["warning_count"]
        )
        if not regressed and improved:
            best_shapes = cand_shapes
            best_qc = cand_qc
            changed_kpts.update(step_changed)
            if best_qc["error_count"] == 0 and best_qc["warning_count"] == 0:
                break
        else:
            break

    if not changed_kpts:
        return {
            "frame": frame_idx,
            "status": "review" if (best_qc["error_count"] > 0 or review_regions) else "pass",
            "before_qc": before_qc,
            "after_qc": best_qc,
            "changed_kpts": [],
            "proposed_shapes": [],
            "before_shapes": [],
            "attempts": attempts,
        }

    # Only return shapes that actually changed
    changed_ids = set()
    for sh_b, sh_a in zip(shapes, best_shapes):
        if sh_b["elements"] != sh_a["elements"]:
            changed_ids.add(int(sh_a["id"]))

    status = "eligible" if (best_qc["error_count"] == 0 and not review_regions) else "review"
    return {
        "frame": frame_idx,
        "status": status,
        "before_qc": before_qc,
        "after_qc": best_qc,
        "changed_kpts": sorted(changed_kpts),
        "before_shapes": [s for s in shapes if int(s["id"]) in changed_ids],
        "proposed_shapes": [s for s in best_shapes if int(s["id"]) in changed_ids],
        "all_before_shapes": shapes,
        "all_proposed_shapes": best_shapes,
        "attempts": attempts,
    }
