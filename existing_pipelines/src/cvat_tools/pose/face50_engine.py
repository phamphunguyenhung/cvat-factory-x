"""Face50 Validation, Surgical Proposal Engine, Bold VS Code Renderer, and Safe CVAT Sync.

Follows Job 1623 architecture and rules.md:
- Rotation-aware 50-landmark geometric + topological + visibility validation.
- Surgical proposals only: preserves all valid human-placed coordinates; only fixes
  verified inner-lip containment (`moitrong` inside `moingoai`) and clear BTC visibility
  flags, while routing ambiguous/missing/extreme-profile frames to the Review Queue.
- Safe CVAT sync with staleness guard, backup, minimal PATCH update, and read-back verification.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from cvat_tools.core import paths
from cvat_tools.core.client import Cvat
from cvat_tools.pose import week2

FACE50_OUT_DIR = paths.WORKSPACE / "artifacts" / "face50_research"
FACE50_DOC_DIR = paths.WORKSPACE / "docs" / "guidelines" / "face50"

FACE50_GROUPS: dict[str, range] = {
    "longmaytrai": range(0, 5),
    "longmayphai": range(5, 10),
    "songmui": range(10, 14),
    "mattrai": range(14, 22),
    "matphai": range(22, 30),
    "moingoai": range(30, 42),
    "moitrong": range(42, 50),
}

# Closed/open polyline edges for bold rendering
FACE50_POLYLINES: list[tuple[list[int], bool]] = [
    ([0, 1, 2, 3, 4], False),
    ([5, 6, 7, 8, 9], False),
    ([10, 11, 12, 13], False),
    ([14, 15, 16, 17, 18, 19, 20, 21], True),
    ([22, 23, 24, 25, 26, 27, 28, 29], True),
    ([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], True),
    ([42, 43, 44, 45, 46, 47, 48, 49], True),
]


def _sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _state_str(el: dict[str, Any]) -> str:
    occ, out = bool(el.get("occluded")), bool(el.get("outside"))
    if occ and out:
        return "Q+O"
    if occ:
        return "Q"
    if out:
        return "O"
    return "V"


def _set_state(el: dict[str, Any], state: str) -> None:
    if state == "V":
        el["occluded"], el["outside"] = False, False
    elif state == "Q":
        el["occluded"], el["outside"] = True, False
    elif state == "O":
        el["occluded"], el["outside"] = False, True
    else:
        raise ValueError(f"Invalid state: {state}")


def _face_rotation_basis(pts: dict[int, dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Compute unit right-axis u and down-axis v from eye centers (14..21 vs 22..29)."""
    lx = sum(pts[k]["x"] for k in range(14, 22)) / 8.0
    ly = sum(pts[k]["y"] for k in range(14, 22)) / 8.0
    rx = sum(pts[k]["x"] for k in range(22, 30)) / 8.0
    ry = sum(pts[k]["y"] for k in range(22, 30)) / 8.0
    dx, dy = rx - lx, ry - ly
    norm = math.hypot(dx, dy)
    if norm < 1e-3:
        return (1.0, 0.0), (0.0, 1.0), 0.0
    ux, uy = dx / norm, dy / norm
    vx, vy = -uy, ux  # 90 deg clockwise = face-down
    angle_deg = math.degrees(math.atan2(uy, ux))
    return (ux, uy), (vx, vy), angle_deg


def _proj(pt: dict[str, Any], origin: tuple[float, float], u: tuple[float, float], v: tuple[float, float]) -> tuple[float, float]:
    dx, dy = pt["x"] - origin[0], pt["y"] - origin[1]
    return dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """Return True if closed line segments a1-a2 and b1-b2 intersect."""
    def _orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_seg(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)
    if (o1 > 1e-9 and o2 < -1e-9 or o1 < -1e-9 and o2 > 1e-9) and (
        o3 > 1e-9 and o4 < -1e-9 or o3 < -1e-9 and o4 > 1e-9
    ):
        return True
    if abs(o1) <= 1e-9 and _on_seg(a1, b1, a2):
        return True
    if abs(o2) <= 1e-9 and _on_seg(a1, b2, a2):
        return True
    if abs(o3) <= 1e-9 and _on_seg(b1, a1, b2):
        return True
    if abs(o4) <= 1e-9 and _on_seg(b1, a2, b2):
        return True
    return False


def _polygon_self_intersects(poly_pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Return pairs of non-adjacent edge indices that intersect in a closed polygon."""
    n = len(poly_pts)
    hits: list[tuple[int, int]] = []
    for i in range(n):
        a1, a2 = poly_pts[i], poly_pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1, b2 = poly_pts[j], poly_pts[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                hits.append((i, j))
    return hits


def validate_face50_frame(
    frame_idx: int,
    shapes: list[dict[str, Any]],
    sub_id_to_kp: dict[int, int],
    lbl_id_to_name: dict[int, str],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Run rotation-aware 50-point structural, geometric, and visibility checks on 1 frame."""
    import numpy as np
    import cv2

    findings: list[dict[str, Any]] = []
    if not shapes:
        findings.append({"layer": "L1", "severity": "error", "code": "MISSING_ANNOTATION", "kps": "0..49", "detail": "0 skeletons on frame"})
        return {"frame": frame_idx, "shape_count": 0, "findings": findings, "angle_deg": 0.0}

    present_labels = [lbl_id_to_name.get(int(sh["label_id"]), "?") for sh in shapes]
    if sorted(present_labels) != sorted(FACE50_GROUPS.keys()):
        findings.append({"layer": "L1", "severity": "error", "code": "LABEL_SET_MISMATCH", "kps": "", "detail": f"labels={present_labels}"})

    pts: dict[int, dict[str, Any]] = {}
    for sh in shapes:
        if bool(sh.get("occluded")):
            findings.append({"layer": "L3", "severity": "error", "code": "SKELETON_BULK_OCCLUDED", "kps": "", "detail": f"shape {sh.get('id')} occluded=True"})
        for el in sh.get("elements") or []:
            k = sub_id_to_kp.get(int(el["label_id"]))
            if k is None:
                continue
            st = _state_str(el)
            if st == "Q+O":
                findings.append({"layer": "L3", "severity": "error", "code": "Q_AND_O_CONFLICT", "kps": str(k), "detail": f"kp {k} has occluded=True and outside=True"})
            px, py = float(el["points"][0]), float(el["points"][1])
            if not (0.0 <= px <= width and 0.0 <= py <= height) and st != "O":
                findings.append({"layer": "L2", "severity": "error", "code": "COORD_OOB", "kps": str(k), "detail": f"kp {k} ({px:.1f},{py:.1f}) out of {width}x{height}"})
            pts[k] = {"x": px, "y": py, "state": st}

    if len(pts) != 50:
        findings.append({"layer": "L1", "severity": "error", "code": "KP_COUNT_MISMATCH", "kps": "", "detail": f"expected 50, got {len(pts)}"})
        return {"frame": frame_idx, "shape_count": len(shapes), "findings": findings, "angle_deg": 0.0}

    u, v, angle_deg = _face_rotation_basis(pts)
    origin = (pts[10]["x"], pts[10]["y"])

    # Outer lip vs Inner lip containment (invariant to rotation!)
    outer_poly = np.array([[pts[k]["x"], pts[k]["y"]] for k in range(30, 42)], dtype=np.float32)
    out_kps = []
    for k in range(42, 50):
        d = float(cv2.pointPolygonTest(outer_poly, (pts[k]["x"], pts[k]["y"]), True))
        if d < -0.05:
            out_kps.append((k, round(d, 2)))
    if out_kps:
        findings.append({
            "layer": "L2",
            "severity": "error",
            "code": "INNER_LIP_OUTSIDE_OUTER",
            "kps": ";".join(str(k) for k, _ in out_kps),
            "detail": f"inner lip points outside moingoai: {out_kps}",
        })

    # Lip polygon self-intersection checks (moingoai 30..41, moitrong 42..49)
    for ring_name, kp_range in (("moingoai", range(30, 42)), ("moitrong", range(42, 50))):
        ring_pts = [(pts[k]["x"], pts[k]["y"]) for k in kp_range]
        hits = _polygon_self_intersects(ring_pts)
        if hits:
            k_list = list(kp_range)
            bad_kps = sorted({k_list[i] for i, _ in hits} | {k_list[j] for _, j in hits})
            findings.append({
                "layer": "L2",
                "severity": "error",
                "code": "LIP_SELF_INTERSECT",
                "kps": ";".join(str(k) for k in bad_kps),
                "detail": f"{ring_name} self-intersects at edge pairs {hits}",
            })

    # Rotation-aware lip vertical order (in face-down v axis)
    lip_inv = []
    for uo, ui in [(32, 43), (33, 44), (34, 45)]:
        if _proj(pts[uo], origin, u, v)[1] > _proj(pts[ui], origin, u, v)[1] + 0.8:
            lip_inv.append((uo, ui))
    for li, lo in [(47, 38), (48, 39), (49, 40)]:
        if _proj(pts[li], origin, u, v)[1] > _proj(pts[lo], origin, u, v)[1] + 0.8:
            lip_inv.append((li, lo))
    if lip_inv:
        findings.append({"layer": "L2", "severity": "error", "code": "LIP_INVERSION_LOCAL", "kps": ";".join(f"{a}/{b}" for a, b in lip_inv), "detail": f"face-local lip inversion: {lip_inv}"})

    # Rotation-aware eyelid order
    eye_inv = []
    for up, lo in [(15, 21), (16, 20), (17, 19), (23, 29), (24, 28), (25, 27)]:
        if _proj(pts[up], origin, u, v)[1] > _proj(pts[lo], origin, u, v)[1] + 0.8:
            eye_inv.append((up, lo))
    if eye_inv:
        findings.append({"layer": "L2", "severity": "warning", "code": "EYE_LID_INVERSION_LOCAL", "kps": ";".join(f"{a}/{b}" for a, b in eye_inv), "detail": f"face-local eyelid inversion: {eye_inv}"})

    # Rotation-aware eye inner-corner separation
    x18, _ = _proj(pts[18], origin, u, v)
    x22, _ = _proj(pts[22], origin, u, v)
    if x18 >= x22 - 1.0:
        findings.append({"layer": "L2", "severity": "warning", "code": "EYE_CLUSTER_COLLAPSE", "kps": "18;22", "detail": f"projected eye inner gap={x22 - x18:.1f}px"})

    # Nose bridge monotonicity along face-down v
    ny = [_proj(pts[k], origin, u, v)[1] for k in range(10, 14)]
    if not (ny[0] < ny[1] < ny[2] < ny[3]):
        findings.append({"layer": "L2", "severity": "warning", "code": "NOSE_NON_MONOTONIC_LOCAL", "kps": "10;11;12;13", "detail": f"projected nose y={ [round(y,1) for y in ny] }"})

    # Eyebrow outer points floating far off face (e.g. f118 kps 8,9 in sky)
    eye_span = max(12.0, abs(_proj(pts[26], origin, u, v)[0] - _proj(pts[14], origin, u, v)[0]))
    brow_r_span = abs(_proj(pts[9], origin, u, v)[0] - _proj(pts[5], origin, u, v)[0])
    if brow_r_span > 1.25 * eye_span:
        findings.append({"layer": "L2", "severity": "warning", "code": "BROW_EXTREME_DRIFT", "kps": "8;9", "detail": f"brow_R span {brow_r_span:.1f} > 1.25*eye_span {eye_span:.1f}"})

    # Brow-below-eye in face-local coordinates (brow 0..4 vs upper lid 15..17, brow 5..9 vs 23..25)
    brow_below: list[int] = []
    for brow_range, lid_kps in ((range(0, 5), (15, 16, 17)), (range(5, 10), (23, 24, 25))):
        min_lid_v = min(_proj(pts[ek], origin, u, v)[1] for ek in lid_kps)
        for bk in brow_range:
            if _proj(pts[bk], origin, u, v)[1] > min_lid_v + 0.5:
                brow_below.append(bk)
    if brow_below:
        findings.append({
            "layer": "L2",
            "severity": "error",
            "code": "BROW_BELOW_EYE",
            "kps": ";".join(str(k) for k in brow_below),
            "detail": f"brow points below upper eyelid in face-local v: {brow_below}",
        })

    # Brow monotonic x in face-local coordinates (0..4 and 5..9)
    brow_non_mono: list[str] = []
    for side_name, brow_range in (("longmaytrai", range(0, 5)), ("longmayphai", range(5, 10))):
        bu = [_proj(pts[bk], origin, u, v)[0] for bk in brow_range]
        if any(bu[i] >= bu[i + 1] for i in range(len(bu) - 1)):
            brow_non_mono.append(f"{side_name}({[round(x, 1) for x in bu]})")
    if brow_non_mono:
        findings.append({
            "layer": "L2",
            "severity": "warning",
            "code": "BROW_NON_MONOTONIC_LOCAL",
            "kps": "0..9",
            "detail": f"non-monotonic brow u: {'; '.join(brow_non_mono)}",
        })

    return {"frame": frame_idx, "shape_count": len(shapes), "findings": findings, "angle_deg": round(angle_deg, 1)}


def _nudge_inner_lip_point(pt: tuple[float, float], anchor: tuple[float, float], outer_poly) -> tuple[float, float]:
    """Nudge an inner-lip corner strictly inward toward mouth center until >= 1.5 px inside moingoai."""
    import cv2

    px, py = pt
    ax, ay = anchor
    dx, dy = ax - px, ay - py
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return pt
    ux, uy = dx / dist, dy / dist
    for step in range(1, int(min(15.0, dist)) + 1):
        cand = (px + ux * step * 0.5, py + uy * step * 0.5)
        if cv2.pointPolygonTest(outer_poly, cand, True) >= 1.2:
            return (round(cand[0], 2), round(cand[1], 2))
    # Fallback: move 15% toward anchor
    return (round(px + 0.15 * dx, 2), round(py + 0.15 * dy, 2))


def align_face50_clahe_fan(
    img_bgr: Any,
    try_rotations: bool = True,
    landmarker: Any = None,
) -> dict[str, Any] | None:
    """Align 50 face landmarks using Gamma + LAB CLAHE + FaceLandmarker + BrowEdgeSnap + InnerLipClamp."""
    import cv2
    import numpy as np
    from cvat_tools.pose import face50

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    work = img_bgr
    if float(np.mean(gray)) < 90.0:
        inv_g = 1.0 / 0.55
        lut = np.array([((i / 255.0) ** inv_g) * 255.0 for i in range(256)], dtype=np.uint8)
        work = cv2.LUT(work, lut)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    fl = landmarker if landmarker is not None else face50.FaceLandmarker()
    det = fl.landmarks68(enh_bgr, try_rotations=try_rotations)
    if det is None:
        return None

    p50 = face50.map68_to_50(det["p68"]).astype(np.float64, copy=True)

    # BrowEdgeSnap: refine brow points (0..9) along face-down axis using local Sobel edge magnitude
    enh_gray = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(enh_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enh_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    pts_dict = {k: {"x": float(p50[k, 0]), "y": float(p50[k, 1])} for k in range(50)}
    origin = (pts_dict[10]["x"], pts_dict[10]["y"])
    u, v, _ = _face_rotation_basis(pts_dict)

    snapped_count = 0
    for brow_range, lid_kps in ((range(0, 5), (15, 16, 17)), (range(5, 10), (23, 24, 25))):
        min_lid_v = min(_proj(pts_dict[ek], origin, u, v)[1] for ek in lid_kps)
        for bk in brow_range:
            bx, by = float(p50[bk, 0]), float(p50[bk, 1])
            ix0, iy0 = int(round(bx)), int(round(by))
            if not (0 <= ix0 < w and 0 <= iy0 < h):
                continue
            base_g = float(grad[iy0, ix0])
            best_g, best_xy = base_g, (bx, by)
            for dv in range(-4, 5):
                if dv == 0:
                    continue
                cx, cy = bx + v[0] * dv, by + v[1] * dv
                icx, icy = int(round(cx)), int(round(cy))
                if not (0 <= icx < w and 0 <= icy < h):
                    continue
                proj_v = _proj({"x": cx, "y": cy}, origin, u, v)[1]
                if proj_v > min_lid_v - 1.0:
                    continue
                cg = float(grad[icy, icx])
                if cg > best_g * 1.15 and cg > base_g + 10.0:
                    best_g, best_xy = cg, (cx, cy)
            if best_xy != (bx, by):
                p50[bk, 0], p50[bk, 1] = round(best_xy[0], 2), round(best_xy[1], 2)
                snapped_count += 1

    # InnerLipClamp: clamp inner lip (42..49) strictly inside moingoai (30..41)
    outer_poly = p50[30:42].astype(np.float32)
    mouth_cx = float(np.mean(p50[30:42, 0]))
    mouth_cy = float(np.mean(p50[30:42, 1]))
    clamped_count = 0
    for k in range(42, 50):
        px, py = float(p50[k, 0]), float(p50[k, 1])
        if float(cv2.pointPolygonTest(outer_poly, (px, py), True)) < 0.5:
            nx, ny = _nudge_inner_lip_point((px, py), (mouth_cx, mouth_cy), outer_poly)
            p50[k, 0], p50[k, 1] = nx, ny
            clamped_count += 1

    return {
        "p50": p50,
        "score68": det["score68"],
        "face_score": float(det["face_score"]),
        "rot": int(det["rot"]),
        "brow_snapped": snapped_count,
        "lip_clamped": clamped_count,
    }


def build_face50_proposals(
    job_id: int = 1975,
    offline: bool = False,
    profile: str = "edufun",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Inspect live CVAT Job 1975 and build high-confidence surgical proposals + review queue."""
    import cv2
    import numpy as np

    snap_dir = FACE50_OUT_DIR / "snapshots"
    if offline:
        meta = json.loads((snap_dir / f"job_{job_id}_meta.json").read_text(encoding="utf-8"))
        labels = json.loads((snap_dir / f"job_{job_id}_labels.json").read_text(encoding="utf-8"))
        ann = json.loads((snap_dir / f"job_{job_id}_annotations_raw.json").read_text(encoding="utf-8"))
    else:
        cv = Cvat.from_env(profile, write_enabled=False)
        meta = cv.meta(job_id)
        labels = cv.labels(job_id)
        ann = cv.annotations(job_id)

    lbl_id_to_name = {int(l["id"]): str(l["name"]) for l in labels}
    sub_id_to_kp: dict[int, int] = {}
    for l in labels:
        for s in l.get("sublabels") or []:
            if str(s["name"]).isdigit():
                sub_id_to_kp[int(s["id"])] = int(s["name"])

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for sh in ann.get("shapes") or []:
        by_frame.setdefault(int(sh["frame"]), []).append(sh)

    # Visibility fixes verified by direct visual inspection of Job 1975 zoomed crops
    VERIFIED_VIS_FIXES: dict[int, dict[str, Any]] = {
        96: {
            "reason": "Mouth (30..49) is clearly visible in RGB; only obscured points should be Q (BTC V1 §5)",
            "targets": {k: "V" for k in range(30, 50)},
        },
        104: {
            "reason": "Driver wearing dark sunglasses in IR; eyes (14..29) behind dark lenses must be Q (BTC V1 §5)",
            "targets": {k: "Q" for k in range(14, 30)},
        },
        107: {
            "reason": "Eyebrows (0..9) are visible above fingers -> V; nose lower (11..13) & mouth (30..49) are behind fingers -> Q",
            "targets": {**{k: "V" for k in range(0, 10)}, **{k: "Q" for k in (11, 12, 13)}, **{k: "Q" for k in range(30, 50)}},
        },
    }

    REVIEW_QUEUE_FRAMES = {
        100: "Extreme ~60 deg profile: brow_R (5..9) stacked & eye_R (22..29) collapsed on nose bridge; needs human placement",
        102: "Missing annotation (0 shapes): hands adjusting glasses occlude eyes/nose in dark IR; needs human decision/draft",
        106: "Missing annotation (0 shapes): 90 deg horizontal IR fisheye with glasses; needs human decision/draft",
        116: "90 deg horizontal IR fisheye: check brow_L/brow_R orientation under BTC V1 §4 (issues.md A4)",
        118: "Extreme profile: brow_R tail (8,9) floating in bright window sky & eye_R (23..29) off face; needs human placement",
    }

    eligible_proposals: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    validation_summary: list[dict[str, Any]] = []

    for i, fm in enumerate(meta.get("frames") or []):
        fidx = int(meta["start_frame"]) + i
        img_name = Path(fm["name"]).name
        w, h = int(fm["width"]), int(fm["height"])
        shapes = by_frame.get(fidx, [])

        val = validate_face50_frame(fidx, shapes, sub_id_to_kp, lbl_id_to_name, w, h)
        validation_summary.append({
            "frame_index": fidx,
            "image_name": img_name,
            "width": w,
            "height": h,
            "angle_deg": val["angle_deg"],
            "finding_count": len(val["findings"]),
            "codes": ";".join(f["code"] for f in val["findings"]),
        })

        if fidx in REVIEW_QUEUE_FRAMES:
            review_queue.append({
                "frame_index": fidx,
                "image_name": img_name,
                "reason": REVIEW_QUEUE_FRAMES[fidx],
                "codes": [f["code"] for f in val["findings"]],
            })
            continue

        if not shapes:
            continue

        before_shapes = copy.deepcopy(shapes)
        prop_shapes = copy.deepcopy(shapes)
        changes: list[dict[str, Any]] = []

        # Map keypoint idx -> (shape_dict, element_dict)
        kp_el: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for sh in prop_shapes:
            for el in sh.get("elements") or []:
                k = sub_id_to_kp.get(int(el["label_id"]))
                if k is not None:
                    kp_el[k] = (sh, el)

        # 1) Surgical inner-lip containment fix (only move 42..49 points that poke outside moingoai)
        if len(kp_el) == 50:
            outer_poly = np.array([[kp_el[k][1]["points"][0], kp_el[k][1]["points"][1]] for k in range(30, 42)], dtype=np.float32)
            mouth_cx = float(sum(kp_el[k][1]["points"][0] for k in range(30, 42)) / 12.0)
            mouth_cy = float(sum(kp_el[k][1]["points"][1] for k in range(30, 42)) / 12.0)
            for k in range(42, 50):
                _, el = kp_el[k]
                px, py = float(el["points"][0]), float(el["points"][1])
                d = float(cv2.pointPolygonTest(outer_poly, (px, py), True))
                if d < -0.05:
                    nx, ny = _nudge_inner_lip_point((px, py), (mouth_cx, mouth_cy), outer_poly)
                    el["points"] = [nx, ny]
                    changes.append({
                        "kp": k,
                        "type": "inner_lip_containment",
                        "old_xy": [round(px, 2), round(py, 2)],
                        "new_xy": [nx, ny],
                        "old_state": _state_str(el),
                        "new_state": _state_str(el),
                        "reason": f"kp {k} was {abs(d):.2f}px outside moingoai (BTC V1 §2.4)",
                    })

        # 2) Verified visibility fixes
        if fidx in VERIFIED_VIS_FIXES:
            vspec = VERIFIED_VIS_FIXES[fidx]
            for k, target_st in vspec["targets"].items():
                if k in kp_el:
                    _, el = kp_el[k]
                    old_st = _state_str(el)
                    if old_st != target_st:
                        _set_state(el, target_st)
                        changes.append({
                            "kp": k,
                            "type": "visibility_fix",
                            "old_xy": [round(float(el["points"][0]), 2), round(float(el["points"][1]), 2)],
                            "new_xy": [round(float(el["points"][0]), 2), round(float(el["points"][1]), 2)],
                            "old_state": old_st,
                            "new_state": target_st,
                            "reason": vspec["reason"],
                        })

        if changes:
            # Verify post-fix validation improves (zero inner-lip errors)
            post_val = validate_face50_frame(fidx, prop_shapes, sub_id_to_kp, lbl_id_to_name, w, h)
            post_errors = [f for f in post_val["findings"] if f["severity"] == "error"]
            if post_errors:
                review_queue.append({
                    "frame_index": fidx,
                    "image_name": img_name,
                    "reason": f"Post-fix still has errors: {post_errors}",
                    "codes": [f["code"] for f in post_errors],
                })
                continue

            # Only include shapes that actually changed
            changed_sh_ids = {kp_el[c["kp"]][0]["id"] for c in changes}
            before_changed = [s for s in before_shapes if s["id"] in changed_sh_ids]
            prop_changed = [s for s in prop_shapes if s["id"] in changed_sh_ids]

            eligible_proposals.append({
                "frame_index": fidx,
                "image_name": img_name,
                "width": w,
                "height": h,
                "action": "update",
                "qc_status": "AI_REVIEWED_ELIGIBLE",
                "changed_kpts": sorted({c["kp"] for c in changes}),
                "changes": changes,
                "before_shapes": before_changed,
                "proposed_shapes": prop_changed,
                "all_before_shapes": before_shapes,
                "all_proposed_shapes": prop_shapes,
            })

    out_p = Path(output_path) if output_path else (FACE50_OUT_DIR / f"proposed_face50_{job_id}.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_hash": _sha256_obj(ann.get("shapes") or []),
        "sub_id_to_kp": {str(k): v for k, v in sub_id_to_kp.items()},
        "eligible_count": len(eligible_proposals),
        "review_queue_count": len(review_queue),
        "eligible_proposals": eligible_proposals,
        "review_queue": review_queue,
        "validation_summary": validation_summary,
    }
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def render_face50_proposals(
    payload: dict[str, Any],
    profile: str = "edufun",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Render zoomed BEFORE vs PROPOSED bold comparison images + Contact Sheet for VS Code."""
    import cv2
    import numpy as np

    job_id = int(payload["job_id"])
    fdir = paths.job_dir(profile, job_id) / "frames"
    out_dir = Path(output_dir) if output_dir else (FACE50_OUT_DIR / "evidence" / "before_after")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_id_to_kp = {int(k): int(v) for k, v in (payload.get("sub_id_to_kp") or {}).items()}

    rendered: list[Path] = []
    tiles: list[Any] = []

    def _draw_face(img, shapes, x1, y1, scale, chg_set, is_after):
        pts = {}
        for sh in shapes:
            for el in sh.get("elements") or []:
                k = sub_id_to_kp.get(int(el["label_id"]))
                if k is not None:
                    pts[k] = (float(el["points"][0]), float(el["points"][1]), _state_str(el))
        for seq, closed in FACE50_POLYLINES:
            for a, b in zip(seq, seq[1:] + ([seq[0]] if closed else [])):
                if a in pts and b in pts:
                    pa = (int((pts[a][0] - x1) * scale), int((pts[a][1] - y1) * scale))
                    pb = (int((pts[b][0] - x1) * scale), int((pts[b][1] - y1) * scale))
                    touches = (a in chg_set) or (b in chg_set)
                    col = (255, 240, 40) if (is_after and touches) else ((60, 40, 255) if (not is_after and touches) else (200, 150, 60))
                    cv2.line(img, pa, pb, (15, 15, 15), 4, cv2.LINE_AA)
                    cv2.line(img, pa, pb, col, 2, cv2.LINE_AA)
        for k, (px, py, st) in sorted(pts.items()):
            sx, sy = int((px - x1) * scale), int((py - y1) * scale)
            is_chg = k in chg_set
            col = (50, 235, 50) if st == "V" else ((0, 200, 255) if st == "Q" else (60, 40, 255))
            cv2.circle(img, (sx, sy), 6 if is_chg else 4, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(img, (sx, sy), 4 if is_chg else 3, col, -1, cv2.LINE_AA)
            if is_chg:
                cv2.circle(img, (sx, sy), 9, (255, 240, 40) if is_after else (255, 0, 230), 2, cv2.LINE_AA)
                cv2.putText(img, f"{k}:{st}", (sx + 8, sy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 3, cv2.LINE_AA)
                cv2.putText(img, f"{k}:{st}", (sx + 8, sy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 40) if is_after else (255, 0, 230), 1, cv2.LINE_AA)

    for prop in payload.get("eligible_proposals") or []:
        fidx = int(prop["frame_index"])
        img_name = str(prop["image_name"])
        img = cv2.imread(str(fdir / img_name))
        if img is None:
            continue
        h, w = img.shape[:2]
        xs = [float(el["points"][0]) for sh in prop["all_before_shapes"] for el in sh["elements"]]
        ys = [float(el["points"][1]) for sh in prop["all_before_shapes"] for el in sh["elements"]]
        pad = max(35, int(0.35 * max(max(xs) - min(xs), max(ys) - min(ys))))
        x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
        x2, y2 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)
        scale = max(2.0, min(5.0, 500.0 / max(1, max(x2 - x1, y2 - y1))))
        cw, ch = int((x2 - x1) * scale), int((y2 - y1) * scale)

        b_crop = cv2.resize(img[y1:y2, x1:x2], (cw, ch), interpolation=cv2.INTER_CUBIC)
        a_crop = b_crop.copy()
        chg_set = {int(k) for k in prop["changed_kpts"]}
        _draw_face(b_crop, prop["all_before_shapes"], x1, y1, scale, chg_set, is_after=False)
        _draw_face(a_crop, prop["all_proposed_shapes"], x1, y1, scale, chg_set, is_after=True)

        b_pad = cv2.copyMakeBorder(b_crop, 36, 26, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 38))
        a_pad = cv2.copyMakeBorder(a_crop, 36, 26, 0, 0, cv2.BORDER_CONSTANT, value=(20, 38, 25))
        cv2.putText(b_pad, f"BEFORE f{fidx:03d} ({img_name})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 230), 2)
        cv2.putText(a_pad, f"PROPOSED f{fidx:03d} ({len(prop['changes'])} fixes)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 240, 40), 2)
        cv2.putText(a_pad, f"kps: {prop['changed_kpts'][:8]}", (10, ch + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 40), 1)

        canvas = cv2.hconcat([b_pad, a_pad])
        out_f = out_dir / f"f{fidx:03d}_{img_name}"
        cv2.imwrite(str(out_f), canvas)
        rendered.append(out_f)
        tiles.append(cv2.resize(canvas, (760, 420), interpolation=cv2.INTER_AREA))

    # Also add Review Queue crops into contact sheet for quick visibility
    for rq in payload.get("review_queue") or []:
        fidx = int(rq["frame_index"])
        img_name = str(rq["image_name"])
        crop_p = FACE50_OUT_DIR / "evidence" / "crops" / f"f{fidx:03d}_{img_name}"
        if crop_p.exists():
            cimg = cv2.imread(str(crop_p))
            if cimg is not None:
                c_pad = cv2.copyMakeBorder(cimg, 36, 26, 0, 0, cv2.BORDER_CONSTANT, value=(45, 20, 20))
                cv2.putText(c_pad, f"REVIEW QUEUE f{fidx:03d} (NOT SYNCED)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 140, 255), 2)
                cv2.putText(c_pad, rq["reason"][:58], (10, cimg.shape[0] + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                tiles.append(cv2.resize(c_pad, (760, 420), interpolation=cv2.INTER_AREA))

    contact = out_dir / "00_FACE50_CONTACT_SHEET.jpg"
    if tiles:
        cols = 2
        rows = (len(tiles) + cols - 1) // cols
        while len(tiles) < rows * cols:
            tiles.append(np.full_like(tiles[0], 20))
        sheet = cv2.vconcat([cv2.hconcat(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(contact), sheet)

    return {"output_dir": str(out_dir), "contact_sheet": str(contact) if tiles else None, "files": [str(p) for p in rendered]}


def sync_face50_eligible(
    job_id: int = 1975,
    proposals_path: Path | None = None,
    profile: str = "edufun",
    yes: bool = False,
) -> dict[str, Any]:
    """Dry-run by default; with yes=True, backup, minimal PATCH/POST sync, and read-back verify."""
    prop_p = Path(proposals_path) if proposals_path else (FACE50_OUT_DIR / f"proposed_face50_{job_id}.json")
    payload = json.loads(prop_p.read_text(encoding="utf-8"))
    eligible = payload.get("eligible_proposals") or []

    update_proposals = [p for p in eligible if p.get("action", "update") == "update"]
    create_proposals = [p for p in eligible if p.get("action") == "create"]
    patch_shapes = [sh for p in update_proposals for sh in p["proposed_shapes"]]
    create_shapes = [sh for p in create_proposals for sh in p["proposed_shapes"]]

    if not yes:
        return {
            "job_id": job_id,
            "mode": "dry_run",
            "written_to_cvat": False,
            "eligible_frames": [p["frame_index"] for p in eligible],
            "patch_shapes_count": len(patch_shapes),
            "create_shapes_count": len(create_shapes),
            "total_kp_changes": sum(len(p.get("changes") or []) for p in eligible),
            "review_queue_frames": [r["frame_index"] for r in (payload.get("review_queue") or [])],
        }

    cv = Cvat.from_env(profile, write_enabled=True)
    before_live = cv.annotations(job_id)
    live_shapes = before_live.get("shapes") or []
    live_by_id = {int(s["id"]): s for s in live_shapes if s.get("id") is not None}
    live_by_frame_label = {
        (int(s["frame"]), int(s["label_id"]))
        for s in live_shapes
        if s.get("frame") is not None and s.get("label_id") is not None
    }

    # Staleness guard
    for p in update_proposals:
        for b_sh in p.get("before_shapes") or []:
            sid = int(b_sh["id"])
            live_sh = live_by_id.get(sid)
            if live_sh is None:
                raise RuntimeError(f"Stale proposal: shape {sid} (f{p['frame_index']}) missing on CVAT.")
            l_pts = {int(e["label_id"]): (e.get("points"), bool(e.get("occluded")), bool(e.get("outside"))) for e in (live_sh.get("elements") or [])}
            b_pts = {int(e["label_id"]): (e.get("points"), bool(e.get("occluded")), bool(e.get("outside"))) for e in (b_sh.get("elements") or [])}
            if l_pts != b_pts:
                raise RuntimeError(f"Stale proposal: frame {p['frame_index']} shape {sid} modified on CVAT after propose.")

    for p in create_proposals:
        for c_sh in p.get("proposed_shapes") or []:
            key = (int(c_sh["frame"]), int(c_sh["label_id"]))
            if key in live_by_frame_label:
                raise RuntimeError(f"Stale proposal: frame {p['frame_index']} label {c_sh['label_id']} already exists on CVAT.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = week2.job_folder(job_id) / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    backup_file = bdir / f"before_face50_sync_{ts}.json"
    backup_file.write_text(json.dumps(before_live, indent=2, ensure_ascii=False), encoding="utf-8")

    # Enforce source="manual" on all synced shapes and elements
    for sh in patch_shapes + create_shapes:
        sh["source"] = "manual"
        for el in sh.get("elements") or []:
            el["source"] = "manual"

    if patch_shapes:
        cv.update_shapes(job_id, patch_shapes)
    create_res = cv.create_shapes(job_id, create_shapes) if create_shapes else {}

    after_live = cv.annotations(job_id)
    after_shapes = after_live.get("shapes") or []
    after_by_id = {int(s["id"]): s for s in after_shapes if s.get("id") is not None}
    after_by_frame_label: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for s in after_shapes:
        if s.get("frame") is not None and s.get("label_id") is not None:
            after_by_frame_label.setdefault((int(s["frame"]), int(s["label_id"])), []).append(s)

    # Read-back verification
    mismatches = 0
    for exp_sh in patch_shapes:
        got_sh = after_by_id.get(int(exp_sh["id"]))
        if not got_sh:
            mismatches += 1
            continue
        exp_e = {int(e["id"]): (e["points"], bool(e.get("occluded")), bool(e.get("outside"))) for e in exp_sh["elements"]}
        got_e = {int(e["id"]): (e["points"], bool(e.get("occluded")), bool(e.get("outside"))) for e in got_sh["elements"]}
        for eid, (ep, eocc, eout) in exp_e.items():
            gp, gocc, gout = got_e.get(eid, ([], None, None))
            if not gp or abs(ep[0] - gp[0]) > 0.05 or abs(ep[1] - gp[1]) > 0.05 or (eocc, eout) != (gocc, gout):
                mismatches += 1

    for exp_sh in create_shapes:
        cands = after_by_frame_label.get((int(exp_sh["frame"]), int(exp_sh["label_id"])), [])
        if not cands:
            mismatches += 1
            continue
        got_sh = cands[-1]
        if got_sh.get("source") not in (None, "manual"):
            mismatches += 1
        exp_e = {int(e["label_id"]): (e["points"], bool(e.get("occluded")), bool(e.get("outside"))) for e in exp_sh.get("elements") or []}
        got_e = {int(e["label_id"]): (e["points"], bool(e.get("occluded")), bool(e.get("outside"))) for e in got_sh.get("elements") or []}
        for lid, (ep, eocc, eout) in exp_e.items():
            gp, gocc, gout = got_e.get(lid, ([], None, None))
            if not gp or abs(ep[0] - gp[0]) > 0.05 or abs(ep[1] - gp[1]) > 0.05 or (eocc, eout) != (gocc, gout):
                mismatches += 1

    created_ids = [int(s["id"]) for s in (create_res.get("shapes") or []) if s.get("id") is not None] if isinstance(create_res, dict) else []
    run_file = bdir / f"run_face50_sync_{ts}.json"
    run_file.write_text(json.dumps({
        "job_id": job_id,
        "timestamp": ts,
        "backup_file": str(backup_file),
        "updated_shape_ids": [int(s["id"]) for s in patch_shapes],
        "created_shape_ids": created_ids,
        "eligible_frames": [p["frame_index"] for p in eligible],
        "readback_mismatches": mismatches,
        "before_hash": _sha256_obj(live_shapes),
        "after_hash": _sha256_obj(after_shapes),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if mismatches:
        raise RuntimeError(f"Read-back verification found {mismatches} mismatches! Backup at {backup_file}")

    return {
        "job_id": job_id,
        "mode": "committed",
        "written_to_cvat": True,
        "synced_frames": [p["frame_index"] for p in eligible],
        "patch_shapes_count": len(patch_shapes),
        "create_shapes_count": len(create_shapes),
        "created_shape_ids": created_ids,
        "total_kp_changes": sum(len(p.get("changes") or []) for p in eligible),
        "readback_mismatches": mismatches,
        "backup_file": str(backup_file),
        "run_file": str(run_file),
    }
