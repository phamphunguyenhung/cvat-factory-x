"""Stage C: Image-Aware Region-Level Quality Control (Geometry + Edge/Gradient + Local Evidence)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cvat_tools.pose.face50_engine import (
    _face_rotation_basis,
    _proj,
    _state_str,
    validate_face50_frame,
)

REGIONS: dict[str, list[int]] = {
    "brow_L": list(range(0, 5)),
    "brow_R": list(range(5, 10)),
    "nose": list(range(10, 14)),
    "eye_L": list(range(14, 22)),
    "eye_R": list(range(22, 30)),
    "outer_lip": list(range(30, 42)),
    "inner_lip": list(range(42, 50)),
}


def compute_image_stats(img_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    luma = float(gray.mean())
    dark_frac = float((gray < 40).mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "luma": round(luma, 2),
        "dark_fraction": round(dark_frac, 3),
        "blur_lap": round(blur_var, 2),
        "is_dark": luma < 60.0,
    }


def _brow_gradient_score(gray_enh: np.ndarray, pts: dict[int, dict[str, Any]], indices: list[int], v_axis: tuple[float, float]) -> float:
    """Mean bright-above minus dark-below contrast along face-down v_axis at brow keypoints."""
    h, w = gray_enh.shape[:2]
    vx, vy = v_axis
    scores = []
    for k in indices:
        px, py = pts[k]["x"], pts[k]["y"]
        ax, ay = int(round(np.clip(px - 2.0 * vx, 0, w - 1))), int(round(np.clip(py - 2.0 * vy, 0, h - 1)))
        bx, by = int(round(np.clip(px + 2.0 * vx, 0, w - 1))), int(round(np.clip(py + 2.0 * vy, 0, h - 1)))
        scores.append(float(gray_enh[ay, ax]) - float(gray_enh[by, bx]))
    return round(float(np.mean(scores)), 2)


def enhance_if_dark(img_bgr: np.ndarray, luma: float) -> np.ndarray:
    """Apply CLAHE + Gamma only when frame is dark (Fast Path First)."""
    if luma >= 60.0:
        return img_bgr
    gamma = 0.55 if luma < 40.0 else 0.70
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enh = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enh, lut)


def evaluate_frame_qc(
    frame_idx: int,
    shapes: list[dict[str, Any]],
    sub_id_to_kp: dict[int, int],
    lbl_id_to_name: dict[int, str],
    width: int,
    height: int,
    img_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate frame geometry + local image edge evidence -> per-region PASS / REFINE / REVIEW."""
    geom = validate_face50_frame(frame_idx, shapes, sub_id_to_kp, lbl_id_to_name, width, height)
    findings = geom["findings"]

    img_stats = {"luma": 0.0, "dark_fraction": 0.0, "blur_lap": 0.0, "is_dark": False}
    gray_enh = None
    if img_path is not None and img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            img_stats = compute_image_stats(img)
            enh = enhance_if_dark(img, img_stats["luma"])
            gray_enh = cv2.GaussianBlur(cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0.8)

    if not shapes:
        return {
            "frame": frame_idx,
            "shape_count": 0,
            "angle_deg": 0.0,
            "img_stats": img_stats,
            "findings": findings,
            "error_count": len([f for f in findings if f["severity"] == "error"]),
            "warning_count": len([f for f in findings if f["severity"] == "warning"]),
            "regions": {r: "REFINE" for r in REGIONS},
            "brow_contrast": {},
        }

    pts: dict[int, dict[str, Any]] = {}
    for sh in shapes:
        for el in sh.get("elements") or []:
            k = sub_id_to_kp.get(int(el["label_id"]))
            if k is not None:
                pts[k] = {"x": float(el["points"][0]), "y": float(el["points"][1]), "state": _state_str(el)}

    region_status = {r: "PASS" for r in REGIONS}
    brow_contrast: dict[str, float] = {}

    # Map findings to exact affected regions (selective refinement!)
    for f in findings:
        code = f["code"]
        kps_str = str(f.get("kps", ""))
        affected_kps = set()
        for part in kps_str.replace("/", ";").split(";"):
            if part.strip().isdigit():
                affected_kps.add(int(part.strip()))

        if code in ("INNER_LIP_OUTSIDE_OUTER",):
            region_status["inner_lip"] = "REFINE"
        elif code == "LIP_SELF_INTERSECT":
            if "moitrong" in f.get("detail", ""):
                region_status["inner_lip"] = "REFINE"
            else:
                region_status["outer_lip"] = "REFINE"
        elif code == "LIP_INVERSION_LOCAL":
            region_status["inner_lip"] = "REFINE"
        elif code in ("BROW_BELOW_EYE", "BROW_NON_MONOTONIC_LOCAL", "BROW_EXTREME_DRIFT"):
            for rname in ("brow_L", "brow_R"):
                if any(k in REGIONS[rname] for k in affected_kps) or not affected_kps:
                    region_status[rname] = "REFINE"
        elif code in ("EYE_LID_INVERSION_LOCAL", "EYE_CLUSTER_COLLAPSE"):
            for rname in ("eye_L", "eye_R"):
                if any(k in REGIONS[rname] for k in affected_kps):
                    region_status[rname] = "REVIEW"
        elif code == "NOSE_NON_MONOTONIC_LOCAL":
            region_status["nose"] = "REVIEW"

    # Local image evidence: check brow contrast if image available
    if gray_enh is not None and len(pts) == 50:
        _, v_axis, _ = _face_rotation_basis(pts)
        for rname in ("brow_L", "brow_R"):
            cval = _brow_gradient_score(gray_enh, pts, REGIONS[rname], v_axis)
            brow_contrast[rname] = cval

    return {
        "frame": frame_idx,
        "shape_count": len(shapes),
        "angle_deg": geom["angle_deg"],
        "img_stats": img_stats,
        "findings": findings,
        "error_count": len([f for f in findings if f["severity"] == "error"]),
        "warning_count": len([f for f in findings if f["severity"] == "warning"]),
        "regions": region_status,
        "brow_contrast": brow_contrast,
    }
