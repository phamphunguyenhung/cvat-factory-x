"""Quantitative evaluation metrics for CVAT drafts vs independent human ground truth."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon


def bbox_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-6, ua + ub - inter)


def _raster_mask(polys: list[list[float]], H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for pts in polys:
        if len(pts) < 6:
            continue
        arr = np.round(np.asarray(pts, np.float32)).astype(np.int32).reshape(-1, 2)
        cv2.fillPoly(m, [arr], 1)
    return m


def _boundary_mask(m: np.ndarray) -> np.ndarray:
    if not m.any():
        return m
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return (m > 0) & (er == 0)


def boundary_f1(pred_m: np.ndarray, gt_m: np.ndarray, tol_px: int = 3) -> float | None:
    pb = _boundary_mask(pred_m).astype(np.uint8)
    gb = _boundary_mask(gt_m).astype(np.uint8)
    if not pb.any() and not gb.any():
        return None
    if not pb.any() or not gb.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    gb_d = cv2.dilate(gb, k) > 0
    pb_d = cv2.dilate(pb, k) > 0
    prec = float((pb > 0)[gb_d].sum()) / max(1, int(pb.sum()))
    rec = float((gb > 0)[pb_d].sum()) / max(1, int(gb.sum()))
    return round(2.0 * prec * rec / max(1e-9, prec + rec), 4) if (prec + rec) > 0 else 0.0


def count_invalid_geometries(items: list[dict[str, Any]]) -> int:
    invalid = 0
    for it in items:
        st = it.get("shape_type")
        pts = it.get("points") or []
        if st == "rectangle":
            if len(pts) != 4 or pts[2] <= pts[0] or pts[3] <= pts[1]:
                invalid += 1
        elif st == "polygon":
            if len(pts) < 6 or len(pts) % 2 != 0:
                invalid += 1
                continue
            arr = np.asarray(pts, dtype=float).reshape(-1, 2)
            if not Polygon(arr).is_valid:
                invalid += 1
    return invalid


def evaluate_split(
    gt_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    frames: list[int],
    *,
    H: int = 720,
    W: int = 1280,
) -> dict[str, Any]:
    """Evaluate AI/rule drafts (excluding cvat_human self-matches) against GT."""
    gt_by_f: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for x in gt_items:
        if int(x["frame"]) in frames:
            gt_by_f[int(x["frame"])].append(x)

    pr_by_f: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for x in pred_items:
        f = int(x["frame"])
        if f in frames and x.get("source") != "cvat_human":
            pr_by_f[f].append(x)

    # BBox draft evaluation (only drafts vs GT rects)
    tp, fp, fn = 0, 0, 0
    ious: list[float] = []
    for f in frames:
        gts = [g for g in gt_by_f[f] if g.get("type") == "rectangle"]
        prs = [p for p in pr_by_f[f] if p.get("shape_type") == "rectangle"]
        matched: set[int] = set()
        for p in prs:
            best_i, best_idx = 0.0, -1
            for idx, g in enumerate(gts):
                if idx in matched or g["label"] != p["label"]:
                    continue
                v = bbox_iou(p["points"], g["points"])
                if v > best_i:
                    best_i, best_idx = v, idx
            if best_i >= 0.5 and best_idx >= 0:
                tp += 1
                matched.add(best_idx)
                ious.append(best_i)
            else:
                fp += 1
        fn += len(gts) - len(matched)

    # Polygon classes evaluation
    poly_labels = ("building", "road", "sidewalk", "vegetation", "sky", "area/direct", "lane/single white", "lane/single yellow")
    class_ious: dict[str, list[float]] = defaultdict(list)
    class_bf1: dict[str, list[float]] = defaultdict(list)
    bld_fp_px = 0
    frame_px = H * W * max(1, len(frames))

    for f in frames:
        for lbl in poly_labels:
            g_polys = [g["points"] for g in gt_by_f[f] if g.get("type") == "polygon" and g["label"] == lbl]
            p_polys = [p["points"] for p in pr_by_f[f] if p.get("shape_type") == "polygon" and p["label"] == lbl]
            if not g_polys and not p_polys:
                continue
            gm = _raster_mask(g_polys, H, W)
            pm = _raster_mask(p_polys, H, W)
            inter = int((gm & pm).sum())
            union = int((gm | pm).sum())
            if union > 0:
                class_ious[lbl].append(inter / union)
            bf = boundary_f1(pm, gm, tol_px=3)
            if bf is not None:
                class_bf1[lbl].append(bf)
            if lbl == "building":
                bld_fp_px += int((pm & (1 - gm)).sum())

    all_preds = [p for f in frames for p in pr_by_f[f]]
    invalid_geom = count_invalid_geometries(all_preds)
    per_class = {
        lbl: {
            "n": len(class_ious[lbl]),
            "iou": round(float(np.mean(class_ious[lbl])), 4) if class_ious[lbl] else None,
            "bf1_3px": round(float(np.mean(class_bf1[lbl])), 4) if class_bf1[lbl] else None,
        }
        for lbl in poly_labels
        if class_ious[lbl]
    }
    macro_miou = round(float(np.mean([v["iou"] for v in per_class.values() if v["iou"] is not None])), 4) if per_class else 0.0
    return {
        "n_frames": len(frames),
        "bbox_drafts": {"tp": tp, "fp": fp, "fn": fn, "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0},
        "macro_miou": macro_miou,
        "area_direct_iou": per_class.get("area/direct", {}).get("iou"),
        "building_fp_area_rate": round(bld_fp_px / frame_px, 5),
        "invalid_geometry": invalid_geom,
        "per_class": per_class,
    }
