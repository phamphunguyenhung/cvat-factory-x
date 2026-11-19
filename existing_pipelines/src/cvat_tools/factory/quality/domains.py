"""Domain-specific quality validators for Face50, Segmentation, BBox, and Tracking."""

from __future__ import annotations

from typing import Any

from .models import QualityIssue

# Face50 anatomical index ranges (standard 50-point facial landmark schema)
FACE50_REGIONS: dict[str, tuple[int, int]] = {
    "left_brow": (0, 5),
    "right_brow": (5, 10),
    "nose": (10, 14),
    "left_eye": (14, 22),
    "right_eye": (22, 30),
    "outer_lip": (30, 42),
    "inner_lip": (42, 50),
}


def _to_face50_array(points: Any):
    """Normalize Face50 points (dict[str, dict] | list[dict] | (50,2) array) to (N, 2) float64, scores, states."""
    import numpy as np

    if isinstance(points, dict):
        n = max([int(k) for k in points.keys()] + [49]) + 1
        coords = np.zeros((n, 2), dtype=float)
        scores = np.ones(n, dtype=float)
        states = ["visible"] * n
        for k, v in points.items():
            idx = int(k)
            coords[idx, 0] = float(v["x"])
            coords[idx, 1] = float(v["y"])
            if "score" in v and v["score"] is not None:
                scores[idx] = float(v["score"])
            if "state" in v and v["state"] is not None:
                states[idx] = str(v["state"]).lower()
            elif v.get("occluded"):
                states[idx] = "occluded"
            elif v.get("outside"):
                states[idx] = "outside"
        return coords, scores, states

    if isinstance(points, list) and points and isinstance(points[0], dict):
        n = len(points)
        coords = np.zeros((n, 2), dtype=float)
        scores = np.ones(n, dtype=float)
        states = ["visible"] * n
        for idx, v in enumerate(points):
            coords[idx, 0] = float(v["x"])
            coords[idx, 1] = float(v["y"])
            if "score" in v and v["score"] is not None:
                scores[idx] = float(v["score"])
            if "state" in v and v["state"] is not None:
                states[idx] = str(v["state"]).lower()
            elif v.get("occluded"):
                states[idx] = "occluded"
        return coords, scores, states

    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    n = len(arr)
    return arr, np.ones(n, dtype=float), ["visible"] * n


def _sample_edge_gradient(image, pts) -> float:
    """Compute mean Sobel gradient magnitude around a set of (x, y) landmarks."""
    import cv2
    import numpy as np

    if image is None or len(pts) == 0:
        return 1.0
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    h, w = gray.shape[:2]
    vals = []
    for x, y in pts:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            x0, x1 = max(0, ix - 2), min(w, ix + 3)
            y0, y1 = max(0, iy - 2), min(h, iy + 3)
            vals.append(float(mag[y0:y1, x0:x1].max()))
    return float(np.mean(vals)) if vals else 0.0


def validate_face50(
    points: Any,
    *,
    image: Any = None,
    prev_points: Any = None,
    scores: Any = None,
    states: list[str] | None = None,
    min_iod_px: float = 3.0,
    min_edge_grad: float = 8.0,
    max_regression_iod: float = 0.65,
    low_conf_threshold: float = 0.30,
) -> list[QualityIssue]:
    """Validate Face50 landmarks across all 9 domain checks.

    Checks:
    1. Brow edge alignment
    2. Eye boundary & openness
    3. Nose alignment
    4. Outer lip geometry
    5. Inner lip topology (contained inside outer lip, non-inverted)
    6. Occluded landmark handling
    7. Profile-aware geometry
    8. Left/right consistency
    9. Landmark movement regression (vs prev_points)
    """
    import numpy as np

    issues: list[QualityIssue] = []
    coords, parsed_scores, parsed_states = _to_face50_array(points)
    if scores is not None:
        parsed_scores = np.asarray(scores, dtype=float)
    if states is not None:
        parsed_states = [str(s).lower() for s in states]

    if len(coords) != 50:
        issues.append(
            QualityIssue(
                code="FACE50_SCHEMA_COUNT",
                domain="face50",
                severity="error",
                region="skeleton",
                message=f"Expected 50 Face50 landmarks, got {len(coords)}.",
                affected_points=list(range(len(coords))),
                repairable=False,
                dimension="schema_valid",
                evidence={"expected": 50, "actual": len(coords)},
            )
        )
        return issues

    e1 = coords[14:22].mean(axis=0)
    e2 = coords[22:30].mean(axis=0)
    eye_vec = e2 - e1
    iod = float(np.linalg.norm(eye_vec))
    if iod < min_iod_px:
        issues.append(
            QualityIssue(
                code="FACE50_IOD_TOO_SMALL",
                domain="face50",
                severity="error",
                region="eye",
                message=f"Inter-ocular distance {iod:.2f}px < {min_iod_px}px.",
                affected_points=list(range(14, 30)),
                repairable=False,
                dimension="geometry_valid",
                evidence={"iod_px": iod, "min_iod_px": min_iod_px},
            )
        )
        return issues

    axis_x = eye_vec / iod
    down = np.array([-axis_x[1], axis_x[0]])
    eye_mid = (e1 + e2) / 2.0

    def proj_down(q: np.ndarray) -> float:
        return float((q - eye_mid) @ down / iod)

    def proj_lat(q: np.ndarray) -> float:
        return float((q - eye_mid) @ axis_x / iod)

    def span_iod(arr: np.ndarray) -> float:
        return float(np.hypot(np.ptp(arr[:, 0]), np.ptp(arr[:, 1])) / iod)

    # 7. Profile-aware geometry detection
    nose_tip = coords[13]
    nose_lat = proj_lat(nose_tip)
    is_profile = abs(nose_lat) > 0.35
    if is_profile and abs(nose_lat) > 0.95:
        issues.append(
            QualityIssue(
                code="FACE50_EXTREME_PROFILE",
                domain="face50",
                severity="warn",
                region="nose",
                message=f"Extreme profile yaw detected (nose lateral offset {nose_lat:.2f} IOD).",
                affected_points=[10, 11, 12, 13],
                repairable=False,
                dimension="confidence_sufficient",
                evidence={"nose_lateral_iod": nose_lat},
            )
        )

    # 1. Brow edge alignment
    lb_up = -proj_down(coords[0:5].mean(axis=0))
    rb_up = -proj_down(coords[5:10].mean(axis=0))
    if not (0.0 <= lb_up <= 1.0):
        issues.append(
            QualityIssue(
                code="FACE50_BROW_ALIGNMENT",
                domain="face50",
                severity="error",
                region="left_brow",
                message=f"Left brow vertical offset {lb_up:.2f} IOD outside [0.0, 1.0].",
                affected_points=list(range(0, 5)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"brow_up_iod": lb_up},
            )
        )
    if not (0.0 <= rb_up <= 1.0):
        issues.append(
            QualityIssue(
                code="FACE50_BROW_ALIGNMENT",
                domain="face50",
                severity="error",
                region="right_brow",
                message=f"Right brow vertical offset {rb_up:.2f} IOD outside [0.0, 1.0].",
                affected_points=list(range(5, 10)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"brow_up_iod": rb_up},
            )
        )

    # 2. Eye boundary
    le_size = span_iod(coords[14:22])
    re_size = span_iod(coords[22:30])
    if le_size > 1.0 or le_size < 0.12:
        issues.append(
            QualityIssue(
                code="FACE50_EYE_BOUNDARY",
                domain="face50",
                severity="error",
                region="left_eye",
                message=f"Left eye span {le_size:.2f} IOD outside [0.12, 1.0].",
                affected_points=list(range(14, 22)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"eye_span_iod": le_size},
            )
        )
    if re_size > 1.0 or re_size < 0.12:
        issues.append(
            QualityIssue(
                code="FACE50_EYE_BOUNDARY",
                domain="face50",
                severity="error",
                region="right_eye",
                message=f"Right eye span {re_size:.2f} IOD outside [0.12, 1.0].",
                affected_points=list(range(22, 30)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"eye_span_iod": re_size},
            )
        )

    # 3. Nose alignment
    nose_down = proj_down(nose_tip)
    if not (0.0 <= nose_down <= 1.4):
        issues.append(
            QualityIssue(
                code="FACE50_NOSE_ALIGNMENT",
                domain="face50",
                severity="error",
                region="nose",
                message=f"Nose tip vertical offset {nose_down:.2f} IOD outside [0.0, 1.4].",
                affected_points=list(range(10, 14)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"nose_down_iod": nose_down},
            )
        )

    # 4. Outer lip
    mouth_down = proj_down(coords[30:42].mean(axis=0))
    mouth_width = float(np.linalg.norm(coords[30] - coords[36]) / iod)
    if not (0.4 <= mouth_down <= 2.8) or not (0.3 <= mouth_width <= 1.6):
        issues.append(
            QualityIssue(
                code="FACE50_OUTER_LIP",
                domain="face50",
                severity="error",
                region="outer_lip",
                message=f"Outer lip geometry anomalous (down={mouth_down:.2f}, width={mouth_width:.2f} IOD).",
                affected_points=list(range(30, 42)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"mouth_down_iod": mouth_down, "mouth_width_iod": mouth_width},
            )
        )

    # 5. Inner lip topology (must stay inside outer lip hull/bbox and not invert)
    outer = coords[30:42]
    inner = coords[42:50]
    ox0, oy0 = outer.min(axis=0)
    ox1, oy1 = outer.max(axis=0)
    margin = 0.08 * iod
    outside_pts = [
        42 + idx
        for idx, (x, y) in enumerate(inner)
        if x < ox0 - margin or x > ox1 + margin or y < oy0 - margin or y > oy1 + margin
    ]
    # Check top inner lip (43,44,45) vs bottom inner lip (47,48,49)
    inner_inverted = proj_down(coords[44]) > proj_down(coords[48]) + 0.05
    if outside_pts or inner_inverted:
        issues.append(
            QualityIssue(
                code="FACE50_INNER_LIP_TOPOLOGY",
                domain="face50",
                severity="error",
                region="inner_lip",
                message="Inner lip topology invalid (outside outer lip or top/bottom inverted).",
                affected_points=outside_pts or list(range(42, 50)),
                repairable=True,
                dimension="geometry_valid",
                evidence={"outside_points": outside_pts, "inverted": inner_inverted},
            )
        )

    # 6. Occluded landmark handling
    occ_indices = [
        idx
        for idx, (st, sc) in enumerate(zip(parsed_states, parsed_scores))
        if st in {"occluded", "outside"} or sc < low_conf_threshold
    ]
    if occ_indices:
        # High fraction of occluded landmarks without profile explanation needs review
        frac = len(occ_indices) / 50.0
        if frac > 0.35:
            issues.append(
                QualityIssue(
                    code="FACE50_HEAVY_OCCLUSION",
                    domain="face50",
                    severity="warn",
                    region="occlusion",
                    message=f"{len(occ_indices)}/50 landmarks occluded or low-confidence (<{low_conf_threshold}).",
                    affected_points=occ_indices,
                    repairable=False,
                    dimension="confidence_sufficient",
                    evidence={"occluded_count": len(occ_indices), "fraction": round(frac, 3)},
                )
            )

    # 8. Left/right consistency (relaxed when profile is detected)
    if not is_profile:
        brow_diff = abs(lb_up - rb_up)
        eye_diff = abs(le_size - re_size)
        if brow_diff > 0.35:
            issues.append(
                QualityIssue(
                    code="FACE50_LR_BROW_INCONSISTENCY",
                    domain="face50",
                    severity="warn",
                    region="left_brow" if lb_up < rb_up else "right_brow",
                    message=f"Left/right brow height asymmetry {brow_diff:.2f} IOD > 0.35.",
                    affected_points=list(range(0, 10)),
                    repairable=True,
                    dimension="geometry_valid",
                    evidence={"brow_diff_iod": brow_diff},
                )
            )
        if eye_diff > 0.30:
            issues.append(
                QualityIssue(
                    code="FACE50_LR_EYE_INCONSISTENCY",
                    domain="face50",
                    severity="warn",
                    region="left_eye" if le_size < re_size else "right_eye",
                    message=f"Left/right eye size asymmetry {eye_diff:.2f} IOD > 0.30.",
                    affected_points=list(range(14, 30)),
                    repairable=True,
                    dimension="geometry_valid",
                    evidence={"eye_diff_iod": eye_diff},
                )
            )

    # 9. Landmark movement regression
    if prev_points is not None:
        prev_coords, _, _ = _to_face50_array(prev_points)
        if len(prev_coords) == 50:
            dists = np.linalg.norm(coords - prev_coords, axis=1) / iod
            reg_pts = [int(i) for i in np.where(dists > max_regression_iod)[0]]
            if reg_pts:
                issues.append(
                    QualityIssue(
                        code="FACE50_MOVEMENT_REGRESSION",
                        domain="face50",
                        severity="error",
                        region="regression",
                        message=f"{len(reg_pts)} landmarks jumped > {max_regression_iod:.2f} IOD from baseline.",
                        affected_points=reg_pts,
                        repairable=False,
                        dimension="no_regression",
                        evidence={"max_jump_iod": float(dists.max()), "threshold": max_regression_iod},
                    )
                )

    # Optional image alignment check (distinguishes geometry_valid from image_alignment_good)
    if image is not None:
        for region_name in ("left_brow", "right_brow", "outer_lip"):
            s, e = FACE50_REGIONS[region_name]
            grad = _sample_edge_gradient(image, coords[s:e])
            if grad < min_edge_grad:
                issues.append(
                    QualityIssue(
                        code="FACE50_IMAGE_MISALIGNMENT",
                        domain="face50",
                        severity="error",
                        region=region_name,
                        message=f"Low image edge gradient ({grad:.1f} < {min_edge_grad}) at {region_name}.",
                        affected_points=list(range(s, e)),
                        repairable=True,
                        dimension="image_alignment_good",
                        evidence={"edge_gradient": round(grad, 2), "min_edge_grad": min_edge_grad},
                    )
                )

    return issues


def validate_segmentation(
    polygons: list[dict[str, Any]],
    *,
    frame_wh: tuple[int, int] | None = None,
    expected_coverage_boxes: list[tuple[float, float, float, float]] | None = None,
    min_area_px: float = 25.0,
    max_overlap_iou: float = 0.15,
    bleed_margin_px: float = 2.0,
) -> list[QualityIssue]:
    """Validate segmentation polygons for bleed, speckle, self-intersection, uncovered regions, and invalid overlaps."""
    from shapely.geometry import Polygon, box as shapely_box
    from shapely.validation import explain_validity

    issues: list[QualityIssue] = []
    built: list[tuple[int, str, Polygon, dict[str, Any]]] = []

    for idx, shape in enumerate(polygons):
        pts = shape.get("points") or []
        label = str(shape.get("label", f"poly_{idx}"))
        shape_id = str(shape.get("id", idx))
        if len(pts) < 3:
            issues.append(
                QualityIssue(
                    code="SEG_DEGENERATE_POLYGON",
                    domain="segmentation",
                    severity="error",
                    region=f"polygon_{shape_id}",
                    message=f"Polygon '{label}' has fewer than 3 vertices ({len(pts)}).",
                    affected_shape_ids=[shape_id],
                    repairable=False,
                    dimension="schema_valid",
                )
            )
            continue

        coord_pairs = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts) - 1, 2)] if isinstance(pts[0], (int, float)) else [(float(p[0]), float(p[1])) for p in pts]
        poly = Polygon(coord_pairs)

        # 3. Self-intersection
        if not poly.is_valid:
            issues.append(
                QualityIssue(
                    code="SEG_SELF_INTERSECTION",
                    domain="segmentation",
                    severity="error",
                    region=f"polygon_{shape_id}",
                    message=f"Self-intersecting polygon '{label}': {explain_validity(poly)}.",
                    affected_shape_ids=[shape_id],
                    repairable=True,
                    dimension="geometry_valid",
                    evidence={"validity": explain_validity(poly)},
                )
            )
            poly = poly.buffer(0)

        # 2. Speckle
        area = float(poly.area)
        if area < min_area_px:
            issues.append(
                QualityIssue(
                    code="SEG_SPECKLE",
                    domain="segmentation",
                    severity="warn",
                    region=f"polygon_{shape_id}",
                    message=f"Speckle polygon '{label}' area {area:.1f}px² < {min_area_px}px².",
                    affected_shape_ids=[shape_id],
                    repairable=True,
                    dimension="geometry_valid",
                    evidence={"area_px": area, "min_area_px": min_area_px},
                )
            )

        # 1. Boundary bleed (outside frame bounds)
        if frame_wh is not None:
            w, h = frame_wh
            minx, miny, maxx, maxy = poly.bounds
            if (
                minx < -bleed_margin_px
                or miny < -bleed_margin_px
                or maxx > w + bleed_margin_px
                or maxy > h + bleed_margin_px
            ):
                issues.append(
                    QualityIssue(
                        code="SEG_BOUNDARY_BLEED",
                        domain="segmentation",
                        severity="error",
                        region=f"polygon_{shape_id}",
                        message=f"Polygon '{label}' bleeds outside frame ({w}x{h}).",
                        affected_shape_ids=[shape_id],
                        repairable=True,
                        dimension="image_alignment_good",
                        evidence={"bounds": [minx, miny, maxx, maxy], "frame_wh": [w, h]},
                    )
                )

        built.append((idx, label, poly, shape))

    # 5. Invalid overlaps
    for i in range(len(built)):
        _, label_i, poly_i, shape_i = built[i]
        id_i = str(shape_i.get("id", i))
        for j in range(i + 1, len(built)):
            _, label_j, poly_j, shape_j = built[j]
            id_j = str(shape_j.get("id", j))
            if not poly_i.is_valid or not poly_j.is_valid:
                continue
            inter_area = float(poly_i.intersection(poly_j).area)
            if inter_area <= 0:
                continue
            min_a = min(float(poly_i.area), float(poly_j.area))
            overlap_ratio = inter_area / min_a if min_a > 0 else 0.0
            if overlap_ratio > max_overlap_iou:
                issues.append(
                    QualityIssue(
                        code="SEG_INVALID_OVERLAP",
                        domain="segmentation",
                        severity="error",
                        region=f"polygon_{id_i}",
                        message=f"Invalid overlap ({overlap_ratio:.2%}) between '{label_i}' and '{label_j}'.",
                        affected_shape_ids=[id_i, id_j],
                        repairable=True,
                        dimension="geometry_valid",
                        evidence={"overlap_ratio": round(overlap_ratio, 3)},
                    )
                )

    # 4. Uncovered regions
    if expected_coverage_boxes:
        for b_idx, (x0, y0, x1, y1) in enumerate(expected_coverage_boxes):
            box_poly = shapely_box(x0, y0, x1, y1)
            box_area = float(box_poly.area)
            if box_area <= 0:
                continue
            covered = sum(float(box_poly.intersection(p).area) for _, _, p, _ in built if p.is_valid)
            cov_ratio = min(1.0, covered / box_area)
            if cov_ratio < 0.25:
                issues.append(
                    QualityIssue(
                        code="SEG_UNCOVERED_REGION",
                        domain="segmentation",
                        severity="warn",
                        region=f"expected_box_{b_idx}",
                        message=f"Expected region {b_idx} has only {cov_ratio:.1%} segmentation coverage.",
                        repairable=False,
                        dimension="image_alignment_good",
                        evidence={"coverage_ratio": round(cov_ratio, 3)},
                    )
                )

    return issues


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def validate_bbox(
    boxes: list[dict[str, Any]],
    *,
    frame_wh: tuple[int, int] | None = None,
    allowed_labels: set[str] | None = None,
    expected_boxes: list[dict[str, Any]] | None = None,
    max_area_ratio: float = 0.85,
    duplicate_iou: float = 0.80,
    occlusion_iou: float = 0.45,
) -> list[QualityIssue]:
    """Validate BBoxes for missing objects, wrong class, oversized box, duplicate box, and occlusion errors."""
    issues: list[QualityIssue] = []
    parsed: list[tuple[str, str, tuple[float, float, float, float], dict[str, Any]]] = []

    frame_area = float(frame_wh[0] * frame_wh[1]) if frame_wh else None

    for idx, b in enumerate(boxes):
        sid = str(b.get("id", idx))
        label = str(b.get("label", ""))
        pts = b.get("points") or []
        if len(pts) != 4:
            issues.append(
                QualityIssue(
                    code="BBOX_INVALID_SCHEMA",
                    domain="bbox",
                    severity="error",
                    region=f"box_{sid}",
                    message=f"Box '{label}' must have 4 coordinates [x0,y0,x1,y1].",
                    affected_shape_ids=[sid],
                    repairable=False,
                    dimension="schema_valid",
                )
            )
            continue
        x0, y0, x1, y1 = float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])
        if x1 <= x0 or y1 <= y0:
            issues.append(
                QualityIssue(
                    code="BBOX_INVERTED_COORDS",
                    domain="bbox",
                    severity="error",
                    region=f"box_{sid}",
                    message=f"Box '{label}' has non-positive dimensions.",
                    affected_shape_ids=[sid],
                    repairable=True,
                    dimension="geometry_valid",
                )
            )
            continue

        # Wrong class
        if allowed_labels is not None and label not in allowed_labels:
            issues.append(
                QualityIssue(
                    code="BBOX_WRONG_CLASS",
                    domain="bbox",
                    severity="error",
                    region=f"box_{sid}",
                    message=f"Label '{label}' is not in allowed schema labels.",
                    affected_shape_ids=[sid],
                    repairable=False,
                    dimension="schema_valid",
                    evidence={"label": label, "allowed": sorted(allowed_labels)},
                )
            )

        # Oversized box
        if frame_area and frame_area > 0:
            area_ratio = ((x1 - x0) * (y1 - y0)) / frame_area
            if area_ratio > max_area_ratio:
                issues.append(
                    QualityIssue(
                        code="BBOX_OVERSIZED",
                        domain="bbox",
                        severity="warn",
                        region=f"box_{sid}",
                        message=f"Box '{label}' covers {area_ratio:.1%} of frame (> {max_area_ratio:.0%}).",
                        affected_shape_ids=[sid],
                        repairable=True,
                        dimension="geometry_valid",
                        evidence={"area_ratio": round(area_ratio, 3)},
                    )
                )

        parsed.append((sid, label, (x0, y0, x1, y1), b))

    # Duplicate & Occlusion errors
    for i in range(len(parsed)):
        sid_i, lab_i, box_i, raw_i = parsed[i]
        for j in range(i + 1, len(parsed)):
            sid_j, lab_j, box_j, raw_j = parsed[j]
            iou = _box_iou(box_i, box_j)
            if lab_i == lab_j and iou >= duplicate_iou:
                issues.append(
                    QualityIssue(
                        code="BBOX_DUPLICATE",
                        domain="bbox",
                        severity="error",
                        region=f"box_{sid_j}",
                        message=f"Duplicate '{lab_i}' boxes (IoU={iou:.2f} >= {duplicate_iou}).",
                        affected_shape_ids=[sid_i, sid_j],
                        repairable=True,
                        dimension="geometry_valid",
                        evidence={"iou": round(iou, 3)},
                    )
                )
            elif iou >= occlusion_iou and not (raw_i.get("occluded") or raw_j.get("occluded")):
                issues.append(
                    QualityIssue(
                        code="BBOX_OCCLUSION_UNFLAGGED",
                        domain="bbox",
                        severity="warn",
                        region=f"box_{sid_i}",
                        message=f"Heavy overlap (IoU={iou:.2f}) between '{lab_i}' and '{lab_j}' without occlusion flag.",
                        affected_shape_ids=[sid_i, sid_j],
                        repairable=True,
                        dimension="image_alignment_good",
                        evidence={"iou": round(iou, 3)},
                    )
                )

    # Missing objects (compared against reference/expected boxes)
    if expected_boxes:
        for e_idx, exp in enumerate(expected_boxes):
            e_pts = tuple(float(v) for v in exp["points"])
            matched = any(_box_iou(e_pts, b_pts) >= 0.30 for _, _, b_pts, _ in parsed)
            if not matched:
                issues.append(
                    QualityIssue(
                        code="BBOX_MISSING_OBJECT",
                        domain="bbox",
                        severity="error",
                        region=f"expected_{e_idx}",
                        message=f"Expected object '{exp.get('label', 'object')}' missing from predictions.",
                        repairable=False,
                        dimension="image_alignment_good",
                        evidence={"expected_box": list(e_pts)},
                    )
                )

    return issues


def validate_tracking(
    tracks: list[dict[str, Any]],
    *,
    max_jump_px: float = 120.0,
    max_gap_frames: int = 3,
) -> list[QualityIssue]:
    """Validate object tracks across frames for ID switches, broken tracks, and temporal jumps.

    Each item in `tracks` is expected to have `track_id`, `frame`, and `points` ([x0,y0,x1,y1]).
    """
    import math

    issues: list[QualityIssue] = []
    by_track: dict[Any, list[dict[str, Any]]] = {}
    for item in tracks:
        tid = item.get("track_id")
        if tid is None:
            continue
        by_track.setdefault(tid, []).append(item)

    def center(pts):
        return ((pts[0] + pts[2]) / 2.0, (pts[1] + pts[3]) / 2.0)

    # Temporal jumps & broken tracks within each track_id
    track_endpoints: list[tuple[Any, int, tuple[float, float], int, tuple[float, float]]] = []
    for tid, obs in by_track.items():
        obs_sorted = sorted(obs, key=lambda o: int(o["frame"]))
        for k in range(1, len(obs_sorted)):
            prev_o = obs_sorted[k - 1]
            curr_o = obs_sorted[k]
            df = int(curr_o["frame"]) - int(prev_o["frame"])
            if df > 1:
                issues.append(
                    QualityIssue(
                        code="TRACK_BROKEN",
                        domain="tracking",
                        severity="warn",
                        region=f"track_{tid}",
                        message=f"Track {tid} broken between frame {prev_o['frame']} and {curr_o['frame']} (gap={df}).",
                        affected_shape_ids=[str(tid)],
                        repairable=True,
                        dimension="geometry_valid",
                        evidence={"track_id": tid, "gap_frames": df},
                    )
                )
            cx0, cy0 = center(prev_o["points"])
            cx1, cy1 = center(curr_o["points"])
            dist = math.hypot(cx1 - cx0, cy1 - cy0)
            allowed = max_jump_px * max(1, df)
            if dist > allowed:
                issues.append(
                    QualityIssue(
                        code="TRACK_TEMPORAL_JUMP",
                        domain="tracking",
                        severity="error",
                        region=f"track_{tid}",
                        message=f"Track {tid} jumped {dist:.1f}px (> {allowed:.1f}px) from frame {prev_o['frame']} to {curr_o['frame']}.",
                        affected_shape_ids=[str(tid)],
                        repairable=False,
                        dimension="no_regression",
                        evidence={"track_id": tid, "distance_px": round(dist, 2)},
                    )
                )

        first_o, last_o = obs_sorted[0], obs_sorted[-1]
        track_endpoints.append(
            (tid, int(first_o["frame"]), center(first_o["points"]), int(last_o["frame"]), center(last_o["points"]))
        )

    # ID switch heuristic: one track ends and another starts nearby within max_gap_frames
    for i in range(len(track_endpoints)):
        tid_a, _, _, end_f_a, end_c_a = track_endpoints[i]
        for j in range(len(track_endpoints)):
            if i == j:
                continue
            tid_b, start_f_b, start_c_b, _, _ = track_endpoints[j]
            gap = start_f_b - end_f_a
            if 0 <= gap <= max_gap_frames:
                dist = math.hypot(start_c_b[0] - end_c_a[0], start_c_b[1] - end_c_a[1])
                if dist < max_jump_px:
                    issues.append(
                        QualityIssue(
                            code="TRACK_ID_SWITCH",
                            domain="tracking",
                            severity="error",
                            region=f"track_{tid_a}->{tid_b}",
                            message=f"Suspected ID switch from track {tid_a} (ends f{end_f_a}) to {tid_b} (starts f{start_f_b}, dist={dist:.1f}px).",
                            affected_shape_ids=[str(tid_a), str(tid_b)],
                            repairable=False,
                            dimension="no_regression",
                            evidence={"from_track": tid_a, "to_track": tid_b, "gap": gap, "dist_px": round(dist, 2)},
                        )
                    )

    return issues
