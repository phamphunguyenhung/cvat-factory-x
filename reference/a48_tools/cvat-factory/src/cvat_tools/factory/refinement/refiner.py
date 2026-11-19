"""Bounded local refinement engine (max 2-3 attempts per region, never resets whole skeleton)."""

from __future__ import annotations

import copy
from typing import Any, Callable

from ..quality.domains import FACE50_REGIONS, _to_face50_array
from ..quality.gate import QualityGate, QualityPolicy
from ..quality.models import QualityIssue, QualityLevel
from .models import RefinementAttempt, RefinementResult

PatchFn = Callable[[str, Any, list[QualityIssue], dict[str, Any]], Any]


def _default_face50_patcher(region: str, payload: Any, issues: list[QualityIssue], ctx: dict[str, Any]) -> Any:
    """Patch only the specified Face50 anatomical region, leaving all other points untouched."""
    import numpy as np

    if region not in FACE50_REGIONS:
        return payload

    coords, _, _ = _to_face50_array(payload)
    if len(coords) != 50:
        return payload

    s, e = FACE50_REGIONS[region]
    patched = coords.copy()

    e1 = coords[14:22].mean(axis=0)
    e2 = coords[22:30].mean(axis=0)
    eye_vec = e2 - e1
    iod = float(np.linalg.norm(eye_vec))
    if iod < 1e-6:
        return payload
    axis_x = eye_vec / iod
    down = np.array([-axis_x[1], axis_x[0]])
    eye_mid = (e1 + e2) / 2.0

    if region in ("left_brow", "right_brow"):
        # Nudge brow into [0.25, 0.75] IOD above eye midline if outside valid band
        brow_pts = patched[s:e]
        up_offsets = -((brow_pts - eye_mid) @ down) / iod
        mean_up = float(up_offsets.mean())
        if mean_up < 0.05 or mean_up > 0.95:
            target_up = 0.45
            delta = (mean_up - target_up) * iod * down
            patched[s:e] = brow_pts + delta
        elif ctx.get("image") is not None:
            patched[s:e] = _snap_region_to_edge(ctx["image"], brow_pts, radius=3)

    elif region == "inner_lip":
        # Clamp inner lip inside outer lip bounds and enforce top < bottom along `down`
        outer = coords[30:42]
        ox0, oy0 = outer.min(axis=0)
        ox1, oy1 = outer.max(axis=0)
        pad_x = max(1.0, (ox1 - ox0) * 0.12)
        pad_y = max(1.0, (oy1 - oy0) * 0.12)
        inner = patched[42:50].copy()
        inner[:, 0] = np.clip(inner[:, 0], ox0 + pad_x, ox1 - pad_x)
        inner[:, 1] = np.clip(inner[:, 1], oy0 + pad_y, oy1 - pad_y)
        # Ensure top inner lip (43,44,45 -> local 1,2,3) is above bottom (47,48,49 -> local 5,6,7)
        top_proj = ((inner[1:4] - eye_mid) @ down).mean()
        bot_proj = ((inner[5:8] - eye_mid) @ down).mean()
        if top_proj > bot_proj:
            inner[[1, 2, 3, 5, 6, 7]] = inner[[7, 6, 5, 3, 2, 1]]
        patched[42:50] = inner

    elif region == "nose":
        # Project nose tip into [0.2, 1.0] IOD below eye midline
        nose_pts = patched[s:e]
        tip_down = float(((nose_pts[-1] - eye_mid) @ down) / iod)
        if tip_down < 0.0 or tip_down > 1.35:
            target_down = 0.65
            shift = (target_down - tip_down) * iod * down
            patched[s:e] = nose_pts + shift

    elif region == "outer_lip" and ctx.get("image") is not None:
        patched[s:e] = _snap_region_to_edge(ctx["image"], patched[s:e], radius=3)

    # Reconstruct in original format without touching other indices
    if isinstance(payload, dict):
        out = copy.deepcopy(payload)
        for idx in range(s, e):
            key = str(idx)
            if key in out:
                out[key]["x"] = float(patched[idx, 0])
                out[key]["y"] = float(patched[idx, 1])
        return out
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        out = copy.deepcopy(payload)
        for idx in range(s, e):
            out[idx]["x"] = float(patched[idx, 0])
            out[idx]["y"] = float(patched[idx, 1])
        return out
    return patched


def _snap_region_to_edge(image: Any, pts: Any, radius: int = 3):
    """Locally snap a small set of landmarks toward strongest Sobel edge within `radius` pixels."""
    import cv2
    import numpy as np

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    h, w = gray.shape[:2]
    out = np.asarray(pts, dtype=float).copy()
    for i, (x, y) in enumerate(out):
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        win = mag[y0:y1, x0:x1]
        if float(win.max()) > 0:
            dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
            out[i, 0] = float(x0 + dx)
            out[i, 1] = float(y0 + dy)
    return out


def _default_segmentation_patcher(region: str, payload: Any, issues: list[QualityIssue], ctx: dict[str, Any]) -> Any:
    """Repair only the flagged segmentation polygon (make_valid / clip to frame)."""
    from shapely.geometry import Polygon, box as shapely_box
    from shapely.validation import make_valid

    if not isinstance(payload, list):
        return payload

    target_id = region.replace("polygon_", "", 1)
    out = copy.deepcopy(payload)
    frame_wh = ctx.get("frame_wh")

    for shape in out:
        sid = str(shape.get("id", ""))
        if sid != target_id:
            continue
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        is_flat = isinstance(pts[0], (int, float))
        coords = [(float(pts[i]), float(pts[i + 1])) for i in range(0, len(pts) - 1, 2)] if is_flat else [(float(p[0]), float(p[1])) for p in pts]
        poly = Polygon(coords)
        if not poly.is_valid:
            repaired = make_valid(poly)
            if repaired.geom_type == "MultiPolygon":
                repaired = max(repaired.geoms, key=lambda g: g.area)
            elif repaired.geom_type == "GeometryCollection":
                polys = [g for g in repaired.geoms if g.geom_type == "Polygon"]
                repaired = max(polys, key=lambda g: g.area) if polys else poly.buffer(0)
            poly = repaired
        if frame_wh is not None and poly.is_valid:
            poly = poly.intersection(shapely_box(0, 0, frame_wh[0], frame_wh[1]))
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        if poly.geom_type == "Polygon" and not poly.is_empty:
            ext = list(poly.exterior.coords)[:-1]
            shape["points"] = [c for xy in ext for c in xy] if is_flat else [[float(x), float(y)] for x, y in ext]
    return out


def _default_bbox_patcher(region: str, payload: Any, issues: list[QualityIssue], ctx: dict[str, Any]) -> Any:
    """Repair only the flagged bbox (fix inverted coords, clip to frame_wh)."""
    if not isinstance(payload, list):
        return payload
    target_id = region.replace("box_", "", 1)
    out = copy.deepcopy(payload)
    frame_wh = ctx.get("frame_wh")
    for idx, b in enumerate(out):
        sid = str(b.get("id", idx))
        if sid != target_id:
            continue
        pts = b.get("points") or []
        if len(pts) != 4:
            continue
        x0, y0, x1, y1 = float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if frame_wh is not None:
            w, h = frame_wh
            x0 = max(0.0, min(float(w), x0))
            x1 = max(0.0, min(float(w), x1))
            y0 = max(0.0, min(float(h), y0))
            y1 = max(0.0, min(float(h), y1))
        b["points"] = [x0, y0, x1, y1]
    return out


class BoundedLocalRefiner:
    """Applies bounded local patches (max 2-3 attempts per region) without resetting whole skeletons."""

    def __init__(
        self,
        *,
        max_attempts_per_region: int = 3,
        gate: QualityGate | None = None,
        patchers: dict[str, PatchFn] | None = None,
    ) -> None:
        if not 1 <= max_attempts_per_region <= 3:
            raise ValueError("max_attempts_per_region must be between 1 and 3.")
        self.max_attempts_per_region = max_attempts_per_region
        self.gate = gate or QualityGate(QualityPolicy(max_refinement_attempts=max_attempts_per_region))
        self.patchers: dict[str, PatchFn] = {
            "face50": _default_face50_patcher,
            "segmentation": _default_segmentation_patcher,
            "bbox": _default_bbox_patcher,
        }
        if patchers:
            self.patchers.update(patchers)

    def refine(
        self,
        domain: str,
        payload: Any,
        *,
        image: Any = None,
        frame_wh: tuple[int, int] | None = None,
        confidence: float | None = None,
        frame_id: int | str | None = None,
        label: str = "",
        **validate_kwargs: Any,
    ) -> RefinementResult:
        """Iteratively patch only repairable regions with error evidence up to max_attempts_per_region."""
        d = domain.lower().strip()
        current = copy.deepcopy(payload)
        attempts_by_region: dict[str, int] = {}
        history: list[RefinementAttempt] = []
        patched_regions: list[str] = []

        report = self.gate.evaluate_domain(
            d,
            current,
            confidence=confidence,
            image=image,
            frame_wh=frame_wh,
            frame_id=frame_id,
            label=label,
            attempts_by_region=attempts_by_region,
            **validate_kwargs,
        )

        if report.level == QualityLevel.PASS:
            return RefinementResult(
                status="pass",
                payload=current,
                report=report,
                attempts_by_region=attempts_by_region,
                history=history,
                patched_regions=patched_regions,
                escalate_to_review=False,
            )

        patcher = self.patchers.get(d)
        ctx = {"image": image, "frame_wh": frame_wh, **validate_kwargs}

        while report.level == QualityLevel.REFINE and patcher is not None:
            progressed = False
            for region in list(report.repairable_regions):
                used = attempts_by_region.get(region, 0)
                if used >= self.max_attempts_per_region:
                    continue

                attempts_by_region[region] = used + 1
                region_issues = [i for i in report.issues if i.region == region and i.repairable]
                affected_pts = sorted({p for iss in region_issues for p in iss.affected_points})

                candidate = patcher(region, copy.deepcopy(current), region_issues, ctx)
                cand_report = self.gate.evaluate_domain(
                    d,
                    candidate,
                    confidence=confidence,
                    image=image,
                    frame_wh=frame_wh,
                    frame_id=frame_id,
                    label=label,
                    attempts_by_region=attempts_by_region,
                    **validate_kwargs,
                )

                accepted = cand_report.score >= report.score and (
                    len(cand_report.issues) < len(report.issues)
                    or cand_report.score > report.score
                    or cand_report.level == QualityLevel.PASS
                )
                history.append(
                    RefinementAttempt(
                        region=region,
                        attempt=attempts_by_region[region],
                        accepted=accepted,
                        reason="improved" if accepted else "no_improvement_or_regression",
                        score_before=report.score,
                        score_after=cand_report.score,
                        affected_points=affected_pts,
                    )
                )
                if accepted:
                    current = candidate
                    report = cand_report
                    progressed = True
                    if region not in patched_regions:
                        patched_regions.append(region)
                    if report.level != QualityLevel.REFINE:
                        break
            if not progressed:
                # Re-evaluate with updated attempts_by_region so exhausted regions escalate cleanly
                report = self.gate.evaluate_domain(
                    d,
                    current,
                    confidence=confidence,
                    image=image,
                    frame_wh=frame_wh,
                    frame_id=frame_id,
                    label=label,
                    attempts_by_region=attempts_by_region,
                    **validate_kwargs,
                )
                if not any(
                    attempts_by_region.get(r, 0) < self.max_attempts_per_region
                    for r in report.repairable_regions
                ):
                    break
                # Stop if no progress was made to guarantee termination
                break

        if report.level == QualityLevel.PASS:
            status = "refined" if patched_regions else "pass"
        elif any(v >= self.max_attempts_per_region for v in attempts_by_region.values()):
            status = "exhausted"
        elif report.level == QualityLevel.REVIEW:
            status = "escalate_review"
        else:
            status = "unchanged"

        return RefinementResult(
            status=status,
            payload=current,
            report=report,
            attempts_by_region=attempts_by_region,
            history=history,
            patched_regions=patched_regions,
            escalate_to_review=(report.level != QualityLevel.PASS),
        )
