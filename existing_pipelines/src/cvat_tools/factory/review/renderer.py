"""Visual evidence renderer for BEFORE / AFTER / OVERLAY + Crop Zoom + Original/Enhanced."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_bgr(image: Any):
    import cv2

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return img
    return image.copy()


def enhance_for_display(bgr_img: Any, *, clip_limit: float = 2.5, gamma: float = 1.35):
    """Create a display-only CLAHE + gamma enhanced image (never used as inference input)."""
    import cv2
    import numpy as np

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    inv_gamma = 1.0 / max(0.1, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(merged, table)


def _extract_xy_list(shapes: Any) -> list[tuple[float, float]]:
    """Extract all (x, y) coordinates from dicts, lists, or arrays for bounding focus."""
    pts: list[tuple[float, float]] = []
    if shapes is None:
        return pts
    if isinstance(shapes, dict):
        if "points" in shapes:
            return _extract_xy_list(shapes["points"])
        for k in sorted(shapes.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            v = shapes[k]
            if isinstance(v, dict) and "x" in v and "y" in v:
                pts.append((float(v["x"]), float(v["y"])))
        return pts
    if isinstance(shapes, list):
        if not shapes:
            return pts
        if isinstance(shapes[0], dict):
            if "x" in shapes[0] and "y" in shapes[0]:
                return [(float(p["x"]), float(p["y"])) for p in shapes]
            for item in shapes:
                pts.extend(_extract_xy_list(item))
            return pts
        if isinstance(shapes[0], (int, float)) and len(shapes) == 4:
            x0, y0, x1, y1 = [float(v) for v in shapes]
            return [(x0, y0), (x1, y1)]
        if isinstance(shapes[0], (list, tuple)):
            return [(float(p[0]), float(p[1])) for p in shapes]
    # numpy array
    try:
        arr = shapes.reshape(-1, 2)
        return [(float(x), float(y)) for x, y in arr]
    except Exception:
        return pts


def _draw_shapes(canvas: Any, shapes: Any, color: tuple[int, int, int], affected_points: set[int] | None = None) -> None:
    """Draw landmarks, bboxes, or polygons onto `canvas` in BGR `color`."""
    import cv2
    import numpy as np

    if shapes is None:
        return

    # Dict of indexed landmarks (Face50 / Pose17)
    if isinstance(shapes, dict) and "points" not in shapes:
        for k, v in shapes.items():
            if not isinstance(v, dict) or "x" not in v or "y" not in v:
                continue
            ix, iy = int(round(float(v["x"]))), int(round(float(v["y"])))
            is_aff = affected_points is not None and str(k).isdigit() and int(k) in affected_points
            r = 4 if is_aff else 2
            c = (0, 165, 255) if is_aff else color
            cv2.circle(canvas, (ix, iy), r, c, -1, lineType=cv2.LINE_AA)
        return

    # List of shape dicts (bbox / polygon / skeleton)
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], dict):
        if "x" in shapes[0] and "y" in shapes[0]:
            for idx, v in enumerate(shapes):
                ix, iy = int(round(float(v["x"]))), int(round(float(v["y"])))
                is_aff = affected_points is not None and idx in affected_points
                cv2.circle(canvas, (ix, iy), 4 if is_aff else 2, (0, 165, 255) if is_aff else color, -1, lineType=cv2.LINE_AA)
            return
        for item in shapes:
            pts = item.get("points")
            stype = str(item.get("type", "")).lower()
            if isinstance(pts, list) and len(pts) == 4 and isinstance(pts[0], (int, float)) and stype != "polygon":
                x0, y0, x1, y1 = [int(round(float(v))) for v in pts]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
            elif isinstance(pts, list) and len(pts) >= 3:
                pairs = (
                    [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
                    if isinstance(pts[0], (int, float))
                    else pts
                )
                arr = np.array([[int(round(float(x))), int(round(float(y)))] for x, y in pairs], dtype=np.int32)
                cv2.polylines(canvas, [arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            elif isinstance(pts, dict):
                _draw_shapes(canvas, pts, color, affected_points)
        return

    # Raw (N, 2) array
    try:
        arr = np.asarray(shapes, dtype=float).reshape(-1, 2)
        for idx, (x, y) in enumerate(arr):
            is_aff = affected_points is not None and idx in affected_points
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                4 if is_aff else 2,
                (0, 165, 255) if is_aff else color,
                -1,
                lineType=cv2.LINE_AA,
            )
    except Exception:
        pass


def render_case_evidence(
    *,
    image: Any,
    before_shapes: Any,
    after_shapes: Any,
    out_dir: Path,
    case_id: str,
    affected_points: list[int] | None = None,
    focus_box: tuple[float, float, float, float] | None = None,
    zoom_target_px: int = 420,
) -> dict[str, str]:
    """Render BEFORE, AFTER, OVERLAY, CROP_ZOOM, ORIGINAL_CROP, and ENHANCED_CROP PNGs."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    base = _load_bgr(image)
    h, w = base.shape[:2]
    aff = set(affected_points or [])

    before_img = base.copy()
    _draw_shapes(before_img, before_shapes, (0, 0, 230), aff)  # Red = before

    after_img = base.copy()
    _draw_shapes(after_img, after_shapes, (0, 190, 0), aff)  # Green = after

    overlay_img = base.copy()
    _draw_shapes(overlay_img, before_shapes, (0, 0, 230), None)
    _draw_shapes(overlay_img, after_shapes, (0, 210, 0), aff)

    # Determine crop window around affected points or focus_box
    if focus_box is not None:
        fx0, fy0, fx1, fy1 = focus_box
    else:
        all_pts = _extract_xy_list(after_shapes) or _extract_xy_list(before_shapes)
        if aff and all_pts:
            sel = [all_pts[i] for i in aff if 0 <= i < len(all_pts)]
            if sel:
                all_pts = sel
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            fx0, fy0, fx1, fy1 = min(xs), min(ys), max(xs), max(ys)
        else:
            fx0, fy0, fx1, fy1 = 0.0, 0.0, float(w), float(h)

    pad = max(24.0, 0.35 * max(fx1 - fx0, fy1 - fy0))
    cx0 = max(0, int(round(fx0 - pad)))
    cy0 = max(0, int(round(fy0 - pad)))
    cx1 = min(w, int(round(fx1 + pad)))
    cy1 = min(h, int(round(fy1 + pad)))
    if cx1 <= cx0:
        cx0, cx1 = 0, w
    if cy1 <= cy0:
        cy0, cy1 = 0, h

    orig_crop = base[cy0:cy1, cx0:cx1].copy()
    enh_crop = enhance_for_display(orig_crop)
    overlay_crop = overlay_img[cy0:cy1, cx0:cx1].copy()

    def _fit(im):
        ch, cw = im.shape[:2]
        scale = min(6.0, zoom_target_px / max(1, max(ch, cw)))
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        return cv2.resize(im, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

    paths = {
        "before_path": str(out_dir / f"{case_id}_before.png"),
        "after_path": str(out_dir / f"{case_id}_after.png"),
        "overlay_path": str(out_dir / f"{case_id}_overlay.png"),
        "crop_zoom_path": str(out_dir / f"{case_id}_crop_zoom.png"),
        "original_crop_path": str(out_dir / f"{case_id}_original_crop.png"),
        "enhanced_crop_path": str(out_dir / f"{case_id}_enhanced_crop.png"),
    }

    cv2.imwrite(paths["before_path"], before_img)
    cv2.imwrite(paths["after_path"], after_img)
    cv2.imwrite(paths["overlay_path"], overlay_img)
    cv2.imwrite(paths["crop_zoom_path"], _fit(overlay_crop))
    cv2.imwrite(paths["original_crop_path"], _fit(orig_crop))
    cv2.imwrite(paths["enhanced_crop_path"], _fit(enh_crop))
    return paths
