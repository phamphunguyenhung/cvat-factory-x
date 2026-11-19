"""Human-in-the-Loop Review Queue & Local HTML Inspector."""
from __future__ import annotations

import html
import json
from pathlib import Path

from harness.contracts import ComposedScene, ElementStatus, LabelCategory, ReviewItem, to_serializable


def build_review_queue(scene: ComposedScene) -> list[ReviewItem]:
    """Collect risky / uncertain elements with explicit evidence for human review."""
    items: list[ReviewItem] = []
    for el in scene.elements:
        if el.status == ElementStatus.SUPPRESSED:
            continue
        reasons: list[str] = list(el.review_flags)
        score = 0.0
        if el.status == ElementStatus.REVIEW:
            score += 2.0
            if not reasons:
                reasons.append("below_accept_threshold")
        if el.occluded and el.occlusion_ratio >= 0.35:
            score += 1.5
            reasons.append(f"heavy_occlusion_{el.occlusion_ratio:.2f}")
        if el.category == LabelCategory.STUFF and el.uncertainty >= 0.40:
            score += 1.2
            reasons.append(f"uncertain_boundary_{el.uncertainty:.2f}")
        if score <= 0.0:
            continue

        bbox = el.points[:4] if el.export_shape_type == "rectangle" and len(el.points) == 4 else None
        items.append(
            ReviewItem(
                frame_id=scene.frame_id,
                element_id=el.element_id,
                label=el.label,
                priority_score=round(score, 2),
                reason_code="|".join(reasons),
                message=f"f{scene.frame_id} {el.label} ({el.element_id}): {', '.join(reasons)}",
                bbox_xyxy=bbox,
                evidence={"conf": el.conf, "occlusion_ratio": el.occlusion_ratio, "uncertainty": el.uncertainty},
            )
        )
    items.sort(key=lambda r: r.priority_score, reverse=True)
    scene.review_items = items
    return items


def write_review_html(scenes: list[ComposedScene], out_html: Path, out_json: Path | None = None) -> Path:
    """Write self-contained local HTML review report + optional JSON queue."""
    all_items = [item for s in scenes for item in s.review_items]
    all_items.sort(key=lambda r: r.priority_score, reverse=True)

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps([to_serializable(i) for i in all_items], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    rows_html = "\n".join(
        f"<tr><td>{i.frame_id}</td><td>{html.escape(i.label)}</td>"
        f"<td>{i.priority_score:.2f}</td><td>{html.escape(i.reason_code)}</td>"
        f"<td>{html.escape(i.message)}</td></tr>"
        for i in all_items[:300]
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>V3 Panoptic Harness — Review Queue</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; font-size: 14px; }}
th {{ background: #e2e8f0; }}
</style></head><body>
<h1>V3 Panoptic Harness — Prioritized Human Review Queue</h1>
<p>Frames: {len(scenes)} | Total flagged items: {len(all_items)}</p>
<table><thead><tr><th>Frame</th><th>Label</th><th>Priority</th><th>Reason</th><th>Details</th></tr></thead>
<tbody>{rows_html}</tbody></table></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    return out_html
