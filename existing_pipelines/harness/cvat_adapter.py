"""Safe CVAT & local ShapeStore adapter (read-only by default)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cvat_tools.core.client import Cvat, WriteNotAuthorized
from cvat_tools.core.shapestore import ShapeStore
from harness.contracts import ComposedScene, ElementStatus, to_serializable


class CvatAdapter:
    """Bridges ComposedScene -> local ShapeStore / JSON draft / gated remote CVAT."""

    def __init__(self, store: ShapeStore | None = None):
        self.store = store

    def save_to_local_store(
        self,
        scenes: list[ComposedScene],
        *,
        include_review: bool = True,
    ) -> dict[str, int]:
        """Persist V3 drafts locally into SQLite ShapeStore under stage='rule_draft'."""
        store = self.store or ShapeStore()
        written = 0
        skipped = 0
        for sc in scenes:
            for el in sc.elements:
                if el.status == ElementStatus.SUPPRESSED:
                    skipped += 1
                    continue
                if el.status == ElementStatus.REVIEW and not include_review:
                    skipped += 1
                    continue
                prov = dict(el.provenance)
                prov["v3_status"] = el.status.value
                prov["occlusion_ratio"] = el.occlusion_ratio
                store.add(
                    sc.job_id,
                    sc.frame_id,
                    el.label,
                    el.export_shape_type,
                    el.points,
                    "rule_draft",
                    z_order=el.z_order,
                    occluded=el.occluded,
                    attributes={"occluded": el.occluded, "truncated": el.truncated},
                    conf=el.conf,
                    issues=el.review_flags,
                    provenance=prov,
                )
                written += 1
        return {"written": written, "skipped": skipped}

    @staticmethod
    def export_draft_json(
        scenes: list[ComposedScene],
        out_path: Path,
        *,
        label_id_map: dict[str, int] | None = None,
        include_review: bool = False,
    ) -> Path:
        shapes: list[dict[str, Any]] = []
        for sc in scenes:
            for el in sc.elements:
                if el.status == ElementStatus.SUPPRESSED:
                    continue
                if el.status == ElementStatus.REVIEW and not include_review:
                    continue
                shapes.append(el.to_cvat_shape(label_id_map=label_id_map))
        payload = {
            "schema": "cvat_v3_draft",
            "job": scenes[0].job_id if scenes else None,
            "frames": [sc.frame_id for sc in scenes],
            "count": len(shapes),
            "shapes": to_serializable(shapes),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path

    @staticmethod
    def push_remote_guarded(
        job_id: int,
        shapes: list[dict[str, Any]],
        *,
        profile: str = "local",
        write_enabled: bool = False,
        confirm_yes: bool = False,
    ) -> dict[str, Any]:
        """Strictly refuse remote writes unless BOTH write_enabled and confirm_yes are True."""
        if not (write_enabled and confirm_yes):
            raise WriteNotAuthorized(
                "Remote CVAT write blocked: requires explicit write_enabled=True AND confirm_yes=True."
            )
        client = Cvat.from_env(profile=profile, write_enabled=True)
        return client.create_shapes(job_id, shapes)
