"""Datumaro-style COCO <-> YOLO <-> COCO round-trip integrity adapter."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoundTripReport:
    """Result of a COCO -> YOLO -> COCO round-trip conversion check."""

    image_count_match: bool
    box_count_match: bool
    class_match: bool
    max_bbox_delta: float
    mean_iou: float
    passed: bool
    source_images: int
    roundtrip_images: int
    source_boxes: int
    roundtrip_boxes: int
    tolerance: float
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    if union <= 0.0:
        return 1.0 if inter == 0.0 else 0.0
    return inter / union


class DatumaroRoundTripAdapter:
    """Converts COCO JSON -> YOLO .txt -> COCO JSON and verifies geometry/class integrity."""

    def coco_to_yolo(self, coco_data: dict[str, Any], yolo_dir: Path) -> dict[int, Path]:
        """Write YOLO .txt files per image into yolo_dir and return {image_id: label_path}."""
        yolo_dir.mkdir(parents=True, exist_ok=True)
        images_by_id = {img["id"]: img for img in coco_data.get("images", [])}
        anns_by_img: dict[int, list[dict[str, Any]]] = {img_id: [] for img_id in images_by_id}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in anns_by_img:
                anns_by_img[img_id].append(ann)

        written: dict[int, Path] = {}
        for img_id, img in images_by_id.items():
            w = float(img["width"])
            h = float(img["height"])
            stem = Path(img["file_name"]).stem
            lbl_path = yolo_dir / f"{stem}.txt"
            lines: list[str] = []
            for ann in anns_by_img[img_id]:
                x_min, y_min, bw, bh = [float(v) for v in ann["bbox"]]
                xc = (x_min + bw / 2.0) / w
                yc = (y_min + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                cid = int(ann["category_id"])
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written[img_id] = lbl_path
        return written

    def yolo_to_coco(
        self,
        yolo_dir: Path,
        images: list[dict[str, Any]],
        categories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reconstruct COCO JSON from YOLO .txt files in yolo_dir."""
        rt_annotations: list[dict[str, Any]] = []
        ann_id = 1
        for img in images:
            w = float(img["width"])
            h = float(img["height"])
            stem = Path(img["file_name"]).stem
            lbl_path = yolo_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue
            for raw in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                xc, yc, nw, nh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                bw = nw * w
                bh = nh * h
                x_min = (xc - nw / 2.0) * w
                y_min = (yc - nh / 2.0) * h
                rt_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img["id"],
                        "category_id": cid,
                        "bbox": [x_min, y_min, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        return {
            "images": list(images),
            "annotations": rt_annotations,
            "categories": list(categories),
        }

    def verify_roundtrip(
        self,
        coco_source: str | Path | dict[str, Any],
        tolerance: float = 1e-4,
    ) -> RoundTripReport:
        """Run COCO -> YOLO -> COCO round-trip and compute integrity metrics."""
        if isinstance(coco_source, dict):
            coco_data = coco_source
        else:
            coco_data = json.loads(Path(coco_source).read_text(encoding="utf-8"))

        src_images = coco_data.get("images", [])
        src_anns = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        with tempfile.TemporaryDirectory(prefix="datumaro_rt_") as tmp:
            yolo_dir = Path(tmp) / "yolo"
            self.coco_to_yolo(coco_data, yolo_dir)
            rt_coco = self.yolo_to_coco(yolo_dir, src_images, categories)

        rt_images = rt_coco["images"]
        rt_anns = rt_coco["annotations"]

        image_count_match = len(src_images) == len(rt_images)
        box_count_match = len(src_anns) == len(rt_anns)

        img_dims = {img["id"]: (float(img["width"]), float(img["height"])) for img in src_images}
        src_by_img: dict[int, list[dict[str, Any]]] = {}
        rt_by_img: dict[int, list[dict[str, Any]]] = {}
        for a in src_anns:
            src_by_img.setdefault(a["image_id"], []).append(a)
        for a in rt_anns:
            rt_by_img.setdefault(a["image_id"], []).append(a)

        class_match = True
        max_delta = 0.0
        ious: list[float] = []
        details: list[str] = []

        for img_id, (w, h) in img_dims.items():
            s_list = src_by_img.get(img_id, [])
            r_list = rt_by_img.get(img_id, [])
            if len(s_list) != len(r_list):
                class_match = False
                details.append(f"image_id={img_id}: box count mismatch {len(s_list)} vs {len(r_list)}")
                continue

            for s_ann, r_ann in zip(s_list, r_list):
                if int(s_ann["category_id"]) != int(r_ann["category_id"]):
                    class_match = False
                    details.append(f"image_id={img_id}: class mismatch {s_ann['category_id']} vs {r_ann['category_id']}")
                sb = [float(v) for v in s_ann["bbox"]]
                rb = [float(v) for v in r_ann["bbox"]]
                # Normalized coordinate delta
                s_norm = [sb[0] / w, sb[1] / h, sb[2] / w, sb[3] / h]
                r_norm = [rb[0] / w, rb[1] / h, rb[2] / w, rb[3] / h]
                delta = max(abs(a - b) for a, b in zip(s_norm, r_norm))
                if delta > max_delta:
                    max_delta = delta
                ious.append(_bbox_iou_xywh(sb, rb))

        mean_iou = sum(ious) / len(ious) if ious else 1.0
        passed = (
            image_count_match
            and box_count_match
            and class_match
            and max_delta <= tolerance
            and (mean_iou >= 0.99 or len(src_anns) == 0)
        )

        return RoundTripReport(
            image_count_match=image_count_match,
            box_count_match=box_count_match,
            class_match=class_match,
            max_bbox_delta=round(max_delta, 8),
            mean_iou=round(mean_iou, 8),
            passed=passed,
            source_images=len(src_images),
            roundtrip_images=len(rt_images),
            source_boxes=len(src_anns),
            roundtrip_boxes=len(rt_anns),
            tolerance=tolerance,
            details=details,
        )
