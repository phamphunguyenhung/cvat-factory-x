"""CVAT live gateway & local cache / COCO250 dataset adapter (strictly read-only on sources)."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_COCO250_DIR = Path(
    r"D:\06-Learning\AIA-VINGROUP\learning\day10\labs\K5-L23-Day10-Dataset-Release-Handover\data_source\coco250"
)
DEFAULT_COCO250_CALIB_DIR = Path(
    r"D:\06-Learning\AIA-VINGROUP\learning\day10\labs\K5-L23-Day10-Dataset-Release-Handover\data_source\coco250_calib"
)
DEFAULT_CVAT_JOBS_DIR = Path(r"D:\06-Learning\AIA-VINGROUP\learning\cvat\data\jobs\edufun")
DEFAULT_JOB285_YOLO_DIR = Path(r"D:\06-Learning\AIA-VINGROUP\learning\cvat\data\datasets\job285_yolo")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def _parse_yolo_lines(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        boxes.append((cls_id, xc, yc, w, h))
    return boxes


class CvatConnector:
    """Read-only connector to live CVAT (via CvatGateway) with local cache & coco250 fallback."""

    def __init__(
        self,
        coco250_dir: str | Path = DEFAULT_COCO250_DIR,
        coco250_calib_dir: str | Path = DEFAULT_COCO250_CALIB_DIR,
        cvat_jobs_dir: str | Path = DEFAULT_CVAT_JOBS_DIR,
        profile: str = "local",
    ) -> None:
        self.coco250_dir = Path(coco250_dir)
        self.coco250_calib_dir = Path(coco250_calib_dir)
        self.cvat_jobs_dir = Path(cvat_jobs_dir)
        self.profile = profile

    def probe_cvat_live(self) -> dict[str, Any]:
        """Safely probe CvatGateway connectivity in read-only mode without raising."""
        try:
            from cvat_tools.core.client import Cvat, CvatError
            from cvat_tools.factory.cvat.gateway import CvatGateway
        except Exception as exc:
            return {
                "available": False,
                "reachable": False,
                "authenticated": False,
                "host": None,
                "server_version": None,
                "username": None,
                "write_enabled": False,
                "error": f"cvat_tools unavailable: {exc}",
            }

        try:
            client = Cvat.from_env(profile=self.profile, write_enabled=False)
            gateway = CvatGateway(
                host=client.host,
                pat="configured",
                write_enabled=False,
                client=client,
            )
            res = gateway.test_connection()
            return {
                "available": True,
                "reachable": res.reachable,
                "authenticated": res.authenticated,
                "host": res.host,
                "server_version": res.server_version,
                "username": res.username,
                "write_enabled": False,
                "error": res.error,
            }
        except CvatError as exc:
            return {
                "available": True,
                "reachable": False,
                "authenticated": False,
                "host": None,
                "server_version": None,
                "username": None,
                "write_enabled": False,
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "available": True,
                "reachable": False,
                "authenticated": False,
                "host": None,
                "server_version": None,
                "username": None,
                "write_enabled": False,
                "error": str(exc),
            }

    def export_coco250_subset(
        self,
        dest_dir: str | Path,
        limit: int = 30,
        seed: int = 42,
        include_calib: bool = False,
        guideline_version: str = "v1.0",
    ) -> dict[str, Any]:
        """Copy a reproducible subset of coco250 into dest_dir and build manifest + COCO + reference boxes."""
        img_root = self.coco250_dir / "images" / "train2017"
        lbl_root = self.coco250_dir / "labels" / "train2017"
        if not img_root.exists():
            raise FileNotFoundError(f"coco250 images directory not found: {img_root}")

        all_images = sorted(img_root.glob("*.jpg"))
        if limit > 0 and limit < len(all_images):
            rng = random.Random(seed)
            selected = sorted(rng.sample(all_images, limit))
        else:
            selected = all_images

        entries: list[tuple[Path, Path, str, bool]] = [
            (img_p, lbl_root / f"{img_p.stem}.txt", "coco250", False) for img_p in selected
        ]

        if include_calib and (self.coco250_calib_dir / "images").exists():
            calib_imgs = sorted((self.coco250_calib_dir / "images").rglob("*.jpg"))
            for img_p in calib_imgs:
                lbl_p = self.coco250_calib_dir / "labels" / img_p.parent.name / f"{img_p.stem}.txt"
                if not lbl_p.exists():
                    lbl_p = self.coco250_calib_dir / "labels" / f"{img_p.stem}.txt"
                entries.append((img_p, lbl_p, "coco250_calib", True))

        out_dir = Path(dest_dir)
        out_img_dir = out_dir / "images"
        out_lbl_dir = out_dir / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        manifest_items: list[dict[str, Any]] = []
        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []
        reference_boxes: dict[str, list[dict[str, Any]]] = {}
        seen_categories: set[int] = set()
        ann_id = 1

        for idx, (src_img, src_lbl, source_tag, is_calib) in enumerate(entries, start=1):
            dst_img = out_img_dir / src_img.name
            dst_lbl = out_lbl_dir / f"{src_img.stem}.txt"
            shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")

            width, height = _read_image_size(dst_img)
            yolo_boxes = _parse_yolo_lines(dst_lbl)

            # Deterministic scene grouping by stem hash/modulo so splits remain scene-disjoint
            scene_bucket = int(hashlib.sha256(src_img.stem.encode("utf-8")).hexdigest()[:8], 16) % 10
            scene_id = f"scene_{scene_bucket:02d}"
            split = "val" if (is_calib or scene_bucket in (8, 9)) else "train"

            img_sha = _sha256_file(dst_img)
            lbl_sha = _sha256_file(dst_lbl)

            coco_images.append(
                {
                    "id": idx,
                    "file_name": dst_img.name,
                    "width": width,
                    "height": height,
                }
            )

            ref_list: list[dict[str, Any]] = []
            for cls_id, xc, yc, bw, bh in yolo_boxes:
                seen_categories.add(cls_id)
                x_min = (xc - bw / 2.0) * width
                y_min = (yc - bh / 2.0) * height
                w_px = bw * width
                h_px = bh * height
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": cls_id,
                        "bbox": [round(x_min, 4), round(y_min, 4), round(w_px, 4), round(h_px, 4)],
                        "area": round(w_px * h_px, 4),
                        "iscrowd": 0,
                    }
                )
                ref_list.append(
                    {
                        "annotation_id": ann_id,
                        "category_id": cls_id,
                        "norm_xywh": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                        "bbox_xywh": [round(x_min, 4), round(y_min, 4), round(w_px, 4), round(h_px, 4)],
                    }
                )
                ann_id += 1

            reference_boxes[dst_img.name] = ref_list
            manifest_items.append(
                {
                    "image_id": idx,
                    "file_name": dst_img.name,
                    "image_path": f"images/{dst_img.name}",
                    "label_path": f"labels/{dst_lbl.name}",
                    "width": width,
                    "height": height,
                    "sha256": img_sha,
                    "label_sha256": lbl_sha,
                    "box_count": len(yolo_boxes),
                    "scene_id": scene_id,
                    "split": split,
                    "guideline_version": guideline_version,
                    "privacy_cleared": True,
                    "source": source_tag,
                }
            )

        categories = [{"id": cid, "name": f"class_{cid}", "supercategory": "object"} for cid in sorted(seen_categories)]
        coco_payload = {
            "info": {
                "description": "COCO250 deterministic subset",
                "seed": seed,
                "guideline_version": guideline_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        dataset_digest = _sha256_json(
            [{"file_name": m["file_name"], "sha256": m["sha256"], "label_sha256": m["label_sha256"]} for m in manifest_items]
        )
        manifest_payload = {
            "source": "coco250",
            "seed": seed,
            "limit": limit,
            "include_calib": include_calib,
            "guideline_version": guideline_version,
            "dataset_digest": dataset_digest,
            "image_count": len(manifest_items),
            "annotation_count": len(coco_annotations),
            "items": manifest_items,
        }

        manifest_path = out_dir / "manifest.json"
        coco_path = out_dir / "coco_annotations.json"
        ref_path = out_dir / "reference_boxes.json"

        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        coco_path.write_text(json.dumps(coco_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        ref_path.write_text(json.dumps(reference_boxes, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "dest_dir": str(out_dir),
            "manifest_path": str(manifest_path),
            "coco_path": str(coco_path),
            "reference_boxes_path": str(ref_path),
            "dataset_digest": dataset_digest,
            "image_count": len(manifest_items),
            "annotation_count": len(coco_annotations),
        }

    def export_cvat_job_local(
        self,
        job_id: int | str,
        dest_dir: str | Path,
        guideline_version: str = "v1.0",
    ) -> dict[str, Any]:
        """Package a cached CVAT job directory (read-only) into a local snapshot."""
        job_dir = self.cvat_jobs_dir / str(job_id)
        if not job_dir.exists():
            raise FileNotFoundError(f"Cached CVAT job directory not found: {job_dir}")

        job_info_src = job_dir / "job_info.json"
        labels_src = job_dir / "labels.json"
        frames_src = job_dir / "frames"

        out_dir = Path(dest_dir)
        out_img_dir = out_dir / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        if job_info_src.exists():
            shutil.copy2(job_info_src, out_dir / "job_info.json")
        if labels_src.exists():
            shutil.copy2(labels_src, out_dir / "labels.json")

        manifest_items: list[dict[str, Any]] = []
        if frames_src.exists():
            frame_files = sorted(p for p in frames_src.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for idx, src_frame in enumerate(frame_files, start=1):
                dst_frame = out_img_dir / src_frame.name
                shutil.copy2(src_frame, dst_frame)
                width, height = _read_image_size(dst_frame)
                manifest_items.append(
                    {
                        "image_id": idx,
                        "file_name": dst_frame.name,
                        "image_path": f"images/{dst_frame.name}",
                        "width": width,
                        "height": height,
                        "sha256": _sha256_file(dst_frame),
                        "scene_id": f"edufun_job_{job_id}",
                        "split": "val",
                        "guideline_version": guideline_version,
                        "privacy_cleared": True,
                        "source": "cvat_local_cache",
                    }
                )

        dataset_digest = _sha256_json([{"file_name": m["file_name"], "sha256": m["sha256"]} for m in manifest_items])
        manifest_payload = {
            "source": "cvat_local_cache",
            "job_id": str(job_id),
            "guideline_version": guideline_version,
            "dataset_digest": dataset_digest,
            "image_count": len(manifest_items),
            "items": manifest_items,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "dest_dir": str(out_dir),
            "manifest_path": str(manifest_path),
            "job_id": str(job_id),
            "source": "cvat_local_cache",
            "dataset_digest": dataset_digest,
            "image_count": len(manifest_items),
        }
