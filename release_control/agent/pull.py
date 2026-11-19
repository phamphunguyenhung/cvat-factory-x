"""Pull: export CVAT local job annotations -> YOLO dataset + manifest.json (structural.py format).

Images/classes/scene_id/split are taken from the SOURCE manifest (demo/coco128_clean);
scene/split/job metadata is simulated by the Day 10 lab and is marked as such.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "demo" / "coco128_clean"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _shapes_of(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return list(payload.get("shapes") or [])
    return list(payload or [])


def _label_names(session: dict[str, Any]) -> dict[Any, str]:
    """Map cvat label id (int or str) -> label name; label_map may be name->id or id->name."""
    out: dict[Any, str] = {}
    for k, v in (session.get("label_map") or {}).items():
        if isinstance(v, int) or (isinstance(v, str) and v.isdigit() and not str(k).isdigit()):
            out[int(v)] = str(k)
        else:
            out[k] = str(v)
            if str(k).isdigit():
                out[int(k)] = str(v)
    return out


def _rect_to_yolo(points: list[float], w: int, h: int) -> tuple[float, float, float, float] | None:
    if len(points) < 4 or w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = (float(p) for p in points[:4])
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def pull_job_to_dataset(
    session: dict[str, Any],
    out_dir: Path | str,
    *,
    client: Any = None,
    source_dir: Path | str | None = None,
) -> Path:
    """Export job annotations to `out_dir` (images/, labels/, classes.txt, manifest.json). Returns out_dir."""
    from PIL import Image

    out = Path(out_dir)
    src = Path(source_dir) if source_dir else DEFAULT_SOURCE
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    classes: list[str] = list(src_manifest["classes"])
    cls_index = {name: i for i, name in enumerate(classes)}
    by_id = {e["image_id"]: e for e in src_manifest["images"]}

    if client is None:
        from release_control.demo_session.session import CvatDemoClient

        client = CvatDemoClient()
    payload = client.get_job_annotations()
    shapes = _shapes_of(payload)
    names = _label_names(session)

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()

    files: list[str] = list(session.get("image_files") or [])
    per_frame: dict[int, list[str]] = {i: [] for i in range(len(files))}
    skipped = {"non_rectangle": 0, "unknown_label": 0, "bad_frame": 0, "degenerate": 0}
    sizes: dict[int, tuple[int, int]] = {}
    for i, fname in enumerate(files):
        sp = src / "images" / Path(fname).name
        with Image.open(sp) as im:
            sizes[i] = im.size

    for sh in shapes:
        fr = sh.get("frame")
        if fr not in per_frame:
            skipped["bad_frame"] += 1
            continue
        if sh.get("type", "rectangle") != "rectangle":
            skipped["non_rectangle"] += 1
            continue
        name = sh.get("label") or sh.get("label_name") or names.get(sh.get("label_id"))
        if name not in cls_index:
            skipped["unknown_label"] += 1
            continue
        box = _rect_to_yolo(list(sh.get("points") or []), *sizes[fr])
        if box is None:
            skipped["degenerate"] += 1
            continue
        per_frame[fr].append(f"{cls_index[name]} " + " ".join(f"{v:.6f}" for v in box))

    entries: list[dict[str, Any]] = []
    for i, fname in enumerate(files):
        iid = Path(fname).stem
        meta = by_id.get(iid, {})
        shutil.copy2(src / "images" / Path(fname).name, out / "images" / Path(fname).name)
        lp = out / "labels" / f"{iid}.txt"
        lp.write_text("".join(line + "\n" for line in per_frame[i]), encoding="utf-8")
        entries.append(
            {
                "image_id": iid,
                "image_path": f"images/{Path(fname).name}",
                "label_path": f"labels/{iid}.txt",
                "sha256": _sha256(out / "images" / Path(fname).name),
                "label_sha256": _sha256(lp),
                "scene_id": meta.get("scene_id"),
                "split": meta.get("split"),
                "job_id": f"cvat-job-{session.get('job_id')}",
                "guideline_version": meta.get("guideline_version", src_manifest["guideline_version"]),
                "n_boxes": len(per_frame[i]),
                "simulated_fields": ["scene_id", "split"],
            }
        )

    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    manifest = {
        "dataset_id": f"rc-demo-task{session.get('task_id')}",
        "guideline_version": src_manifest["guideline_version"],
        "classes": classes,
        "num_classes": len(classes),
        "license": src_manifest.get("license", "CC BY 4.0 (COCO)"),
        "source": {
            "cvat_host": session.get("cvat_host"),
            "task_id": session.get("task_id"),
            "job_id": session.get("job_id"),
            "task_name": session.get("task_name"),
            "scene_split_simulated": True,
            "skipped_shapes": skipped,
        },
        "images": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
