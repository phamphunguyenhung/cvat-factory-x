"""Deterministic fault injection lab for CVAT Factory X datasets (T1-T8)."""

from __future__ import annotations

from collections import Counter
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable
import uuid

from release_control.models import (
    FaultManifest,
    FaultRecord,
    FaultType,
    Severity,
)

_SHORT_TO_FAULT_TYPE: dict[str, FaultType] = {
    "T1": FaultType.T1_MISSING_IMAGE,
    "T2": FaultType.T2_WRONG_HASH,
    "T3": FaultType.T3_INVALID_CLASS,
    "T4": FaultType.T4_BBOX_OUT_OF_BOUNDS,
    "T5": FaultType.T5_BBOX_DRIFT,
    "T6": FaultType.T6_CLASS_SWAP,
    "T7": FaultType.T7_SCENE_LEAK,
    "T8": FaultType.T8_GUIDELINE_MISMATCH,
}


def normalize_fault_type(fault_type: FaultType | str) -> FaultType:
    """Normalize a FaultType enum, full name, or short code (T1..T8) to FaultType."""
    if isinstance(fault_type, FaultType):
        return fault_type
    key = str(fault_type).strip().upper()
    if key in _SHORT_TO_FAULT_TYPE:
        return _SHORT_TO_FAULT_TYPE[key]
    return FaultType(key)


def _resolve_image_relpath(entry: dict[str, Any]) -> str | None:
    rel = entry.get("image_path") or entry.get("relative_path") or entry.get("file_name")
    return str(rel) if isinstance(rel, str) and rel.strip() else None


def _resolve_label_relpath(entry: dict[str, Any]) -> str | None:
    rel = entry.get("label_path")
    if isinstance(rel, str) and rel.strip():
        return rel
    img_rel = _resolve_image_relpath(entry)
    if img_rel:
        p = Path(img_rel)
        return str(Path("labels") / f"{p.stem}.txt")
    return None


def _resolve_image_id(entry: dict[str, Any], idx: int) -> str:
    img_id = entry.get("image_id")
    if isinstance(img_id, str) and img_id.strip():
        return img_id
    return f"img_{idx}"


def _parse_yolo_rows(label_path: Path) -> list[list[str]]:
    if not label_path.is_file():
        return []
    rows: list[list[str]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        rows.append(stripped.split())
    return rows


def _write_yolo_rows(label_path: Path, rows: list[list[str]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(" ".join(parts) for parts in rows)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


class FaultInjector:
    """Isolated, deterministic fault injector covering T1..T8."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def inject(
        self,
        source_dir: str | Path,
        dest_dir: str | Path,
        fault_types: Iterable[FaultType | str],
        seed: int | None = None,
    ) -> FaultManifest:
        """Copy source_dir to an isolated dest_dir and inject requested faults deterministically."""
        active_seed = self.seed if seed is None else seed
        rng = random.Random(active_seed)

        src_root = Path(source_dir).resolve()
        dst_root = Path(dest_dir).resolve()
        if src_root == dst_root:
            raise ValueError("dest_dir must be distinct from source_dir to prevent source mutation")
        if not src_root.is_dir():
            raise FileNotFoundError(f"Source dataset directory not found: {src_root}")

        if dst_root.exists():
            shutil.rmtree(dst_root)
        shutil.copytree(src_root, dst_root)

        manifest_path = dst_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.json not found in {dst_root}")

        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        images: list[dict[str, Any]] = manifest.get("images", [])
        if not images:
            raise ValueError("Dataset manifest contains no images for fault injection")

        normalized_types = [normalize_fault_type(ft) for ft in fault_types]
        records: list[FaultRecord] = []

        for ft in normalized_types:
            record = self._apply_fault(dst_root, manifest, ft, rng)
            records.append(record)

        # Keep label_sha256 in sync for label/semantic faults (T3-T6) so they don't falsely trigger T2
        if FaultType.T2_WRONG_HASH not in normalized_types:
            import hashlib

            for entry in images:
                if isinstance(entry, dict) and "label_sha256" in entry and entry.get("label_path"):
                    lbl = dst_root / str(entry["label_path"])
                    if lbl.is_file():
                        entry["label_sha256"] = hashlib.sha256(lbl.read_bytes()).hexdigest()

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        summary = dict(Counter(rec.fault_type.value for rec in records))
        run_id = f"fault-run-{active_seed}-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{src_root}:{active_seed}:{normalized_types}').hex[:8]}"
        base_snapshot_id = str(manifest.get("dataset_id", src_root.name))

        fault_manifest = FaultManifest(
            run_id=run_id,
            base_snapshot_id=base_snapshot_id,
            faults=records,
            summary=summary,
        )

        # Persist fault_manifest.json with seed, provenance, and explicit target/ground_truth fields
        payload = fault_manifest.model_dump(mode="json")
        payload["seed"] = active_seed
        payload["provenance"] = {
            "source_dir": str(src_root),
            "working_dir": str(dst_root),
            "seed": active_seed,
            "requested_fault_types": [ft.value for ft in normalized_types],
        }
        for idx, rec in enumerate(records):
            payload["faults"][idx]["target"] = rec.target_image or rec.target_label
            payload["faults"][idx]["seed"] = active_seed
            payload["faults"][idx]["ground_truth"] = {
                "before": rec.before_value,
                "after": rec.after_value,
            }

        (dst_root / "fault_manifest.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return fault_manifest

    def _apply_fault(
        self,
        work_dir: Path,
        manifest: dict[str, Any],
        fault_type: FaultType,
        rng: random.Random,
    ) -> FaultRecord:
        images: list[dict[str, Any]] = manifest["images"]
        fault_id = f"{fault_type.value}-{rng.randint(1000, 9999)}"

        if fault_type == FaultType.T1_MISSING_IMAGE:
            idx = rng.randrange(len(images))
            entry = images[idx]
            img_id = _resolve_image_id(entry, idx)
            img_rel = _resolve_image_relpath(entry)
            if img_rel is None:
                raise ValueError(f"Image '{img_id}' has no image_path")
            img_path = work_dir / img_rel
            existed = img_path.is_file()
            if existed:
                img_path.unlink()
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.CRITICAL,
                target_image=img_rel,
                description=f"Removed image file '{img_rel}' while keeping manifest entry '{img_id}'.",
                before_value={"exists": existed, "image_id": img_id},
                after_value={"exists": False, "image_id": img_id},
            )

        if fault_type == FaultType.T2_WRONG_HASH:
            idx = rng.randrange(len(images))
            entry = images[idx]
            img_id = _resolve_image_id(entry, idx)
            img_rel = _resolve_image_relpath(entry)
            if img_rel is None:
                raise ValueError(f"Image '{img_id}' has no image_path")
            img_path = work_dir / img_rel
            before_sha = entry.get("sha256")
            original = img_path.read_bytes() if img_path.is_file() else b""
            corrupted = original + b"-corrupted-t2"
            img_path.write_bytes(corrupted)
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.CRITICAL,
                target_image=img_rel,
                description=f"Corrupted bytes of '{img_rel}' without updating manifest sha256.",
                before_value={"sha256": before_sha, "size_bytes": len(original)},
                after_value={"sha256_stale_in_manifest": before_sha, "size_bytes": len(corrupted)},
            )

        if fault_type == FaultType.T3_INVALID_CLASS:
            idx = rng.randrange(len(images))
            entry = images[idx]
            img_id = _resolve_image_id(entry, idx)
            lbl_rel = _resolve_label_relpath(entry)
            if lbl_rel is None:
                raise ValueError(f"Image '{img_id}' has no label_path")
            lbl_path = work_dir / lbl_rel
            rows = _parse_yolo_rows(lbl_path)
            invalid_cid = 255 if rng.random() < 0.5 else 99
            before_rows = [list(r) for r in rows]
            if rows:
                row_idx = rng.randrange(len(rows))
                rows[row_idx][0] = str(invalid_cid)
            else:
                rows.append([str(invalid_cid), "0.5", "0.5", "0.2", "0.2"])
            _write_yolo_rows(lbl_path, rows)
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.CRITICAL,
                target_image=_resolve_image_relpath(entry),
                target_label=lbl_rel,
                description=f"Injected out-of-range class_id={invalid_cid} into '{lbl_rel}'.",
                before_value=before_rows,
                after_value=rows,
            )

        if fault_type == FaultType.T4_BBOX_OUT_OF_BOUNDS:
            idx = rng.randrange(len(images))
            entry = images[idx]
            img_id = _resolve_image_id(entry, idx)
            lbl_rel = _resolve_label_relpath(entry)
            if lbl_rel is None:
                raise ValueError(f"Image '{img_id}' has no label_path")
            lbl_path = work_dir / lbl_rel
            rows = _parse_yolo_rows(lbl_path)
            before_rows = [list(r) for r in rows]
            oob_box = ["0", "1.25", "0.5", "0.4", "0.3"]
            rows.append(oob_box)
            _write_yolo_rows(lbl_path, rows)
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.CRITICAL,
                target_image=_resolve_image_relpath(entry),
                target_label=lbl_rel,
                description=f"Injected out-of-bounds YOLO bbox (cx=1.25, cy=0.5, w=0.4, h=0.3) into '{lbl_rel}'.",
                before_value=before_rows,
                after_value=rows,
            )

        if fault_type == FaultType.T5_BBOX_DRIFT:
            candidates = [
                (i, e) for i, e in enumerate(images)
                if (_resolve_label_relpath(e) and (work_dir / str(_resolve_label_relpath(e))).is_file()
                    and _parse_yolo_rows(work_dir / str(_resolve_label_relpath(e))))
            ]
            idx, entry = rng.choice(candidates) if candidates else (0, images[0])
            lbl_rel = _resolve_label_relpath(entry)
            if lbl_rel is None:
                raise ValueError("No label_path available for T5_BBOX_DRIFT")
            lbl_path = work_dir / lbl_rel
            rows = _parse_yolo_rows(lbl_path)
            before_rows = [list(r) for r in rows]
            if not rows:
                rows = [["0", "0.5", "0.5", "0.2", "0.2"]]
            drifted_rows: list[list[str]] = []
            for parts in rows:
                if len(parts) >= 5:
                    cid = parts[0]
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Shift by (+0.15, +0.12), flipping sign if needed to keep [0, 1] bounds valid
                    dx = 0.15 if (cx + 0.15 + w / 2.0 <= 1.0) else -0.15
                    dy = 0.12 if (cy + 0.12 + h / 2.0 <= 1.0) else -0.12
                    new_cx = max(w / 2.0 + 1e-4, min(1.0 - w / 2.0 - 1e-4, cx + dx))
                    new_cy = max(h / 2.0 + 1e-4, min(1.0 - h / 2.0 - 1e-4, cy + dy))
                    drifted_rows.append([cid, f"{new_cx:.6f}", f"{new_cy:.6f}", f"{w:.6f}", f"{h:.6f}"])
                else:
                    drifted_rows.append(parts)
            _write_yolo_rows(lbl_path, drifted_rows)
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.WARNING,
                target_image=_resolve_image_relpath(entry),
                target_label=lbl_rel,
                description=f"Shifted valid YOLO boxes by dx=±0.15, dy=±0.12 in '{lbl_rel}'.",
                before_value=before_rows,
                after_value=drifted_rows,
            )

        if fault_type == FaultType.T6_CLASS_SWAP:
            # Prefer swapping car (2) <-> truck (7); fallback to another class pair present
            car_id, truck_id = 2, 7
            target_idx: int | None = None
            swap_a, swap_b = car_id, truck_id

            indices = list(range(len(images)))
            rng.shuffle(indices)
            for i in indices:
                lbl_rel = _resolve_label_relpath(images[i])
                if not lbl_rel:
                    continue
                rows = _parse_yolo_rows(work_dir / lbl_rel)
                cids = {int(r[0]) for r in rows if r and r[0].lstrip("-").isdigit()}
                if car_id in cids or truck_id in cids:
                    target_idx = i
                    break

            if target_idx is None:
                for i in indices:
                    lbl_rel = _resolve_label_relpath(images[i])
                    if not lbl_rel:
                        continue
                    rows = _parse_yolo_rows(work_dir / lbl_rel)
                    cids = sorted({int(r[0]) for r in rows if r and r[0].lstrip("-").isdigit()})
                    if cids:
                        target_idx = i
                        swap_a = cids[0]
                        classes = manifest.get("classes", [])
                        num_classes = len(classes) if isinstance(classes, (list, dict)) and len(classes) > 1 else 80
                        swap_b = (swap_a + 1) % num_classes
                        break

            if target_idx is None:
                target_idx = 0

            entry = images[target_idx]
            lbl_rel = _resolve_label_relpath(entry)
            if lbl_rel is None:
                raise ValueError("No label_path available for T6_CLASS_SWAP")
            lbl_path = work_dir / lbl_rel
            rows = _parse_yolo_rows(lbl_path)
            before_rows = [list(r) for r in rows]
            swapped_rows: list[list[str]] = []
            for parts in rows:
                if parts and parts[0].lstrip("-").isdigit():
                    cid = int(parts[0])
                    if cid == swap_a:
                        parts = [str(swap_b), *parts[1:]]
                    elif cid == swap_b:
                        parts = [str(swap_a), *parts[1:]]
                swapped_rows.append(parts)
            _write_yolo_rows(lbl_path, swapped_rows)
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.WARNING,
                target_image=_resolve_image_relpath(entry),
                target_label=lbl_rel,
                description=f"Swapped class IDs {swap_a} <-> {swap_b} in '{lbl_rel}'.",
                before_value={"swap_pair": [swap_a, swap_b], "rows": before_rows},
                after_value={"swap_pair": [swap_a, swap_b], "rows": swapped_rows},
            )

        if fault_type == FaultType.T7_SCENE_LEAK:
            train_imgs = [e for e in images if str(e.get("split", "")).lower() == "train"]
            val_imgs = [e for e in images if str(e.get("split", "")).lower() in ("val", "valid", "validation")]
            if not train_imgs:
                train_imgs = [images[0]]
                train_imgs[0]["split"] = "train"
            if not val_imgs:
                candidate = images[1] if len(images) > 1 else images[0]
                candidate["split"] = "val"
                val_imgs = [candidate]

            t_entry = rng.choice(train_imgs)
            v_entry = rng.choice(val_imgs)
            leaked_scene = str(t_entry.get("scene_id") or f"scene_leak_{rng.randint(100, 999)}")
            before_state = {
                "train_image": _resolve_image_relpath(t_entry),
                "train_scene": t_entry.get("scene_id"),
                "val_image": _resolve_image_relpath(v_entry),
                "val_scene": v_entry.get("scene_id"),
            }
            t_entry["scene_id"] = leaked_scene
            v_entry["scene_id"] = leaked_scene
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.CRITICAL,
                target_image=_resolve_image_relpath(v_entry),
                description=f"Assigned shared scene_id '{leaked_scene}' to both train and val splits.",
                before_value=before_state,
                after_value={
                    "train_image": _resolve_image_relpath(t_entry),
                    "val_image": _resolve_image_relpath(v_entry),
                    "shared_scene_id": leaked_scene,
                },
            )

        if fault_type == FaultType.T8_GUIDELINE_MISMATCH:
            idx = rng.randrange(len(images))
            entry = images[idx]
            before_img_gv = entry.get("guideline_version")
            before_manifest_gv = manifest.get("guideline_version")
            legacy_version = "v1.0-legacy"
            entry["guideline_version"] = legacy_version
            return FaultRecord(
                fault_id=fault_id,
                fault_type=fault_type,
                severity=Severity.CRITICAL,
                target_image=_resolve_image_relpath(entry),
                description=f"Set image guideline_version to '{legacy_version}' on '{_resolve_image_relpath(entry)}'.",
                before_value={"image_guideline": before_img_gv, "manifest_guideline": before_manifest_gv},
                after_value={"image_guideline": legacy_version, "manifest_guideline": before_manifest_gv},
            )

        raise ValueError(f"Unsupported fault type: {fault_type}")


def inject_faults(
    source_dir: str | Path,
    dest_dir: str | Path,
    fault_types: Iterable[FaultType | str],
    seed: int = 42,
) -> FaultManifest:
    """Convenience function to inject faults into an isolated copy of source_dir."""
    return FaultInjector(seed=seed).inject(
        source_dir=source_dir,
        dest_dir=dest_dir,
        fault_types=fault_types,
        seed=seed,
    )
