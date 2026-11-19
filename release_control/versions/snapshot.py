"""Content-addressed immutable dataset snapshotting, verification, and diffing."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from release_control.config import ControlTowerConfig
from release_control.db import ControlTowerDB
from release_control.models import SnapshotRecord

_SIDECAR_FILES = frozenset({"dataset.dvc", "snapshot_meta.json"})
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a file in streaming chunks (for DVC sidecar compatibility)."""
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class SnapshotManager:
    """Manages content-addressed immutable dataset snapshots, verification, and diffs."""

    def __init__(
        self,
        snapshots_root: Path | str | None = None,
        config: ControlTowerConfig | None = None,
    ) -> None:
        if snapshots_root is not None:
            self.snapshots_root = Path(snapshots_root).resolve()
        elif config is not None:
            self.snapshots_root = config.snapshots_dir.resolve()
        else:
            self.snapshots_root = ControlTowerConfig.auto_discover().snapshots_dir.resolve()
        self.snapshots_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_directory_digest(directory: Path | str) -> tuple[str, dict[str, dict[str, Any]]]:
        """Compute deterministic SHA-256 content-addressed digest over sorted dataset files.

        Excludes snapshot sidecars (`dataset.dvc`, `snapshot_meta.json`) at the root so
        source, snapshot, and restored directories yield identical digests.
        """
        root = Path(directory).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        file_entries: list[tuple[str, Path]] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_posix = path.relative_to(root).as_posix()
            if rel_posix in _SIDECAR_FILES:
                continue
            file_entries.append((rel_posix, path))

        file_entries.sort(key=lambda item: item[0])

        overall_hasher = hashlib.sha256()
        files_meta: dict[str, dict[str, Any]] = {}
        for rel_posix, path in file_entries:
            file_hash = _sha256_file(path)
            file_size = path.stat().st_size
            overall_hasher.update(f"{rel_posix}:{file_hash}\n".encode("utf-8"))
            files_meta[rel_posix] = {
                "sha256": file_hash,
                "size": file_size,
            }

        return overall_hasher.hexdigest(), files_meta

    def _resolve_snapshot_dir(self, snapshot_dir_or_digest: Path | str) -> Path:
        candidate = Path(snapshot_dir_or_digest)
        if candidate.is_dir():
            return candidate.resolve()
        by_digest = (self.snapshots_root / str(snapshot_dir_or_digest)).resolve()
        return by_digest

    @staticmethod
    def _count_assets(root: Path, files_meta: dict[str, dict[str, Any]]) -> tuple[int, int]:
        manifest_path = root / "manifest.json"
        image_count: int | None = None
        label_count: int | None = None

        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest_data.get("images"), list):
                    image_count = len(manifest_data["images"])
                if isinstance(manifest_data.get("annotations"), dict):
                    label_count = len(manifest_data["annotations"])
            except (json.JSONDecodeError, OSError):
                pass

        if image_count is None:
            image_count = sum(
                1
                for rel in files_meta
                if rel.startswith("images/") or Path(rel).suffix.lower() in _IMAGE_EXTENSIONS
            )
        if label_count is None:
            label_count = sum(
                1
                for rel in files_meta
                if rel.startswith("labels/") and Path(rel).suffix.lower() in {".txt", ".json"}
            )
        return image_count, label_count

    def create_snapshot(
        self,
        dataset_dir: Path | str,
        dataset_id: str,
        version_tag: str,
        db: ControlTowerDB | None = None,
        parent_digest: str | None = None,
    ) -> SnapshotRecord:
        """Create an immutable content-addressed snapshot with DVC sidecar and metadata."""
        source_dir = Path(dataset_dir).resolve()
        digest, files_meta = self.compute_directory_digest(source_dir)
        snapshot_dir = (self.snapshots_root / digest).resolve()

        total_size = sum(int(meta["size"]) for meta in files_meta.values())
        nfiles = len(files_meta)
        image_count, label_count = self._count_assets(source_dir, files_meta)
        created_at = datetime.now(timezone.utc)

        if not snapshot_dir.exists():
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for rel_posix in files_meta:
                src_file = source_dir / rel_posix
                dst_file = snapshot_dir / rel_posix
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)

            meta_payload = {
                "digest": digest,
                "dataset_id": dataset_id,
                "version_tag": version_tag,
                "parent_digest": parent_digest,
                "source_path": str(source_dir),
                "created_at": created_at.isoformat(),
                "nfiles": nfiles,
                "size": total_size,
                "image_count": image_count,
                "label_count": label_count,
                "files": files_meta,
            }
            meta_path = snapshot_dir / "snapshot_meta.json"
            meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

            dvc_md5 = hashlib.md5(digest.encode("utf-8")).hexdigest()
            dvc_payload = {
                "md5": dvc_md5,
                "sha256": digest,
                "nfiles": nfiles,
                "size": total_size,
                "dataset_id": dataset_id,
                "version_tag": version_tag,
                "parent_digest": parent_digest,
                "outs": [
                    {
                        "path": ".",
                        "md5": f"{dvc_md5}.dir",
                        "sha256": digest,
                        "nfiles": nfiles,
                        "size": total_size,
                    }
                ],
            }
            dvc_path = snapshot_dir / "dataset.dvc"
            dvc_path.write_text(yaml.safe_dump(dvc_payload, sort_keys=False), encoding="utf-8")

        manifest_sha256 = (
            files_meta["manifest.json"]["sha256"]
            if "manifest.json" in files_meta
            else digest
        )

        record = SnapshotRecord(
            snapshot_id=digest,
            dataset_name=dataset_id,
            version=version_tag,
            source_path=str(source_dir),
            snapshot_path=str(snapshot_dir),
            manifest_sha256=manifest_sha256,
            image_count=image_count,
            label_count=label_count,
            created_at=created_at,
            metadata={
                "digest": digest,
                "parent_digest": parent_digest,
                "nfiles": nfiles,
                "size": total_size,
                "dvc_path": str(snapshot_dir / "dataset.dvc"),
            },
        )

        if db is not None:
            db.upsert_snapshot(record)

        return record

    def verify_snapshot(self, snapshot_dir_or_digest: Path | str) -> dict[str, Any]:
        """Recompute every file's SHA-256 and the overall digest to prove immutability."""
        snapshot_dir = self._resolve_snapshot_dir(snapshot_dir_or_digest)
        if not snapshot_dir.is_dir():
            return {
                "verified": False,
                "digest": None,
                "expected_digest": str(snapshot_dir_or_digest),
                "file_count": 0,
                "mismatches": [
                    {
                        "type": "missing_snapshot_dir",
                        "path": str(snapshot_dir),
                        "detail": "Snapshot directory does not exist.",
                    }
                ],
                "snapshot_dir": str(snapshot_dir),
            }

        mismatches: list[dict[str, Any]] = []
        meta_path = snapshot_dir / "snapshot_meta.json"
        expected_digest: str | None = None
        expected_files: dict[str, dict[str, Any]] | None = None

        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                expected_digest = meta.get("digest")
                expected_files = meta.get("files")
            except (json.JSONDecodeError, OSError) as exc:
                mismatches.append(
                    {
                        "type": "corrupt_metadata",
                        "path": str(meta_path),
                        "detail": str(exc),
                    }
                )
        else:
            expected_digest = snapshot_dir.name

        actual_digest, actual_files = self.compute_directory_digest(snapshot_dir)

        if expected_files is not None:
            for rel_posix, exp_info in expected_files.items():
                if rel_posix not in actual_files:
                    mismatches.append(
                        {
                            "type": "missing_file",
                            "path": rel_posix,
                            "expected_sha256": exp_info.get("sha256"),
                            "actual_sha256": None,
                        }
                    )
                elif actual_files[rel_posix]["sha256"] != exp_info.get("sha256"):
                    mismatches.append(
                        {
                            "type": "hash_mismatch",
                            "path": rel_posix,
                            "expected_sha256": exp_info.get("sha256"),
                            "actual_sha256": actual_files[rel_posix]["sha256"],
                        }
                    )

            for rel_posix in actual_files:
                if rel_posix not in expected_files:
                    mismatches.append(
                        {
                            "type": "unexpected_file",
                            "path": rel_posix,
                            "expected_sha256": None,
                            "actual_sha256": actual_files[rel_posix]["sha256"],
                        }
                    )

        if expected_digest is not None and actual_digest != expected_digest:
            mismatches.append(
                {
                    "type": "digest_mismatch",
                    "path": ".",
                    "expected_digest": expected_digest,
                    "actual_digest": actual_digest,
                }
            )

        return {
            "verified": len(mismatches) == 0,
            "digest": actual_digest,
            "expected_digest": expected_digest,
            "file_count": len(actual_files),
            "mismatches": mismatches,
            "snapshot_dir": str(snapshot_dir),
        }

    @staticmethod
    def _extract_annotations_summary(
        snapshot_dir: Path, files_meta: dict[str, dict[str, Any]]
    ) -> tuple[int, int, Counter[str], dict[str, list[tuple[str, str]]]]:
        """Extract image count, total box count, class distribution, and per-target boxes."""
        image_count, _ = SnapshotManager._count_assets(snapshot_dir, files_meta)
        class_counts: Counter[str] = Counter()
        per_target_boxes: dict[str, list[tuple[str, str]]] = {}
        box_count = 0

        manifest_path = snapshot_dir / "manifest.json"
        parsed_from_manifest = False
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                annotations = manifest_data.get("annotations")
                if isinstance(annotations, dict) and annotations:
                    parsed_from_manifest = True
                    for image_key, boxes in annotations.items():
                        if not isinstance(boxes, list):
                            continue
                        entries: list[tuple[str, str]] = []
                        for box in boxes:
                            cls_key = str(
                                box.get("class_name")
                                if box.get("class_name") is not None
                                else box.get("class_id", "unknown")
                            )
                            class_counts[cls_key] += 1
                            box_count += 1
                            entries.append((cls_key, json.dumps(box, sort_keys=True)))
                        per_target_boxes[image_key] = entries
            except (json.JSONDecodeError, OSError):
                pass

        if not parsed_from_manifest:
            for rel_posix in sorted(files_meta):
                if not rel_posix.startswith("labels/"):
                    continue
                label_path = snapshot_dir / rel_posix
                if label_path.suffix.lower() == ".txt":
                    entries = []
                    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls_key = parts[0]
                        class_counts[cls_key] += 1
                        box_count += 1
                        entries.append((cls_key, line))
                    per_target_boxes[rel_posix] = entries

        return image_count, box_count, class_counts, per_target_boxes

    def diff_snapshots(self, digest_a: str, digest_b: str) -> dict[str, Any]:
        """Compare two snapshots file-by-file and annotation-by-annotation."""
        dir_a = self._resolve_snapshot_dir(digest_a)
        dir_b = self._resolve_snapshot_dir(digest_b)

        resolved_a, files_a = self.compute_directory_digest(dir_a)
        resolved_b, files_b = self.compute_directory_digest(dir_b)

        keys_a = set(files_a)
        keys_b = set(files_b)

        added_files = sorted(keys_b - keys_a)
        removed_files = sorted(keys_a - keys_b)
        modified_files = sorted(
            k for k in (keys_a & keys_b) if files_a[k]["sha256"] != files_b[k]["sha256"]
        )
        unchanged_count = len((keys_a & keys_b)) - len(modified_files)

        img_a, box_a, cls_a, targets_a = self._extract_annotations_summary(dir_a, files_a)
        img_b, box_b, cls_b, targets_b = self._extract_annotations_summary(dir_b, files_b)

        all_classes = sorted(set(cls_a) | set(cls_b))
        class_distribution_delta = {
            cls_name: cls_b.get(cls_name, 0) - cls_a.get(cls_name, 0)
            for cls_name in all_classes
            if cls_b.get(cls_name, 0) - cls_a.get(cls_name, 0) != 0
        }

        all_targets = sorted(set(targets_a) | set(targets_b))
        annotation_diffs: list[dict[str, Any]] = []
        for target in all_targets:
            boxes_a = targets_a.get(target, [])
            boxes_b = targets_b.get(target, [])
            if boxes_a != boxes_b:
                annotation_diffs.append(
                    {
                        "target": target,
                        "box_count_a": len(boxes_a),
                        "box_count_b": len(boxes_b),
                        "delta": len(boxes_b) - len(boxes_a),
                    }
                )

        return {
            "digest_a": resolved_a,
            "digest_b": resolved_b,
            "added_files": added_files,
            "removed_files": removed_files,
            "modified_files": modified_files,
            "unchanged_file_count": unchanged_count,
            "image_count_a": img_a,
            "image_count_b": img_b,
            "image_count_delta": img_b - img_a,
            "box_count_a": box_a,
            "box_count_b": box_b,
            "box_count_delta": box_b - box_a,
            "class_distribution_a": dict(cls_a),
            "class_distribution_b": dict(cls_b),
            "class_distribution_delta": class_distribution_delta,
            "annotation_diffs": annotation_diffs,
        }
