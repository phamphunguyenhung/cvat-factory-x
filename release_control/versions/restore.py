"""Time Machine dataset restoration with cryptographic integrity verification."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from release_control.config import ControlTowerConfig
from release_control.db import ControlTowerDB
from release_control.versions.snapshot import SnapshotManager, _SIDECAR_FILES


class SnapshotIntegrityError(ValueError):
    """Raised when a snapshot or restored target fails cryptographic integrity checks."""


class RestoreManager:
    """Restores immutable dataset snapshots into clean working directories."""

    def __init__(
        self,
        snapshots_root: Path | str | None = None,
        config: ControlTowerConfig | None = None,
    ) -> None:
        self._snapshot_manager = SnapshotManager(
            snapshots_root=snapshots_root,
            config=config,
        )

    def restore_snapshot(
        self,
        snapshot_digest: str,
        target_dir: Path | str,
        db: ControlTowerDB | None = None,
    ) -> dict[str, Any]:
        """Verify an immutable snapshot and materialize a clean verified copy into target_dir."""
        snapshot_ref: Path | str = snapshot_digest
        if db is not None:
            record = db.get_snapshot(snapshot_digest)
            if record is not None and Path(record.snapshot_path).is_dir():
                snapshot_ref = Path(record.snapshot_path)

        verification = self._snapshot_manager.verify_snapshot(snapshot_ref)
        if not verification["verified"]:
            raise SnapshotIntegrityError(
                f"Snapshot '{snapshot_digest}' failed integrity verification: "
                f"{verification['mismatches']}"
            )

        snapshot_dir = Path(verification["snapshot_dir"]).resolve()
        expected_digest = str(verification["digest"])
        target_path = Path(target_dir).resolve()

        if target_path == snapshot_dir:
            raise ValueError("target_dir cannot be the immutable snapshot directory itself.")

        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

        copied_files = 0
        for src_file in snapshot_dir.rglob("*"):
            if not src_file.is_file():
                continue
            rel_path = src_file.relative_to(snapshot_dir)
            if rel_path.as_posix() in _SIDECAR_FILES:
                continue
            dst_file = target_path / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied_files += 1

        restored_digest, _ = SnapshotManager.compute_directory_digest(target_path)
        if restored_digest != expected_digest:
            raise SnapshotIntegrityError(
                f"Restored directory digest mismatch: expected {expected_digest}, "
                f"got {restored_digest}"
            )

        return {
            "snapshot_digest": expected_digest,
            "restored_digest": restored_digest,
            "integrity_verified": True,
            "restore_integrity": True,
            "file_count": copied_files,
            "target_dir": str(target_path),
        }
