"""Nguon frame: thu muc anh local (vd MOT17-09 .lab-local) hoac job CVAT. Frame id = so frame CVAT (0-based)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}


class DirSource:
    """Anh trong thu muc, sap theo ten; frame i (0-based) = file thu i. Ten file KHONG dua vao khoa, chi (ten,kich thuoc)."""

    def __init__(self, folder: str | Path, start: int = 0, limit: int | None = None):
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
        self.files = files[start:start + limit if limit else None]
        self.start = start
        self.folder = str(Path(folder).resolve())

    def __len__(self):
        return len(self.files)

    def frame_ids(self) -> list[int]:
        return [self.start + i for i in range(len(self.files))]

    def image(self, fid: int) -> np.ndarray:
        import cv2
        return cv2.imread(str(self.files[fid - self.start]), cv2.IMREAD_COLOR)

    @property
    def identity(self) -> dict:
        h = hashlib.sha256()
        for p in self.files:
            h.update(f"{p.name}:{p.stat().st_size}".encode())
        return {"kind": "dir", "folder": self.folder, "start": self.start, "n": len(self.files), "digest": h.hexdigest()[:16]}
