"""Tải checkpoint YOLO26 detect (n/s/l) vào models/checkpoints/ (gitignored). Chạy: python scripts/fetch_checkpoints.py"""
from pathlib import Path

from ultralytics import YOLO

DEST = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
DEST.mkdir(parents=True, exist_ok=True)
for name in ("yolo26n.pt", "yolo26s.pt", "yolo26l.pt"):
    if not (DEST / name).exists():
        import os
        os.chdir(DEST)
        YOLO(name)
    print(name, (DEST / name).stat().st_size)
