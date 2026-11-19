"""Nguon frame + shape hien co (CHI DOC): cache offline cua copilot truoc, fallback client CVAT chi doc.

  src = FrameSource(285); img = src.image(71); shapes = src.shapes(71)   # [{label,type,points,(z_order,occluded)}]
Frame 71 job 285 co them `golden/frame71_target.json` (nguoi dung ve, MOC NGHIEM THU): src.target(71).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from cvat_tools.core import paths

FRAMES_DIR = paths.COPILOT / "frames"


class FrameSource:
    def __init__(self, job: int, frames_dir: Optional[Path] = None, client=None, jobs_dir: Optional[Path] = None):
        self.job = job
        self.dir = Path(frames_dir or FRAMES_DIR)
        self.jobs_dir = Path(jobs_dir or (paths.DATA / "jobs" / str(job)))
        self._client = client
        self._ann: Optional[list] = None

    def client(self):
        if self._client is None:
            from cvat_tools.core.client import Cvat
            self._client = Cvat.from_env("local", write_enabled=False)   # CHI DOC
        return self._client

    def image(self, frame: int) -> np.ndarray:
        p = self.dir / f"j{self.job}_f{frame}.jpg"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(np.frombuffer(self.client().frame(self.job, frame), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"khong doc duoc anh frame {frame} (job {self.job})")
        return img

    def _load_ann(self) -> list:
        if self._ann is None:
            p = self.dir / f"j{self.job}_ann.json"
            if p.exists():
                self._ann = json.loads(p.read_text(encoding="utf8"))
            else:
                cv = self.client()
                names = {l["id"]: l["name"] for l in cv.labels(self.job)}
                self._ann = [{"frame": int(s["frame"]), "label": names.get(s["label_id"], "?"), "type": s["type"],
                              "points": s["points"], "z_order": s.get("z_order", 0), "occluded": s.get("occluded", False)}
                             for s in cv.annotations(self.job).get("shapes", [])]
        return self._ann

    def shapes(self, frame: int) -> List[dict]:
        return [dict(s) for s in self._load_ann() if int(s["frame"]) == frame]

    def roads(self, frame: int) -> List[List[float]]:
        return [list(map(float, s["points"])) for s in self.shapes(frame) if s["label"] == "road" and s["type"] == "polygon"]

    def target(self, frame: int) -> Optional[List[dict]]:
        p = self.jobs_dir / "golden" / f"frame{frame}_target.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf8"))["shapes"]
