"""Detector nguoi, lazy-load, nhieu backend (pytorch / onnx / openvino) qua ultralytics.

Moi lan chi giu MOT model trong bo nho; export backend duoc cache o models/<backend>/ theo hash checkpoint.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cvat_tools.core import paths
from cvat_tools.core.cache import file_hash
from cvat_tools.core.config import load_models

BACKENDS = ("pytorch", "onnx", "openvino", "openvino_fp16", "openvino_int8")


def checkpoint_path(model_id: str) -> Path:
    spec = load_models()[model_id]
    return paths.CHECKPOINTS / spec["checkpoint"]


def export_path(model_id: str, backend: str, imgsz: int) -> Path:
    h = file_hash(checkpoint_path(model_id))
    base = paths.MODELS / backend.split("_")[0] / f"{model_id}_{h}_{imgsz}_{backend}"
    if backend == "onnx":
        return base.with_suffix(".onnx")
    return base.with_name(base.name + "_openvino_model")  # ultralytics nhan dang theo hau to nay


def ensure_backend(model_id: str, backend: str, imgsz: int, calib_data: str | None = None) -> Path:
    """Tra duong dan weights cho backend; export neu chua co. INT8 can calib_data (yaml dataset) -> neu thieu raise."""
    from ultralytics import YOLO

    ck = checkpoint_path(model_id)
    if backend == "pytorch":
        return ck
    out = export_path(model_id, backend, imgsz)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(ck))
    kw = dict(imgsz=imgsz, device="cpu", verbose=False)
    if backend == "onnx":
        src = m.export(format="onnx", simplify=False, **kw)
    elif backend in ("openvino", "openvino_fp16"):
        src = m.export(format="openvino", half=backend.endswith("fp16"), **kw)
    elif backend == "openvino_int8":
        if not calib_data:
            raise RuntimeError("BLOCKED: INT8 can du lieu calibration dai dien (dataset yaml); chua co")
        src = m.export(format="openvino", int8=True, data=calib_data, **kw)
    else:
        raise ValueError(backend)
    import shutil
    shutil.move(str(src), str(out))
    return out


class Detector:
    def __init__(self, model_id: str, backend: str = "pytorch", imgsz: int = 640, conf: float = 0.25,
                 threads: int | None = None, iou: float = 0.7):
        self.model_id, self.backend, self.imgsz, self.conf, self.iou = model_id, backend, imgsz, conf, iou
        self.threads = threads
        self._m = None
        self.weights = None

    def load(self):
        if self._m is None:
            from ultralytics import YOLO
            if self.threads:
                import torch
                torch.set_num_threads(self.threads)
            self.weights = ensure_backend(self.model_id, self.backend, self.imgsz)
            self._m = YOLO(str(self.weights), task="detect")
        return self

    def unload(self):
        self._m = None

    @property
    def identity(self) -> dict:
        """Dung cho khoa cache: hash checkpoint goc + backend + cau hinh suy luan."""
        return {"model": self.model_id, "ckpt": file_hash(checkpoint_path(self.model_id)), "backend": self.backend,
                "imgsz": self.imgsz, "conf": self.conf, "iou": self.iou}

    def predict(self, img: np.ndarray) -> list[list[float]]:
        """-> [[x1,y1,x2,y2,score], ...] chi class person (0)."""
        self.load()
        r = self._m.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=[0], device="cpu", verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return [[*map(float, b), float(c)] for b, c in zip(xyxy, cf)]

    def smoke(self, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        self.load()
        t1 = time.perf_counter()
        n = len(self.predict(img))
        return {"load_s": round(t1 - t0, 2), "first_infer_s": round(time.perf_counter() - t1, 2), "n_det": n}
