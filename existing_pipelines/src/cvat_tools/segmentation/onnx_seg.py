"""SegFormer ONNX (Cityscapes) chay QUA INPUT CONTRACT: prepare_input -> onnxruntime -> restore -> class map goc.

Khac evidence._B2 (dung SceneParser cua copilot, tien xu ly rieng): o day moi buoc resize/chuan hoa/anh xa nguoc
la inputspec, nen policy (long_side, enhance cho frame toi) do config/model_inputs.yaml quyet dinh. Model thieu spec -> RAISE.

  seg = OnnxSeg("b2"); cm, info = seg.class_map(frame_bgr)   # cm: uint8 HxW id Cityscapes
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from cvat_tools.core import paths
from cvat_tools.segmentation.inputspec import get_spec, prepare_input, restore

ONNX_FILES = {"b2": paths.COPILOT / "models" / "segformer_b2_cityscapes.onnx"}
CITYSCAPES = {"road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4, "pole": 5, "traffic light": 6,
              "traffic sign": 7, "vegetation": 8, "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
              "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18}


class OnnxSeg:
    def __init__(self, model_id: str = "b2", run: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 onnx_path: Optional[Path] = None, threads: Optional[int] = None):
        sp = get_spec(model_id)  # khong co spec -> UnknownModelSpec, khong nap model
        if sp.get("backend") != "onnx" and run is None:
            raise ValueError(f"model {model_id!r} khong phai backend onnx (spec: {sp.get('backend')})")
        self.model_id = model_id
        if run is None:
            import onnxruntime as ort

            p = Path(onnx_path or ONNX_FILES[model_id])
            if not p.exists():
                raise FileNotFoundError(p)
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            run = lambda x: sess.run(None, {name: x})[0]  # noqa: E731
        self._run = run

    def logits(self, frame_bgr: np.ndarray, policy: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        x, meta = prepare_input(frame_bgr, self.model_id, policy=policy)
        out = np.concatenate([self._run(x[i:i + 1]) for i in range(len(x))]) if meta["tiles"] else self._run(x)
        return restore(out if meta["tiles"] else out[0], meta, nearest=False), meta

    def class_map(self, frame_bgr: np.ndarray, policy: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        t0 = time.time()
        lg, meta = self.logits(frame_bgr, policy)
        cm = lg.argmax(0).astype(np.uint8)
        return cm, {"s": round(time.time() - t0, 3), "in_hw": meta["in_hw"], "enhance": meta["enhance"]}
