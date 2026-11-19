"""Prelabel: YOLO draft boxes written into the demo job (source=auto).

Honesty rules: draft boxes are model output, not ground truth. If no local
checkpoint / ultralytics is available the fallback detector is marked
`simulated` (model_version="simulated-fallback") and never passed off as YOLO.
Idempotent: put_annotations replaces the job's annotation set with the full
deterministic shape list, so re-running never duplicates boxes.
"""
from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
LEARNING = PROJECT.parents[1]
IMAGES_DIR = PROJECT / "demo" / "coco128_clean" / "images"
CLASSES_TXT = PROJECT / "demo" / "coco128_clean" / "classes.txt"
CHECKPOINT_CANDIDATES = [
    LEARNING / "cvat" / "models" / "checkpoints" / "yolo26n.pt",
    LEARNING / "models" / "yolo11n.pt",
]
ACTOR = "agent.prelabel"


def resolve_image(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    return IMAGES_DIR / p.name


def _coco_names() -> dict[int, str]:
    out = {}
    if CLASSES_TXT.exists():
        for line in CLASSES_TXT.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                out[int(parts[0])] = parts[1]
    return out


def _image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (640, 480)


class YoloDetector:
    simulated = False

    def __init__(self, checkpoint: Path, conf: float = 0.25):
        from ultralytics import YOLO  # local checkpoint only, no download
        self.model = YOLO(str(checkpoint))
        self.conf = conf
        self.model_version = f"{checkpoint.stem} (ultralytics, local checkpoint)"

    def detect(self, image: Path) -> list[dict]:
        res = self.model.predict(str(image), conf=self.conf, verbose=False, device="cpu")[0]
        names = res.names
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append({"label": names[int(b.cls[0])], "box": [x1, y1, x2, y2],
                        "confidence": round(float(b.conf[0]), 4)})
        return out


class SimulatedDetector:
    """Explicit fallback: one fixed centre box, low confidence. NOT a real model."""
    simulated = True
    model_version = "simulated-fallback"

    def detect(self, image: Path) -> list[dict]:
        w, h = _image_size(image)
        return [{"label": "person", "box": [w * .3, h * .3, w * .7, h * .7], "confidence": 0.1}]


def load_detector(checkpoint: Path | None = None):
    cands = [Path(checkpoint)] if checkpoint else CHECKPOINT_CANDIDATES
    for c in cands:
        if c.exists():
            try:
                return YoloDetector(c)
            except Exception:
                continue
    return SimulatedDetector()


def _audit(session: dict, action: str, target: str, **kw):
    # TODO: remove guard once release_control.audit is guaranteed present.
    try:
        from release_control.audit import AuditLog
    except Exception:
        return None
    try:
        from release_control import config
        path = Path(getattr(config, "AUDIT_PATH", PROJECT / "artifacts" / "release_control_demo" / "audit.jsonl"))
    except Exception:
        path = PROJECT / "artifacts" / "release_control_demo" / "audit.jsonl"
    try:
        return AuditLog(path).append(ACTOR, action, target, **kw)
    except Exception:
        return None


def build_shapes(session: dict, detector) -> list[dict]:
    label_map = session.get("label_map", {})
    version = getattr(detector, "model_version", "unknown")
    shapes = []
    for frame, name in enumerate(session.get("image_files", [])):
        for d in detector.detect(resolve_image(name)):
            lid = label_map.get(d["label"])
            if lid is None:
                continue  # label not in the job's taxonomy: skip, don't invent
            shapes.append({
                "type": "rectangle", "frame": frame, "label_id": lid,
                "label": d["label"], "points": [round(v, 2) for v in d["box"]],
                "source": "auto", "occluded": False, "z_order": 0,
                "attributes": [{"name": "model_version", "value": version},
                               {"name": "confidence", "value": f"{d['confidence']:.4f}"}],
                "confidence": d["confidence"], "model_version": version,
                "simulated": bool(getattr(detector, "simulated", False)),
            })
    return shapes


def run_prelabel(session: dict, client=None, detector=None) -> dict:
    if client is None:
        from release_control.demo_session.session import CvatDemoClient
        client = CvatDemoClient()
    detector = detector or load_detector()
    shapes = build_shapes(session, detector)
    target = f"task:{session.get('task_id')}/job:{session.get('job_id')}"
    try:
        client.put_annotations(shapes)
    except Exception as e:  # incl. WriteDenied: audit and re-raise
        _audit(session, "prelabel.DENIED", target, evidence={"error": repr(e)})
        raise
    simulated = bool(getattr(detector, "simulated", False))
    result = {"job_id": session.get("job_id"), "task_id": session.get("task_id"),
              "frames": len(session.get("image_files", [])), "shapes_written": len(shapes),
              "model_version": getattr(detector, "model_version", "unknown"),
              "simulated": simulated, "source": "auto",
              "note": "draft boxes are model output, not verified ground truth"
                      + ("; SIMULATED fallback, no real model ran" if simulated else "")}
    _audit(session, "prelabel", target, inputs={"images": result["frames"]},
           outputs={k: result[k] for k in ("shapes_written", "model_version", "simulated")})
    return result
