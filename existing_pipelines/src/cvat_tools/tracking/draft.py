"""Sinh NHÁP track: detector + tracker mã nguồn mở của ultralytics (ByteTrack, BoT-SORT, TrackTrack, OC-SORT, Deep OC-SORT).

Chỉ là gợi ý cho người gán duyệt. RULES.md của lab Ngày 5 cấm nộp box/ID do AI sinh như nhãn của mình:
chỉ chạy khi người dùng đã có cho phép của giảng viên (cờ --allow-ai-draft) và phải ghi vào báo cáo.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from cvat_tools.tracking import geometry as core
from cvat_tools.core import paths

TRACKERS = ("botsort", "bytetrack", "tracktrack", "ocsort", "deepocsort")
DEFAULT_MODEL = "yolo26l.pt"
NOTICE = ("Nháp AI. RULES.md của lab Ngày 5 cấm nộp box/track ID do AI sinh như nhãn của mình. "
          "Chỉ dùng khi giảng viên đã cho phép, và ghi model + số track/keyframe sinh/sửa vào báo cáo.")


def decode(b: bytes) -> np.ndarray:
    import cv2
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def tracker_yaml(name: str, reid: bool) -> str:
    """Đường dẫn cấu hình tracker. Với --reid: bản sao của yaml gốc, bật with_reid (chỉ botsort/deepocsort/tracktrack hỗ trợ)."""
    import ultralytics
    import yaml

    src = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
    if not src.exists():
        raise SystemExit(f"Phiên bản ultralytics này không có {name}.yaml (có: {sorted(p.stem for p in src.parent.glob('*.yaml'))})")
    if not reid:
        return str(src)
    cfg = yaml.safe_load(src.read_text(encoding="utf8"))
    if "with_reid" not in cfg:
        raise SystemExit(f"{name} không hỗ trợ ReID (dùng botsort, deepocsort hoặc tracktrack).")
    cfg["with_reid"] = True
    out = paths.DATA / "runs" / f"{name}_reid.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg), encoding="utf8")
    return str(out)


def load_model(name: str | None = None):
    from ultralytics import YOLO
    name = name or DEFAULT_MODEL
    paths.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    return YOLO(name if Path(name).exists() else str(paths.CHECKPOINTS / name)), name


def run_tracker(model, images: dict, tracker: str, reid: bool, conf: float, imgsz: int) -> dict:
    """images: {frame_cvat: BGR}. Phải đi tuần tự để tracker giữ ID. Trả {track_id: {frame: xyxy}} (chỉ class person)."""
    cfg = tracker_yaml(tracker, reid)
    raw: dict = {}
    for f in sorted(images):
        r = model.track(images[f], persist=True, tracker=cfg, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        for tid, xyxy in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
            raw.setdefault(tid, {})[f] = xyxy
    model.predictor = None  # reset tracker giữa các lần chạy (bench)
    return raw


def build_draft(raw: dict, label_id: int, stop_frame: int, n: int, anchors: dict | None, ref_frame: int,
                tol: float, max_gap: int, min_len: int) -> tuple[dict, list]:
    """Chọn track (theo điểm neo P1..Pn nếu có, không thì n track dài nhất) -> payload track + cảnh báo."""
    filled = {t: core.fill_gaps(tr, max_gap) for t, tr in raw.items()}
    warns: list = []
    if anchors:
        pick, warns = core.select_by_anchors(filled, anchors, ref_frame)
        chosen = {name: tid for name, tid in pick.items()}
    else:
        top = core.select_longest(filled, n, min_len)
        chosen = {f"T{i + 1}": tid for i, tid in enumerate(top)}
    tracks, info = [], {}
    for name, tid in chosen.items():
        tr = filled[tid]
        kf = core.simplify_keyframes(tr, tol)
        tracks.append({"name": name, "tracker_id": tid, "payload": core.to_cvat_track(label_id, tr, kf, stop_frame)})
        info[name] = {"tracker_id": tid, "first": min(tr), "last": max(tr), "boxes": len(tr), "keyframes": len(kf)}
    return {"tracks": tracks, "info": info}, warns


def save_draft(job: int, draft: dict) -> Path:
    d = paths.DATA / "drafts"
    d.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    p = d / f"job{job}_{ts}.json"
    p.write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf8")
    (d / f"LATEST_job{job}.json").write_text(p.read_text(encoding="utf8"), encoding="utf8")
    return p


def load_draft(job: int, path: str | None = None) -> dict:
    p = Path(path) if path else paths.DATA / "drafts" / f"LATEST_job{job}.json"
    return json.loads(p.read_text(encoding="utf8"))
