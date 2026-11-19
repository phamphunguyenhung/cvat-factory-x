"""Chay tracker cua ultralytics tren detection DA CACHE (khong chay lai detector), giong het nhau cho moi tracker.

Tracker chi duoc dua detection + frame (cho GMC cua BoT-SORT/TrackTrack). ReID mac dinh TAT; ReID chi la bang chung, khong gan lai ID.
Tra ve bieu dien track doc lap tracker: {track_id: {frame: [x1,y1,x2,y2]}} + score + thoi gian association tung frame.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

SUPPORTED = ("bytetrack", "botsort", "tracktrack", "ocsort", "deepocsort", "fasttrack")


class Dets:
    """Hinh dang Boxes numpy ma tracker doi hoi: conf, cls, xywh (+ xyxy), boolean-index, len."""

    def __init__(self, arr: np.ndarray):
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 5)
        self.xyxy, self.conf = a[:, :4], a[:, 4]
        self.cls = np.zeros(len(a), dtype=np.float32)
        self.xywh = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2, a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]], 1) if len(a) else np.zeros((0, 4), np.float32)
        self.data = a

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return Dets(self.data[idx])


def tracker_cfg(name: str, reid: bool = False, overrides: dict | None = None):
    import ultralytics
    import yaml
    from ultralytics.utils import IterableSimpleNamespace

    src = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
    if not src.exists():
        raise ValueError(f"ultralytics {ultralytics.__version__} khong co {name}.yaml")
    cfg = yaml.safe_load(src.read_text(encoding="utf8"))
    if reid:
        if "with_reid" not in cfg:
            raise ValueError(f"{name} khong ho tro ReID")
        cfg["with_reid"] = True
    cfg.update(overrides or {})
    ns = IterableSimpleNamespace(**cfg)
    ns.device = "cpu"
    return ns


def make_tracker(name: str, reid: bool = False, overrides: dict | None = None):
    from ultralytics.trackers.track import TRACKER_MAP
    cfg = tracker_cfg(name, reid, overrides)
    return TRACKER_MAP[cfg.tracker_type](args=cfg)


def run_on_cache(dets: dict, source, tracker: str, reid: bool = False, overrides: dict | None = None) -> dict:
    """dets: {frame: [[x1,y1,x2,y2,s]]}. source.image(fid) cho GMC. Tuan tu, tracker moi (reset xac dinh).
    -> {"tracks": {tid: {frame: box}}, "scores": {tid: {frame: s}}, "assoc_s": [per-frame], "tracker": name}"""
    tr = make_tracker(tracker, reid, overrides)
    tracks: dict = {}
    scores: dict = {}
    per = []
    for f in sorted(dets):
        img = source.image(f) if source is not None else np.zeros((10, 10, 3), np.uint8)
        d = Dets(dets[f])
        t0 = time.perf_counter()
        out = tr.update(d, img)
        per.append(time.perf_counter() - t0)
        for row in np.asarray(out).reshape(-1, out.shape[-1] if len(out) else 8):
            tid = int(row[4])
            tracks.setdefault(tid, {})[f] = [float(v) for v in row[:4]]
            scores.setdefault(tid, {})[f] = float(row[5])
    return {"tracker": tracker + ("+reid" if reid else ""), "tracks": tracks, "scores": scores, "assoc_s": per}
