"""Detection cache: moi tracker thay CUNG detection. Khoa = nguon + anh xa frame + hash checkpoint + cau hinh suy luan."""
from __future__ import annotations

import time

import numpy as np

from cvat_tools.core.cache import Cache, make_key
from cvat_tools.detection.detector import Detector


def cache_key(source, det: Detector) -> str:
    return make_key(source=source.identity, frames=[source.frame_ids()[0], source.frame_ids()[-1]], det=det.identity)


def detect_cached(source, det: Detector, force: bool = False) -> dict:
    """-> {"dets": {fid: [[x1,y1,x2,y2,s],..]}, "timing": {...}, "key": str, "hit": bool}.
    Timing la cua LAN CHAY DAU (luu cung cache) de benchmark detector tach khoi tracker."""
    c = Cache("detections")
    key = cache_key(source, det)
    got = None if force else c.get(key)
    if got:
        v = got["value"]
        return {"dets": {int(k): x for k, x in v["dets"].items()}, "timing": v["timing"], "key": key, "hit": True}
    import tracemalloc  # noqa: F401  (RAM do bang psutil o benchmark)
    import psutil
    psutil.cpu_percent(interval=None)
    dets, per = {}, []
    ids = source.frame_ids()
    det.load()
    det.predict(source.image(ids[0]))  # warm-up, khong tinh
    for f in ids:
        img = source.image(f)
        t0 = time.perf_counter()
        dets[f] = det.predict(img)
        per.append(time.perf_counter() - t0)
    p = np.array(per)
    timing = {"n": len(ids), "median_s": float(np.median(p)), "p95_s": float(np.percentile(p, 95)), "total_s": float(p.sum()),
              "fps": float(len(ids) / p.sum()),
              "cpu_load_avg": psutil.cpu_percent(interval=None)}
    c.put(key, {"dets": dets, "timing": timing}, meta={"source": source.identity, "det": det.identity})
    return {"dets": dets, "timing": timing, "key": key, "hit": False}
