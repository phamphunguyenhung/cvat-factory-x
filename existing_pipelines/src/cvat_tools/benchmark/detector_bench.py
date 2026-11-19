"""Benchmark detector (1 cau hinh / 1 tien trinh con de peak RAM sach). CSV: data/benchmarks/detector_bench.csv.

KHONG co GT -> khong tinh mAP/recall that. Cot `agree_*` la do khop voi DETECTION THAM CHIEU (pseudo, khong phai recall).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from cvat_tools.core import paths

CSV_PATH = paths.DATA / "benchmarks" / "detector_bench.csv"
FIELDS = ["ts", "model", "backend", "imgsz", "n_frames", "warm_load_s", "median_ms", "p95_ms", "total_s", "fps",
          "peak_ram_mb", "n_det_total", "cpu_load_before", "noisy", "status", "note"]


def cpu_load() -> float:
    import psutil
    return psutil.cpu_percent(interval=1.0)


def wait_idle(max_wait: float = 0, thr: float = 35.0) -> float:
    """Cho CPU roi (<thr %) toi da max_wait giay; tra ve tai CPU cuoi cung. Khong ep: het gio van chay va danh dau noisy."""
    t0 = time.time()
    load = cpu_load()
    while load > thr and time.time() - t0 < max_wait:
        time.sleep(5)
        load = cpu_load()
    return load


def run_one(model: str, backend: str, imgsz: int, frames_dir: str, start: int, limit: int, conf: float, threads: int | None) -> dict:
    import psutil

    from cvat_tools.detection.cached import detect_cached
    from cvat_tools.detection.detector import Detector
    from cvat_tools.detection.frames import DirSource

    src = DirSource(frames_dir, start, limit)
    det = Detector(model, backend, imgsz, conf, threads)
    t0 = time.perf_counter()
    det.load()
    det.predict(src.image(src.frame_ids()[0]))
    load_s = time.perf_counter() - t0
    r = detect_cached(src, det, force=True)
    tm = r["timing"]
    return {"warm_load_s": round(load_s, 2), "median_ms": round(tm["median_s"] * 1000, 1), "p95_ms": round(tm["p95_s"] * 1000, 1),
            "total_s": round(tm["total_s"], 2), "fps": round(tm["fps"], 3),
            "peak_ram_mb": round(psutil.Process().memory_info().peak_wset / 2**20), "n_det_total": sum(len(v) for v in r["dets"].values()),
            "cache_key": r["key"]}


def bench(model: str, backend: str, imgsz: int, frames_dir: str, start: int = 0, limit: int = 30, conf: float = 0.25,
          threads: int | None = None, timeout: int = 3600, idle_wait: float = 0) -> dict:
    load = wait_idle(idle_wait)
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "backend": backend, "imgsz": imgsz, "n_frames": limit,
           "cpu_load_before": load, "noisy": load > 35}
    cmd = [sys.executable, "-m", "cvat_tools.benchmark.detector_bench", "--child", model, backend, str(imgsz), frames_dir,
           str(start), str(limit), str(conf), str(threads or 0)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf8")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode == 0 and line:
            row.update(json.loads(line[-1][7:]), status="measured", note="")
        else:
            row.update(status="FAILED", note=(p.stderr.strip().splitlines() or ["?"])[-1][:200])
    except subprocess.TimeoutExpired:
        row.update(status="TIMEOUT", note=f">{timeout}s")
    row.pop("cache_key", None)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    return row


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    _, _, m, b, sz, fd, st, lm, cf, th = sys.argv
    print("RESULT " + json.dumps(run_one(m, b, int(sz), fd, int(st), int(lm), float(cf), int(th) or None)))
