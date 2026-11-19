"""Benchmark tracker tren CUNG detection cache. Thoi gian association tach khoi detector.

Chi so la HEURISTIC tu nhat quan (khong GT -> KHONG phai IDF1/MOTA/HOTA). Co GT that (`gt`) thi them IDF1/MOTA qua motmetrics neu cai.
Cot P1-P5: voi moi neo (nguoi gan), track chua neo -> do lien tuc = so frame co box / (last-first+1); so neo dung chung 1 track = nghi ngo.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from cvat_tools.core import paths
from cvat_tools.tracking import engine
from cvat_tools.tracking import geometry as G

CSV_PATH = paths.DATA / "benchmarks" / "tracker_bench.csv"
FIELDS = ["ts", "det_cache", "tracker", "n_frames", "assoc_median_ms", "assoc_p95_ms", "assoc_total_s", "assoc_fps", "peak_ram_mb",
          "raw_tracks", "long_tracks_ge30", "top5_coverage", "fragmentation", "id_switch_candidates", "anchors_found",
          "anchors_shared_track", "anchor_continuity_mean", "ai_provenance", "noisy", "status", "note"]


def metrics(tracks: dict, n_people: int, anchors: dict | None, ref_frame: int, n_frames: int) -> dict:
    top = G.select_longest(tracks, n_people, 10)
    q = G.qa_tracks(top)
    cnt = lambda k: sum(1 for x in q if x["kind"] == k)
    cover = sum(len(t) for t in top.values()) / max(1, n_people * n_frames)
    m = {"raw_tracks": len(tracks), "long_tracks_ge30": sum(len(t) >= 30 for t in tracks.values()),
         "top5_coverage": round(cover, 3), "fragmentation": cnt("fragmentation"), "id_switch_candidates": cnt("id_switch")}
    if anchors:
        pick, _ = G.select_by_anchors(tracks, anchors, ref_frame)
        conts = [len(tracks[t]) / (max(tracks[t]) - min(tracks[t]) + 1) for t in pick.values()]
        vals = list(pick.values())
        m.update(anchors_found=len(pick), anchors_shared_track=len(vals) - len(set(vals)),
                 anchor_continuity_mean=round(float(np.mean(conts)), 3) if conts else 0.0)
    return m


def run_all(dets: dict, source, trackers: list[str], det_cache: str, n_people: int = 5, anchors: dict | None = None,
            ref_frame: int = 0, reid: bool = False, out_csv: Path = CSV_PATH, save_dir: Path | None = None) -> list[dict]:
    import json

    import psutil
    rows, results = [], {}
    n = len(dets)
    for name in trackers:
        for rid in ([False, True] if reid and name in ("botsort", "tracktrack", "deepocsort") else [False]):
            load = psutil.cpu_percent(interval=0.5)
            row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "det_cache": det_cache, "tracker": name + ("+reid" if rid else ""),
                   "n_frames": n, "noisy": load > 35, "ai_provenance": "ai_draft (khong dung lam nhan nop)"}
            try:
                r = engine.run_on_cache(dets, source, name, rid)
                a = np.array(r["assoc_s"])
                row.update(assoc_median_ms=round(float(np.median(a)) * 1000, 2), assoc_p95_ms=round(float(np.percentile(a, 95)) * 1000, 2),
                           assoc_total_s=round(float(a.sum()), 3), assoc_fps=round(n / max(a.sum(), 1e-9), 1),
                           peak_ram_mb=round(psutil.Process().memory_info().peak_wset / 2**20),
                           status="measured", note="", **metrics(r["tracks"], n_people, anchors, ref_frame, n))
                results[row["tracker"]] = r
            except Exception as e:  # tracker khong chay duoc -> BLOCKED, khong bia
                row.update(status="BLOCKED", note=repr(e)[:200])
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for k, r in results.items():
            (save_dir / f"{k}.json").write_text(json.dumps({"tracker": k, "tracks": r["tracks"], "scores": r["scores"]}), encoding="utf8")
    return rows
