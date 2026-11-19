"""Ablation co dinh tren mot doan tuan tu MOT17-09 (A-G). Moi cau hinh chay trong tien trinh con (peak RAM sach).

Khong co GT -> IDF1/HOTA/MOTA = "n/a (no GT)". `pseudo_recall_vs_A` = ty le detection conf>=0.5 cua cau hinh A (YOLO26l PyTorch,
tham chieu GIA, khong phai nhan that) ma cau hinh nay cung thay (IoU>=0.5): chi do khop voi A, KHONG phai recall.
Chay: python -m cvat_tools.benchmark.ablation --project day05-mot17 --limit 60 [--only A B C] [--idle-wait 600]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time

import numpy as np

from cvat_tools.core import paths

CSV_PATH = paths.DATA / "benchmarks" / "ablation.csv"
CONFIGS = {
    "A": dict(model="yolo26l", backend="pytorch", imgsz=1920, tracker="botsort", note="baseline hien tai (mtm mac dinh)"),
    "B": dict(model="yolo26n", backend="openvino", imgsz=640, tracker="bytetrack"),
    "C": dict(model="yolo26s", backend="openvino", imgsz=640, tracker="tracktrack"),
    "D": dict(model="yolo26s", backend="openvino", imgsz=640, tracker="ocsort"),
    "E": dict(model="yolo26s", backend="openvino", imgsz=640, tracker="fasttrack"),
    "F": dict(model="yolo26s", backend="openvino", imgsz=640, tracker="tracktrack", reid="osnet_x0_25"),
    "G": dict(model="yolo26s", backend="openvino", imgsz=640, tracker="tracktrack", adaptive=dict(model="yolo26l", backend="openvino", imgsz=1280)),
}
FIELDS = ["ts", "id", "model", "tracker", "resolution", "backend", "frames", "det_fps", "e2e_fps", "peak_ram_mb", "pseudo_recall_vs_A",
          "pseudo_recall_small_vs_A", "IDF1", "anchor_continuity", "anchors_shared_track", "frag", "idsw_cand", "keyframes_opt",
          "keyframes_dp", "adaptive_rerun_frac", "cpu_load", "noisy", "status", "note"]


def _agree(ref: dict, dets: dict, small: bool = False) -> float | None:
    from cvat_tools.tracking import geometry as G
    tot = hit = 0
    for f, rd in ref.items():
        for d in rd:
            if d[4] < 0.5 or (small and (d[3] - d[1]) >= 100):
                continue
            tot += 1
            hit += any(G.box_iou(d[:4], x[:4]) >= 0.5 for x in dets.get(f, []))
    return round(hit / tot, 3) if tot else None


def run_one(cid: str, project: str, limit: int) -> dict:
    import psutil

    from cvat_tools.core.config import load_project
    from cvat_tools.detection.adaptive import hard_frames, merge_dets
    from cvat_tools.detection.cached import detect_cached
    from cvat_tools.detection.detector import Detector
    from cvat_tools.detection.frames import DirSource
    from cvat_tools.tracking import engine
    from cvat_tools.tracking import geometry as G
    from cvat_tools.tracking import keyframes as K
    from cvat_tools.benchmark.tracker_bench import metrics

    cfg = CONFIGS[cid]
    proj = load_project(project)
    src = DirSource(paths.REPO / proj["dataset"]["frames_dir"], 0, limit)
    out = {"id": cid, "model": cfg["model"], "tracker": cfg["tracker"], "resolution": cfg["imgsz"], "backend": cfg["backend"], "frames": len(src)}
    if cfg.get("reid"):
        out.update(status="BLOCKED", note="ReID chon loc can OSNet x0.25 (thieu trong so, khong cai lan man); xem tracking/reid.py. Chay nhu C neu bo ReID.")
        return out
    det = Detector(cfg["model"], cfg["backend"], cfg["imgsz"], 0.25)
    r = detect_cached(src, det)
    dets, det_total = r["dets"], r["timing"]["total_s"]
    out["det_fps"] = round(r["timing"]["fps"], 2)
    load = r["timing"].get("cpu_load_avg")
    if cfg.get("adaptive"):
        hf = hard_frames(dets)
        a = cfg["adaptive"]
        big = Detector(a["model"], a["backend"], a["imgsz"], 0.25)
        big.load()
        big.predict(src.image(0))
        import time as _t
        t0 = _t.perf_counter()
        ref = {f: big.predict(src.image(f)) for f in hf}
        det_total += _t.perf_counter() - t0
        dets = merge_dets(dets, ref)
        out["adaptive_rerun_frac"] = round(len(hf) / max(1, len(dets)), 3)
        out["note"] = f"{len(hf)} frame kho chay lai bang {a['model']}/{a['backend']}/{a['imgsz']}"
    tr = engine.run_on_cache(dets, src, cfg["tracker"])
    assoc = float(np.sum(tr["assoc_s"]))
    pd = paths.DATA / "predictions" / project / r["key"]
    pd.mkdir(parents=True, exist_ok=True)
    (pd / f"{cid}_{cfg['tracker']}.json").write_text(json.dumps({"tracker": cfg["tracker"], "tracks": tr["tracks"], "scores": tr["scores"],
                                                                 "provenance": "ai_draft"}), encoding="utf8")
    n = len(dets)
    anchors = {k: tuple(v) for k, v in json.loads((paths.REPO / proj["anchors_file"]).read_text(encoding="utf8")).items()} if proj.get("anchors_file") else None
    anchors = {k: v for k, v in (anchors or {}).items() if v[2] < n} or None  # chi neo nam trong doan
    m = metrics(tr["tracks"], 5, anchors, 0, n)
    kf_opt = kf_dp = 0
    pick, _ = G.select_by_anchors(tr["tracks"], anchors, 0) if anchors else ({}, [])
    for tid in set(pick.values()):
        t = G.fill_gaps(tr["tracks"][tid], 5)
        kf_opt += len(K.optimal_keyframes(t, 4.0))
        kf_dp += len(G.simplify_keyframes(t, 4.0))
    # tham chieu A (chi khi da co cache cua A cung doan)
    ref_dets = None
    try:
        a_det = Detector("yolo26l", "pytorch", 1920, 0.25)
        from cvat_tools.detection.cached import cache_key
        from cvat_tools.core.cache import Cache
        got = Cache("detections").get(cache_key(src, a_det))
        ref_dets = {int(k): v for k, v in got["value"]["dets"].items()} if got else None
    except Exception:
        pass
    out.update(e2e_fps=round(n / (det_total + assoc), 3), peak_ram_mb=round(psutil.Process().memory_info().peak_wset / 2**20),
               IDF1="n/a (no GT)", anchor_continuity=m.get("anchor_continuity_mean"), anchors_shared_track=m.get("anchors_shared_track"),
               frag=m["fragmentation"], idsw_cand=m["id_switch_candidates"], keyframes_opt=kf_opt, keyframes_dp=kf_dp,
               cpu_load=load, noisy=(load or 0) > 35, status="measured")
    if ref_dets and cid != "A":
        out["pseudo_recall_vs_A"] = _agree(ref_dets, dets)
        out["pseudo_recall_small_vs_A"] = _agree(ref_dets, dets, True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="day05-mot17")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--only", nargs="*", default=list(CONFIGS))
    ap.add_argument("--idle-wait", type=float, default=0)
    ap.add_argument("--child")
    a = ap.parse_args(argv)
    if a.child:
        print("RESULT " + json.dumps(run_one(a.child, a.project, a.limit)))
        return 0
    from cvat_tools.benchmark.detector_bench import wait_idle
    rows = []
    for cid in a.only:
        wait_idle(a.idle_wait)
        p = subprocess.run([sys.executable, "-m", "cvat_tools.benchmark.ablation", "--child", cid, "--project", a.project, "--limit", str(a.limit)],
                           capture_output=True, text=True, encoding="utf8")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        row = json.loads(line[-1][7:]) if p.returncode == 0 and line else {"id": cid, "status": "FAILED", "note": (p.stderr.strip().splitlines() or ["?"])[-1][:200]}
        row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rows.append(row)
        print({k: row.get(k) for k in ("id", "model", "tracker", "resolution", "backend", "e2e_fps", "peak_ram_mb", "keyframes_opt", "status", "noisy", "note")}, flush=True)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not CSV_PATH.exists()
        with CSV_PATH.open("a", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
