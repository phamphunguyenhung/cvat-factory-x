"""Đo nháp pose 17 điểm so với điểm BẠN ĐÃ SỬA TAY trên CVAT (gold). Chỉ đọc CVAT.

Gold = frame có >= --min-moved điểm bị dời so với nháp lần đẩy gần nhất. Mỗi điểm đo:
  lệch px, và OKS = exp(-d^2 / (2 s^2 k^2)), k = 2*sigma COCO (sigmas lấy từ spm config/schema.json), s^2 = diện tích bbox các điểm gold của frame.

  python scripts/eval_pose17.py --job 1623 --refresh-gold
  python scripts/eval_pose17.py --job 1623 --draft week02/.../drafts/person17_A.json week02/.../drafts/person17_B.json   # so nhiều nháp
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 làm vỡ tiếng Việt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvat_tools.core import paths  # noqa: E402
from cvat_tools.core.client import Cvat  # noqa: E402
from cvat_tools.pose import gold as G, week2  # noqa: E402

NAMES = ["mũi", "mắt T", "mắt P", "tai T", "tai P", "vai T", "vai P", "khuỷu T", "khuỷu P", "cổ tay T", "cổ tay P",
         "hông T", "hông P", "gối T", "gối P", "cổ chân T", "cổ chân P"]


def sigmas() -> np.ndarray:
    f = paths.REPO / "tools" / "skeleton-pose-manager" / "config" / "schema.json"
    return np.array(json.loads(f.read_text(encoding="utf8"))["person"]["sigmas"], float)


def evaluate(gold: dict, draft: dict) -> dict:
    k = 2 * sigmas()
    per = {str(i): [] for i in range(1, 18)}
    frames = {}
    for fr, pts in gold.items():
        fr = int(fr)
        if fr not in draft:
            continue
        xs = [v[0] for v in pts.values()]; ys = [v[1] for v in pts.values()]
        area = max(np.ptp(xs) * np.ptp(ys), 50.0 ** 2)
        errs, oks = [], []
        for n, v in pts.items():
            if n not in draft[fr]:
                continue
            d = math.hypot(v[0] - draft[fr][n][0], v[1] - draft[fr][n][1])
            per[n].append(d); errs.append(d); oks.append(math.exp(-d * d / (2 * area * k[int(n) - 1] ** 2)))
        if errs:
            frames[fr] = {"mean_px": float(np.mean(errs)), "oks": float(np.mean(oks)), "n": len(errs)}
    allo = [v["oks"] for v in frames.values()]
    return {"frames": frames, "point_px": {n: (float(np.mean(v)) if v else None) for n, v in per.items()},
            "oks_mean": float(np.mean(allo)) if allo else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, required=True); ap.add_argument("--profile", default="edufun")
    ap.add_argument("--draft", nargs="*", default=None); ap.add_argument("--refresh-gold", action="store_true")
    ap.add_argument("--tol", type=float, default=1.5); ap.add_argument("--min-moved", type=int, default=8)
    a = ap.parse_args()
    folder = week2.job_folder(a.job)
    gold = G.refresh_gold(Cvat.from_env(a.profile), a.job, folder, a.tol, a.min_moved) if a.refresh_gold else G.load_gold(folder)
    if not gold:
        print("Chưa có gold: bạn chưa sửa frame nào (>= %d điểm) trong job %d. Sửa 5–8 frame đại diện rồi chạy lại với --refresh-gold." % (a.min_moved, a.job))
        return
    files = [Path(f) for f in a.draft] if a.draft else sorted((folder / "drafts").glob("person17_*.json"))[-2:]
    print(f"gold: frame {sorted(map(int, gold))}")
    for f in files:
        r = evaluate(gold, G.draft_points(f))
        print(f"\n== {f.name}\n  OKS trung bình: {None if r['oks_mean'] is None else round(r['oks_mean'], 3)}")
        print("  lệch px theo điểm:", {NAMES[int(n) - 1]: round(v, 1) for n, v in r["point_px"].items() if v is not None})
        print("  theo frame (px, OKS):", {k: (round(v["mean_px"], 1), round(v["oks"], 2)) for k, v in r["frames"].items()})


if __name__ == "__main__":
    main()
