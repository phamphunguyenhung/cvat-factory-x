"""Đo độ lệch nháp mặt 50 điểm so với nhãn bạn đã SỬA TAY trên CVAT (gold). Chỉ đọc CVAT.

Gold = frame mà nhãn auto đã đẩy lên bị sửa nhiều (>= --min-moved điểm lệch > --tol px so với nháp lúc đẩy).
Gold lưu ở <job>/metrics/gold.json (giữ lại cho lần đo sau). NME = lệch TB / khoảng cách hai khóe mắt ngoài (14 và 26).

  python scripts/eval_face50.py --job 1975 --refresh-gold                      # cập nhật gold từ CVAT rồi đo nháp mới nhất
  python scripts/eval_face50.py --job 1975 --draft week02/.../drafts/face50_XXX.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 làm vỡ tiếng Việt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvat_tools.core.client import Cvat  # noqa: E402
from cvat_tools.pose import gold as G, week2  # noqa: E402

GROUPS = week2.FACE_LABELS


_draft_points, _label_of, refresh_gold = G.draft_points, G.label_of, G.refresh_gold   # logic gold dùng chung: pose/gold.py


def evaluate(gold: dict, draft: dict, grp: dict) -> dict:
    rows, per = {}, {g: [] for g in GROUPS}
    nme = []
    for fr, pts in gold.items():
        fr = int(fr)
        if fr not in draft:
            continue
        iod = math.hypot(pts["14"][0] - pts["26"][0], pts["14"][1] - pts["26"][1]) if "14" in pts and "26" in pts else float("nan")
        errs = {n: math.hypot(pts[n][0] - draft[fr][n][0], pts[n][1] - draft[fr][n][1]) for n in pts if n in draft[fr]}
        for n, e in errs.items():
            per[grp[n]].append(e)
        nme.append(float(np.mean(list(errs.values()))) / iod)
        rows[fr] = {"mean_px": float(np.mean(list(errs.values()))), "nme": nme[-1], "iod": iod}
    return {"frames": rows, "group_px": {g: (float(np.mean(v)) if v else None) for g, v in per.items()},
            "nme_mean": float(np.mean(nme)) if nme else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, required=True); ap.add_argument("--profile", default="edufun")
    ap.add_argument("--draft", nargs="*", default=None, help="file nháp cần đo (mặc định: nháp đang đẩy + nháp mới nhất)")
    ap.add_argument("--refresh-gold", action="store_true"); ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--min-moved", type=int, default=30)
    a = ap.parse_args()
    c = Cvat.from_env(a.profile); folder = week2.job_folder(a.job)
    gold = refresh_gold(c, a.job, folder, a.tol, a.min_moved) if a.refresh_gold else json.loads((folder / "metrics" / "gold.json").read_text(encoding="utf8"))
    grp = _label_of(c.labels(a.job))
    files = [Path(f) for f in a.draft] if a.draft else sorted((folder / "drafts").glob("face50_*.json"))[-2:]
    print(f"gold: frame {sorted(map(int, gold))} ({len(gold)} frame)")
    for f in files:
        r = evaluate(gold, _draft_points(f), grp)
        print(f"\n== {f.name}")
        print("  lệch TB theo nhóm (px):", {g: (round(v, 1) if v is not None else None) for g, v in r["group_px"].items()})
        print("  theo frame:", {k: (round(v["mean_px"], 1), round(v["nme"], 3)) for k, v in r["frames"].items()}, "(px, NME)")
        print("  NME trung bình:", None if r["nme_mean"] is None else round(r["nme_mean"], 3))


if __name__ == "__main__":
    main()
