"""Hinh so sanh tu CACHE co san (khong chay lai model): overlay detection giua cac cau hinh + quy dao tracker.
Chay: python -m cvat_tools.benchmark.figures --project day05-mot17 --limit 60 --frame 30
Dau ra: data/exports/<project>/figures/{detect_overlay_f<N>.jpg, tracks_<tracker>.jpg}
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from cvat_tools.core import paths
from cvat_tools.core.cache import Cache
from cvat_tools.core.config import load_project
from cvat_tools.detection.cached import cache_key
from cvat_tools.detection.detector import Detector
from cvat_tools.detection.frames import DirSource
from cvat_tools.tracking import geometry as G


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="day05-mot17")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--configs", nargs="+", default=["yolo26l:pytorch:1920", "yolo26s:openvino:640", "yolo26n:openvino:640"])
    a = ap.parse_args(argv)
    proj = load_project(a.project)
    src = DirSource(paths.REPO / proj["dataset"]["frames_dir"], 0, a.limit)
    out = paths.DATA / "exports" / a.project / "figures"
    out.mkdir(parents=True, exist_ok=True)
    img0 = src.image(a.frame)
    panels, keys = [], {}
    cols = [(0, 0, 255), (0, 160, 0), (255, 100, 0)]
    for i, c in enumerate(a.configs):
        m, b, sz = c.split(":")
        det = Detector(m, b, int(sz), 0.25)
        k = cache_key(src, det)
        got = Cache("detections").get(k)
        if not got:
            print("thieu cache", c)
            continue
        keys[c] = k
        d = {int(x): v for x, v in got["value"]["dets"].items()}[a.frame]
        im = img0.copy()
        for x in d:
            cv2.rectangle(im, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), cols[i % 3], 3)
        cv2.putText(im, f"{c}  n={len(d)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
        panels.append(cv2.resize(im, (960, 540)))
    if panels:
        cv2.imwrite(str(out / f"detect_overlay_f{a.frame}.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 80])
    # quy dao: dung cache dau tien co du doan
    for k in keys.values():
        pdir = paths.DATA / "predictions" / a.project / k
        if not (pdir.exists() and list(pdir.glob("*.json"))):
            continue
        for f in sorted(pdir.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf8"))
            top = G.select_longest({int(t): {int(fr): b for fr, b in v.items()} for t, v in d["tracks"].items()}, 5, 10)
            im = img0.copy()
            for j, (tid, tr) in enumerate(top.items()):
                col = [(0, 0, 255), (0, 160, 0), (255, 100, 0), (200, 0, 200), (0, 160, 200)][j]
                pts = [(int((b[0] + b[2]) / 2), int(b[3])) for _, b in sorted(tr.items())]
                for p, q in zip(pts, pts[1:]):
                    cv2.line(im, p, q, col, 4)
                cv2.putText(im, f"#{tid} ({len(tr)}f)", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
            cv2.imwrite(str(out / f"tracks_{f.stem}_{k[:6]}.jpg"), cv2.resize(im, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        break
    print("hinh ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
