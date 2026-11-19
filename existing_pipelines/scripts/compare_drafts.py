"""So sánh hai nháp person17 cạnh nhau (cũ | mới), chỉ frame có điểm đổi trạng thái. Chạy offline bằng cache ảnh.

Ký hiệu: tròn đặc xanh = thấy (V), vòng vàng = che/Q, X đỏ = v0/outside (O).
  python scripts/compare_drafts.py --job 1623 --old person17_auto_20261116_195704.json --new person17_auto_20261117_101316.json
Ra: <job folder>/review/compare/cmp_fNN.jpg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvat_tools.core import paths  # noqa: E402
from cvat_tools.pose.week2 import job_folder  # noqa: E402

GREEN, YELLOW, RED = (0, 200, 0), (0, 215, 255), (0, 0, 255)


def _draw(img, fr, title):
    im = img.copy()
    p = fr["people"][0]["points"]
    for n, pt in p.items():
        c = (int(pt["x"]), int(pt["y"]))
        st = pt["state"]
        if st == "outside":
            cv2.line(im, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), RED, 3)
            cv2.line(im, (c[0] - 8, c[1] + 8), (c[0] + 8, c[1] - 8), RED, 3)
        elif st == "occluded":
            cv2.circle(im, c, 8, YELLOW, 3)
        else:
            cv2.circle(im, c, 6, GREEN, -1)
        cv2.putText(im, n, (c[0] + 9, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(im, n, (c[0] + 9, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.rectangle(im, (0, 0), (im.shape[1], 44), (255, 255, 255), -1)
    cv2.putText(im, title, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, required=True)
    ap.add_argument("--profile", default="edufun")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--scale", type=float, default=0.75)
    a = ap.parse_args()
    folder = job_folder(a.job)
    ld = lambda n: {f["frame"]: f for f in json.loads((folder / "drafts" / n).read_text(encoding="utf8"))["frames"]}
    old, new = ld(a.old), ld(a.new)
    cache = paths.DATA / "jobs" / a.profile / str(a.job)
    info = json.loads((cache / "job_info.json").read_text(encoding="utf8"))
    frames_dir = paths.DATA.parent / info["frames_dir"]
    out = folder / "review" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for n in sorted(set(old) & set(new)):
        so = {k: v["state"] for k, v in old[n]["people"][0]["points"].items()}
        sn = {k: v["state"] for k, v in new[n]["people"][0]["points"].items()}
        diff = [k for k in so if so[k] != sn[k]]
        if not diff:
            continue
        name = info["names"][n - info["start_frame"]].split("/")[-1]
        img = cv2.imread(str(frames_dir / name))
        pair = np.hstack([_draw(img, old[n], f"CU f{n}"), _draw(img, new[n], f"MOI f{n}  doi: {','.join(sorted(diff, key=int))}")])
        pair = cv2.resize(pair, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"cmp_f{n}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles.append(pair)
        print(f"f{n}: {len(diff)} điểm đổi ({','.join(sorted(diff, key=int))})")
    print(f"-> {out} ({len(tiles)} frame)")


if __name__ == "__main__":
    main()
