"""Kiểm tra quy tắc hình học xác định (chỉ đọc CVAT). Không sửa gì.

Mỗi vi phạm có mức:
  CONFIRMED  : vi phạm chắc chắn theo guideline BTC (ví dụ: thiếu điểm, điểm visible nằm ngoài ảnh, hai điểm visible trùng tọa độ).
  PROBABLE   : nghi ngờ, cần ảnh hoặc người nhìn (ví dụ: thứ tự trái/phải, mắt dồn, miệng trong nằm ngoài miệng ngoài).

Ghi reports/w2_qc_rules_<job>.csv (một dòng mỗi vi phạm). Ghi thêm reports/w2_qc_summary.json.

  python -X utf8 scripts/w2_qc_rules.py --job 1623 --job 1975
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))
import cv2  # noqa: E402
from cvat_tools.core.client import Cvat  # noqa: E402

FACE_SKEL = {"longmaytrai": range(0, 5), "longmayphai": range(5, 10), "songmui": range(10, 14),
             "mattrai": range(14, 22), "matphai": range(22, 30), "moingoai": range(30, 42), "moitrong": range(42, 50)}
PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]  # (R chẵn, L lẻ)


def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def load_points(c: Cvat, job: int):
    """{frame: {skeleton_or_person: {point_name: {x,y,occluded,outside}}}}"""
    subn = {}
    for l in c.labels(job):
        for s in l["sublabels"]:
            subn[s["id"]] = (l["name"], s["name"])
    out: dict = defaultdict(lambda: defaultdict(dict))
    for s in c.annotations(job)["shapes"]:
        for e in s.get("elements", []):
            skel, pname = subn[e["label_id"]]
            pts = e["points"]
            out[s["frame"]][skel][pname] = {"x": pts[0], "y": pts[1],
                                            "occluded": bool(e.get("occluded")), "outside": bool(e.get("outside"))}
        if not s.get("elements"):  # skeleton không có element -> ghi nhận để báo thiếu
            skel, _ = subn.get(s["label_id"], ("?", "?"))
            out[s["frame"]][skel] = out[s["frame"]].get(skel, {})
    return out


def frame_size(job: int, n: int):
    p = ROOT / "data" / "jobs" / "edufun" / str(job) / "frames" / f"G06_{n + 1:03d}.jpg"
    if not p.exists():
        return None
    h, w = cv2.imread(str(p)).shape[:2]
    return w, h


def check_face(fr, pts, size, add):
    for skel, rng in FACE_SKEL.items():
        got = pts.get(skel, {})
        missing = [str(i) for i in rng if str(i) not in got]
        if skel not in pts or not got:
            add("CONFIRMED", "face_skeleton_missing", skel, "không có điểm nào")
        elif missing:
            add("CONFIRMED", "face_points_missing", skel, f"thiếu {len(missing)} điểm: {','.join(missing)}")
    allp = [(n, v) for sk in pts.values() for n, v in sk.items()]
    # trùng tọa độ hai điểm visible
    vis = [(n, v) for n, v in allp if not v["outside"]]
    for i in range(len(vis)):
        for j in range(i + 1, len(vis)):
            (n1, a), (n2, b) = vis[i], vis[j]
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 2.0:
                # BTC face cho phép cặp mí/môi trùng khi nhắm mắt hoặc ngậm miệng -> chỉ PROBABLE
                add("PROBABLE", "face_duplicate_point", f"{n1}/{n2}", "hai điểm không outside trùng tọa độ (<2px); có thể hợp lệ khi nhắm mắt/ngậm miệng")
    # vị trí mắt trái/phải (trên ảnh)
    def cx(skel):
        xs = [v["x"] for v in pts.get(skel, {}).values()]
        return sum(xs) / len(xs) if xs else None
    cl, cr = cx("mattrai"), cx("matphai")
    if cl is not None and cr is not None and cl > cr:
        add("PROBABLE", "face_eye_lr_swapped", "mattrai/matphai", "mắt trái (trên ảnh) nằm bên phải mắt phải; có thể do ảnh xoay")
    ml, mr = cx("longmaytrai"), cx("longmayphai")
    if ml is not None and mr is not None and ml > mr:
        add("PROBABLE", "face_brow_lr_swapped", "longmaytrai/longmayphai", "lông mày trái nằm bên phải lông mày phải")
    # môi trong nằm trong môi ngoài
    outer = [(v["x"], v["y"]) for k, v in sorted(pts.get("moingoai", {}).items(), key=lambda t: int(t[0]))]
    if len(outer) == 12:
        for n, v in pts.get("moitrong", {}).items():
            if not v["outside"] and not point_in_poly((v["x"], v["y"]), outer):
                add("PROBABLE", "face_inner_mouth_outside_outer", n, "điểm môi trong nằm ngoài vòng môi ngoài")
    # dồn điểm: toàn bộ mặt trong vùng quá nhỏ so với ảnh
    if size and allp:
        xs = [v["x"] for _, v in allp]; ys = [v["y"] for _, v in allp]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area < 0.01 * size[0] * size[1]:
            add("PROBABLE", "face_cluster_collapsed", "all", f"toàn bộ 50 điểm nằm trong {int(area)} px² (<1% ảnh)")


def check_pose(fr, pts, size, add):
    person = pts.get("person_vf17", {})
    names = {str(i) for i in range(1, 18)}
    missing = sorted(names - set(person), key=int)
    if not person:
        add("CONFIRMED", "pose_skeleton_missing", "person_vf17", "không có skeleton")
        return
    if missing:
        add("CONFIRMED", "pose_points_missing", "person_vf17", f"thiếu {len(missing)} điểm: {','.join(missing)}")
    vis = {n: v for n, v in person.items() if not v["outside"]}
    for n, v in vis.items():
        if size and not (0 <= v["x"] <= size[0] and 0 <= v["y"] <= size[1]):
            add("CONFIRMED", "pose_visible_outside_image", n, "điểm không outside nhưng nằm ngoài ảnh")
    ks = sorted(vis, key=int)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = vis[ks[i]], vis[ks[j]]
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 2.0:
                add("CONFIRMED", "pose_duplicate_point", f"{ks[i]}/{ks[j]}", "hai điểm không outside trùng tọa độ (<2px)")
    for r, l in PAIRS:
        R, L = person.get(str(r)), person.get(str(l))
        if R and L and not R["outside"] and not L["outside"]:
            if R["x"] <= L["x"]:
                add("PROBABLE", "pose_lr_swapped", f"{r}/{l}", "điểm R (chẵn) không nằm bên phải điểm L (lẻ) trên ảnh; có thể do người nằm ngang hoặc ảnh xoay")
    # outside nhưng nằm trong khung: chỉ cảnh báo để kiểm tra
    if size:
        for n, v in person.items():
            if v["outside"] and 0 <= v["x"] <= size[0] and 0 <= v["y"] <= size[1]:
                add("PROBABLE", "pose_outside_but_inside_frame", n, "điểm outside nhưng tọa độ nằm trong khung; kiểm tra có phải occluded không")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, action="append", required=True)
    ap.add_argument("--profile", default="edufun")
    a = ap.parse_args()
    c = Cvat.from_env(a.profile)
    summary = {}
    for job in a.job:
        pts_all = load_points(c, job)
        rows = []
        for n in range(90, 120):
            size = frame_size(job, n)
            pts = pts_all.get(n, {})
            def add(sev, rule, where, detail, _n=n):
                rows.append({"job_id": job, "frame_id": _n, "severity": sev, "rule": rule, "where": where, "detail": detail})
            if not pts:
                add("CONFIRMED", "frame_no_skeleton", "all", "frame chưa có nhãn")
                continue
            (check_face if job == 1975 else check_pose)(n, pts, size, add)
        out = ROOT / "reports" / f"w2_qc_rules_{job}.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=["job_id", "frame_id", "severity", "rule", "where", "detail"])
            w.writeheader(); w.writerows(rows)
        summary[job] = {"violations": dict(Counter(r["severity"] for r in rows)),
                        "rules": dict(Counter(r["rule"] for r in rows)),
                        "frames_with_confirmed": sorted({r["frame_id"] for r in rows if r["severity"] == "CONFIRMED"})}
        print(f"job {job}: {summary[job]['violations']} -> {out}")
        print("  quy tắc:", summary[job]["rules"])
        print("  frame có CONFIRMED:", summary[job]["frames_with_confirmed"])
    (ROOT / "reports" / "w2_qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
