"""Patch an toàn face50/pose: chỉ UPDATE toạ độ/occluded của element hiện có (không xóa/tạo lại skeleton).

Quy trình: (1) tải server, kiểm tra không khác snapshot live (version/hash) -> nếu khác thì DỪNG;
(2) backup nguyên trạng server vào reports/w2_recovery/backups/; (3) PATCH action=update chỉ các shape của frame;
(4) đọc lại, so sánh từng điểm, ghi SYNC_VERIFICATION.csv; (5) lỗi/khác -> rollback từ backup.
Mặc định dry-run. Source giữ nguyên (KHÔNG giả lập manual/human).

  python -X utf8 scripts/w2_face_sync.py --job 1975 --frames 91 --snapshot reports/w2_recovery/snapshots/live_1975_*.json [--yes]
Đề xuất lấy từ reports/w2_recovery/proposals/prop_<job>_<frame>.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cvat_tools.core.client import Cvat  # noqa: E402
import w2_recovery_compare as w  # noqa: E402

OUT = ROOT / "reports" / "w2_recovery"


def h(an) -> str:
    return hashlib.sha256(json.dumps(an, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--snapshot", required=True, help="snapshot live đã dùng để lập đề xuất")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    snap = json.loads(Path(a.snapshot).read_text(encoding="utf8"))
    c = Cvat.from_env("edufun", write_enabled=a.yes)
    server = c.annotations(a.job)
    lab, sub = w.labels(a.job)
    name2sub = {v: k for k, v in sub.items()}
    # (1) server phải giống snapshot ở các frame sắp sửa (không đè chỉnh sửa mới của người dùng)
    for f in a.frames:
        s_now = sorted((s["id"], json.dumps(s["points"]), json.dumps([(e["id"], e["points"], e.get("occluded"), e.get("outside")) for e in s["elements"]])) for s in server["shapes"] if s["frame"] == f)
        s_old = sorted((s["id"], json.dumps(s["points"]), json.dumps([(e["id"], e["points"], e.get("occluded"), e.get("outside")) for e in s["elements"]])) for s in snap["shapes"] if s["frame"] == f)
        if s_now != s_old:
            sys.exit(f"DỪNG: frame {f} trên server khác snapshot (người dùng vừa sửa?). Không ghi.")
    print("server version", server.get("version"), "hash", h(server), "- khớp snapshot ở các frame đích")
    # (2) tính patch
    patch_shapes, log = [], []
    for f in a.frames:
        prop = json.loads((OUT / "proposals" / f"prop_{a.job}_{f}.json").read_text(encoding="utf8"))
        pf = {p["label"]: p["points"] for p in prop["frames"][0]["people"]}
        for s in server["shapes"]:
            if s["frame"] != f or s["type"] != "skeleton":
                continue
            ns = copy.deepcopy(s)
            for e in ns["elements"]:
                nm = sub[e["label_id"]]
                p = pf[lab[s["label_id"]]][nm]
                b = (round(e["points"][0], 1), round(e["points"][1], 1), bool(e.get("occluded")), bool(e.get("outside")))
                nx = (p["x"], p["y"], p["state"] == "occluded", p["state"] == "outside")
                if b != nx:
                    log.append([a.job, f, lab[s["label_id"]], nm, e["id"], *b, *nx])
                e["points"] = [nx[0], nx[1]]
                e["occluded"], e["outside"] = nx[2], nx[3]
            patch_shapes.append(ns)
    print(f"{len(log)} keypoint thay đổi trên {len(patch_shapes)} skeleton")
    if not a.yes:
        print("dry-run (thêm --yes để ghi)")
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = OUT / "backups"
    bdir.mkdir(exist_ok=True)
    bk = bdir / f"server_{a.job}_before_{ts}.json"
    bk.write_text(json.dumps(server, ensure_ascii=False), encoding="utf8")
    print("backup:", bk.name)
    with (OUT / "KEYPOINT_CHANGES.csv").open("a", newline="", encoding="utf8") as fh:
        cw = csv.writer(fh)
        if fh.tell() == 0:
            cw.writerow("job frame skeleton point elem_id x_before y_before occ_before out_before x_after y_after occ_after out_after".split())
        cw.writerows(log)
    c._patch(a.job, "update", {"shapes": patch_shapes, "tracks": [], "tags": []})
    # (4) read-back
    back = c.annotations(a.job)
    bad, rows = 0, []
    for f in a.frames:
        exp = {e["id"]: (e["points"], e.get("occluded"), e.get("outside")) for s in patch_shapes if s["frame"] == f for e in s["elements"]}
        got = {e["id"]: (e["points"], e.get("occluded"), e.get("outside")) for s in back["shapes"] if s["frame"] == f for e in s["elements"]}
        nsk = sum(1 for s in back["shapes"] if s["frame"] == f and s["type"] == "skeleton")
        ok = set(exp) == set(got) and all(abs(exp[i][0][0] - got[i][0][0]) < 0.06 and abs(exp[i][0][1] - got[i][0][1]) < 0.06 and exp[i][1:] == got[i][1:] for i in exp) and nsk == 7
        bad += not ok
        rows.append([a.job, f, ts, nsk, len(got), "SYNC_VERIFIED" if ok else "MISMATCH", bk.name, h(back)])
    with (OUT / "SYNC_VERIFICATION.csv").open("a", newline="", encoding="utf8") as fh:
        cw = csv.writer(fh)
        if fh.tell() == 0:
            cw.writerow("job frame ts n_skeleton n_keypoints status backup readback_hash".split())
        cw.writerows(rows)
    (OUT / "snapshots" / f"live_{a.job}_{ts}_postsync.json").write_text(json.dumps(back, ensure_ascii=False), encoding="utf8")
    print(rows)
    if bad:
        print("MISMATCH -> rollback bằng backup (chưa tự động; chạy lại với backup)")
        sys.exit(2)


if __name__ == "__main__":
    main()
