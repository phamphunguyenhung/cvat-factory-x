#!/usr/bin/env python3
"""Kiểm phần bàn giao của nhóm (không cần ground truth).

  python scripts/check_handover.py --run runs/r11

Kiểm định dạng và tính nhất quán: v1 không bị sửa, lô v2 đủ phần, hash đúng, số liệu khớp,
truy vết khớp jobs.csv, kết luận đúng ngưỡng, lineage có HOLD v1 và release v2, zip khớp .sha256.
PASS ở đây KHÔNG có nghĩa nhãn đúng.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

# Lô v2 là sản phẩm DETECTION (images + labels). Ảnh có lỗi ở nhãn detection hoặc thiếu file phải gán lại:
# vào rework_queue và rời v2. Lỗi panoptic không làm sai nhãn detection: ảnh ở lại v2, báo team panoptic trong README Ghi chú.
# wrong_hash, guideline_mismatch, scene_leak sửa được ngay trong v2.
DET_REWORK = {"missing_image", "invalid_class", "bbox_out_of_bounds", "duplicate_box", "class_swap", "bbox_drift", "missing_box"}
THRESHOLD = 0.02  # Day 7: > 2% ảnh lỗi thì không Release


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Report:
    def __init__(self):
        self.rows = []

    def check(self, code, ok, good, bad):
        self.rows.append((code, bool(ok), good if ok else bad))

    def done(self) -> int:
        for code, ok, msg in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {code:<4} {msg}")
        fails = sum(1 for _, ok, _ in self.rows if not ok)
        print(f"\nTổng: {len(self.rows) - fails} PASS, {fails} FAIL")
        print("=> ĐỦ ĐIỀU KIỆN ĐỊNH DẠNG (PASS không có nghĩa nhãn đúng)." if not fails else "=> CHƯA ĐẠT: sửa các dòng FAIL.")
        return 1 if fails else 0


def file_rule_errors(d: Path, man: dict, meta: dict, jobs: dict) -> set:
    """Lỗi luật mức file bộ kiểm tự thấy được, để errors.json rỗng hoặc thiếu không qua được."""
    out = set()
    scenes = {}
    for name, it in man.items():
        m = meta[name]
        scenes.setdefault(m["scene_id"], set()).add(m["split"])
        img = d / it["image"]
        if not img.is_file():
            out.add(("missing_image", name))
            continue
        if sha(img) != it["image_sha256"]:
            out.add(("wrong_hash", name))
            continue
        if m["guideline_version"] != jobs[m["job_id"]]["guideline"]:
            out.add(("guideline_mismatch", name))
            continue
        for ln in (d / it["label"]).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            c, x, y, w, h = ln.split()[:5]
            c, x, y, w, h = int(float(c)), float(x), float(y), float(w), float(h)
            if not 0 <= c < 80:
                out.add(("invalid_class", name))
                break
            if x - w / 2 < -0.01 or y - h / 2 < -0.01 or x + w / 2 > 1.01 or y + h / 2 > 1.01:
                out.add(("bbox_out_of_bounds", name))
                break
    out |= {("scene_leak", sc) for sc, v in scenes.items() if len(v) > 1}
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def kv(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Kiểm phần bàn giao")
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    d = run / "delivery"
    r = Report()
    rid = json.loads((d / "release.json").read_text(encoding="utf-8"))["release_id"]
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["items"]
    meta = {m["image"]: m for m in json.loads((d / "metadata.json").read_text(encoding="utf-8"))["items"]}
    jobs = {row["job_id"]: row for row in read_csv(d / "jobs.csv")}
    names80 = [ln.split(" ", 1)[1].strip() for ln in (d / "classes.txt").read_text(encoding="utf-8").splitlines()]

    # H1 v1 giữ nguyên
    now = "lab250_" + hashlib.sha256((d / "manifest.json").read_bytes() + (d / "metadata.json").read_bytes()).hexdigest()[:12]
    rel = json.loads((d / "release.json").read_text(encoding="utf-8"))
    labels_ok = all(sha(d / it["label"]) == it["label_sha256"] and sha(d / it["mask"]) == it["mask_sha256"] for it in man.values()) \
        and rel.get("panoptic_sha256") in (None, sha(d / "panoptic.json"))
    r.check("H1", now == rid and labels_ok, f"v1 ({rid}) giữ nguyên", "delivery v1 đã bị sửa: không ghi đè lô đã giao, làm bản v2")

    # H2 errors.json + review_log
    errors = json.loads((run / "errors.json").read_text(encoding="utf-8"))["errors"] if (run / "errors.json").is_file() else None
    errors = errors or []
    must = file_rule_errors(d, man, meta, jobs)
    have = {(e["type"], e["target"]) for e in errors}
    r.check("H2", (run / "errors.json").is_file() and must <= have, f"errors.json có {len(errors)} lỗi, gồm đủ lỗi luật mức file",
            f"errors.json thiếu {len(must - have)} lỗi luật mức file (missing_image, wrong_hash, guideline_mismatch, scene_leak, "
            "invalid_class, bbox_out_of_bounds) mà bộ kiểm tự thấy được: chạy lại validator")
    log = read_csv(run / "review_log.csv") if (run / "review_log.csv").is_file() else []
    confirmed = {(row.get("type", "").strip(), row.get("image", "").strip()) for row in log if row.get("verdict", "").strip() == "confirmed"}
    need = {(e["type"], e["target"]) for e in errors if e["type"] in metrics.SEMANTIC_TYPES}
    notes = [(row.get("note") or "").strip() for row in log]
    notes_ok = notes and all(len(n) >= 20 for n in notes) and len(set(notes)) >= 0.7 * len(notes)
    r.check("H3", log and need <= confirmed and notes_ok,
            f"review_log.csv: {len(log)} dòng, mọi lỗi gán nhãn đã được xem ảnh và xác nhận, ghi chú cụ thể",
            "review_log.csv phải có dòng verdict=confirmed cho mỗi lỗi gán nhãn trong errors.json "
            f"(thiếu {len(need - confirmed)}), và mỗi ghi chú ≥ 20 ký tự, không chép giống nhau (nói rõ thấy gì trong ảnh)")

    # Lô v2
    v2 = run / "handover" / f"{rid}_v2"
    parts = ["images", "labels", "manifest.csv", "label_summary.csv", "truyvet.csv", "README.txt", "rework_queue.csv"]
    missing = [p for p in parts if not (v2 / p).exists()]
    r.check("H4", not missing, f"{v2.name}/ đủ {len(parts)} phần", f"{v2} thiếu: {missing}")
    if missing:
        return r.done()
    stems = sorted(p.stem for p in (v2 / "images").glob("*.jpg"))
    lab_stems = sorted(p.stem for p in (v2 / "labels").glob("*.txt"))
    rework = read_csv(v2 / "rework_queue.csv")
    rq = {row["image"] for row in rework}
    bad_imgs = {e["target"] for e in errors if e["target"] in man and e["type"] in DET_REWORK}
    reported = {e["target"] for e in errors}
    r.check("H5", stems == lab_stems and bad_imgs <= rq and rq <= reported and not (rq & set(stems)),
            f"v2 có {len(stems)} ảnh; {len(rq)} ảnh lỗi chuyển sang rework_queue, không nằm trong v2",
            "mỗi ảnh có lỗi nhãn detection hoặc thiếu file phải vào rework_queue.csv và rời khỏi v2 (lỗi panoptic thì ảnh ở lại); "
            "ảnh trong rework_queue phải có lỗi trong errors.json; images/ và labels/ phải khớp nhau")
    wrong_owner = [row["image"] for row in rework if row["image"] in meta and
                   (row.get("annotator") != jobs[meta[row["image"]]["job_id"]]["annotator"] or row.get("reviewer") != jobs[meta[row["image"]]["job_id"]]["reviewer"])]
    r.check("H6", not wrong_owner, "rework_queue ghi đúng người gán/người kiểm theo jobs.csv", f"sai người gán/kiểm: {wrong_owner[:4]}")
    changed = [s for s in lab_stems if s in man and sha(v2 / "labels" / f"{s}.txt") != man[s]["label_sha256"]]
    unknown = [s for s in stems if s not in man]
    r.check("H7", not changed and not unknown, "nhãn trong v2 giữ nguyên như đã gán (không sửa tay)",
            f"nhãn bị sửa tay hoặc ảnh lạ: {(changed + unknown)[:4]}")

    # manifest.csv
    rows = read_csv(v2 / "manifest.csv")
    listed = {row["file"]: row["sha256"] for row in rows}
    files = {p.relative_to(v2).as_posix(): p for sub in ("images", "labels") for p in (v2 / sub).iterdir() if p.is_file()}
    bad = [f for f, p in files.items() if listed.get(f) != sha(p)]
    r.check("H8", set(listed) == set(files) and not bad, f"manifest.csv: {len(listed)} file, sha256 đúng",
            f"manifest.csv phải liệt kê đúng mọi file trong images/ và labels/ với sha256 thật (lệch: {(bad or sorted(set(listed) ^ set(files)))[:3]})")

    # label_summary.csv
    counts = {}
    for s in lab_stems:
        for ln in (v2 / "labels" / f"{s}.txt").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c = int(float(ln.split()[0]))
                counts[names80[c] if 0 <= c < 80 else str(c)] = counts.get(names80[c] if 0 <= c < 80 else str(c), 0) + 1
    summ = {row["class"]: int(row["count"]) for row in read_csv(v2 / "label_summary.csv")}
    total = summ.pop("TOTAL", None)
    r.check("H9", summ == counts and total == sum(counts.values()), f"label_summary.csv khớp ({sum(counts.values())} nhãn)",
            "label_summary.csv (cột class,count, tên lớp theo classes.txt, dòng TOTAL) phải khớp nhãn trong v2")

    # truyvet.csv
    tv = {row["image"]: row for row in read_csv(v2 / "truyvet.csv")}
    wrong = []
    for s in stems:
        row, m = tv.get(s), meta.get(s, {})
        job = jobs.get(m.get("job_id"), {})
        if not row or row.get("job_id") != m.get("job_id") or row.get("annotator") != job.get("annotator") \
           or row.get("reviewer") != job.get("reviewer") or row.get("guideline") != job.get("guideline") or row.get("scene_id") != m.get("scene_id"):
            wrong.append(s)
    scene_splits = {}
    for s in stems:
        if s in tv:
            scene_splits.setdefault(tv[s].get("scene_id"), set()).add(tv[s].get("split"))
    leak = [sc for sc, v in scene_splits.items() if len(v) > 1]
    r.check("H10", not wrong and set(tv) == set(stems), "truyvet.csv: mỗi ảnh một dòng, job/người/guideline theo jobs.csv",
            f"truyvet.csv sai hoặc thiếu (guideline lấy theo job, không theo metadata ảnh): {wrong[:4]}")
    r.check("H11", not leak, "v2 không còn cảnh nằm ở cả train và val", f"cảnh còn rò train/val: {leak[:4]}")

    # README.txt
    info = kv(v2 / "README.txt")
    n_lab = sum(counts.values())
    err_imgs = {e["target"] for e in errors}  # mỗi ảnh (hoặc cảnh, với scene_leak) tính một lần
    qc = re.search(r"(\d+)\s*/\s*(\d+)", info.get("QC v1", ""))
    v1_bad = qc and int(qc.group(1)) == len(err_imgs) and int(qc.group(2)) == len(man)
    rate = len(err_imgs) / len(man)
    v1_decision = info.get("Kết luận v1", "").split(" ")[0]
    ok_v1 = v1_decision in ("Hold", "Rework") if rate > THRESHOLD else v1_decision == "Release"
    r.check("H12", info.get("Release") == f"{rid}_v2" and info.get("Parent") == rid
            and info.get("Số ảnh") == str(len(stems)) and info.get("Số nhãn") == str(n_lab),
            "README.txt: tên lô, lô cha, số ảnh, số nhãn khớp",
            f"README.txt cần Release: {rid}_v2, Parent: {rid}, Số ảnh: {len(stems)}, Số nhãn: {n_lab}")
    r.check("H13", v1_bad and ok_v1 and info.get("Kết luận v2", "").split(" ")[0] in ("Release", "Hold", "Rework"),
            f"QC v1 {len(err_imgs)}/{len(man)} = {rate:.1%} ⇒ {v1_decision} (ngưỡng 2%); có kết luận v2",
            f"README.txt cần 'QC v1: {len(err_imgs)}/{len(man)} ...' (số ảnh lỗi trong errors.json / tổng ảnh), "
            "'Kết luận v1:' đúng ngưỡng 2% và 'Kết luận v2:'")

    # lineage
    events = []
    lp = run.parent / "lineage.jsonl"
    if lp.is_file():
        events = [json.loads(x) for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
    held = any(e.get("event") == "decision" and e.get("release_id") == rid and e.get("decision") == "HOLD" for e in events)
    if held and v1_decision == "Release":
        r.check("H13b", False, "", "README ghi 'Kết luận v1: Release' nhưng lineage ghi HOLD: hai nơi phải khớp")
    rel = any(e.get("event") == "release" and e.get("release_id") == f"{rid}_v2" and e.get("parent") == rid for e in events)
    r.check("H14", held and rel, "lineage: HOLD cho v1 và release v2 có parent v1",
            f"{lp}: cần decision HOLD cho {rid} và event release {rid}_v2 --parent {rid}")

    # zip + sha256
    # zip tái lập được: không có file zip (không commit) thì đóng gói lại trong bộ nhớ rồi so.
    from pack_release import build_zip
    s = v2.with_suffix(".sha256")
    zok = s.is_file() and s.read_text(encoding="utf-8").split()[0] == hashlib.sha256(build_zip(v2)).hexdigest()
    r.check("H15", zok, "đóng gói lại v2 ra đúng mã trong .sha256",
            "chạy python scripts/pack_release.py <thư mục v2> sau khi chốt lô (mã .sha256 phải khớp nội dung hiện tại)")

    # trace + security
    tr = json.loads((run / "trace.json").read_text(encoding="utf-8")) if (run / "trace.json").is_file() else {}
    people = {j["annotator"] for j in jobs.values()} | {j["reviewer"] for j in jobs.values()}
    tr_ok = all(isinstance(tr.get(k), list) for k in ("tier_a", "tier_b", "notify")) and tr.get("tier_a") and tr.get("notify") \
        and set(tr.get("tier_a", []) + tr.get("tier_b", [])) <= set(man) and set(tr.get("notify", [])) <= people
    r.check("H16", tr_ok, f"trace.json: tier_a {len(tr.get('tier_a', []))} ảnh, tier_b {len(tr.get('tier_b', []))} ảnh, notify {len(tr.get('notify', []))} người",
            "trace.json cần tier_a (không rỗng), tier_b (có thể rỗng) là tên ảnh trong lô, notify (không rỗng) là mã người có trong jobs.csv")
    sec = (run / "security.md").read_text(encoding="utf-8") if (run / "security.md").is_file() else ""
    parts_ = [p for p in re.split(r"^##\s+", sec, flags=re.M)[1:] if len(re.sub(r"\s+", "", p.split("\n", 1)[-1])) >= 80]
    face = [p for p in parts_ if "khuôn mặt" in p.split("\n")[0].lower()]
    names_in = face and any(n in face[0] for n in man)
    r.check("H17", len(parts_) >= 3 and face and "hold" in face[0].lower() and names_in and "<<" not in sec,
            "security.md đủ 3 mục, ca khuôn mặt kết luận Hold và nêu ảnh cụ thể trong lô",
            "security.md cần ≥ 3 mục có nội dung (không còn <<...>>); mục 'khuôn mặt' phải kết luận Hold và nêu tên ít nhất một ảnh có người trong lô")
    return r.done()


if __name__ == "__main__":
    raise SystemExit(main())
