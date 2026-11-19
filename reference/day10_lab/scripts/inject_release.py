#!/usr/bin/env python3
"""Tạo MỘT lô bàn giao (detection + panoptic cùng 250 ảnh) có lỗi gài theo seed.

  python scripts/inject_release.py --seed 11 --out runs/r11
  python scripts/inject_release.py --seed 5 --no-faults --out runs/clean        # lô sạch (pool calib)

Mọi lựa chọn ngẫu nhiên đi qua random.Random(seed). Cùng seed, cùng pool, cùng dữ liệu
nguồn thì ra cùng delivery và cùng ground truth.

Số lỗi mỗi loại là ngẫu nhiên (0-3), không cố định. Mỗi ảnh mang nhiều nhất một lỗi.
Lỗi gán nhãn (class_swap, bbox_drift, missing_box, category_swap, instance_merge) không
có nhãn gốc nào trong delivery để so: chỉ thấy khi đối chiếu detection với panoptic,
rồi nhìn ảnh để quyết định bên nào sai.

Đề ẩn do giảng viên tạo bằng công cụ riêng (--opaque, --profile và các bước xử lý ảnh khác);
mức lỗi trong đề ẩn có thể khác mặc định ở đây.
Validator không được đọc <out>/_private.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pngutil  # noqa: E402
from download_public import COCO80  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DET_RULE = ["missing_image", "wrong_hash", "invalid_class", "bbox_out_of_bounds", "duplicate_box", "scene_leak", "guideline_mismatch"]
DET_CROSS = ["class_swap", "bbox_drift", "missing_box"]
PAN_RULE = ["invalid_category_id", "thing_overlap"]
PAN_CROSS = ["category_swap", "instance_merge"]
# Mặc định cho lô luyện tập. Đề ẩn dùng --profile (JSON) của giảng viên để đổi các mức này.
COUNTS = {
    "missing_image": (0, 2), "wrong_hash": (1, 2), "invalid_class": (0, 2), "bbox_out_of_bounds": (1, 2),
    "duplicate_box": (1, 2), "scene_leak": (1, 1), "guideline_mismatch": (1, 2),
    "class_swap": (1, 3), "bbox_drift": (1, 3), "missing_box": (1, 2),
    "invalid_category_id": (0, 2), "thing_overlap": (1, 2), "category_swap": (1, 2), "instance_merge": (1, 2),
}
# Cặp lớp dễ nhầm (tên COCO). Đổi lớp chỉ trong các nhóm này nên "lớp lạ trong ảnh" không lộ lỗi.
CONFUSABLE = [
    ["car", "truck", "bus"], ["bicycle", "motorcycle"], ["cat", "dog"], ["horse", "cow", "sheep"],
    ["cup", "wine glass", "bowl"], ["chair", "couch", "bench"], ["knife", "fork", "spoon"],
    ["laptop", "tv"], ["cell phone", "remote"], ["handbag", "backpack", "suitcase"],
    ["orange", "apple"], ["donut", "cake"], ["bird", "kite"],
]
PROFILE = {
    "counts": COUNTS,
    "drift": [0.12, 0.3],          # độ lệch tâm, tính theo kích thước box
    "drift_scale": 0.2,            # đổi cỡ box tối đa ±20% (một nửa số ca)
    "swap_confusable": 0.7,        # xác suất đổi sang lớp dễ nhầm (còn lại: lớp bất kỳ)
    "min_box_area": 0.005,         # box bị đổi lớp/lệch phải chiếm >= 0.5% ảnh (nhìn được bằng mắt)
    "min_segment_px": 1024,        # segment bị đổi lớp/gộp phải >= 1024 pixel
}
ANNOTATORS = ["A01", "A02", "A03", "A04", "A05"]
REVIEWERS = ["Q01", "Q02", "Q03"]
SCENE_SIZE = 5
JOB_SIZE = 25


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_yolo(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip()]


def _iou(a: list[str], b: list[str]) -> float:
    ax, ay, aw, ah = map(float, a[1:5])
    bx, by, bw, bh = map(float, b[1:5])
    ix = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
    iy = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-12)


def dedupe(rows: list[list[str]]) -> list[list[str]]:
    """Bỏ box trùng có sẵn trong COCO (cùng lớp, IoU > 0.7), để duplicate_box chỉ còn lỗi gài."""
    kept: list[list[str]] = []
    for row in rows:
        if not any(k[0] == row[0] and _iou(k, row) > 0.7 for k in kept):
            kept.append(row)
    return kept


def write_yolo(rows: list[list[str]]) -> str:
    return "".join(" ".join(row) + "\n" for row in rows)


def perturb_jpg(data: bytes, rng: random.Random) -> bytes:
    """Chèn một COM segment ngay sau SOI: ảnh hiển thị như cũ, hash khác bản COCO."""
    if not data.startswith(b"\xff\xd8"):
        return data
    payload = bytes(rng.randrange(32, 127) for _ in range(24))
    return data[:2] + b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:]


class Builder:
    def __init__(self, args):
        self.rng = random.Random(args.seed)
        self.p = dict(PROFILE)
        if getattr(args, "profile", None):
            self.p.update(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        self.seed = args.seed
        suffix = {"practice": "", "test": "_test", "calib": "_calib"}[args.pool]
        self.det_src = ROOT / "data_source" / f"coco250{suffix}"
        self.pan_src = ROOT / "data_source" / f"coco_panoptic250{suffix}"
        if not (self.pan_src / "panoptic.json").is_file():
            raise SystemExit(f"Chưa có dữ liệu {self.pan_src}. Chạy: python scripts/download_public.py" + (" --pool test" if suffix == "_test" else ""))
        self.opaque = args.opaque
        self.out = Path(args.out)
        self.truth: list[dict] = []
        self.used: set[str] = set()

    # ------------------------------------------------------------ nạp nguồn
    def load(self):
        doc = json.loads((self.pan_src / "panoptic.json").read_text(encoding="utf-8"))
        self.categories = doc["categories"]
        self.cat_by_id = {c["id"]: c for c in self.categories}
        self.cat_by_name = {c["name"]: c["id"] for c in self.categories}
        self.things = {c["id"] for c in self.categories if c.get("isthing")}
        images = sorted(doc["images"], key=lambda im: im["id"])
        anns = {a["image_id"]: json.loads(json.dumps(a)) for a in doc["annotations"]}
        order = list(range(len(images)))
        if self.opaque:
            self.rng.shuffle(order)
        self.items = []
        for new_idx, src_idx in enumerate(order):
            im = images[src_idx]
            stem = Path(im["file_name"]).stem
            name = f"img_{new_idx + 1:04d}" if self.opaque else stem
            self.items.append({
                "name": name, "stem": stem, "width": im["width"], "height": im["height"],
                "image_id": new_idx + 1 if self.opaque else im["id"],
                "jpg": (self.det_src / "images" / "train2017" / f"{stem}.jpg").read_bytes(),
                "boxes": dedupe(read_yolo((self.det_src / "labels" / "train2017" / f"{stem}.txt").read_text(encoding="utf-8"))),
                "ann": anns[im["id"]],
                "mask": pngutil.read_rgb(self.pan_src / "masks" / Path(anns[im["id"]]["file_name"]).name),
            })
        if self.opaque:
            self.items.sort(key=lambda it: it["name"])
            for it in self.items:
                it["jpg"] = perturb_jpg(it["jpg"], self.rng)
                self.remap_segments(it)
        self.by_name = {it["name"]: it for it in self.items}

    def remap_segments(self, it):
        """Đánh số lại segment id (đổi cả màu trong mask) để mask không trùng byte với COCO."""
        width, height, pixels = it["mask"]
        old_ids = [s["id"] for s in it["ann"]["segments_info"]]
        new_ids = self.rng.sample(range(1000, 16_000_000), len(old_ids))
        mapping = dict(zip(old_ids, new_ids))
        out = bytearray(pixels)
        for i in range(0, len(out), 3):
            sid = pngutil.segment_id(out, i)
            if sid in mapping:
                out[i:i + 3] = bytes(pngutil.segment_rgb(mapping[sid]))
        for s in it["ann"]["segments_info"]:
            s["id"] = mapping[s["id"]]
        it["mask"] = (width, height, out)

    # ------------------------------------------------------------ cảnh, job, guideline
    def organize(self):
        rng = self.rng
        n_scenes = (len(self.items) + SCENE_SIZE - 1) // SCENE_SIZE
        scene_codes = rng.sample(range(100, 999), n_scenes)
        # Ảnh của một cảnh là các ảnh liền nhau theo thứ tự nguồn (giống một đoạn quay).
        src_order = sorted(self.items, key=lambda it: it["stem"])
        for idx, it in enumerate(src_order):
            it["scene"] = f"scene_{scene_codes[idx // SCENE_SIZE]}"
        scenes = sorted({it["scene"] for it in self.items})
        val = set(rng.sample(scenes, round(len(scenes) * 0.3)))
        for it in self.items:
            it["split"] = "val" if it["scene"] in val else "train"
        # Job = các cảnh liền nhau; job đầu theo guideline cũ v1.2, các job sau v1.3.
        self.jobs = []
        n_jobs = (len(src_order) + JOB_SIZE - 1) // JOB_SIZE
        old_jobs = set(range(rng.randint(3, 4)))
        annot = ANNOTATORS * 4
        rng.shuffle(annot)
        for j in range(n_jobs):
            reviewer = rng.choice([q for q in REVIEWERS])
            self.jobs.append({"job_id": f"J{j + 1:02d}", "annotator": annot[j], "reviewer": reviewer,
                              "guideline": "v1.2" if j in old_jobs else "v1.3"})
        for idx, it in enumerate(src_order):
            job = self.jobs[idx // JOB_SIZE]
            it["job"] = job["job_id"]
            it["guideline"] = job["guideline"]
        self.job_by_id = {j["job_id"]: j for j in self.jobs}

    # ------------------------------------------------------------ helper lỗi
    def free(self, pred=lambda it: True):
        return [it for it in self.items if it["name"] not in self.used and pred(it)]

    def mark(self, it, kind, **extra):
        self.used.add(it["name"])
        self.truth.append({"type": kind, "target": it["name"], "unit": "image", **extra})

    def box_px(self, it, row):
        _, x, y, w, h = row
        W, H = it["width"], it["height"]
        x, y, w, h = float(x), float(y), float(w), float(h)
        return [(x - w / 2) * W, (y - h / 2) * H, w * W, h * H]

    def name80(self, cls: str) -> str:
        return self.cat_by_id[COCO80[int(cls)]]["name"]

    def partner(self, name: str):
        for group in CONFUSABLE:
            if name in group:
                return [g for g in group if g != name and g in self.cat_by_name]
        return []

    # ------------------------------------------------------------ lỗi detection
    def det_faults(self, kind):
        rng = self.rng
        if kind == "missing_image":
            it = rng.choice(self.free())
            it["missing"] = True
            self.mark(it, kind)
        elif kind == "wrong_hash":
            it = rng.choice(self.free())
            other = rng.choice([o for o in self.items if o is not it])
            it["fake_hash"] = sha(other["jpg"])  # mã băm hợp lệ, nhưng của ảnh khác
            self.mark(it, kind)
        elif kind == "invalid_class":
            it = rng.choice(self.free(lambda i: i["boxes"]))
            row = rng.choice(it["boxes"])
            row[0] = "80"  # lệch một: 0..79 là hợp lệ
            self.mark(it, kind)
        elif kind == "bbox_out_of_bounds":
            def ok(i):
                return any(float(r[1]) + float(r[3]) / 2 > 0.9 for r in i["boxes"])
            it = rng.choice(self.free(ok))
            row = max(it["boxes"], key=lambda r: float(r[1]) + float(r[3]) / 2)
            x, w = float(row[1]), float(row[3])
            over = rng.uniform(0.02, 0.08)
            right = 1.0 + over
            left = x - w / 2
            row[1], row[3] = f"{(left + right) / 2:.6f}", f"{right - left:.6f}"
            self.mark(it, kind)
        elif kind == "duplicate_box":
            it = rng.choice(self.free(lambda i: i["boxes"]))
            row = list(rng.choice(it["boxes"]))
            x, y, w, h = map(float, row[1:5])
            # lệch 1-2 pixel, giữ nguyên kích thước, không tràn mép (chỉ mang một lỗi)
            x = min(1 - w / 2, max(w / 2, x + rng.choice([-2, -1, 1, 2]) / it["width"]))
            y = min(1 - h / 2, max(h / 2, y + rng.choice([-2, -1, 1, 2]) / it["height"]))
            it["boxes"].insert(rng.randrange(len(it["boxes"]) + 1), [row[0], f"{x:.6f}", f"{y:.6f}", row[3], row[4]])
            self.mark(it, kind)
        elif kind == "scene_leak":
            it = rng.choice(self.free())
            it["split"] = "val" if it["split"] == "train" else "train"
            self.used.add(it["name"])
            self.truth.append({"type": kind, "target": it["scene"], "unit": "scene", "image": it["name"]})
        elif kind == "guideline_mismatch":
            it = rng.choice(self.free())
            it["guideline"] = "v1.2" if it["guideline"] == "v1.3" else "v1.3"
            self.mark(it, kind)
        elif kind == "class_swap":
            big = self.p["min_box_area"]
            conf = rng.random() < self.p["swap_confusable"]

            def eligible(r):
                return float(r[3]) * float(r[4]) >= big and (not conf or self.partner(self.name80(r[0])))
            it = rng.choice(self.free(lambda i: any(eligible(r) for r in i["boxes"])))
            row = rng.choice([r for r in it["boxes"] if eligible(r)])
            before = self.name80(row[0])
            pool = self.partner(before) if conf else [n for n in (self.cat_by_id[c]["name"] for c in COCO80) if n != before]
            after = rng.choice(pool)
            row[0] = str(COCO80.index(self.cat_by_name[after]))
            self.mark(it, kind, from_class=before, to_class=after)
        elif kind == "bbox_drift":
            import math

            def ok_box(r):
                return float(r[3]) > 0.08 and float(r[4]) > 0.08 and float(r[3]) * float(r[4]) >= self.p["min_box_area"]
            it = rng.choice(self.free(lambda i: any(ok_box(r) for r in i["boxes"])))
            row = rng.choice([r for r in it["boxes"] if ok_box(r)])
            x, y, w, h = map(float, row[1:5])
            ang = rng.uniform(0, 2 * math.pi)  # hướng bất kỳ, không chỉ ngang
            mag = rng.uniform(*self.p["drift"])
            if rng.random() < 0.5:             # một nửa số ca đổi cỡ box
                s = 1 + rng.uniform(-self.p["drift_scale"], self.p["drift_scale"])
                w, h = min(1.0, w * s), min(1.0, h * s)
            x = min(1 - w / 2, max(w / 2, x + math.cos(ang) * mag * w))
            y = min(1 - h / 2, max(h / 2, y + math.sin(ang) * mag * h))
            row[1:5] = [f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]
            self.mark(it, kind)
        elif kind == "missing_box":
            def ok(i):
                return len(i["boxes"]) >= 2 and any(float(r[3]) * float(r[4]) > 0.01 for r in i["boxes"])
            it = rng.choice(self.free(ok))
            row = rng.choice([r for r in it["boxes"] if float(r[3]) * float(r[4]) > 0.01])
            it["boxes"].remove(row)
            self.mark(it, kind, removed_class=self.name80(row[0]))

    # ------------------------------------------------------------ lỗi panoptic
    def segs(self, it):
        return [s for s in it["ann"]["segments_info"] if s["category_id"] in self.things and not s.get("iscrowd")]

    def counts(self, it):
        width, _, pixels = it["mask"]
        found = {}
        for i in range(0, len(pixels), 3):
            sid = pngutil.segment_id(pixels, i)
            if sid:
                x, y = (i // 3) % width, (i // 3) // width
                f = found.setdefault(sid, [0, x, y, x, y])
                f[0] += 1
                f[1], f[2], f[3], f[4] = min(f[1], x), min(f[2], y), max(f[3], x), max(f[4], y)
        return found

    def pan_faults(self, kind):
        rng = self.rng
        if kind == "invalid_category_id":
            it = rng.choice(self.free(lambda i: self.segs(i)))
            gaps = [c for c in range(1, max(self.cat_by_id)) if c not in self.cat_by_id]
            seg = rng.choice(self.segs(it))
            seg["category_id"] = rng.choice(gaps)  # id nằm trong khoảng trống của danh sách lớp
            self.mark(it, kind)
        elif kind == "thing_overlap":
            cands = self.free(lambda i: len(self.segs(i)) >= 2)
            rng.shuffle(cands)
            for it in cands:
                stats = self.counts(it)
                viable = [s for s in self.segs(it) if stats.get(s["id"], [0])[0] >= 30]
                if len(viable) < 2:
                    continue
                donor, receiver = rng.sample(viable, 2)
                width, height, pixels = it["mask"]
                pixels = bytearray(pixels)
                offs = [i for i in range(0, len(pixels), 3) if pngutil.segment_id(pixels, i) == donor["id"]]
                steal = max(1, int(len(offs) * rng.uniform(0.15, 0.3)))
                for i in offs[:steal]:
                    pixels[i:i + 3] = bytes(pngutil.segment_rgb(receiver["id"]))
                it["mask"] = (width, height, pixels)
                new = self.counts(it)[receiver["id"]]
                receiver["area"] = new[0]
                receiver["bbox"] = [new[1], new[2], new[3] - new[1] + 1, new[4] - new[2] + 1]
                self.mark(it, kind)
                return
            raise SystemExit("không đủ ảnh cho thing_overlap")
        elif kind == "category_swap":
            conf = rng.random() < self.p["swap_confusable"]

            def eligible(s):
                return s.get("area", 0) >= self.p["min_segment_px"] and (not conf or self.partner(self.cat_by_id[s["category_id"]]["name"]))
            it = rng.choice(self.free(lambda i: any(eligible(s) for s in self.segs(i))))
            seg = rng.choice([s for s in self.segs(it) if eligible(s)])
            before = self.cat_by_id[seg["category_id"]]["name"]
            pool = self.partner(before) if conf else sorted(self.cat_by_id[c]["name"] for c in self.things if self.cat_by_id[c]["name"] != before)
            after = rng.choice(pool)
            seg["category_id"] = self.cat_by_name[after]
            self.mark(it, kind, from_class=before, to_class=after)
        elif kind == "instance_merge":
            def groups(i):
                g = {}
                for s in self.segs(i):
                    g.setdefault(s["category_id"], []).append(s)
                return [v for v in g.values() if len(v) >= 2 and sorted(s.get("area", 0) for s in v)[0] >= self.p["min_segment_px"] // 2]
            it = rng.choice(self.free(lambda i: groups(i)))
            pair = sorted(rng.choice(groups(it)), key=lambda s: s.get("area", 0))
            removed, kept = pair[0], pair[-1]
            width, height, pixels = it["mask"]
            pixels = bytearray(pixels)
            for i in range(0, len(pixels), 3):
                if pngutil.segment_id(pixels, i) == removed["id"]:
                    pixels[i:i + 3] = bytes(pngutil.segment_rgb(kept["id"]))
            it["mask"] = (width, height, pixels)
            new = self.counts(it)[kept["id"]]
            kept["area"] = new[0]
            kept["bbox"] = [new[1], new[2], new[3] - new[1] + 1, new[4] - new[2] + 1]
            it["ann"]["segments_info"] = [s for s in it["ann"]["segments_info"] if s["id"] != removed["id"]]
            self.mark(it, kind)

    def inject(self):
        kinds = []
        for kind, (lo, hi) in self.p["counts"].items():
            kinds += [kind] * self.rng.randint(lo, hi)
        self.rng.shuffle(kinds)
        for kind in kinds:
            if kind in DET_RULE or kind in DET_CROSS:
                self.det_faults(kind)
            else:
                self.pan_faults(kind)

    # ------------------------------------------------------------ câu chuyện truy vết
    def story(self):
        rng = self.rng
        old = [it for it in self.items if self.job_by_id[it["job"]]["guideline"] == "v1.2"]
        freq = {}
        for it in old:
            for name in {self.name80(r[0]) for r in it["boxes"] if r[0].isdigit() and int(r[0]) < 80}:
                freq[name] = freq.get(name, 0) + 1
        old_jobs = sorted({it["job"] for it in old})

        def jobs_with(name):
            return {it["job"] for it in old if name in {self.name80(r[0]) for r in it["boxes"] if int(r[0]) < 80}}
        base = sorted(n for n, c in freq.items() if 6 <= c and n != "person") or sorted(freq)
        # ưu tiên lớp KHÔNG có mặt ở mọi job cũ: báo hết người của mọi job cũ sẽ bị trừ điểm
        choices = [n for n in base if len(jobs_with(n)) < len(old_jobs)] or base
        self.rule_class = rng.choice(choices)
        has = [it for it in old if self.rule_class in {self.name80(r[0]) for r in it["boxes"] if int(r[0]) < 80}]
        scenes = sorted({it["scene"] for it in has})
        self.reported = sorted(rng.sample(scenes, min(2, len(scenes))))
        tier_a = sorted(it["name"] for it in has if it["scene"] in self.reported)
        tier_b = sorted(it["name"] for it in has if it["scene"] not in self.reported)
        jobs = {self.by_name[n]["job"] for n in tier_a + tier_b}
        notify = sorted({self.job_by_id[j]["annotator"] for j in jobs} | {self.job_by_id[j]["reviewer"] for j in jobs})
        self.trace_truth = {"seed": self.seed, "rule_class": self.rule_class, "reported_scenes": self.reported,
                            "tier_a": tier_a, "tier_b": tier_b, "notify": notify}

    # ------------------------------------------------------------ ghi ra đĩa
    def write(self):
        if self.out.exists():
            shutil.rmtree(self.out)
        d = self.out / "delivery"
        for sub in ("images", "labels", "masks", "guideline", "incoming"):
            (d / sub).mkdir(parents=True)
        manifest = {}
        anns, imgs = [], []
        for it in self.items:
            jpg_name, label_name, mask_name = f"{it['name']}.jpg", f"{it['name']}.txt", f"{it['name']}.png"
            if not it.get("missing"):
                (d / "images" / jpg_name).write_bytes(it["jpg"])
            label = write_yolo(it["boxes"]).encode()
            (d / "labels" / label_name).write_bytes(label)
            width, height, pixels = it["mask"]
            pngutil.write_rgb(d / "masks" / mask_name, width, height, bytes(pixels))
            manifest[it["name"]] = {
                "image": f"images/{jpg_name}", "label": f"labels/{label_name}", "mask": f"masks/{mask_name}",
                "image_sha256": it.get("fake_hash") or sha(it["jpg"]),
                "label_sha256": sha(label),
                "mask_sha256": sha((d / "masks" / mask_name).read_bytes()),
            }
            ann = it["ann"]
            ann["image_id"], ann["file_name"] = it["image_id"], mask_name
            anns.append(ann)
            imgs.append({"id": it["image_id"], "file_name": jpg_name, "width": it["width"], "height": it["height"]})
        names = [self.cat_by_id[c]["name"] for c in COCO80]
        (d / "classes.txt").write_text("".join(f"{i} {n}\n" for i, n in enumerate(names)), encoding="utf-8")
        (d / "panoptic.json").write_text(json.dumps({"categories": self.categories, "images": imgs, "annotations": anns}) + "\n", encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({"dataset": "lab250", "num_classes": 80, "items": manifest}, indent=1) + "\n", encoding="utf-8")
        meta = [{"image": it["name"], "scene_id": it["scene"], "split": it["split"], "job_id": it["job"],
                 "guideline_version": it["guideline"]} for it in self.items]
        (d / "metadata.json").write_text(json.dumps({"note": "scene_id là giả lập cho bài lab", "items": meta}, indent=1) + "\n", encoding="utf-8")
        with (d / "jobs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["job_id", "annotator", "reviewer", "guideline"])
            w.writeheader()
            w.writerows(self.jobs)
        (d / "guideline" / "changelog.md").write_text(
            "# Changelog guideline\n\n"
            "## v1.3\n"
            f"- Lớp `{self.rule_class}`: vật bị che trên 50% hoặc cắt ở mép ảnh **vẫn phải gán box** "
            "(v1.2 cho phép bỏ qua). Áp dụng cho mọi job tạo sau khi v1.3 ban hành.\n"
            "- Không đổi quy tắc của lớp khác.\n\n"
            "## v1.2\n- Bản dùng cho các job đầu của lô.\n", encoding="utf-8")
        (d / "incoming" / "ticket.md").write_text(
            "# Báo lỗi từ team AI\n\n"
            f"Model train trên lô này hay bỏ sót `{self.rule_class}` bị che hoặc ở mép ảnh. "
            f"Bọn mình xem tay thấy rõ ở {' và '.join(self.reported)}: nhiều `{self.rule_class}` bị che không có box. "
            "Nhờ team dữ liệu kiểm tra và cho biết còn ảnh nào khác bị như vậy không.\n", encoding="utf-8")
        release_id = "lab250_" + sha((d / "manifest.json").read_bytes() + (d / "metadata.json").read_bytes())[:12]
        (d / "release.json").write_text(json.dumps({"release_id": release_id, "version": "v1",
                                                     "panoptic_sha256": sha((d / "panoptic.json").read_bytes()),
                                                     "note": "Ảnh COCO val2017; scene, job, người gán là giả lập."}, indent=2) + "\n", encoding="utf-8")
        p = self.out / "_private"
        p.mkdir()
        (p / "truth.json").write_text(json.dumps({"seed": self.seed, "task": "release", "errors": self.truth}, indent=2) + "\n", encoding="utf-8")
        (p / "trace_truth.json").write_text(json.dumps(self.trace_truth, indent=2) + "\n", encoding="utf-8")
        return release_id


def main() -> int:
    ap = argparse.ArgumentParser(description="Tạo lô bàn giao có lỗi gài theo seed")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="practice", choices=["practice", "calib", "test"])
    ap.add_argument("--opaque", action="store_true", help="đề ẩn: đổi tên, đánh số lại id, đổi byte ảnh/mask")
    ap.add_argument("--profile", help="JSON đổi mức lỗi (giảng viên dùng cho đề ẩn)")
    ap.add_argument("--dev-truth", action="store_true", help="in ground truth ra màn hình")
    ap.add_argument("--no-faults", action="store_true", help="lô sạch, không gài lỗi, trên pool calib (50 ảnh khác bộ luyện tập) để đo nhiễu tự nhiên")
    args = ap.parse_args()
    if args.no_faults and args.pool == "practice":
        args.pool = "calib"  # lô sạch KHÔNG dùng 250 ảnh luyện tập: diff với lô luyện tập sẽ lộ đáp án
    b = Builder(args)
    b.load()
    b.organize()
    if not args.no_faults:
        b.inject()
    b.story()
    rid = b.write()
    if args.dev_truth:
        for row in b.truth:
            print(row)
        print(b.trace_truth)
    print(f"seed={args.seed} pool={args.pool} images={len(b.items)} release_id={rid}")
    print(f"delivery: {b.out / 'delivery'}   (validator KHÔNG đọc {b.out / '_private'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
