"""Validator luật cho lô Day 10. Chỉ đọc thư mục delivery được truyền vào.

Chỉ dùng thư viện chuẩn; không mạng; không đọc _private, data_source, annotation COCO.

Cách dùng:
  python validator/validate.py <delivery> --out errors.json [--candidates candidates.json]

- errors.json: chín loại lỗi luật (xem docs/DELIVERY.md mục 2).
- candidates.json (tùy chọn): ứng viên lỗi gán nhãn từ đối chiếu detection <-> panoptic.
  Đây chỉ là GIẢ THUYẾT để người xem ảnh quyết định; không tự thành lỗi xác nhận.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pngutil  # noqa: E402  (thư viện chuẩn, đọc PNG)

NUM_CLASSES = 80
BOX_EDGE_TOL_PX = 1.0       # tràn mép tối đa coi là làm tròn
DUP_SIZE_TOL_PX = 1         # duplicate: cùng kích thước sai khác tối đa 1 px
DUP_CENTER_TOL_PX = 2       # duplicate: lệch tâm tối đa 2 px
AREA_TOL_ABS = 3            # thing_overlap: sai số tuyệt đối tối thiểu cho area
AREA_TOL_REL = 0.02         # thing_overlap: sai số tương đối
PAIR_MIN_MASK_IOU = 0.5     # cross-check: ghép box với segment khi IoU(mask, box) >= ngưỡng này
MISSING_BOX_MIN_FRAC = 0.005  # cross-check: segment thing không có box, diện tích tối thiểu (tỉ lệ ảnh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem(name: str) -> str:
    return Path(name).stem


def load_json(path: Path, errors_out: list, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[warn] không đọc được {path.name}: {exc}", file=sys.stderr)
        return default


def load_jobs(path: Path) -> dict[str, dict]:
    jobs: dict[str, dict] = {}
    if not path.exists():
        return jobs
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            jobs[row["job_id"].strip()] = row
    return jobs


def load_classes(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            names[int(parts[0])] = parts[1].strip()
    return names


def read_yolo(path: Path) -> list[tuple[int | None, list[float] | None]]:
    """Trả về danh sách (class_id, [xc, yc, w, h]); dòng hỏng cho (None, None)."""
    rows: list[tuple[int | None, list[float] | None]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:5]]
            if len(parts) != 5:
                raise ValueError
            rows.append((cls, vals))
        except (ValueError, IndexError):
            rows.append((None, None))
    return rows


def yolo_to_pixels(vals: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    xc, yc, w, h = vals
    return ((xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def segment_pixel_counts(pixels: bytes | bytearray) -> Counter:
    """Đếm pixel theo segment id = R + G*256 + B*65536, bỏ qua nền (id 0)."""
    mv = memoryview(bytes(pixels))
    counts = Counter(zip(mv[0::3], mv[1::3], mv[2::3]))
    out: Counter = Counter()
    for (r, g, b), n in counts.items():
        sid = r + g * 256 + b * 65536
        if sid:
            out[sid] += n
    return out


def check_rules(delivery: Path) -> list[dict]:
    errors: list[dict] = []

    def add(kind: str, target: str, why: str) -> None:
        errors.append({"type": kind, "target": target, "why": why})

    manifest = load_json(delivery / "manifest.json", errors, {"items": {}})
    metadata = load_json(delivery / "metadata.json", errors, {"items": []})
    panoptic = load_json(delivery / "panoptic.json", errors, {"categories": [], "images": [], "annotations": []})
    jobs = load_jobs(delivery / "jobs.csv")
    class_names = load_classes(delivery / "classes.txt")
    items = manifest.get("items", {})
    meta_items = metadata.get("items", [])

    image_size = {stem(im["file_name"]): (im["width"], im["height"]) for im in panoptic.get("images", [])}
    category_ids = {c["id"] for c in panoptic.get("categories", [])}

    # --- File: missing_image, wrong_hash (so với manifest, tính sha256 thật) ---
    for name, entry in sorted(items.items()):
        for kind, path_key, hash_key in (
            ("image", "image", "image_sha256"),
            ("label", "label", "label_sha256"),
            ("mask", "mask", "mask_sha256"),
        ):
            rel = entry.get(path_key)
            if not rel:
                continue
            path = delivery / rel
            if not path.is_file():
                add("missing_image", name, f"thiếu file {kind}: {rel}")
                break
            expected = entry.get(hash_key)
            if expected and sha256_file(path) != expected:
                add("wrong_hash", name, f"sha256 {kind} không khớp manifest")
                break

    # --- Nhãn detection: invalid_class, bbox_out_of_bounds, duplicate_box ---
    for name, entry in sorted(items.items()):
        label_rel = entry.get("label")
        if not label_rel:
            continue
        size = image_size.get(name)
        rows = read_yolo(delivery / label_rel)
        pixel_boxes: list[tuple[int, tuple[float, float, float, float], tuple[float, float]]] = []
        bad_class = False
        bad_bounds = False
        for cls, vals in rows:
            if cls is None or not (0 <= cls < NUM_CLASSES):
                bad_class = True
                continue
            if size is None:
                continue
            width, height = size
            x1, y1, x2, y2 = yolo_to_pixels(vals, width, height)
            if x1 < -BOX_EDGE_TOL_PX or y1 < -BOX_EDGE_TOL_PX or x2 > width + BOX_EDGE_TOL_PX or y2 > height + BOX_EDGE_TOL_PX:
                bad_bounds = True
            pixel_boxes.append((cls, (x1, y1, x2, y2), (x2 - x1, y2 - y1)))
        if bad_class:
            add("invalid_class", name, "class_id ngoài 0–79 hoặc dòng nhãn hỏng")
        if bad_bounds:
            add("bbox_out_of_bounds", name, "box vượt ra ngoài ảnh")
        dup = False
        for i in range(len(pixel_boxes)):
            for j in range(i + 1, len(pixel_boxes)):
                ci, bi, si = pixel_boxes[i]
                cj, bj, sj = pixel_boxes[j]
                if ci != cj:
                    continue
                if abs(si[0] - sj[0]) > DUP_SIZE_TOL_PX or abs(si[1] - sj[1]) > DUP_SIZE_TOL_PX:
                    continue
                cxi, cyi = (bi[0] + bi[2]) / 2, (bi[1] + bi[3]) / 2
                cxj, cyj = (bj[0] + bj[2]) / 2, (bj[1] + bj[3]) / 2
                if abs(cxi - cxj) <= DUP_CENTER_TOL_PX and abs(cyi - cyj) <= DUP_CENTER_TOL_PX:
                    dup = True
        if dup:
            add("duplicate_box", name, "box lặp cùng lớp, cùng kích thước, lệch 1–2 px")

    # --- Metadata: scene_leak, guideline_mismatch ---
    splits_by_scene: dict[str, set] = defaultdict(set)
    for m in meta_items:
        splits_by_scene[m["scene_id"]].add(m["split"])
    for scene, splits in sorted(splits_by_scene.items()):
        if len(splits) > 1:
            add("scene_leak", scene, f"cảnh có ảnh ở nhiều split: {sorted(splits)}")

    for m in meta_items:
        job = jobs.get(m["job_id"])
        if job is None:
            continue
        if m["guideline_version"] != job["guideline"]:
            add("guideline_mismatch", m["image"],
                f"ảnh ghi {m['guideline_version']}, job {m['job_id']} là {job['guideline']}")

    # --- Panoptic: invalid_category_id, thing_overlap ---
    ann_by_stem = {stem(a["file_name"]): a for a in panoptic.get("annotations", [])}
    for name, entry in sorted(items.items()):
        ann = ann_by_stem.get(name)
        if ann is None:
            continue
        bad_cat = any(seg["category_id"] not in category_ids for seg in ann.get("segments_info", []))
        if bad_cat:
            add("invalid_category_id", name, "category_id không có trong categories")
        mask_rel = entry.get("mask")
        if not mask_rel or not (delivery / mask_rel).is_file():
            continue
        width, height, pixels = pngutil.read_rgb(delivery / mask_rel)
        counts = segment_pixel_counts(pixels)
        overlap = False
        for seg in ann.get("segments_info", []):
            area = seg.get("area", 0)
            actual = counts.get(seg["id"], 0)
            if area - actual > max(AREA_TOL_ABS, AREA_TOL_REL * area):
                overlap = True
        if overlap:
            add("thing_overlap", name, "area khai báo lớn hơn số pixel thật của segment")

    return errors


def cross_check_candidates(delivery: Path) -> list[dict]:
    """Ứng viên lỗi gán nhãn: detection <-> panoptic. Chỉ là giả thuyết cần người xem ảnh.

    Ghép mỗi box detection với segment thing có tỉ lệ pixel nằm trong box cao nhất.
    - Khác tên lớp: class_swap (nếu box sai) hoặc category_swap (nếu segment sai).
    - Segment thing có diện tích đáng kể mà không có box nào bao phủ: missing_box.
    - Một segment bao hai box cùng lớp: instance_merge.
    - Cùng lớp nhưng bbox lệch khỏi mask: bbox_drift.
    """
    manifest = load_json(delivery / "manifest.json", [], {"items": {}})
    panoptic = load_json(delivery / "panoptic.json", [], {"categories": [], "images": [], "annotations": []})
    class_names = load_classes(delivery / "classes.txt")
    items = manifest.get("items", {})
    image_size = {stem(im["file_name"]): (im["width"], im["height"]) for im in panoptic.get("images", [])}
    cat_name = {c["id"]: c["name"] for c in panoptic.get("categories", []) if c.get("isthing", 1)}
    ann_by_stem = {stem(a["file_name"]): a for a in panoptic.get("annotations", [])}
    candidates: list[dict] = []

    for name, entry in sorted(items.items()):
        size = image_size.get(name)
        ann = ann_by_stem.get(name)
        if size is None or ann is None or not entry.get("mask"):
            continue
        width, height = size
        rows = read_yolo(delivery / entry["label"])
        boxes = [(cls, yolo_to_pixels(vals, width, height)) for cls, vals in rows if cls is not None and vals]
        mask_path = delivery / entry["mask"]
        if not mask_path.is_file():
            continue
        _, _, pixels = pngutil.read_rgb(mask_path)
        mv = memoryview(bytes(pixels))
        # Pixel-level: segment id cho từng pixel (bỏ crowd/nền ở bước sau)
        seg_pixels: dict[int, list[int]] = defaultdict(list)
        for idx, (r, g, b) in enumerate(zip(mv[0::3], mv[1::3], mv[2::3])):
            sid = r + g * 256 + b * 65536
            if sid:
                seg_pixels[sid].append(idx)
        seg_info = {s["id"]: s for s in ann.get("segments_info", []) if cat_name.get(s["category_id"])}

        def inside(idx: int, box: tuple[float, float, float, float]) -> bool:
            x, y = idx % width, idx // width
            return box[0] <= x < box[2] and box[1] <= y < box[3]

        covered_seg: set[int] = set()
        for cls, box in boxes:
            det_name = class_names.get(cls, "")
            box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            best_sid, best_frac = None, 0.0
            for sid, info in seg_info.items():
                px = seg_pixels.get(sid, [])
                if not px:
                    continue
                inter = sum(1 for p in px if inside(p, box))
                # IoU mask-vs-box: segment nhỏ nằm gọn trong box lớn không được ghép
                mask_iou = inter / (len(px) + box_area - inter)
                if mask_iou > best_frac:
                    best_sid, best_frac = sid, mask_iou
            if best_sid is None or best_frac < PAIR_MIN_MASK_IOU:
                continue
            covered_seg.add(best_sid)
            seg_name = cat_name[seg_info[best_sid]["category_id"]]
            seg_box = seg_info[best_sid]["bbox"]
            seg_pix_box = (seg_box[0], seg_box[1], seg_box[0] + seg_box[2], seg_box[1] + seg_box[3])
            if seg_name != det_name:
                candidates.append({"type": "class_swap", "target": name, "score": round(best_frac, 3),
                                   "evidence": f"box={det_name}, segment={seg_name}"})
            elif iou(box, seg_pix_box) < 0.5:
                candidates.append({"type": "bbox_drift", "target": name, "score": round(1 - iou(box, seg_pix_box), 3),
                                   "evidence": f"lớp {det_name}, IoU box/segment thấp"})

        # Segment được coi là "có box" nếu có ÍT NHẤT một box chứa >= 50% pixel của nó
        # (không chỉ box được ghép IoU). Tránh báo missing_box cho vật đã có box lệch nhẹ.
        for sid, info in seg_info.items():
            px = seg_pixels.get(sid, [])
            if px and any(sum(1 for p in px if inside(p, b)) >= 0.5 * len(px) for _, b in boxes):
                covered_seg.add(sid)
        for sid, info in seg_info.items():
            if sid not in covered_seg and info["area"] >= MISSING_BOX_MIN_FRAC * width * height:
                candidates.append({"type": "missing_box", "target": name, "score": round(info["area"] / (width * height), 4),
                                   "evidence": f"segment {cat_name[info['category_id']]} lớn không có box"})

    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description="Validator luật Day 10 (chỉ thư viện chuẩn)")
    ap.add_argument("delivery", help="thư mục delivery (chỉ đọc)")
    ap.add_argument("--out", required=True, help="đường dẫn errors.json")
    ap.add_argument("--candidates", help="ghi thêm candidates.json (ứng viên lỗi gán nhãn)")
    args = ap.parse_args()

    delivery = Path(args.delivery)
    if not (delivery / "manifest.json").is_file():
        print(f"Lỗi: không thấy manifest.json trong {delivery}", file=sys.stderr)
        return 2

    rules = check_rules(delivery)
    out = {"errors": [{"type": e["type"], "target": e["target"]} for e in rules]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"errors": sorted(out["errors"], key=lambda e: (e["type"], e["target"]))}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    counts = Counter(e["type"] for e in rules)
    print(f"lỗi luật: {len(rules)} -> {args.out}")
    for kind in sorted(counts):
        print(f"  {kind}: {counts[kind]}")

    if args.candidates:
        cands = cross_check_candidates(delivery)
        cands.sort(key=lambda c: (c["type"], c["target"]))
        Path(args.candidates).write_text(json.dumps({"candidates": cands}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ứng viên gán nhãn: {len(cands)} -> {args.candidates}")
        for kind, n in sorted(Counter(c["type"] for c in cands).items()):
            print(f"  {kind}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
