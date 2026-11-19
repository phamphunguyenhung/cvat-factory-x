#!/usr/bin/env python3
"""Download the lab datasets. Python 3.10+, standard library only.

  python scripts/download_public.py
  python scripts/download_public.py --pool test        # giảng viên
  python scripts/download_public.py --dataset coco128

coco250 is the lab set: 250 COCO val2017 images, YOLO detection labels, and the
matching panoptic masks. The ids are fixed: images that have a non-crowd box and
a panoptic annotation, sorted by image id. `--pool practice` (default) takes ids
1-250 of that list; `--pool test` takes ids 251-500 and is for the instructor's
hidden test delivery. coco128 remains available as the small Ultralytics zip.

Images download in parallel (16 threads) with retries; the mask zip is opened once.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
import urllib.request
import zipfile
from pathlib import Path

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
INSTANCES_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PANOPTIC_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{name}"
SAMPLE_COUNT = 250
POOLS = {"practice": (0, 250), "test": (250, 500), "calib": (500, 550)}
SUFFIX = {"practice": "", "test": "_test", "calib": "_calib"}
# Official 80 instance categories, in the order used by YOLO COCO labels.
COCO80 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print("cached", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print("downloading", url)
    request = urllib.request.Request(url, headers={"User-Agent": "day10-lab"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    partial.replace(dest)


def zip_names(archive: Path) -> dict[str, str]:
    with zipfile.ZipFile(archive) as zipped:
        return {Path(name).name: name for name in zipped.namelist() if not name.endswith("/")}


def member(archive: Path, suffix: str) -> str:
    with zipfile.ZipFile(archive) as zipped:
        matches = [
            name for name in zipped.namelist()
            if name.endswith(suffix) and not name.endswith("/") and "__MACOSX" not in name
        ]
    if len(matches) != 1:
        raise SystemExit(f"{archive} has {len(matches)} members ending in {suffix}: {matches}")
    return matches[0]


def read_zip_json(archive: Path, suffix: str) -> dict:
    name = member(archive, suffix)
    with zipfile.ZipFile(archive) as zipped:
        return json.loads(zipped.read(name))


def extract_zip_member(archive: Path, member_name: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo(member_name)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise SystemExit(f"symlink rejected: {member_name}")
        with zipped.open(info) as source, dest.open("wb") as handle:
            shutil.copyfileobj(source, handle)


def yolo_lines(image: dict, annotations: list[dict], class_of: dict[int, int]) -> list[str]:
    width = image["width"]
    height = image["height"]
    lines = []
    for ann in annotations:
        if ann.get("iscrowd"):
            continue
        class_id = class_of.get(ann["category_id"])
        if class_id is None:
            continue
        x, y, box_w, box_h = [float(value) for value in ann["bbox"]]
        x = min(max(x, 0.0), width - 1)
        y = min(max(y, 0.0), height - 1)
        box_w = min(box_w, width - x)
        box_h = min(box_h, height - y)
        if box_w < 1 or box_h < 1:
            continue
        lines.append(
            f"{class_id} {(x + box_w / 2) / width:.6f} {(y + box_h / 2) / height:.6f} "
            f"{box_w / width:.6f} {box_h / height:.6f}"
        )
    return lines


def choose_ids(instances: dict, panoptic: dict, pool: str = "practice") -> list[int]:
    boxed = set()
    for ann in instances["annotations"]:
        if not ann.get("iscrowd"):
            boxed.add(ann["image_id"])
    panoptic_ids = {ann["image_id"] for ann in panoptic["annotations"]}
    start, stop = POOLS[pool]
    chosen = sorted(boxed & panoptic_ids)[start:stop]
    if len(chosen) < stop - start:
        raise SystemExit(f"only {len(chosen)} images have both a box and a panoptic mask")
    return chosen


def link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, dest)


def download_coco128(root: Path) -> None:
    archive = root / "cache" / "coco128.zip"
    target = root / "coco128"
    download(COCO128_URL, archive)
    if target.exists() and list(target.rglob("*.jpg")) and list(target.rglob("*.txt")):
        print("already extracted", target)
        return
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            dest = (root / info.filename).resolve()
            if dest != root and root not in dest.parents:
                raise SystemExit(f"unsafe zip path: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise SystemExit(f"symlink rejected: {info.filename}")
        zipped.extractall(root)
    images = list(target.rglob("*.jpg"))
    labels = list(target.rglob("*.txt"))
    if len(images) < 100 or len(labels) < 100:
        raise SystemExit(f"unexpected coco128 layout: {len(images)} images / {len(labels)} labels")
    print(f"ready coco128: {len(images)} images, {len(labels)} labels")


def download_coco250(root: Path, pool: str = "practice") -> None:
    suffix_, count_ = SUFFIX[pool], POOLS[pool][1] - POOLS[pool][0]
    done_dir = root / f"coco_panoptic250{suffix_}"
    if (done_dir / "panoptic.json").is_file() and len(list((done_dir / "masks").glob("*.png"))) >= count_ \
            and len(list((root / f"coco250{suffix_}" / "images" / "train2017").glob("*.jpg"))) >= count_:
        print(f"đã có coco250{suffix_} ({count_} ảnh), bỏ qua")
        return
    cache = root / "cache"
    instances_zip = cache / "annotations_trainval2017.zip"
    panoptic_zip = cache / "panoptic_annotations_trainval2017.zip"
    download(INSTANCES_URL, instances_zip)
    download(PANOPTIC_URL, panoptic_zip)
    instances = read_zip_json(instances_zip, "instances_val2017.json")
    panoptic = read_zip_json(panoptic_zip, "panoptic_val2017.json")
    mask_zip = cache / "panoptic_val2017.zip"
    extract_zip_member(panoptic_zip, member(panoptic_zip, "panoptic_val2017.zip"), mask_zip)
    chosen = choose_ids(instances, panoptic, pool)
    by_id = {image["id"]: image for image in instances["images"]}
    panoptic_by_id = {ann["image_id"]: ann for ann in panoptic["annotations"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in instances["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    class_of = {category_id: index for index, category_id in enumerate(COCO80)}
    panoptic_members = zip_names(mask_zip)

    suffix = SUFFIX[pool]
    count = POOLS[pool][1] - POOLS[pool][0]
    detect = root / f"coco250{suffix}"
    images_dir = detect / "images" / "train2017"
    labels_dir = detect / "labels" / "train2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    panoptic_root = root / f"coco_panoptic250{suffix}"
    mask_dir = panoptic_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    todo = [by_id[i] for i in chosen if not (images_dir / by_id[i]["file_name"]).is_file()]
    with ThreadPoolExecutor(max_workers=16) as pool_:
        for done, _ in enumerate(pool_.map(lambda im: download(IMAGE_URL.format(name=im["file_name"]), images_dir / im["file_name"]), todo), start=1):
            if done % 50 == 0 or done == len(todo):
                print(f"images {done}/{len(todo)}", flush=True)
    with zipfile.ZipFile(mask_zip) as zmask:
        for index, image_id in enumerate(chosen, start=1):
            image = by_id[image_id]
            jpg = images_dir / image["file_name"]
            label = labels_dir / f"{jpg.stem}.txt"
            if not label.is_file():
                label.write_text("\n".join(yolo_lines(image, anns_by_image.get(image_id, []), class_of)) + "\n", encoding="utf-8")
            mask_name = Path(panoptic_by_id[image_id]["file_name"]).name
            mask = mask_dir / mask_name
            if mask_name not in panoptic_members:
                raise SystemExit(f"panoptic zip has no {mask_name}")
            if not (mask.is_file() and mask.stat().st_size > 0):
                info = zmask.getinfo(panoptic_members[mask_name])
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise SystemExit(f"symlink rejected: {mask_name}")
                with zmask.open(info) as src, mask.open("wb") as handle:
                    shutil.copyfileobj(src, handle)
            link_or_copy(jpg, panoptic_root / "images" / jpg.name)
            if index % 50 == 0 or index == count:
                print(f"coco250{suffix} {index}/{count}", flush=True)

    chosen_set = set(chosen)
    subset = {
        "info": panoptic.get("info", {}),
        "licenses": panoptic.get("licenses", []),
        "categories": panoptic["categories"],
        "images": [image for image in panoptic["images"] if image["id"] in chosen_set],
        "annotations": [panoptic_by_id[image_id] for image_id in chosen],
    }
    selection = {
        "source": "COCO val2017",
        "task_detection": "YOLO labels converted from instances_val2017.json",
        "task_panoptic": "COCO panoptic val2017",
        "count": count,
        "rule": f"Image ids present in both non-crowd instance boxes and panoptic_val2017, sorted ascending, positions {POOLS[pool][0] + 1}-{POOLS[pool][1]}.",
        "image_ids": chosen,
        "detection_dir": f"data_source/coco250{suffix}",
        "panoptic_dir": f"data_source/coco_panoptic250{suffix}",
        "image_terms": "https://cocodataset.org/dataset/termsofuse.htm",
        "annotation_license": "CC BY 4.0",
    }
    (detect / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    (panoptic_root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    (panoptic_root / "panoptic.json").write_text(json.dumps(subset) + "\n", encoding="utf-8")
    jpg_count = len(list(images_dir.glob("*.jpg")))
    txt_count = len(list(labels_dir.glob("*.txt")))
    png_count = len(list(mask_dir.glob("*.png")))
    if jpg_count < count or txt_count < count or png_count < count:
        raise SystemExit(f"incomplete coco250{suffix}: {jpg_count} jpg, {txt_count} txt, {png_count} png")
    print(f"ready coco250{suffix}: {jpg_count} images, {txt_count} labels")
    print(f"ready coco_panoptic250{suffix}: {png_count} masks, panoptic.json")
    print("Image rights: https://cocodataset.org/dataset/termsofuse.htm")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public lab dataset")
    parser.add_argument("--dataset", default="coco250", choices=["coco250", "coco128"])
    parser.add_argument("--start", type=int, help="(giảng viên) vị trí bắt đầu của pool test trong danh sách ảnh hợp lệ")
    parser.add_argument("--pool", default="practice", choices=sorted(POOLS), help="practice (mặc định, tải kèm calib) = 250 ảnh luyện tập; calib = 50 ảnh khác để đo nhiễu lô sạch; test = đề ẩn (giảng viên)")
    parser.add_argument("--root", default="data_source")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "coco128":
        download_coco128(root)
    else:
        if args.start is not None and args.pool == "test":
            if args.start < 550:
                raise SystemExit("--start phải >= 550 (không trùng ảnh luyện tập và calib)")
            POOLS["test"] = (args.start, args.start + 250)
        pools = ["practice", "calib"] if args.pool == "practice" else [args.pool]
        for pool in pools:
            download_coco250(root, pool)


if __name__ == "__main__":
    main()
