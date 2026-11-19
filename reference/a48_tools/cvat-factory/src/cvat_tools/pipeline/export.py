"""Xuat anh + bao cao cua pipeline: f<N>_tool.jpg, f<N>_side_by_side.jpg, contact_sheet.jpg, run_report.md/.csv.

Mau theo bang LABEL_COLOR (BGR); vien toi quanh vien mau de nhin duoc ca tren nen sang lan toi; frame toi thi anh nen
chi duoc LAM SANG khi hien thi (gamma), du lieu shape khong doi.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

STEP_ORDER = ("seg_b2", "yolo", "stuff", "lanes", "resolve", "subtract", "refine", "decision", "store", "total")
OBJECT_LABELS = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign")
LABEL_ORDER = ("sky", "building", "vegetation", "terrain", "wall", "fence", "sidewalk", "road", "pole",
               "area/direct", "area/alternative", "lane/crosswalk", "lane/road curb", "lane/single white", "lane/double white",
               "lane/single yellow", "lane/double yellow", "lane/single other", "lane/double other") + OBJECT_LABELS
LABEL_COLOR = {   # BGR
    "sky": (235, 206, 135), "building": (60, 60, 255), "vegetation": (60, 200, 60), "terrain": (120, 220, 170),
    "wall": (180, 130, 255), "fence": (160, 160, 255), "sidewalk": (255, 120, 200), "road": (160, 160, 160),
    "pole": (0, 215, 255), "area/direct": (255, 0, 255), "area/alternative": (255, 255, 0),
    "lane/crosswalk": (255, 255, 255), "lane/road curb": (0, 128, 255), "lane/single white": (240, 240, 240),
    "lane/double white": (200, 200, 255), "lane/single yellow": (0, 230, 255), "lane/double yellow": (0, 180, 255),
    "lane/single other": (255, 0, 0), "lane/double other": (200, 0, 0),
    "car": (0, 0, 255), "truck": (0, 128, 255), "bus": (0, 200, 255), "train": (128, 0, 255), "motorcycle": (255, 0, 255),
    "bicycle": (255, 0, 128), "pedestrian": (255, 128, 0), "rider": (255, 200, 0),
    "traffic light": (0, 255, 255), "traffic sign": (0, 255, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def color_of(label: str):
    return LABEL_COLOR.get(label, DEFAULT_COLOR)


def _text(img, txt, org, color=(255, 255, 255), sc=0.45, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def brighten_for_display(img: np.ndarray, thresh: float = 80.0) -> np.ndarray:
    mg = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    if mg >= thresh:
        return img
    g = float(np.clip(np.log(110 / 255.0) / np.log(max(mg, 1.0) / 255.0), 0.4, 1.0))
    lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    return cv2.LUT(img, lut)


def _pts(s) -> np.ndarray:
    p = s["points"]
    if s["type"] == "rectangle":
        x1, y1, x2, y2 = p[:4]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return np.asarray(p, np.float32).reshape(-1, 2)


def render(img: np.ndarray, shapes: List[dict], title: str, fill: float = 0.28, names: bool = True) -> np.ndarray:
    """Ve shape len anh (thu tu z_order tang dan). Polygon: to nhe + vien toi/sang; hop: chi vien."""
    base = brighten_for_display(img)
    ov = base.copy()
    order = sorted(shapes, key=lambda s: s.get("z_order", 0))
    for s in order:
        if s["type"] == "polygon":
            cv2.fillPoly(ov, [np.round(_pts(s)).astype(np.int32)], color_of(s["label"]))
    out = cv2.addWeighted(ov, fill, base, 1 - fill, 0)
    for s in order:
        p = np.round(_pts(s)).astype(np.int32).reshape(-1, 1, 2)
        col = color_of(s["label"])
        th = 1 if s["label"].startswith("lane/") else 2
        cv2.polylines(out, [p], True, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.polylines(out, [p], True, col, th, cv2.LINE_AA)
    if names:
        for s in order:
            xy = _pts(s)
            i = int(np.argmin(xy[:, 1] * 10000 + xy[:, 0]))
            x, y = int(xy[i, 0]), int(xy[i, 1])
            _text(out, s["label"].replace("lane/", "l/").replace("area/", "a/"), (max(2, min(x, out.shape[1] - 90)), max(12, y + 12)),
                  color_of(s["label"]), 0.38)
    bar = np.zeros((26, out.shape[1], 3), np.uint8)
    _text(bar, title, (8, 18), (255, 255, 255), 0.55)
    return np.vstack([bar, out])


def render_tool(img, result: dict) -> np.ndarray:
    shapes = result["road_input"] + result["shapes"]
    t = result["timings"]["total"]
    title = (f"f{result['frame']} TOOL  {len(result['shapes'])} shape  {t:.1f}s{' (>10s)' if result['over_budget'] else ''}"
             f"{'  TOI' if result['dark'] else ''}  [road = nguoi ve, dau vao]")
    return render(img, shapes, title)


def render_human(img, frame: int, human_shapes: List[dict], tag: str = "NGUOI VE") -> np.ndarray:
    return render(img, human_shapes, f"f{frame} {tag}  {len(human_shapes)} shape")


def side_by_side(img, frame: int, human_shapes: List[dict], result: dict, tag: str = "NGUOI VE") -> np.ndarray:
    left, right = render_human(img, frame, human_shapes, tag), render_tool(img, result)
    return np.hstack([left, right])


def contact_sheet(tiles: Dict[int, np.ndarray], cols: int = 5, tw: int = 384) -> np.ndarray:
    ims = []
    for f in sorted(tiles):
        t = tiles[f]
        h = int(round(t.shape[0] * tw / t.shape[1]))
        ims.append(cv2.resize(t, (tw, h), interpolation=cv2.INTER_AREA))
    th = max(i.shape[0] for i in ims)
    rows = []
    for r in range(0, len(ims), cols):
        row = ims[r:r + cols]
        row += [np.zeros((th, tw, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack([np.vstack([i, np.zeros((th - i.shape[0], tw, 3), np.uint8)]) if i.shape[0] < th else i for i in row]))
    return np.vstack(rows)


# ------------------------------------------------------------------ bao cao
def _labels_in(results: List[dict]) -> List[str]:
    seen = {s["label"] for r in results for s in r["shapes"]}
    return [l for l in LABEL_ORDER if l in seen] + sorted(seen - set(LABEL_ORDER))


def write_csv(path: Path, results: List[dict]) -> None:
    labels = _labels_in(results)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "mean_gray", "dark"] + [f"n:{l}" for l in labels] + [f"dup:{l}" for l in labels]
                   + [f"s:{k}" for k in STEP_ORDER] + ["over_10s", "val_err_before", "val_warn_before", "val_err_after", "val_warn_after"])
        for r in results:
            n = {l: sum(s["label"] == l for s in r["shapes"]) for l in labels}
            d = {l: sum(s["label"] == l and s.get("dup_human") for s in r["shapes"]) for l in labels}
            T = r["timings"]
            w.writerow([r["frame"], r["mean_gray"], int(r["dark"])] + [n[l] for l in labels] + [d[l] for l in labels]
                       + [T.get(k, "") for k in STEP_ORDER] + [int(r["over_budget"]), r["issues_before"]["error"], r["issues_before"]["warn"],
                                                             r["issues_after"]["error"], r["issues_after"]["warn"]])


def _md_table(head: List[str], rows: List[List]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_markdown(path: Path, results: List[dict], *, job: int, budget_s: float = 10.0, models_load_s: Optional[float] = None,
                   target_cmp: Optional[dict] = None, target_weak: Optional[List[str]] = None, notes: Optional[List[str]] = None,
                   frame_target: Optional[int] = None) -> None:
    labels = _labels_in(results)
    L = [f"# Run report - job {job} ({len(results)} frame, moi frame doc lap, CHI DOC CVAT)\n"]
    if notes:
        L += [f"- {n}" for n in notes] + [""]
    L.append("## 1. So shape tool theo nhan x frame (trong ngoac: so shape trung shape nguoi IoU>=0.5)\n")
    rows = []
    for r in results:
        cells = []
        for l in labels:
            n = sum(s["label"] == l for s in r["shapes"])
            d = sum(s["label"] == l and s.get("dup_human") for s in r["shapes"])
            cells.append(f"{n}({d})" if n else "0")
        rows.append([f"f{r['frame']}"] + cells + [len(r["shapes"])])
    tot = ["TONG"] + [str(sum(sum(s["label"] == l for s in r["shapes"]) for r in results)) for l in labels] + [sum(len(r["shapes"]) for r in results)]
    L.append(_md_table(["frame"] + labels + ["tong"], rows + [tot]))
    L.append("\n## 2. Thoi gian tung buoc (s/frame; **dam** = frame >10 s)\n")
    rows = []
    for r in results:
        T = r["timings"]
        rows.append([f"f{r['frame']}"] + [T.get(k, "-") for k in STEP_ORDER[:-1]] + [f"**{T['total']}**" if r["over_budget"] else T["total"]])
    mean = ["TB"] + [round(float(np.mean([r["timings"].get(k, 0.0) for r in results])), 2) for k in STEP_ORDER]
    L.append(_md_table(["frame"] + list(STEP_ORDER), rows + [mean]))
    nb = sum(r["over_budget"] for r in results)
    L.append(f"\n{nb}/{len(results)} frame vuot {budget_s:g} s."
             + (f" Nap model 1 lan: {models_load_s} s (khong tinh vao tung frame)." if models_load_s is not None else ""))
    L.append("\n## 3. Validator truoc / sau decision (loi do duoc)\n")
    rows = [[f"f{r['frame']}", r["issues_before"]["error"], r["issues_before"]["warn"], r["issues_after"]["error"], r["issues_after"]["warn"],
             ", ".join(f"{k}:{v}" for k, v in r["codes_after"].items()) or "-"] for r in results]
    L.append(_md_table(["frame", "error truoc", "warn truoc", "error sau", "warn sau", "ma con lai sau"], rows))
    L.append("\n## 4. Co trung shape nguoi\n")
    dup = [(r["frame"], sum(1 for s in r["shapes"] if s.get("dup_human")), len(r["shapes"])) for r in results]
    L.append(_md_table(["frame", "shape tool trung nguoi", "tong shape tool"], [[f"f{a}", b, c] for a, b, c in dup]))
    if target_cmp is not None:
        L.append(f"\n## 5. Frame {frame_target}: tool so voi frame{frame_target}_target.json (MOC NGHIEM THU)\n")
        rows = []
        for lab in [l for l in LABEL_ORDER if l in target_cmp] + sorted(set(target_cmp) - set(LABEL_ORDER)):
            c = target_cmp[lab]
            rows.append([lab, c["n_target"], c["n_tool"], c["matched"],
                         c["union_iou"] if c["kind"] == "polygon" else "-",
                         c["mean_pair_iou"] if c["matched"] else "-",
                         c["recall"] if c["recall"] is not None else "-", c["precision"] if c["precision"] is not None else "-"])
        L.append(_md_table(["nhan", "target", "tool", "khop IoU>=0.5", "IoU hop nhat (polygon)", "IoU TB cap khop", "recall", "precision"], rows))
        L.append("\n**Nhan tool lam kem (quy tac co dinh: polygon IoU hop nhat <0.5; vat the recall <0.5; thieu; thua):**\n")
        L += [f"- {w}" for w in (target_weak or ["(khong co)"])]
    path.write_text("\n".join(L) + "\n", encoding="utf8")
