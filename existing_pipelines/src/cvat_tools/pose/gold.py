"""Gold = điểm BẠN ĐÃ SỬA TAY trên CVAT so với nháp auto đã đẩy lên (dùng chung cho mặt 50 điểm và pose 17 điểm). Chỉ đọc CVAT.

Chỉ lưu điểm bị dời (> tol px) hoặc đổi cờ, vì điểm không đổi có thể chỉ là nháp cũ chưa ai duyệt.
Gold lưu ở <job>/metrics/gold.json: {frame: {tên_điểm: [x, y]}}.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def draft_points(path: Path) -> dict:
    """{frame: {tên_điểm: (x, y)}} từ file nháp (mọi label gộp; tên điểm là duy nhất trong frame)."""
    d = json.loads(Path(path).read_text(encoding="utf8"))
    return {fr["frame"]: {n: (pt["x"], pt["y"]) for p in fr["people"] for n, pt in p["points"].items()} for fr in d["frames"]}


def label_of(labels: list) -> dict:
    """{tên_điểm: tên_label}."""
    return {s["name"]: l["name"] for l in labels for s in l["sublabels"]}


def refresh_gold(c, job: int, folder: Path, tol: float = 1.5, min_moved: int = 30) -> dict:
    """Cập nhật metrics/gold.json từ CVAT: frame có >= min_moved điểm bị dời so với nháp của lần đẩy gần nhất."""
    runs = sorted((folder / "backups").glob("run_*.json"))
    if not runs:
        raise SystemExit("Chưa có run_*.json: chưa đẩy nháp nào lên CVAT nên chưa có gì để so.")
    run = json.loads(runs[-1].read_text(encoding="utf8"))
    pushed = draft_points(folder / "drafts" / run["draft"])
    subn = {s["id"]: s["name"] for l in c.labels(job) for s in l["sublabels"]}
    ids = set(run["created_ids"])
    cur: dict = {}
    for s in c.annotations(job)["shapes"]:
        if s["id"] in ids:
            for e in s["elements"]:
                cur.setdefault(s["frame"], {})[subn[e["label_id"]]] = tuple(e["points"])
    gf = folder / "metrics" / "gold.json"
    gold = json.loads(gf.read_text(encoding="utf8")) if gf.exists() else {}
    for fr, pts in cur.items():
        if fr not in pushed:
            continue
        moved = {n: list(v) for n, v in pts.items() if n in pushed[fr] and math.hypot(v[0] - pushed[fr][n][0], v[1] - pushed[fr][n][1]) > tol}
        if len(moved) >= min_moved:
            gold[str(fr)] = moved
    gf.parent.mkdir(exist_ok=True)
    gf.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf8")
    return gold


def load_gold(folder: Path) -> dict:
    gf = folder / "metrics" / "gold.json"
    return json.loads(gf.read_text(encoding="utf8")) if gf.exists() else {}
