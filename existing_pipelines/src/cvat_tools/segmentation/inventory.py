"""Kiem ke canh: moi frame co label nao, bao nhieu, o dau. Dau vao = shape tu layers.resolve."""
from __future__ import annotations

from shapely.geometry import Polygon


def _area_bbox(flat):
    pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
    if len(pts) < 3:
        return 0.0, [0, 0, 0, 0]
    xs, ys = zip(*pts)
    return Polygon(pts).buffer(0).area, [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def inventory(frame_instances) -> dict:
    """-> {label: {count, total_area, bbox_list, occluded_count, conf_mean}}."""
    inv: dict = {}
    for s in frame_instances:
        a, bb = _area_bbox(s["polygon"])
        e = inv.setdefault(s["label"], {"count": 0, "total_area": 0.0, "bbox_list": [], "occluded_count": 0, "_c": 0.0})
        e["count"] += 1
        e["total_area"] += a
        e["bbox_list"].append(bb)
        e["occluded_count"] += bool(s.get("occluded"))
        e["_c"] += float(s.get("conf", 1.0))
    for e in inv.values():
        e["conf_mean"] = round(e.pop("_c") / e["count"], 3)
        e["total_area"] = round(e["total_area"], 1)
    return dict(sorted(inv.items()))


def frame_markdown(frame_id, frame_instances, expected_labels=None) -> str:
    inv = inventory(frame_instances)
    lines = [f"### Frame {frame_id}: {sum(e['count'] for e in inv.values())} instance, {len(inv)} label",
             "", "| label | n | dien tich | khuat | conf | vi tri (bbox) |", "|---|---:|---:|---:|---:|---|"]
    for lb, e in inv.items():
        bbs = " ".join(str([int(v) for v in b]) for b in e["bbox_list"][:4]) + (" ..." if e["count"] > 4 else "")
        lines.append(f"| {lb} | {e['count']} | {e['total_area']:.0f} | {e['occluded_count']} | {e['conf_mean']} | {bbs} |")
    if expected_labels:
        miss = sorted(set(expected_labels) - set(inv))
        if miss:
            lines += ["", "Thieu: " + ", ".join(miss)]
    return "\n".join(lines)


def summary_markdown(frames: dict, expected_labels=None) -> str:
    """frames: {frame_id: shapes}. Tom tat N frame: moi label -> so frame co mat, tong, max/frame, khuat, frame thieu."""
    invs = {f: inventory(s) for f, s in frames.items()}
    n = len(invs)
    labels = sorted({lb for i in invs.values() for lb in i} | set(expected_labels or []))
    lines = [f"## Tom tat {n} frame", "", "| label | frame co | tong | max/frame | khuat | frame thieu |",
             "|---|---:|---:|---:|---:|---|"]
    for lb in labels:
        have = [f for f, i in invs.items() if lb in i]
        miss = [str(f) for f in invs if f not in have]
        tot = sum(invs[f][lb]["count"] for f in have)
        mx = max((invs[f][lb]["count"] for f in have), default=0)
        occ = sum(invs[f][lb]["occluded_count"] for f in have)
        ms = (",".join(miss[:8]) + ("..." if len(miss) > 8 else "")) if miss else "-"
        lines.append(f"| {lb} | {len(have)}/{n} | {tot} | {mx} | {occ} | {ms} |")
    return "\n".join(lines)
