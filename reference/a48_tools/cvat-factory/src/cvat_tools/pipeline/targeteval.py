"""So tool voi shape nguoi ve (frame71_target.json) theo nhan: so luong, IoU polygon, khop hop IoU>=0.5.

Nguong CO DINH (0.5), khong chinh theo frame. `weak()` liet ke nhan tool lam kem theo quy tac co dinh:
  polygon: IoU hop nhat < 0.5 | vat the: recall hop < 0.5 | target co ma tool 0 shape = THIEU.
"""
from __future__ import annotations

from typing import Dict, List

from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union
from shapely.validation import make_valid

IOU_MATCH = 0.5
OBJECT_LABELS = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign")


def geom(s: dict):
    p = s["points"]
    if s["type"] == "rectangle":
        return sbox(min(p[0], p[2]), min(p[1], p[3]), max(p[0], p[2]), max(p[1], p[3]))
    g = Polygon([(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]) if len(p) >= 6 else Polygon()
    return g if g.is_valid else make_valid(g)


def iou(a, b) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def greedy_match(tg: list, xg: list, thr: float = IOU_MATCH):
    """1-1 theo IoU giam dan. -> [(i_target, j_tool, iou)]"""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(tg) for j, b in enumerate(xg)), reverse=True)
    ut, ux, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in ut or j in ux:
            continue
        ut.add(i), ux.add(j), out.append((i, j, v))
    return out


def compare(target: List[dict], tool: List[dict]) -> Dict[str, dict]:
    labels = sorted({s["label"] for s in target} | {s["label"] for s in tool})
    res = {}
    for lab in labels:
        t = [s for s in target if s["label"] == lab]
        x = [s for s in tool if s["label"] == lab]
        tg, xg = [geom(s) for s in t], [geom(s) for s in x]
        r = {"n_target": len(t), "n_tool": len(x), "kind": "object" if lab in OBJECT_LABELS else "polygon"}
        ut = unary_union(tg) if tg else Polygon()
        ux = unary_union(xg) if xg else Polygon()
        r["union_iou"] = round(iou(ut, ux), 3) if (tg or xg) else None
        m = greedy_match(tg, xg)
        r["matched"] = len(m)
        r["recall"] = round(len(m) / len(t), 3) if t else None
        r["precision"] = round(len(m) / len(x), 3) if x else None
        r["mean_pair_iou"] = round(sum(v for _, _, v in m) / len(m), 3) if m else 0.0
        res[lab] = r
    return res


def weak(res: Dict[str, dict]) -> List[str]:
    out = []
    for lab, r in res.items():
        if r["n_target"] and not r["n_tool"]:
            out.append(f"{lab}: THIEU (target {r['n_target']}, tool 0)")
        elif r["n_target"] and r["kind"] == "polygon" and (r["union_iou"] or 0) < IOU_MATCH:
            out.append(f"{lab}: IoU hop nhat {r['union_iou']} < {IOU_MATCH} (target {r['n_target']}, tool {r['n_tool']})")
        elif r["n_target"] and r["kind"] == "object" and (r["recall"] or 0) < IOU_MATCH:
            out.append(f"{lab}: recall hop {r['recall']} < {IOU_MATCH} (khop {r['matched']}/{r['n_target']}, tool {r['n_tool']})")
        elif not r["n_target"] and r["n_tool"]:
            out.append(f"{lab}: THUA {r['n_tool']} shape (target 0)")
    return out
