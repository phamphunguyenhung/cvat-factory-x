"""Hình học track thuần (không mạng): chọn track, rút gọn keyframe, payload CVAT, QA, xuất MOT 1.1.

Track = {frame: [xtl, ytl, xbr, ybr]}, frame = số frame CVAT. CVAT nội suy TUYẾN TÍNH giữa hai keyframe,
nên keyframe chỉ cần ở chỗ nội suy lệch quá `tol` px so với box của detector.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

Box = list
Track = dict


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def lerp_box(f: int, f0: int, b0: Sequence[float], f1: int, b1: Sequence[float]) -> Box:
    t = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
    return [b0[i] + (b1[i] - b0[i]) * t for i in range(4)]


def max_dev(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(a[i] - b[i]) for i in range(4))


# ---------------------------------------------------------------- chọn track
def fill_gaps(track: Track, max_gap: int) -> Track:
    """Lấp lỗ <= max_gap frame bằng nội suy để detector hụt ngắn không làm track đứt (fragmentation)."""
    fr, out = sorted(track), dict(track)
    for a, b in zip(fr, fr[1:]):
        if 1 < b - a <= max_gap + 1:
            for f in range(a + 1, b):
                out[f] = lerp_box(f, a, track[a], b, track[b])
    return out


def select_longest(tracks: dict, n: int, min_len: int = 10) -> dict:
    def score(it):
        tr = it[1]
        return len(tr), sum((b[2] - b[0]) * (b[3] - b[1]) for b in tr.values()) / max(1, len(tr))
    c = sorted(((t, tr) for t, tr in tracks.items() if len(tr) >= min_len), key=score, reverse=True)
    return dict(c[:n])


def select_by_anchors(tracks: dict, anchors: dict, ref_frame: int) -> tuple[dict, list]:
    """P1..P5 -> track: chọn track có box tại ref_frame chứa điểm neo (nhiều track thì lấy tâm gần nhất).
    Trả (map tên->track_id, cảnh báo). Hai neo trùng một track hoặc không track nào chứa neo đều bị báo."""
    chosen, warns = {}, []
    for name, anchor in anchors.items():
        x, y = anchor[0], anchor[1]
        fr = int(anchor[2]) if len(anchor) > 2 else ref_frame  # neo có thể kèm frame riêng của từng người
        best = None
        for tid, tr in tracks.items():
            b = tr.get(fr)
            if not b:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            inside = b[0] <= x <= b[2] and b[1] <= y <= b[3]
            d = math.hypot(x - cx, y - cy)
            if inside or d < 0.3 * (b[3] - b[1]):
                if best is None or d < best[0]:
                    best = (d, tid)
        if best is None:
            warns.append(f"{name}: không track nào chứa điểm neo ({x:.0f},{y:.0f}) ở frame {fr}")
        else:
            chosen[name] = best[1]
    dup = {t for t in chosen.values() if list(chosen.values()).count(t) > 1}
    for name, t in chosen.items():
        if t in dup:
            warns.append(f"{name}: trùng track {t} với neo khác (hai người đi sát nhau, cần gán tay)")
    return chosen, warns


def same_person(a: Track, b: Track, min_frames: int = 10, iou: float = 0.5) -> bool:
    """Hai track là cùng một người nếu chồng nhau (IoU >= iou) ở >= min_frames frame chung. Dùng để không ghi trùng."""
    return sum(1 for f in set(a) & set(b) if box_iou(a[f], b[f]) >= iou) >= min_frames


# ---------------------------------------------------------------- keyframe
def simplify_keyframes(track: Track, tol: float = 4.0) -> list[int]:
    """Douglas-Peucker theo frame: tập keyframe nhỏ nhất để nội suy tuyến tính lệch <= tol px ở mọi tọa độ."""
    fr = sorted(track)
    if len(fr) <= 2:
        return fr
    keep, stack = {fr[0], fr[-1]}, [(fr[0], fr[-1])]
    while stack:
        f0, f1 = stack.pop()
        worst, wf = 0.0, None
        for f in fr:
            if f0 < f < f1:
                d = max_dev(track[f], lerp_box(f, f0, track[f0], f1, track[f1]))
                if d > worst:
                    worst, wf = d, f
        if wf is not None and worst > tol:
            keep.add(wf)
            stack += [(f0, wf), (wf, f1)]
    return sorted(keep)


def to_cvat_track(label_id: int, track: Track, keyframes: Sequence[int], stop_frame: int) -> dict:
    """Payload track hình chữ nhật. source=manual (không để 'auto'). Người rời khung trước stop_frame thì thêm
    keyframe outside=True ngay sau frame cuối để track không kéo dài qua frame trống."""
    shapes = [{"type": "rectangle", "frame": int(f), "points": [round(v, 2) for v in track[f]], "occluded": False,
               "outside": False, "z_order": 0, "rotation": 0.0, "attributes": []} for f in keyframes]
    last = max(keyframes)
    if last < stop_frame:
        shapes.append({"type": "rectangle", "frame": int(last) + 1, "points": [round(v, 2) for v in track[last]],
                       "occluded": False, "outside": True, "z_order": 0, "rotation": 0.0, "attributes": []})
    return {"frame": int(min(keyframes)), "label_id": label_id, "group": 0, "source": "manual",
            "attributes": [], "shapes": shapes}


def expand_cvat_track(trk: dict, stop_frame: int) -> Track:
    """Ngược của to_cvat_track: box từng frame theo nội suy tuyến tính, tôn trọng `outside`."""
    sh = sorted(trk.get("shapes", []), key=lambda s: s["frame"])
    out: Track = {}
    for s0, s1 in zip(sh, sh[1:] + [None]):
        if s0.get("outside"):
            continue
        end = s1["frame"] if s1 else stop_frame + 1
        for f in range(s0["frame"], end):
            if s1 is not None and not s1.get("outside"):
                out[f] = lerp_box(f, s0["frame"], s0["points"], s1["frame"], s1["points"])
            else:
                out[f] = list(s0["points"])
    return out


# ---------------------------------------------------------------- QA
def qa_tracks(tracks: dict, short_len: int = 10, handoff_gap: int = 5, handoff_iou: float = 0.3,
              merge_iou: float = 0.7, merge_frames: int = 5) -> list[dict]:
    """Gợi ý (heuristic, không phải phán quyết) cho các lỗi của lab: false_track, fragmentation, id_switch, merge."""
    out, ids = [], sorted(tracks)
    for t in ids:
        fr = sorted(tracks[t])
        if len(fr) < short_len:
            out.append({"kind": "false_track", "msg": f"track {t} chỉ {len(fr)} frame"})
        for a, b in zip(fr, fr[1:]):
            if b - a > 1:
                out.append({"kind": "fragmentation", "msg": f"track {t} thiếu frame {a + 1}..{b - 1}"})
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for x, y in ((a, b), (b, a)):
                tx, ty = tracks[x], tracks[y]
                end, start = max(tx), min(ty)
                if 0 < start - end <= handoff_gap and box_iou(tx[end], ty[start]) >= handoff_iou:
                    out.append({"kind": "id_switch", "msg": f"track {x} kết thúc f{end}, track {y} bắt đầu f{start} cùng vị trí (cùng một người? merge bằng M)"})
            both = [f for f in set(tracks[a]) & set(tracks[b]) if box_iou(tracks[a][f], tracks[b][f]) >= merge_iou]
            if len(both) >= merge_frames:
                out.append({"kind": "merge", "msg": f"track {a} và {b} chồng nhau (IoU>={merge_iou}) {len(both)} frame, từ f{min(both)}"})
    return out


def qa_keyframes(cvat_tracks: list[dict], density_limit: float = 50.0, gap_limit: int = 60, sparse_span: int = 30) -> list[dict]:
    """Ngưỡng giống src/check_submission.py của lab: >50 keyframe/100 frame, <=2 keyframe cho >30 frame,
    khuất (outside) > 60 frame rồi nối lại cùng ID."""
    out = []
    for i, t in enumerate(cvat_tracks, 1):
        sh = sorted(t.get("shapes", []), key=lambda s: s["frame"])
        kfs = [s for s in sh if not s.get("outside") and s.get("keyframe", True)]
        if not kfs:
            continue
        span = max(s["frame"] for s in sh) - kfs[0]["frame"] + 1
        if len(kfs) / span * 100 > density_limit:
            out.append({"kind": "keyframe_dense", "msg": f"track {i}: {len(kfs)} keyframe / {span} frame"})
        if len(kfs) <= 2 and span > sparse_span:
            out.append({"kind": "drift_risk", "msg": f"track {i}: chỉ {len(kfs)} keyframe cho {span} frame"})
        for a, b in zip(sh, sh[1:]):
            if a.get("outside") and not b.get("outside") and b["frame"] - a["frame"] > gap_limit:
                out.append({"kind": "gap_over_2s", "msg": f"track {i}: outside {b['frame'] - a['frame']} frame (>{gap_limit}) rồi nối lại cùng ID, nên là track mới"})
    return out


def write_mot(tracks: dict, path: Path, base_frame: int | None = None) -> int:
    """MOT 1.1 gt.txt: frame,id,x,y,w,h,1,1,1 với frame bắt đầu từ 1 (= frame CVAT + 1) và id liên tục từ 1."""
    ids = {t: i + 1 for i, t in enumerate(sorted(tracks))}
    f0 = base_frame if base_frame is not None else min((min(t) for t in tracks.values() if t), default=0)
    rows = sorted((f - f0 + 1, ids[t], b[0], b[1], b[2] - b[0], b[3] - b[1]) for t, tr in tracks.items() for f, b in tr.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{f},{i},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n" for f, i, x, y, w, h in rows), encoding="utf-8")
    return len(rows)
