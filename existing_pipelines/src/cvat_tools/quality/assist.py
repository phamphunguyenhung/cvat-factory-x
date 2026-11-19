"""MANUAL-ASSIST: doc track NGUOI DUNG DA TU GAN, bao cao chat luong, KHONG sinh/sua nhan, KHONG ghi CVAT.

Dung duoc ca khi chua duoc phep prelabel (RULES Day 5). Chi dung client chi doc hoac file JSON annotation (backup/fixture).
Dau ra: danh sach van de + frame can mo lai tren CVAT (so frame 0-based theo CVAT; MOT 1-based = +1).
"""
from __future__ import annotations

import json
from pathlib import Path

from cvat_tools.tracking import geometry as G


def load_tracks(source, job: int | None = None, stop_frame: int | None = None) -> tuple[list[dict], int]:
    """source: Cvat (chi doc) hoac duong dan JSON annotation {tracks:[...]}. -> (cvat_tracks, stop_frame)."""
    if isinstance(source, (str, Path)):
        d = json.loads(Path(source).read_text(encoding="utf8"))
        return d["tracks"], stop_frame if stop_frame is not None else int(d.get("stop_frame", 149))
    return source.annotations(job)["tracks"], source.job(job)["stop_frame"]


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def analyze(cvat_tracks: list[dict], stop_frame: int, jump_h: float = 0.5, jump_iou: float = 0.3, size_ratio: float = 1.4,
            sparse_span: int = 30, dup_iou: float = 0.7, dup_frames: int = 5, accel_ratio: float = 3.0) -> dict:
    """-> {"tracks": {...tom tat}, "issues": [{kind, track, frames, msg, severity}], "reopen": [frame,...]}."""
    issues: list[dict] = []
    summary = {}
    expanded: dict = {}

    def add(kind, tid, frames, msg, sev="warn"):
        issues.append({"kind": kind, "track": tid, "frames": sorted(set(frames)), "msg": msg, "severity": sev})

    for i, t in enumerate(cvat_tracks, 1):
        tid = t.get("id", i)
        label = t.get("label_id")
        shapes = sorted(t.get("shapes", []), key=lambda s: s["frame"])
        tr = G.expand_cvat_track(t, stop_frame)
        expanded[tid] = tr
        kfs = [s for s in shapes if not s.get("outside") and s.get("keyframe", True)]
        summary[tid] = {"label_id": label, "first": min(tr) if tr else None, "last": max(tr) if tr else None,
                        "boxes": len(tr), "keyframes": len(kfs), "source": t.get("source")}
        if not tr:
            add("empty_track", tid, [], f"track {tid} khong co box nao", "error")
            continue
        # 1) nhay bbox bat thuong giua hai frame lien tiep (tren box DA noi suy; nhay nam o keyframe)
        fr = sorted(tr)
        for a, b in zip(fr, fr[1:]):
            if b - a != 1:
                continue
            ba, bb = tr[a], tr[b]
            h = max(1.0, ba[3] - ba[1])
            (ax, ay), (bx, by) = _center(ba), _center(bb)
            disp = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / h
            wa, wb = max(1.0, ba[2] - ba[0]), max(1.0, bb[2] - bb[0])
            ha, hb = h, max(1.0, bb[3] - bb[1])
            if disp > jump_h or G.box_iou(ba, bb) < jump_iou:
                add("bbox_jump", tid, [a, b], f"track {tid}: box nhay {disp:.2f}x chieu cao giua f{a}->f{b} (IoU {G.box_iou(ba, bb):.2f})")
            elif max(wb / wa, wa / wb, hb / ha, ha / hb) > size_ratio:
                add("bbox_size_jump", tid, [a, b], f"track {tid}: kich thuoc doi dot ngot f{a}->f{b}")
        # 2) khoang trong / track dut (outside roi noi lai, hoac thieu frame)
        for k, (a, b) in enumerate(zip(shapes, shapes[1:])):
            seen_before = any(not x.get("outside") for x in shapes[:k])  # outside luc dau = chua vao khung, khong phai khoang trong
            if a.get("outside") and not b.get("outside") and seen_before:
                gap = b["frame"] - a["frame"]
                add("outside_gap", tid, [a["frame"], b["frame"]],
                    f"track {tid}: outside {gap} frame roi noi lai cung ID" + (" (>60: nen la ID moi)" if gap > 60 else ""),
                    "error" if gap > 60 else "warn")
        if max(tr) < stop_frame - 1 and not shapes[-1].get("outside"):
            add("ends_early", tid, [max(tr)], f"track {tid} dung o f{max(tr)} nhung job den f{stop_frame} (thieu outside/keyframe cuoi?)", "info")
        # 3) keyframe thieu / qua thua / qua day
        span = (max(tr) - min(tr) + 1)
        if len(kfs) <= 2 and span > sparse_span:
            add("missing_keyframes", tid, [min(tr), max(tr)], f"track {tid}: chi {len(kfs)} keyframe cho {span} frame (nguy co lech)")
        if kfs and len(kfs) / span * 100 > 50:
            add("keyframe_dense", tid, [], f"track {tid}: {len(kfs)} keyframe / {span} frame (>50/100)", "info")
        # 4) khoang noi suy bat thuong: van toc ngam dinh khac xa hai khoang ke ben
        kf = [s for s in shapes if not s.get("outside")]
        speeds = []
        for a, b in zip(kf, kf[1:]):
            if b["frame"] > a["frame"] and not any(s.get("outside") and a["frame"] < s["frame"] <= b["frame"] for s in shapes):
                (ax, ay), (bx, by) = _center(a["points"]), _center(b["points"])
                speeds.append((a["frame"], b["frame"], (((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5) / (b["frame"] - a["frame"]),
                               max(1.0, a["points"][3] - a["points"][1])))
        for k in range(1, len(speeds) - 1):
            f0, f1, sp, h = speeds[k]
            nb = (speeds[k - 1][2] + speeds[k + 1][2]) / 2
            if sp > max(accel_ratio * nb, 0.1 * h) and sp > 3 * (1 + 0):
                add("odd_interpolation", tid, [f0, f1], f"track {tid}: f{f0}->f{f1} noi suy nhanh {sp:.1f}px/frame, hai khoang ke ben ~{nb:.1f}px/frame")
    # 5) ID trung / hai track chong nhau (cung mot nguoi)
    ids = sorted(expanded)
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            both = [f for f in set(expanded[a]) & set(expanded[b]) if G.box_iou(expanded[a][f], expanded[b][f]) >= dup_iou]
            if len(both) >= dup_frames:
                add("duplicate_track", a, both, f"track {a} va {b} chong nhau (IoU>={dup_iou}) {len(both)} frame tu f{min(both)}: trung nguoi?", "error")
    # 6) id_switch/fragmentation (heuristic san co) cho nguoi dung doi chieu
    for q in G.qa_tracks(expanded):
        if q["kind"] in ("id_switch", "fragmentation"):
            add(q["kind"], None, [], q["msg"], "info")
    reopen = sorted({f for it in issues for f in it["frames"]})
    sev = {"error": 0, "warn": 0, "info": 0}
    for it in issues:
        sev[it["severity"]] += 1
    return {"tracks": summary, "issues": issues, "reopen": reopen, "counts": sev, "stop_frame": stop_frame,
            "note": "Bao cao chat luong chi doc; KHONG sua nhan. Frame 0-based (MOT = +1)."}


def to_markdown(rep: dict) -> str:
    L = ["# Bao cao manual-assist (chi doc, khong sua nhan)", "",
         f"Track: {len(rep['tracks'])} · loi {rep['counts']['error']} · canh bao {rep['counts']['warn']} · thong tin {rep['counts']['info']}", ""]
    for tid, s in rep["tracks"].items():
        L.append(f"- track {tid}: f{s['first']}..f{s['last']} · {s['boxes']} box · {s['keyframes']} keyframe · source={s['source']}")
    L += ["", "## Van de"]
    for it in sorted(rep["issues"], key=lambda x: ({"error": 0, "warn": 1, "info": 2}[x["severity"]], x["frames"][:1])):
        L.append(f"- [{it['severity']}/{it['kind']}] {it['msg']}  -> mo lai frame {it['frames'][:6]}")
    L += ["", "## Frame can mo lai tren CVAT", ", ".join(map(str, rep["reopen"])) or "(khong co)"]
    return "\n".join(L) + "\n"
