"""Keyframe optimizer thich ung. CVAT noi suy TUYEN TINH tung toa do giua hai keyframe (rectangle track), nen:

- `optimal_keyframes`: so keyframe TOI THIEU (quy hoach dong tren do thi, khong chi Douglas-Peucker) sao cho
  sai so noi suy <= tol px o MOI frame va moi toa do. Frame bat buoc (`must_keep`) luon duoc giu.
- `must_keep_frames`: vao/ra khung, doi huong, doi ti le bbox dot ngot, gia toc, giao nhau/che khuat (cac su kien rui ro).
- Tuyet doi khong giu frame lam sai so vuot tol (chan cung); quyet dinh danh tinh khong chac chan KHONG duoc tu chon o day,
  chi duoc dua vao `review_flags` cho nguoi duyet.
"""
from __future__ import annotations

import math
from typing import Sequence

from cvat_tools.tracking import geometry as G


def interp_error(track: dict, kfs: Sequence[int]) -> dict:
    """Sai so noi suy tuyen tinh theo kfs so voi box that: {max, mean, worst_frame}."""
    kfs = sorted(kfs)
    worst, wf, tot, n = 0.0, None, 0.0, 0
    for a, b in zip(kfs, kfs[1:]):
        for f in range(a + 1, b):
            if f in track:
                d = G.max_dev(track[f], G.lerp_box(f, a, track[a], b, track[b]))
                tot += d
                n += 1
                if d > worst:
                    worst, wf = d, f
    return {"max": worst, "mean": tot / n if n else 0.0, "worst_frame": wf}


def optimal_keyframes(track: dict, tol: float = 4.0, must_keep: Sequence[int] = ()) -> list[int]:
    """Tap keyframe nho nhat (bao gom dau, cuoi, must_keep) voi sai so noi suy <= tol. Gia dinh frame lien tuc (da fill_gaps)."""
    fr = sorted(track)
    n = len(fr)
    if n <= 2:
        return fr
    forced = {fr[0], fr[-1], *[f for f in must_keep if f in track]}
    idx = {f: i for i, f in enumerate(fr)}
    INF = 10 ** 9
    best = [INF] * n
    prev = [-1] * n
    best[0] = 1
    forced_idx = sorted(idx[f] for f in forced)
    for i in range(n - 1):
        if best[i] >= INF:
            continue
        # khong duoc nhay qua frame bat buoc ke tiep
        limit = next((j for j in forced_idx if j > i), n - 1)
        for j in range(i + 1, limit + 1):
            fi, fj = fr[i], fr[j]
            ok = True
            for k in range(i + 1, j):
                fk = fr[k]
                if G.max_dev(track[fk], G.lerp_box(fk, fi, track[fi], fj, track[fj])) > tol:
                    ok = False
                    break
            if ok and best[i] + 1 < best[j]:
                best[j], prev[j] = best[i] + 1, i
            elif not ok and j > i + 1:
                # khong the keo dai hon: khong chac don dieu nen van thu tiep (dung chinh xac), nhung nhay het chan
                pass
    out, j = [], n - 1
    while j != -1:
        out.append(fr[j])
        j = prev[j]
    return sorted(out)


def must_keep_frames(track: dict, others: dict | None = None, dir_deg: float = 35.0, scale: float = 0.12,
                     accel: float = 0.5, cross_iou: float = 0.3, min_speed: float = 2.0) -> dict[int, list[str]]:
    """Frame can giu theo ly do. -> {frame: [ly do]}. `others` = {tid: track} de bat giao nhau/che khuat."""
    fr = sorted(track)
    why: dict[int, list[str]] = {}

    def add(f, r):
        why.setdefault(f, []).append(r)

    if not fr:
        return why
    add(fr[0], "vao_khung/bat_dau")
    add(fr[-1], "ra_khung/ket_thuc")
    for a, b in zip(fr, fr[1:]):
        if b - a > 1:
            add(a, "bien_khoang_trong")
            add(b, "bien_khoang_trong")

    def c(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    for p, q, r in zip(fr, fr[1:], fr[2:]):
        if q - p != 1 or r - q != 1:
            continue
        v1 = (c(track[q])[0] - c(track[p])[0], c(track[q])[1] - c(track[p])[1])
        v2 = (c(track[r])[0] - c(track[q])[0], c(track[r])[1] - c(track[q])[1])
        s1, s2 = math.hypot(*v1), math.hypot(*v2)
        if s1 > min_speed and s2 > min_speed:
            ang = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])))
            if ang > dir_deg:
                add(q, f"doi_huong {ang:.0f}deg")
        h = max(1.0, track[q][3] - track[q][1])
        if abs(s2 - s1) / h > accel:
            add(q, f"gia_toc dv={abs(s2 - s1):.1f}px")
        h1, h2 = track[q][3] - track[q][1], track[r][3] - track[r][1]
        if h1 > 0 and abs(math.log(max(h2, 1e-3) / h1)) > scale:
            add(r, f"doi_ti_le bbox {h2 / h1:.2f}x")
    for oid, ot in (others or {}).items():
        for f in fr:
            if f in ot and G.box_iou(track[f], ot[f]) >= cross_iou:
                add(f, f"giao_nhau/che_khuat voi track {oid} iou={G.box_iou(track[f], ot[f]):.2f}")
    return why


def plan_track(track: dict, tol: float = 4.0, others: dict | None = None, force_events: bool = True, **kw) -> dict:
    """Ke hoach keyframe cho 1 track: {keyframes, forced, error, baseline{dp, all}, reduction}."""
    why = must_keep_frames(track, others, **kw) if force_events else {}
    # su kien vao/ra luon giu; cac su kien khac (doi huong, ti le...) chi giu NEU anh huong sai so? Khong: giu cung,
    # nhung giu rieng "bat_buoc" va "goi_y" de bao cao muc giam.
    hard = {f for f, rs in why.items() if any(r.startswith(("vao_khung", "ra_khung", "bien_khoang")) for r in rs)}
    kf_min = optimal_keyframes(track, tol, hard)
    kf_ev = optimal_keyframes(track, tol, set(why)) if force_events else kf_min
    dp = G.simplify_keyframes(track, tol)
    return {"keyframes": kf_min, "keyframes_with_events": kf_ev, "events": {f: r for f, r in why.items()},
            "error": interp_error(track, kf_min), "baseline": {"all_frames": len(track), "douglas_peucker": len(dp)},
            "n_min": len(kf_min), "n_with_events": len(kf_ev),
            "reduction_vs_all": 1 - len(kf_min) / max(1, len(track)), "reduction_vs_dp": 1 - len(kf_min) / max(1, len(dp))}
