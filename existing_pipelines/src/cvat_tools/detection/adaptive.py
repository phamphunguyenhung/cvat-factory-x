"""Do phan giai thich ung: chay detector re/nho cho moi frame, chi frame KHO chay lai bang model/do phan giai lon hon.

Frame kho = co detection conf trung binh (nguong thap..cao), so nguoi lech so voi lang gieng, hoac nguoi nho (cao < small_h px).
Tra ve detection hop nhat + ty le frame chay lai (de tinh chi phi that).
"""
from __future__ import annotations

import numpy as np


def hard_frames(dets: dict, low=0.25, high=0.5, small_h: float = 70.0, count_dev: int = 1) -> dict[int, list[str]]:
    fr = sorted(dets)
    counts = {f: sum(1 for d in dets[f] if d[4] >= low) for f in fr}
    why: dict[int, list[str]] = {}
    for i, f in enumerate(fr):
        r = []
        mid = [d for d in dets[f] if low <= d[4] < high]
        if mid:
            r.append(f"{len(mid)} nguoi conf {low}-{high}")
        nb = [counts[g] for g in fr[max(0, i - 2):i + 3] if g != f]
        if nb and abs(counts[f] - int(np.median(nb))) >= count_dev:
            r.append(f"so nguoi {counts[f]} lech lang gieng {int(np.median(nb))}")
        if any((d[3] - d[1]) < small_h for d in dets[f] if d[4] >= low):
            r.append("nguoi nho")
        if r:
            why[f] = r
    return why


def merge_dets(base: dict, refined: dict) -> dict:
    out = dict(base)
    out.update(refined)
    return out
