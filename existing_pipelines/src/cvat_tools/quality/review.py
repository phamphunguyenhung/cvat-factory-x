"""Review queue + HTML local. Xep hang theo rui ro uoc luong, moi muc co LY DO + BANG CHUNG. Khong tu duyet theo dong thuan model.

Tin hieu: detector bat dong (nhieu cache detection), tracker bat dong ID (so sanh nhieu tracker), conf giam, nguoi chong lan,
track dut, noi suy lech qua dung sai. Quyet dinh cua nguoi duyet luu RIENG (JSON tai ve) -- khong tron voi du doan tho.
"""
from __future__ import annotations

import base64
import html
import json
from collections import defaultdict

import numpy as np

from cvat_tools.tracking import geometry as G

W = {"detector_disagree": 2.0, "tracker_disagree": 3.0, "conf_drop": 1.5, "overlap": 2.0, "track_break": 2.5, "interp_error": 3.0}


def _match(a: list, b: list, thr: float = 0.5) -> int:
    """So box cua a khong khop (IoU>=thr) voi bat ky box cua b."""
    return sum(1 for x in a if not any(G.box_iou(x[:4], y[:4]) >= thr for y in b))


def build_queue(det_sets: dict, trk_sets: dict, scores: dict | None = None, people: int = 5, kf_plans: dict | None = None,
                tol: float = 4.0, overlap_iou: float = 0.3, min_det_conf: float = 0.5) -> list[dict]:
    """det_sets {ten: {frame: dets}}; trk_sets {tracker: {tid: {frame: box}}}; scores {tracker: {tid: {frame: s}}}."""
    items: dict[int, list[dict]] = defaultdict(list)

    def add(f, kind, msg, evidence=None, w=None):
        items[f].append({"kind": kind, "msg": msg, "weight": w or W[kind], "evidence": evidence or {}})

    names = list(det_sets)
    frames = sorted(set().union(*[set(d) for d in det_sets.values()])) if det_sets else []
    # 1) detector bat dong: so nguoi conf cao ma detector kia khong thay
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for f in frames:
                da = [d for d in det_sets[a].get(f, []) if d[4] >= min_det_conf]
                db = det_sets[b].get(f, [])
                miss_b, miss_a = _match(da, db), _match([d for d in db if d[4] >= min_det_conf], det_sets[a].get(f, []))
                if miss_a + miss_b:
                    add(f, "detector_disagree", f"{a} vs {b}: {miss_b} nguoi chi {a} thay, {miss_a} nguoi chi {b} thay",
                        {"only_" + a: miss_b, "only_" + b: miss_a}, W["detector_disagree"] * min(3, miss_a + miss_b))
    # 2) tracker bat dong: moi cap tracker, track A khop track B khac voi anh xa thuong gap
    tn = list(trk_sets)
    for i, a in enumerate(tn):
        for b in tn[i + 1:]:
            ta, tb = trk_sets[a], trk_sets[b]
            vote: dict = defaultdict(lambda: defaultdict(int))
            pairs: dict = defaultdict(dict)
            for x, tr_a in ta.items():
                for f, bx in tr_a.items():
                    best = max(((G.box_iou(bx, tb_[f]), y) for y, tb_ in tb.items() if f in tb_), default=(0, None))
                    if best[0] >= 0.5:
                        vote[x][best[1]] += 1
                        pairs[x][f] = best[1]
            for x, pf in pairs.items():
                main = max(vote[x], key=vote[x].get)
                for f, y in pf.items():
                    if y != main and vote[x][main] >= 5:
                        add(f, "tracker_disagree", f"{a}#{x} va {b}: ID {b}#{y} thay vi {b}#{main} (mot trong hai doi ID?)",
                            {"tracker_a": f"{a}#{x}", "tracker_b_expected": f"{b}#{main}", "tracker_b_actual": f"{b}#{y}"})
    # 3) conf giam, 4) chong lan, 5) track dut, theo tung tracker
    for tname, tr in trk_sets.items():
        sc = (scores or {}).get(tname, {})
        for tid, t in tr.items():
            fr = sorted(t)
            s = sc.get(tid, {})
            for p, q in zip(fr, fr[1:]):
                if q - p > 1:
                    add(p, "track_break", f"{tname}#{tid} dut f{p+1}..f{q-1}", {"gap": q - p - 1})
                    add(q, "track_break", f"{tname}#{tid} noi lai sau {q-p-1} frame (cung ID?)", {"gap": q - p - 1})
                if p in s and q in s and s[q] < 0.4 and s[p] - s[q] > 0.25:
                    add(q, "conf_drop", f"{tname}#{tid} conf {s[p]:.2f}->{s[q]:.2f}", {"from": s[p], "to": s[q]})
        ids = sorted(tr)
        for i, x in enumerate(ids):
            for y in ids[i + 1:]:
                for f in set(tr[x]) & set(tr[y]):
                    iou = G.box_iou(tr[x][f], tr[y][f])
                    if iou >= overlap_iou:
                        add(f, "overlap", f"{tname}#{x} chong {tname}#{y} IoU {iou:.2f} (nguy co doi ID)", {"iou": iou})
    # 6) noi suy lech > tol (so voi keyframe da chon) - ke hoach tu keyframes.plan_track
    for key, plan in (kf_plans or {}).items():
        err = plan["error"]
        if err["max"] > tol + 1e-6 and err["worst_frame"] is not None:
            add(err["worst_frame"], "interp_error", f"{key}: noi suy lech {err['max']:.1f}px > {tol}px", {"max": err["max"]})
    out = []
    for f, its in items.items():
        agg = defaultdict(float)
        for it in its:
            agg[it["kind"]] = max(agg[it["kind"]], it["weight"])
        out.append({"frame": f, "risk": round(sum(agg.values()), 2), "reasons": its})
    return sorted(out, key=lambda x: -x["risk"])


def render_frame(img, trackers: dict, dets: list | None, f: int, width: int = 720) -> str:
    import cv2
    im = img.copy()
    pal = [(0, 128, 255), (0, 170, 0), (200, 0, 160), (0, 0, 220), (150, 120, 0), (0, 160, 160)]
    for d in dets or []:
        cv2.rectangle(im, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (160, 160, 160), 1)
    for k, (name, tr) in enumerate(trackers.items()):
        col = pal[k % len(pal)]
        for tid, t in tr.items():
            if f in t:
                b = t[f]
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2 if k == 0 else 1)
                cv2.putText(im, f"{name[:3]}{tid}", (int(b[0]), max(12, int(b[1]) - 3 - 14 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    s = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0] * s)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode()


def write_html(path, queue: list[dict], source, trk_sets: dict, dets: dict | None, title: str, top: int = 30,
               identities=("P1", "P2", "P3", "P4", "P5"), meta: dict | None = None) -> int:
    cards = []
    for it in queue[:top]:
        f = it["frame"]
        b64 = render_frame(source.image(f), trk_sets, (dets or {}).get(f), f)
        why = "".join(f"<li><b>{html.escape(r['kind'])}</b>: {html.escape(r['msg'])}</li>" for r in it["reasons"][:8])
        cards.append(f"""<section class="card" data-frame="{f}"><h3>Frame {f} <small>(MOT {f+1}) risk {it['risk']}</small></h3>
<img src="data:image/jpeg;base64,{b64}"><ul>{why}</ul>
<label>Quyet dinh: <select class="dec"><option value="">chua duyet</option><option>dung</option><option>sai - sua tay</option><option>khong chac - mo CVAT</option></select></label>
<label>Ghi chu / anh xa {'/'.join(identities)}: <input class="note" size="40" placeholder="vd P2=botsort#7"></label></section>""")
    doc = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>:root{{--bg:#fff;--fg:#111;--mut:#555;--line:#ccc}}body{{font:14px system-ui;background:var(--bg);color:var(--fg);margin:0 auto;max-width:900px;padding:0 16px}}
.card{{border:1px solid var(--line);padding:12px;margin:16px 0;border-radius:6px}}img{{max-width:100%}}small{{color:var(--mut)}}li{{margin:2px 0}}button{{padding:8px 14px}}</style>
<h1>{html.escape(title)}</h1>
<p>Hang doi review xep theo rui ro. <b>Day la goi y cua tracker AI (ai_draft), khong phai nhan.</b> Mau cac khung = tracker khac nhau; khung xam = detection. Quyet dinh cua ban luu rieng:
<button onclick="dl()">Tai quyet dinh (JSON)</button></p><p><small>{html.escape(json.dumps(meta or {}, ensure_ascii=False))}</small></p>
{''.join(cards)}
<script>const K='cet-review-'+document.title;function save(){{const d={{}};document.querySelectorAll('.card').forEach(c=>{{d[c.dataset.frame]={{decision:c.querySelector('.dec').value,note:c.querySelector('.note').value}}}});try{{localStorage.setItem(K,JSON.stringify(d))}}catch(e){{}}return d}}
try{{const d=JSON.parse(localStorage.getItem(K)||'{{}}');document.querySelectorAll('.card').forEach(c=>{{const x=d[c.dataset.frame];if(x){{c.querySelector('.dec').value=x.decision;c.querySelector('.note').value=x.note}}}})}}catch(e){{}}
document.addEventListener('change',save);document.addEventListener('input',save);
function dl(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(save(),null,1)],{{type:'application/json'}}));a.download='review_decisions.json';a.click()}}</script></html>"""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(doc, encoding="utf8")
    return min(top, len(queue))
