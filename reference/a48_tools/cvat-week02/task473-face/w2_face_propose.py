"""Sinh đề xuất face50 cho 1 frame từ các mốc giải phẫu đã đọc bằng mắt (ảnh gốc), theo công thức BTC.
Ghi reports/w2_recovery/proposals/prop_<job>_<frame>.json (định dạng draft). KHÔNG ghi CVAT.
Mốc đầu vào (spec JSON): eyeL/eyeR {c:[x,y], w}, roll_deg, browL/browR (5 điểm top-edge), nose [x,y]x4, mouth {cl,cr,upper[5],lower[5],closure_y[3]}
python -X utf8 scripts/w2_face_propose.py spec.json"""
import json, math, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
ROOT = Path(__file__).resolve().parents[1]

def eye(c, w, roll, first, h=(3.0, 4.0, 3.0), hl=(2.5, 3.0, 2.5)):
    a = math.radians(roll); u = (math.cos(a), -math.sin(a)); n = (math.sin(a), math.cos(a))
    P = lambda t, off: (c[0] + u[0] * w * (t - .5) + n[0] * off, c[1] + u[1] * w * (t - .5) + n[1] * off)
    pts = [P(0, 0)] + [P(r, -h[i]) for i, r in enumerate((.25, .5, .75))] + [P(1, 0)] + [P(r, hl[i]) for i, r in zip((2, 1, 0), (.75, .5, .25))]
    return {str(first + i): p for i, p in enumerate(pts)}

pts = {}
roll = spec["roll_deg"]
L = eye(spec["eyeL"]["c"], spec["eyeL"]["w"], roll, 14)           # 14 ngoài ... 18 trong
# mắt phải: 22 là khoé TRONG (trái ảnh) -> 26 khoé ngoài; cùng công thức, bắt đầu từ trái ảnh
R = eye(spec["eyeR"]["c"], spec["eyeR"]["w"], roll, 22)
pts.update({k: v for k, v in L.items()}); pts.update(R)
for i, p in enumerate(spec["browL"]): pts[str(i)] = p
for i, p in enumerate(spec["browR"]): pts[str(5 + i)] = p
for i, p in enumerate(spec["nose"]): pts[str(10 + i)] = p
m = spec["mouth"]; cl, cr = m["cl"], m["cr"]
pts["30"], pts["36"] = cl, cr
for i, p in enumerate(m["upper"]): pts[str(31 + i)] = p
for i, p in enumerate(m["lower"]): pts[str(37 + i)] = p       # 37 gần khoé phải -> 41 gần khoé trái
pts["42"], pts["46"] = m["icl"], m["icr"]
for i, p in enumerate(m["inner_up"]): pts[str(43 + i)] = p
for i, p in enumerate(m["inner_lo"]): pts[str(47 + i)] = p
LAB = {"longmaytrai": range(0, 5), "longmayphai": range(5, 10), "songmui": range(10, 14), "mattrai": range(14, 22), "matphai": range(22, 30), "moingoai": range(30, 42), "moitrong": range(42, 50)}
occ = set(map(str, spec.get("occluded", [])))
people = [{"label": lab, "points": {str(i): {"x": round(pts[str(i)][0], 1), "y": round(pts[str(i)][1], 1), "state": "occluded" if str(i) in occ else "visible"} for i in r}} for lab, r in LAB.items()]
out = {"job": spec["job"], "kind": "face50", "status": "proposal_local_only", "frames": [{"frame": spec["frame"], "people": people}]}
o = ROOT / "reports/w2_recovery/proposals" / f"prop_{spec['job']}_{spec['frame']}.json"
o.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8"); print(o)
