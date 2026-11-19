"""Schema LENH SUA (JSON) cho buoc review. LLM tra mot list lenh; apply.py thuc hien + bam canh tat dinh.

Quy tac chong toa do tho: voi label HEP (lane/*, curb, pole) CAM toa do tuyet doi ('to', 'at'):
dung dich tuong doi (dx,dy), phan so doc canh ('t') va 'snap_band' de bam canh.

Lenh (moi lenh co 'op', 'shape' (id), tuy chon 'snap_band' cho lenh doi hinh):
  move_vertex    {shape, idx, dx, dy | to:[x,y], snap_band?}   (rectangle: idx 0=TL 1=TR 2=BR 3=BL)
  delete_vertex  {shape, idx}
  add_vertex     {shape, after_idx, t | at:[x,y]}              t in [0,1] doc canh after_idx -> after_idx+1
  delete_shape   {shape}
  relabel        {shape, label}
  merge          {shapes:[id,id,...], label?}                  hop nhat (union / hop bao); giu id dau
  split          {shape, line:[[x,y],[x,y]]}                   polygon, cat theo duong thang
  snap_to_edge   {shape, band_px}
  set_flag       {shape, flag: occluded|truncated, value?: bool}
"""
from __future__ import annotations

from cvat_tools.review import validators as V

MAX_MOVE_NARROW = 40.0
MAX_BAND = 12.0
OPS = ("move_vertex", "delete_vertex", "add_vertex", "delete_shape", "relabel", "merge", "split",
       "snap_to_edge", "set_flag")

EDIT_SCHEMA = {
    "format": "list of commands, e.g. [{\"op\": \"move_vertex\", \"shape\": \"f61_003\", \"idx\": 2, \"dx\": -3, \"dy\": 0, \"snap_band\": 4}]",
    "ops": {
        "move_vertex": "shape, idx, (dx,dy) | to[x,y], snap_band?",
        "delete_vertex": "shape, idx",
        "add_vertex": "shape, after_idx, t | at[x,y]",
        "delete_shape": "shape",
        "relabel": "shape, label",
        "merge": "shapes[...], label?",
        "split": "shape, line[[x,y],[x,y]]",
        "snap_to_edge": "shape, band_px",
        "set_flag": "shape, flag(occluded|truncated), value?",
    },
    "narrow_labels_rule": f"lane/*, lane/road curb, pole: forbid 'to'/'at' (absolute); use dx,dy (<= {MAX_MOVE_NARROW}px), t, snap_band",
}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pt(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(_num(x) for x in v)


def validate_command(cmd, shapes: dict | None = None, rs: dict | None = None) -> list[str]:
    """Tra danh sach loi (rong = hop le). shapes: {id: shape} de kiem tham chieu/label hep/kieu hinh."""
    rs = rs or V.load_ruleset()
    if not isinstance(cmd, dict):
        return ["lenh phai la object"]
    op = cmd.get("op")
    if op not in OPS:
        return [f"op khong hop le: {op!r} (cho phep {', '.join(OPS)})"]
    errs: list[str] = []
    ids = cmd.get("shapes") if op == "merge" else [cmd.get("shape")]
    if op == "merge":
        if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(i, str) for i in ids):
            errs.append("merge can 'shapes': list >= 2 id")
            ids = []
    elif not isinstance(ids[0], str):
        errs.append("thieu 'shape' (id chuoi)")
        ids = []
    sh = None
    if shapes is not None:
        for i in ids:
            if i not in shapes:
                errs.append(f"shape '{i}' khong ton tai")
        sh = shapes.get(ids[0]) if ids else None
    narrow = bool(sh) and V.is_narrow(sh["label"], rs)
    rect = bool(sh) and sh["type"] == "rectangle"
    sb = cmd.get("snap_band")
    if sb is not None and (not _num(sb) or not 0 < sb <= MAX_BAND):
        errs.append(f"snap_band phai trong (0,{MAX_BAND}]")

    def idx(name):
        v = cmd.get(name)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errs.append(f"'{name}' phai la so nguyen >= 0")
            return None
        if sh is not None:
            n = len(V.vertices(sh))
            if v >= n:
                errs.append(f"'{name}'={v} vuot so dinh ({n})")
        return v

    if op == "move_vertex":
        idx("idx")
        has_d, has_to = ("dx" in cmd or "dy" in cmd), "to" in cmd
        if has_d == has_to:
            errs.append("move_vertex can dung MOT trong (dx,dy) hoac 'to'")
        if has_d:
            for k in ("dx", "dy"):
                if k in cmd and not _num(cmd[k]):
                    errs.append(f"'{k}' phai la so")
            lim = MAX_MOVE_NARROW if narrow else 200
            if all(_num(cmd.get(k, 0)) for k in ("dx", "dy")) and max(abs(cmd.get("dx", 0)), abs(cmd.get("dy", 0))) > lim:
                errs.append(f"dich > {lim}px")
        if has_to:
            if not _pt(cmd["to"]):
                errs.append("'to' phai la [x,y]")
            if narrow:
                errs.append("label hep: cam toa do tuyet doi 'to' (dung dx,dy + snap_band)")
    elif op == "delete_vertex":
        idx("idx")
        if rect:
            errs.append("rectangle khong xoa dinh")
        elif sh is not None and len(V.vertices(sh)) <= 3:
            errs.append("polygon con 3 dinh, khong xoa them")
    elif op == "add_vertex":
        idx("after_idx")
        if ("t" in cmd) == ("at" in cmd):
            errs.append("add_vertex can dung MOT trong 't' hoac 'at'")
        if "t" in cmd and not (_num(cmd["t"]) and 0 <= cmd["t"] <= 1):
            errs.append("'t' phai trong [0,1]")
        if "at" in cmd:
            if not _pt(cmd["at"]):
                errs.append("'at' phai la [x,y]")
            if narrow:
                errs.append("label hep: cam toa do tuyet doi 'at' (dung 't')")
        if rect:
            errs.append("rectangle khong them dinh")
    elif op == "relabel":
        lab = cmd.get("label")
        if not isinstance(lab, str) or V.group_of(lab, rs) is None:
            errs.append(f"label moi {lab!r} khong thuoc ruleset")
        elif sh is not None and V.expected_type(lab, rs) != sh["type"]:
            errs.append(f"label '{lab}' can kieu {V.expected_type(lab, rs)}, shape la {sh['type']}")
    elif op == "merge":
        if shapes is not None and ids and all(i in shapes for i in ids):
            if len({shapes[i]["type"] for i in ids}) > 1:
                errs.append("merge: cac shape phai cung kieu")
            labs = {shapes[i]["label"] for i in ids}
            if len(labs) > 1 and "label" not in cmd:
                errs.append("merge khac label: can 'label' dich")
        if "label" in cmd and V.group_of(cmd["label"], rs) is None:
            errs.append("label dich khong thuoc ruleset")
    elif op == "split":
        ln = cmd.get("line")
        if not (isinstance(ln, (list, tuple)) and len(ln) == 2 and all(_pt(p) for p in ln)):
            errs.append("'line' phai la [[x,y],[x,y]]")
        if rect:
            errs.append("rectangle khong split")
    elif op == "snap_to_edge":
        b = cmd.get("band_px")
        if not _num(b) or not 0 < b <= MAX_BAND:
            errs.append(f"'band_px' phai trong (0,{MAX_BAND}]")
    elif op == "set_flag":
        if cmd.get("flag") not in ("occluded", "truncated"):
            errs.append("'flag' phai la occluded|truncated")
        if "value" in cmd and not isinstance(cmd["value"], bool):
            errs.append("'value' phai la bool")
        if cmd.get("flag") == "truncated" and sh is not None and V.group_of(sh["label"], rs) != "object":
            errs.append("truncated chi cho 10 label vat the")
    return errs


def validate_commands(cmds, shapes=None, rs=None) -> dict:
    """shapes: list shape hoac {id: shape}. -> {'ok': bool, 'errors': {index: [msg]}}."""
    if isinstance(shapes, list):
        shapes = {s["id"]: s for s in shapes}
    if not isinstance(cmds, list):
        return {"ok": False, "errors": {-1: ["goc phai la list lenh"]}}
    errs = {}
    for i, c in enumerate(cmds):
        e = validate_command(c, shapes, rs)
        if e:
            errs[i] = e
    return {"ok": not errs, "errors": errs}
