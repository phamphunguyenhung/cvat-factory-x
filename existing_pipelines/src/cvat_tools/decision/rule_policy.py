"""Luat tat dinh: loi cua review.validators -> action (action_schema). Khong dung model, khong doan.

Moi shape nhan toi da 1 action/vong theo thu tu uu tien PRECEDENCE. Shape do nguoi ve chi nhan KEEP/FLAG_REVIEW.
Loi con lai sau khi luat da thu (persistent) -> FLAG_REVIEW.

review_priority_score (0-100, CHUA HIEU CHINH; thu tu xep hang, khong phai xac suat):
    score = 100 * (0.40*severity + 0.25*uncertainty + 0.20*area + 0.15*disagreement)
  severity    = max(error 1.0, warn 0.5, info 0.2 cua shape) + 0.05*(so loi - 1), cat o 1
  uncertainty = 1 - rule_confidence(rule_id)   (bang RULE_CONF ben duoi, dat tay; FLAG_REVIEW = 1.0)
  area        = sqrt(min(1, dien_tich_shape / (0.05 * dien_tich_anh)))   (anh huong; bao hoa o 5% anh)
  disagreement= |p1 - p2| neu co 2 model bat dong (votes), nguoc lai 0
"""
from __future__ import annotations

from cvat_tools.decision import action_schema as S
from cvat_tools.review import validators as V

# code -> (rule_id, do tin cay luat DAT TAY, chua hieu chinh)
RULE_CONF = {
    "R_DUP": ("R_DUP_DELETE_LOWER_CONF", 0.80),
    "R_INVALID": ("R_POLY_MAKE_VALID", 0.85),
    "R_TINY": ("R_POLY_DEGENERATE_DELETE", 0.70),
    "R_ROAD": ("R_CLIP_TO_ROAD", 0.70),
    "R_VMAX": ("R_SIMPLIFY_TO_MAX_VERTICES", 0.75),
    "R_NARROW": ("R_SNAP_NARROW", 0.50),
    "R_AREA": ("R_RESHARE_BOUNDARY", 0.65),
    "R_STUFF_MASK": ("R_SUBTRACT_THINGS_MASK", 0.70),
    "R_STUFF_BOX": ("R_SUBTRACT_THINGS_BOX", 0.45),   # vung vat the chi la hop: kem tin cay hon
    "R_FLAG": ("R_FLAG_UNKNOWN", 0.0),
}
ACTIVE_WARN = {"STUFF_OVER_THING"}   # ma muc warn van duoc luat xu ly (min_severity khong loai)
PRECEDENCE = ["LABEL_UNKNOWN", "SHAPE_TYPE", "POLY_TOO_FEW", "POLY_TINY", "POLY_INVALID", "POLY_HOLE", "DUPLICATE",
              "NOT_IN_ROAD", "VERTEX_MAX", "VERTEX_MIN", "AREA_OVERLAP", "LANE_TOO_WIDE", "POLE_SHAPE", "BOX_TINY",
              "Z_ORDER", "STUFF_OVER_THING"]
DEFAULTS = {"min_severity": "error", "not_in_road_max_cut": 0.5, "snap_band_lane": 6.0, "snap_band_pole": 4.0,
            "simplify_band": 2.0}
SEV_W = {"error": 1.0, "warn": 0.5, "info": 0.2}


def cfg_of(rs: dict) -> dict:
    return {**DEFAULTS, **((rs.get("decision") or {}).get("policy") or {})}


def build_ctx(job: int, frame: int, shapes: list[dict], issues: list[dict], rs: dict, thing_masks: dict | None = None) -> dict:
    """ctx cho action_schema: shapes, roi_id theo shape, proposal_id theo cap AREA_OVERLAP. ID do he thong dat."""
    ctx = {"job": job, "frame": frame, "shapes": {s["id"]: s for s in shapes}, "rs": rs,
           "rois": {f"roi:{s['id']}": s["id"] for s in shapes}, "proposals": {},
           "thing_masks": thing_masks or {}}
    for i in issues:
        if i["code"] == "AREA_OVERLAP":
            pair = [i["shape_id"], i["data"]["other"]]
            ctx["proposals"][_pid(pair)] = {"kind": "shared_boundary", "shapes": pair}
    return ctx


def _pid(pair) -> str:
    return "prop:shared:" + "|".join(pair)


def _area_frac(shape: dict, frame_wh) -> float:
    g = V.poly_of(shape)
    if frame_wh is None or g.is_empty:
        return 0.5
    return min(1.0, g.area / (0.05 * frame_wh[0] * frame_wh[1])) ** 0.5


def priority_score(shape_issues: list[dict], rule_conf: float, shape: dict, frame_wh=None, votes=None) -> float:
    sev = min(1.0, max((SEV_W[i["severity"]] for i in shape_issues), default=0.2) + 0.05 * max(0, len(shape_issues) - 1))
    dis = abs(votes[0] - votes[1]) if votes and len(votes) == 2 else 0.0
    return round(100 * (0.40 * sev + 0.25 * (1 - rule_conf) + 0.20 * _area_frac(shape, frame_wh) + 0.15 * dis), 1)


def _flag(sid_list, code, note, evidence):
    return {"schema_version": S.SCHEMA_VERSION, "action": "FLAG_REVIEW", "shape_ids": list(sid_list),
            "reason_code": code, "evidence_refs": evidence, "params": {"note": note}}


def decide(shapes: list[dict], issues: list[dict], rs: dict, frame_wh=None, job: int = 0, frame: int = 0,
           persistent: set | None = None, thing_masks: dict | None = None) -> tuple[list[dict], dict]:
    """-> (records, ctx). records = ban ghi quyet dinh (action_schema.new_record) da sap theo priority giam dan.

    persistent: {(shape_id, code)} da duoc luat xu ly o vong truoc nhung van con -> FLAG_REVIEW.
    """
    cfg = cfg_of(rs)
    persistent = persistent or set()
    minr = V.SEV_RANK[cfg["min_severity"]]
    ctx = build_ctx(job, frame, shapes, issues, rs, thing_masks)
    smap = ctx["shapes"]
    by_shape: dict = {}
    for i in issues:
        by_shape.setdefault(i["shape_id"], []).append(i)
    claimed: set = set()
    out: list[dict] = []

    def emit(act, rid, rc, sid):
        errs = S.validate_action(act, ctx)
        if errs:  # luat khong duoc phep sinh action sai schema: ha cap thanh FLAG_REVIEW
            act = _flag(act["shape_ids"], act["reason_code"], "luat sinh action sai schema: " + "; ".join(errs)[:200], act["evidence_refs"])
            rid, rc = RULE_CONF["R_FLAG"]
        pr = priority_score(by_shape.get(sid, []), rc, smap[sid], frame_wh)
        out.append(S.new_record(job, frame, act, rule_id=rid, priority=pr))
        out[-1]["confidence"] = rc
        out[-1]["calibrated"] = False

    for sid, lst in by_shape.items():
        if sid in claimed or sid not in smap:
            continue
        sh = smap[sid]
        errs = [i for i in lst if V.SEV_RANK[i["severity"]] >= minr or i["code"] in ACTIVE_WARN]
        if not errs:
            continue
        byc = {i["code"]: i for i in errs}
        code = next((c for c in PRECEDENCE if c in byc), None)
        ev = [f"issue:{sid}:{i['code']}" for i in errs]
        human = S.origin_of(sh) != "tool"

        def mk(action, params=None, ids=None):
            return {"schema_version": S.SCHEMA_VERSION, "action": action, "shape_ids": ids or [sid],
                    "reason_code": code or errs[0]["code"], "evidence_refs": ev, "params": params or {}}

        if code is None:   # loi error ngoai bang luat
            code = errs[0]["code"]
            emit(mk("FLAG_REVIEW", {"note": f"ma loi chua co luat: {code}"}), *RULE_CONF["R_FLAG"], sid)
            claimed.add(sid)
            continue
        if human and code != "DUPLICATE":
            emit(mk("FLAG_REVIEW", {"note": "shape cua nguoi: khong sua tu dong"}), *RULE_CONF["R_FLAG"], sid)
            claimed.add(sid)
            continue
        if (sid, code) in persistent:
            emit(mk("FLAG_REVIEW", {"note": "luat da thu nhung loi con"}), *RULE_CONF["R_FLAG"], sid)
            claimed.add(sid)
            continue
        roi = f"roi:{sid}"
        if code in ("POLY_TOO_FEW", "POLY_TINY"):
            emit(mk("DELETE_DRAFT"), *RULE_CONF["R_TINY"], sid)
        elif code == "POLY_INVALID":
            emit(mk("SNAP_EDGE", {"roi_id": roi, "band_px": cfg["simplify_band"], "mode": "make_valid"}), *RULE_CONF["R_INVALID"], sid)
        elif code == "DUPLICATE":
            other = smap.get(byc["DUPLICATE"]["data"]["other"])
            if other is None or other["id"] in claimed:
                continue
            a, b = other, sh                      # a = ban truoc, b = ban bi danh dau
            ha, hb = S.origin_of(a) != "tool", S.origin_of(b) != "tool"
            ca, cb = (a.get("raw") or {}).get("_conf"), (b.get("raw") or {}).get("_conf")
            if ha and hb:
                emit(mk("FLAG_REVIEW", {"note": "ca hai ban trung deu do nguoi ve"}, [a["id"], b["id"]]), *RULE_CONF["R_FLAG"], sid)
            else:
                if ha:
                    victim = b
                elif hb:
                    victim = a
                elif ca is not None and cb is not None and ca != cb:
                    victim = a if ca < cb else b
                else:
                    victim = b                    # hoa/khong co conf: bo ban bi danh dau (ve sau)
                emit(mk("DELETE_DRAFT", ids=[victim["id"]]), *RULE_CONF["R_DUP"], victim["id"] if victim["id"] in by_shape else sid)
            claimed.update([a["id"], b["id"]])
            continue
        elif code == "NOT_IN_ROAD":
            if byc["NOT_IN_ROAD"]["data"].get("outside", 1.0) > cfg["not_in_road_max_cut"]:
                emit(mk("FLAG_REVIEW", {"note": "cat vao road se mat qua nhieu"}), *RULE_CONF["R_FLAG"], sid)
            else:
                emit(mk("SNAP_EDGE", {"roi_id": roi, "band_px": cfg["simplify_band"], "mode": "clip_road"}), *RULE_CONF["R_ROAD"], sid)
        elif code == "VERTEX_MAX":
            emit(mk("SNAP_EDGE", {"roi_id": roi, "band_px": cfg["simplify_band"], "mode": "simplify"}), *RULE_CONF["R_VMAX"], sid)
        elif code == "AREA_OVERLAP":
            pair = [sid, byc["AREA_OVERLAP"]["data"]["other"]]
            if pair[1] in claimed or S.origin_of(smap[pair[1]]) != "tool":
                emit(mk("FLAG_REVIEW", {"note": "bien chung voi shape khong sua duoc"}), *RULE_CONF["R_FLAG"], sid)
            else:
                emit(mk("SPLIT_PROPOSAL", {"proposal_id": _pid(pair)}, pair), *RULE_CONF["R_AREA"], sid)
                claimed.update(pair)
                continue
        elif code == "STUFF_OVER_THING":
            only_box = all(t["source"] == "box" for t in byc["STUFF_OVER_THING"]["data"]["things"])
            emit(mk("SNAP_EDGE", {"roi_id": roi, "band_px": cfg["simplify_band"], "mode": "subtract_things"}),
                 *RULE_CONF["R_STUFF_BOX" if only_box else "R_STUFF_MASK"], sid)
        elif code in ("LANE_TOO_WIDE", "POLE_SHAPE"):
            band = cfg["snap_band_pole"] if code == "POLE_SHAPE" else cfg["snap_band_lane"]
            emit(mk("SNAP_EDGE", {"roi_id": roi, "band_px": band, "mode": "snap"}), *RULE_CONF["R_NARROW"], sid)
        else:  # LABEL_UNKNOWN, SHAPE_TYPE, POLY_HOLE, VERTEX_MIN, BOX_TINY, Z_ORDER: khong du bang chung de tu sua
            emit(mk("FLAG_REVIEW", {"note": f"{code}: can nguoi/VLM xem"}), *RULE_CONF["R_FLAG"], sid)
        claimed.add(sid)
    out.sort(key=lambda r: -(r["priority_score"] or 0))
    return out, ctx
