"""Schema co phien ban cho ACTION va ban ghi quyet dinh.

Bat bien:
- Action chi THAM CHIEU id da ton tai (shape_id / roi_id / proposal_id); khong cho phep tu dat id, khong toa do tho.
- DELETE_DRAFT chi cho shape do TOOL tao (origin=tool); shape cua nguoi (origin=human) chi duoc KEEP / FLAG_REVIEW.
- Khong du bang chung -> FLAG_REVIEW.
Ngu canh (ctx): {"job","frame","shapes":{id:shape},"rois":{roi_id:shape_id},"proposals":{pid:{...}},"rs":ruleset}.
"""
from __future__ import annotations

import re

from cvat_tools.review import edits as E
from cvat_tools.review import validators as V

SCHEMA_VERSION = 1
ACTIONS = ("KEEP", "RELABEL", "DELETE_DRAFT", "SPLIT_PROPOSAL", "MERGE_PROPOSALS", "SNAP_EDGE", "FLAG_REVIEW")
READONLY_ACTIONS = ("KEEP", "FLAG_REVIEW")           # duoc phep tren shape cua nguoi
SNAP_MODES = ("snap", "simplify", "make_valid", "clip_road", "subtract_things")
APPROVALS = ("pending", "approved", "rejected")
VALIDATION_STATUS = ("pending", "schema_invalid", "schema_ok", "applied_ok", "applied_reverted", "no_effect", "stale")
ORIGINS = ("tool", "human")
MAX_BAND = E.MAX_BAND

PARAM_KEYS = {
    "KEEP": set(), "DELETE_DRAFT": set(), "RELABEL": {"label"},
    "SPLIT_PROPOSAL": {"proposal_id"}, "MERGE_PROPOSALS": {"proposal_id"},
    "SNAP_EDGE": {"roi_id", "band_px", "mode"}, "FLAG_REVIEW": {"note"},
}
TOP_KEYS = {"schema_version", "action", "shape_ids", "params", "reason_code", "evidence_refs", "confidence"}
FORBIDDEN_KEYS = {"points", "polygon", "polygons", "coords", "coordinates", "to", "at", "line", "bbox", "xy",
                  "vertices", "mask", "id", "new_id", "new_ids"}
_NUMS_IN_TEXT = re.compile(r"(\d+(\.\d+)?\s*[,; ]\s*){3,}\d")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{1,39}$")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def has_raw_coords(v) -> bool:
    """True neu cau truc chua toa do tho: khoa cam, hoac list >=2 so lien tiep, hoac chuoi nhieu so."""
    if isinstance(v, dict):
        return any(k in FORBIDDEN_KEYS or has_raw_coords(x) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        if len(v) >= 2 and all(_num(x) for x in v):
            return True
        return any(has_raw_coords(x) for x in v)
    if isinstance(v, str):
        return bool(_NUMS_IN_TEXT.search(v))
    return False


def origin_of(shape: dict) -> str:
    return shape.get("origin", "tool")


def validate_action(act, ctx: dict) -> list[str]:
    """Tra danh sach loi (rong = hop le)."""
    if not isinstance(act, dict):
        return ["action phai la object"]
    errs: list[str] = []
    extra = set(act) - TOP_KEYS
    if extra:
        errs.append(f"khoa la: {sorted(extra)}")
    if act.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version phai = {SCHEMA_VERSION}")
    kind = act.get("action")
    if kind not in ACTIONS:
        return errs + [f"action khong hop le: {kind!r} (cho phep {', '.join(ACTIONS)})"]
    rc = act.get("reason_code")
    if not isinstance(rc, str) or not _REASON.match(rc):
        errs.append("reason_code phai la CHU_HOA_GACH_DUOI")
    ev = act.get("evidence_refs", [])
    if not isinstance(ev, list) or not all(isinstance(x, str) and len(x) <= 200 for x in ev):
        errs.append("evidence_refs phai la list chuoi")
    conf = act.get("confidence")
    if conf is not None and not (_num(conf) and 0 <= conf <= 1):
        errs.append("confidence phai trong [0,1]")
    params = act.get("params", {})
    if not isinstance(params, dict):
        return errs + ["params phai la object"]
    if has_raw_coords(params) or has_raw_coords({k: v for k, v in act.items() if k not in ("params", "shape_ids")}):
        errs.append("cam toa do tho / id tu dat trong action (chi tham chieu id co san)")
    bad = set(params) - PARAM_KEYS[kind]
    if bad:
        errs.append(f"params khong cho phep cho {kind}: {sorted(bad)}")
    ids = act.get("shape_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        return errs + ["shape_ids phai la list chuoi khong rong"]
    shapes, rs = ctx["shapes"], ctx.get("rs") or V.load_ruleset()
    missing = [i for i in ids if i not in shapes]
    if missing:
        return errs + [f"shape_id khong ton tai: {missing}"]
    if len(set(ids)) != len(ids):
        errs.append("shape_ids trung")
    if kind not in READONLY_ACTIONS:
        hum = [i for i in ids if origin_of(shapes[i]) != "tool"]
        if hum:
            errs.append(f"{kind} bi cam tren shape cua nguoi/goc: {hum} (chi KEEP/FLAG_REVIEW)")
    one = kind in ("KEEP", "RELABEL", "DELETE_DRAFT", "SNAP_EDGE")
    if one and len(ids) != 1:
        errs.append(f"{kind} can dung 1 shape_id")
    sh = shapes[ids[0]]
    if kind == "RELABEL":
        lab = params.get("label")
        if not isinstance(lab, str) or V.group_of(lab, rs) is None:
            errs.append(f"label moi {lab!r} khong thuoc ruleset")
        elif lab == sh["label"]:
            errs.append("RELABEL sang chinh label hien tai")
        elif V.expected_type(lab, rs) != sh["type"]:
            errs.append(f"label '{lab}' can kieu {V.expected_type(lab, rs)}, shape la {sh['type']}")
    elif kind == "SNAP_EDGE":
        roi = params.get("roi_id")
        if roi not in ctx.get("rois", {}):
            errs.append(f"roi_id khong ton tai: {roi!r}")
        elif ctx["rois"][roi] != ids[0]:
            errs.append("roi_id khong thuoc shape_id")
        b = params.get("band_px")
        if not _num(b) or not 0 < b <= MAX_BAND:
            errs.append(f"band_px phai trong (0,{MAX_BAND}]")
        mode = params.get("mode", "snap")
        if mode not in SNAP_MODES:
            errs.append(f"mode phai thuoc {SNAP_MODES}")
        elif sh["type"] == "rectangle" and mode != "snap":
            errs.append("rectangle chi ho tro mode snap")
    elif kind in ("SPLIT_PROPOSAL", "MERGE_PROPOSALS"):
        pid = params.get("proposal_id")
        prop = ctx.get("proposals", {}).get(pid)
        if prop is None:
            errs.append(f"proposal_id khong ton tai: {pid!r}")
        else:
            want = {"SPLIT_PROPOSAL": ("split", "shared_boundary"), "MERGE_PROPOSALS": ("merge",)}[kind]
            if prop["kind"] not in want:
                errs.append(f"proposal {pid} kieu {prop['kind']} khong hop voi {kind}")
            if set(prop["shapes"]) != set(ids):
                errs.append("shape_ids khong khop shapes cua proposal")
        if kind == "MERGE_PROPOSALS" and len(ids) < 2:
            errs.append("MERGE_PROPOSALS can >= 2 shape_ids")
        if kind == "SPLIT_PROPOSAL" and prop and prop["kind"] == "split" and sh["type"] != "polygon":
            errs.append("split chi cho polygon")
    elif kind == "FLAG_REVIEW":
        note = params.get("note")
        if note is not None and (not isinstance(note, str) or len(note) > 300):
            errs.append("note phai la chuoi <= 300 ky tu")
    return errs


def new_record(job: int, frame: int, act: dict, *, rule_id: str | None = None, model_id: str | None = None,
               model_version: str | None = None, priority: float | None = None) -> dict:
    """Ban ghi quyet dinh tu action. before/after_revision dien khi ap dung."""
    return {
        "schema_version": SCHEMA_VERSION, "job_id": job, "frame_id": frame, "shape_ids": list(act["shape_ids"]),
        "proposed_action": act["action"], "params": dict(act.get("params") or {}), "reason_code": act["reason_code"],
        "evidence_refs": list(act.get("evidence_refs") or []), "rule_id": rule_id, "model_id": model_id,
        "model_version": model_version, "confidence": act.get("confidence"), "calibrated": False,
        "priority_score": priority, "before_revision": None, "after_revision": None,
        "validation_status": "pending", "approval_status": "pending",
    }


def record_action(rec: dict) -> dict:
    return {"schema_version": rec.get("schema_version"), "action": rec.get("proposed_action"),
            "shape_ids": rec.get("shape_ids"), "params": rec.get("params", {}), "reason_code": rec.get("reason_code"),
            "evidence_refs": rec.get("evidence_refs", []), "confidence": rec.get("confidence")}


def validate_record(rec: dict, ctx: dict) -> list[str]:
    errs = []
    if rec.get("approval_status") not in APPROVALS:
        errs.append("approval_status khong hop le")
    if rec.get("validation_status") not in VALIDATION_STATUS:
        errs.append("validation_status khong hop le")
    if not (rec.get("rule_id") or rec.get("model_id")):
        errs.append("can rule_id hoac model_id")
    return errs + validate_action(record_action(rec), ctx)
