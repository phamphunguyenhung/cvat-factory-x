"""Thuc thi action: review/apply.py (lenh sua, bam canh) + segmentation/geomrefine (hinh hoc) -> revision bat bien trong shapestore.

KHONG ghi CVAT. Moi action da validate (action_schema) moi duoc chay; shape cua nguoi khong bao gio bi sua/xoa.
Revision: moi action doi shape -> them dong moi (stage llm_edit, provenance.actor = rule|llm, parent_id = ban truoc);
shape bi xoa -> dong status='rejected' (dong cu van con). 'rule_edit' khong co trong CHECK(stage) cua bang shapes nen
dung llm_edit + provenance.actor='rule' (tranh phai dung lai bang).
"""
from __future__ import annotations

import copy
import math

from shapely.ops import unary_union

from cvat_tools.decision import action_schema as S
from cvat_tools.decision import rule_policy as RP
from cvat_tools.review import apply as A
from cvat_tools.review import validators as V
from cvat_tools.segmentation import geomrefine as G
from cvat_tools.segmentation import stuff_things as ST


class GuardError(Exception):
    pass


def _flat_ring(poly, min_gap: float = 0.6):
    """Vong ngoai -> list phang, bo dinh lien tiep gan nhau (< min_gap px) de khong sinh DUP_VERTEX."""
    pts = [(A._r(x), A._r(y)) for x, y in list(poly.exterior.coords)[:-1]]
    keep = [p for k, p in enumerate(pts) if not k or math.dist(p, pts[k - 1]) >= min_gap]
    return [c for xy in keep for c in xy]


def _exec_subtract_things(sh, rs, shapes, thing_masks):
    """Tru vung vat the khoi polygon nen (stuff_things.subtract_things); khong lo; manh phu -> shape moi `<id>.<k>`."""
    from shapely.geometry import Polygon
    regs = V.thing_regions(list(shapes.values()), rs, thing_masks)
    hit = V.stuff_intrusions(sh, regs, rs)
    if not hit:
        raise ValueError("khong con vat the bi de")
    cfg = rs["review"]
    fix = ST.subtract_things(sh["points"], [h["region"] for h in hit], min_ratio=cfg["stuff_min_ratio"],
                             box_inset=cfg["stuff_box_inset"])
    parts = sorted((Polygon(zip(f[0::2], f[1::2])) for f in fix.polygons), key=lambda p: -p.area)
    parts = [p for p in parts if p.is_valid and not p.is_empty]
    if not fix.changed or not parts:
        raise ValueError("tru vat the cho ket qua rong")
    raw = sh.setdefault("raw", {})
    if any(h["source"] == "box" for h in hit):
        raw["_issues"] = list(raw.get("_issues") or []) + ["cat_theo_hop"]
    sh["points"] = _flat_ring(parts[0])
    for k, p in enumerate(parts[1:], 1):
        ns = copy.deepcopy(sh)
        ns["id"], ns["points"] = f"{sh['id']}.{k}", _flat_ring(p)
        shapes[ns["id"]] = ns
    return {"mode": "subtract_things", "removed_px2": fix.removed_area, "parts": len(parts),
            "things": [h["thing_id"] for h in hit], "sources": sorted({h["source"] for h in hit})}


def _exec_snap(sh, mode, band, img, rs, shapes, thing_masks=None):
    if mode == "snap":
        return A.snap_shape(sh, img, band, rs)
    p = V.poly_of(sh)
    if mode == "make_valid":
        q = G.valid_polygon(p)
        if q is None:
            raise ValueError("khong sua duoc polygon")
    elif mode == "simplify":
        grp = V.group_of(sh["label"], rs)
        cap = rs["max_vertices"].get(grp) or 8
        q = G._cap_points(G.valid_polygon(p) or p, 0.6, cap)
    elif mode == "subtract_things":
        return _exec_subtract_things(sh, rs, shapes, thing_masks)
    elif mode == "clip_road":
        roads = [V._valid_geom(s) for s in shapes.values() if V.canon_label(s["label"], rs) == "road"]
        roads = [r for r in roads if not r.is_empty]
        if not roads:
            raise ValueError("khong co road")
        q = G.valid_polygon(V._valid_geom(sh).intersection(unary_union(roads).buffer(rs["review"]["road_tol"])))
        if q is None:
            raise ValueError("cat vao road rong")
    else:
        raise ValueError(mode)
    if not q.is_valid or q.is_empty:
        raise ValueError("ket qua khong hop le")
    sh["points"] = _flat_ring(q)
    return {"mode": mode}


def execute(rec: dict, shapes: dict, ctx: dict, img=None, wh=None) -> str:
    """Ap dung 1 record da validate len `shapes` (dict, doi tai cho). Tra detail. Nem GuardError/ValueError."""
    rs = ctx["rs"]
    act, p, ids = rec["proposed_action"], rec["params"], rec["shape_ids"]
    for i in ids:  # chot chan cuoi: khong bao gio sua/xoa shape cua nguoi
        if act not in S.READONLY_ACTIONS and S.origin_of(shapes[i]) != "tool":
            raise GuardError(f"{act} tren shape khong phai tool: {i}")
    if act in ("KEEP", "FLAG_REVIEW"):
        return "no-op"
    if act == "DELETE_DRAFT":
        detail, _ = A._apply_one({"op": "delete_shape", "shape": ids[0]}, shapes, img, wh, rs)
        return detail
    if act == "RELABEL":
        detail, _ = A._apply_one({"op": "relabel", "shape": ids[0], "label": p["label"]}, shapes, img, wh, rs)
        return detail
    if act == "SNAP_EDGE":
        return str(_exec_snap(shapes[ids[0]], p.get("mode", "snap"), p["band_px"], img, rs, shapes, ctx.get("thing_masks")))
    prop = ctx["proposals"][p["proposal_id"]]
    if prop["kind"] == "shared_boundary":
        a, b = shapes[prop["shapes"][0]], shapes[prop["shapes"][1]]
        pa, pb = G.smooth_shared_boundary(V.flat_of(a), V.flat_of(b))
        a["points"], b["points"] = [A._r(v) for v in pa], [A._r(v) for v in pb]
        return "shared boundary"
    cmd = ({"op": "split", "shape": prop["shapes"][0], "line": prop["line"]} if prop["kind"] == "split"
           else {"op": "merge", "shapes": prop["shapes"]})
    detail, _ = A._apply_one(cmd, shapes, img, wh, rs)
    return detail


def _same(a, b) -> bool:
    return (a["points"], a["label"], a["z_order"]) == (b["points"], b["label"], b["z_order"])


def run_frame(job: int, frame: int, shapes: list[dict], img=None, rs: dict | None = None, store=None,
              rounds: int = 2, adjudicator=None, frame_wh=None, apply: bool = True, thing_masks: dict | None = None) -> dict:
    """validate -> luat -> (VLM) -> thuc thi tren BAN SAO -> validate lai. Tra dict ket qua, gom records + so loi truoc/sau."""
    rs = rs or V.load_ruleset(job)
    if frame_wh is None and img is not None:
        frame_wh = (img.shape[1], img.shape[0])
    cur = {s["id"]: copy.deepcopy(s) for s in shapes}
    issues0 = V.validate_frame(list(cur.values()), frame_wh, rs, thing_masks)
    rowmap = _load_store(store, job, frame, cur) if store is not None else {}
    all_recs: list[dict] = []
    persistent: set = set()
    seen_flags: set = set()
    issues = issues0
    for rnd in range(1, rounds + 1):
        recs, ctx = RP.decide(list(cur.values()), issues, rs, frame_wh, job, frame, persistent, thing_masks)
        if adjudicator is not None:
            recs = _with_adjudication(recs, ctx, adjudicator, job, frame)
        recs = [r for r in recs if not (r["proposed_action"] == "FLAG_REVIEW"
                                        and (tuple(r["shape_ids"]), r["reason_code"]) in seen_flags)]
        for r in recs:
            r["round"] = rnd
            if r["proposed_action"] == "FLAG_REVIEW":
                seen_flags.add((tuple(r["shape_ids"]), r["reason_code"]))
            errs = S.validate_record(r, ctx)
            if errs:
                r["validation_status"], r["errors"] = "schema_invalid", errs
            else:
                r["validation_status"] = "schema_ok"
        if apply:
            for r in recs:
                if r["validation_status"] != "schema_ok" or r["proposed_action"] in S.READONLY_ACTIONS:
                    continue
                if any(i not in cur for i in r["shape_ids"]):
                    r["validation_status"] = "stale"
                    continue
                before = copy.deepcopy(cur)
                try:
                    r["detail"] = execute(r, cur, ctx, img, frame_wh)
                    changed = [i for i in cur if i not in before or not _same(cur[i], before[i])]
                    gone = [i for i in before if i not in cur]
                    bad = [i for i in changed if V.poly_of(cur[i]).is_empty or not V.poly_of(cur[i]).is_valid]
                    if bad:
                        raise ValueError(f"shape hong sau khi sua: {bad}")
                    r["validation_status"] = "applied_ok" if (changed or gone) else "no_effect"
                    if changed or gone:
                        _record_revisions(store, rowmap, before, cur, changed, gone, r)
                except (GuardError, ValueError, KeyError) as e:
                    cur.clear()
                    cur.update(before)
                    r["validation_status"], r["errors"] = "applied_reverted", [f"{type(e).__name__}: {e}"]
        for r in recs:
            if store is not None:
                if r["before_revision"] is None:
                    r["before_revision"] = rowmap.get(r["shape_ids"][0])
                store.add_decision(r)
        all_recs += recs
        issues = V.validate_frame(list(cur.values()), frame_wh, rs, thing_masks)
        if not apply:
            break
        persistent |= {(r["shape_ids"][0], r["reason_code"]) for r in recs
                       if r["proposed_action"] not in S.READONLY_ACTIONS and r["validation_status"] in ("applied_ok", "no_effect", "applied_reverted")}
        if not any(r["validation_status"] == "applied_ok" for r in recs):
            break
    if apply:  # loi error con lai (luat het vong / khong co luat) -> FLAG_REVIEW de khong bo sot
        left = {(i["shape_id"], i["code"]) for i in issues if i["severity"] == "error"}
        last, ctx = RP.decide(list(cur.values()), issues, rs, frame_wh, job, frame, persistent | left, thing_masks)
        last = [r for r in last if r["proposed_action"] == "FLAG_REVIEW"
                and (tuple(r["shape_ids"]), r["reason_code"]) not in seen_flags]
        for r in last:
            r["round"] = rounds + 1
            r["validation_status"] = "schema_ok" if not S.validate_record(r, ctx) else "schema_invalid"
            if store is not None:
                r["before_revision"] = rowmap.get(r["shape_ids"][0])
                store.add_decision(r)
        all_recs += last
    return {"frame": frame, "shapes": list(cur.values()), "records": all_recs,
            "issues_before": issues0, "issues_after": issues,
            "counts_before": V.count_by_code(issues0), "counts_after": V.count_by_code(issues)}


def _with_adjudication(recs, ctx, adj, job, frame):
    """Voi FLAG_REVIEW cua luat: xin VLM phan xu (neu bat); action VLM chi nhan khi qua schema, nguoc lai giu FLAG."""
    out = []
    flagged = [r for r in recs if r["proposed_action"] == "FLAG_REVIEW"]
    props = adj.adjudicate(ctx, flagged) if flagged else {}
    for r in recs:
        new = props.get(tuple(r["shape_ids"])) if r["proposed_action"] == "FLAG_REVIEW" else None
        out.append(new or r)
    return out


# ------------------------------------------------------------------ shapestore
def _load_store(store, job, frame, cur) -> dict:
    """Dua ban nhap vao kho (stage rule_draft; human_final neu origin=human). Tra {shape_id: row_id}."""
    m = {}
    for sid, s in cur.items():
        stage = "human_final" if S.origin_of(s) == "human" else "rule_draft"
        m[sid] = store.add(job, frame, s["label"], s["type"], s["points"], stage, z_order=s["z_order"], occluded=s["occluded"],
                           conf=(s.get("raw") or {}).get("_conf"), provenance={"shape_id": sid, "src": (s.get("raw") or {}).get("_src")})
    return m


def _record_revisions(store, rowmap, before, cur, changed, gone, rec):
    if store is None:
        return
    actor = "llm" if rec.get("model_id") else "rule"
    prov = {"actor": actor, "rule_id": rec.get("rule_id"), "model_id": rec.get("model_id"), "action": rec["proposed_action"],
            "reason_code": rec["reason_code"]}
    first_after = None
    for sid in gone:
        row = store.get(rowmap[sid])
        rid = store.add(row["job"], row["frame"], row["label"], row["shape_type"], row["points"], "llm_edit", z_order=row["z_order"],
                        occluded=row["occluded"], parent_id=rowmap[sid], status="rejected", provenance=prov)
        rec["before_revision"] = rec.get("before_revision") or rowmap[sid]
        first_after = first_after or rid
    for sid in changed:
        s = cur[sid]
        src = sid if sid in rowmap else sid.rsplit(".", 1)[0]
        parent = rowmap.get(src)
        rid = store.add(store.get(parent)["job"], store.get(parent)["frame"], s["label"], s["type"], s["points"], "llm_edit",
                        z_order=s["z_order"], occluded=s["occluded"], parent_id=parent, provenance={**prov, "shape_id": sid})
        rec["before_revision"] = rec.get("before_revision") or parent
        rowmap[sid] = rid
        first_after = first_after or rid
    rec["after_revision"] = first_after
