import copy
import json
import math
import sqlite3

import pytest
from cvat_tools.core import shapestore as SS
from cvat_tools.decision import action_schema as S
from cvat_tools.decision import engine as EN
from cvat_tools.decision import rule_policy as RP
from cvat_tools.decision import vlm_adjudicator as VA
from cvat_tools.review import validators as V

WH = (400, 300)
RS = V.load_ruleset()


def sh(i, label, pts, typ="polygon", origin="tool", conf=None, **kw):
    z = V.expected_z(label, RS)
    return {"id": i, "frame": 1, "label": label, "type": typ, "points": [float(p) for p in pts],
            "z_order": -8 if z is None else z, "occluded": False, "truncated": False, "origin": origin,
            "raw": {"frame": 1, "type": typ, "_conf": conf}, **kw}


def rect(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


ROAD = sh("road", "road", rect(0, 150, 400, 300))


def run(shapes, **kw):
    return EN.run_frame(1, 1, shapes, None, RS, kw.pop("store", None), frame_wh=WH, **kw)


def acts(res):
    return [(r["reason_code"], r["proposed_action"], tuple(r["shape_ids"]), r["validation_status"]) for r in res["records"]]


def ctx_of(shapes, issues=()):
    return RP.build_ctx(1, 1, shapes, list(issues), RS)


def good(**kw):
    a = {"schema_version": 1, "action": "FLAG_REVIEW", "shape_ids": ["road"], "reason_code": "X_Y", "evidence_refs": [], "params": {}}
    a.update(kw)
    return a


# ---------------- schema
def test_schema_accepts_valid_and_rejects_fake_ids():
    ctx = ctx_of([ROAD])
    assert S.validate_action(good(), ctx) == []
    assert any("khong ton tai" in e for e in S.validate_action(good(shape_ids=["ghost"]), ctx))
    a = good(action="SNAP_EDGE", params={"roi_id": "roi:ghost", "band_px": 3})
    assert any("roi_id" in e for e in S.validate_action(a, ctx))
    a = good(action="SPLIT_PROPOSAL", params={"proposal_id": "prop:made_up"})
    assert any("proposal_id" in e for e in S.validate_action(a, ctx))


@pytest.mark.parametrize("bad", [
    {"params": {"note": "x", "points": [1, 2, 3, 4]}},
    {"params": {"note": [10.0, 20.0]}},
    {"params": {"note": "10,20,30,40,50,60"}},
    {"new_id": "f1_999"},
    {"id": "f1_999"},
])
def test_schema_rejects_raw_coords_and_self_made_ids(bad):
    assert S.validate_action(good(**bad), ctx_of([ROAD]))


def test_schema_rejects_wrong_action_type_and_convention():
    car = sh("c", "car", [10, 200, 50, 250], typ="rectangle")
    ctx = ctx_of([ROAD, car])
    assert S.validate_action(good(action="DELETE_EVERYTHING"), ctx)
    a = good(action="RELABEL", shape_ids=["c"], params={"label": "sky"})      # rectangle -> label polygon
    assert any("can kieu" in e for e in S.validate_action(a, ctx))
    a = good(action="RELABEL", shape_ids=["c"], params={"label": "not a label"})
    assert S.validate_action(a, ctx)
    roi = {"action": "SNAP_EDGE", "shape_ids": ["c"], "params": {"roi_id": "roi:c", "band_px": 99}}
    assert any("band_px" in e for e in S.validate_action(good(**roi), ctx))


def test_human_shape_only_keep_or_flag():
    h = sh("h", "car", [10, 200, 50, 250], typ="rectangle", origin="human")
    ctx = ctx_of([ROAD, h])
    for kind, par in (("DELETE_DRAFT", {}), ("RELABEL", {"label": "bus"}), ("SNAP_EDGE", {"roi_id": "roi:h", "band_px": 3})):
        assert any("cam" in e for e in S.validate_action(good(action=kind, shape_ids=["h"], params=par), ctx)), kind
    assert S.validate_action(good(action="KEEP", shape_ids=["h"]), ctx) == []
    assert S.validate_action(good(action="FLAG_REVIEW", shape_ids=["h"]), ctx) == []


def test_record_defaults_uncalibrated_pending():
    rec = S.new_record(1, 1, good(), rule_id="R")
    assert rec["calibrated"] is False and rec["approval_status"] == "pending" and rec["validation_status"] == "pending"
    assert S.validate_record(rec, ctx_of([ROAD])) == []
    rec["approval_status"] = "maybe"
    assert S.validate_record(rec, ctx_of([ROAD]))
    assert S.validate_record(S.new_record(1, 1, good()), ctx_of([ROAD]))  # thieu rule_id/model_id


# ---------------- luat
def test_rule_duplicate_deletes_lower_conf():
    a = sh("a", "car", [10, 200, 80, 260], typ="rectangle", conf=5.0)
    b = sh("b", "car", [10, 200, 80, 260], typ="rectangle", conf=9.0)
    res = run([ROAD, a, b])
    assert ("DUPLICATE", "DELETE_DRAFT", ("a",), "applied_ok") in acts(res)
    assert {s["id"] for s in res["shapes"]} == {"road", "b"}
    assert "DUPLICATE" not in res["counts_after"]


def test_rule_duplicate_keeps_human():
    a = sh("a", "car", [10, 200, 80, 260], typ="rectangle", conf=99.0, origin="human")
    b = sh("b", "car", [10, 200, 80, 260], typ="rectangle", conf=1.0)
    res = run([ROAD, a, b])
    assert ("DUPLICATE", "DELETE_DRAFT", ("b",), "applied_ok") in acts(res)
    assert "a" in {s["id"] for s in res["shapes"]}
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)           # nguoc lai: a la tool conf cao, b la nguoi -> xoa a
    a2["origin"], b2["origin"] = "tool", "human"
    res = run([ROAD, a2, b2])
    assert {s["id"] for s in res["shapes"]} == {"road", "b"}


def test_rule_duplicate_two_humans_flag_only():
    a = sh("a", "car", [10, 200, 80, 260], typ="rectangle", origin="human")
    b = sh("b", "car", [10, 200, 80, 260], typ="rectangle", origin="human")
    res = run([ROAD, a, b])
    assert {s["id"] for s in res["shapes"]} == {"road", "a", "b"}
    assert any(r["proposed_action"] == "FLAG_REVIEW" and r["reason_code"] == "DUPLICATE" for r in res["records"])


def test_rule_vertex_max_simplifies():
    pts = []
    for k in range(12):          # canh trai co nhieu dinh thua, van la cot hep
        pts += [300 + (k % 2) * 0.2, 20 + k * 10]
    pts += [306, 140, 306, 20]
    pole = sh("p", "pole", pts)
    res = run([ROAD, pole])
    rec = [r for r in res["records"] if r["reason_code"] == "VERTEX_MAX"]
    assert rec and rec[0]["proposed_action"] == "SNAP_EDGE" and rec[0]["params"]["mode"] == "simplify"
    new = next(s for s in res["shapes"] if s["id"] == "p")
    assert len(new["points"]) // 2 <= RS["max_vertices"]["pole"]
    assert math.isfinite(sum(new["points"]))


def test_rule_not_in_road_clip_or_flag():
    lane = sh("l", "lane/single white", [100, 120, 104, 120, 130, 290, 126, 290])
    r1 = run([ROAD, lane])
    assert any(r["reason_code"] == "NOT_IN_ROAD" and r["proposed_action"] == "SNAP_EDGE" and r["params"]["mode"] == "clip_road"
               for r in r1["records"])
    assert "NOT_IN_ROAD" not in r1["counts_after"]
    lane2 = sh("l2", "lane/single white", [100, 10, 104, 10, 110, 140, 106, 140])
    r2 = run([ROAD, lane2])
    assert any(r["reason_code"] == "NOT_IN_ROAD" and r["proposed_action"] == "FLAG_REVIEW" for r in r2["records"])
    assert next(s for s in r2["shapes"] if s["id"] == "l2")["points"] == lane2["points"]


def test_rule_area_overlap_reshares_boundary():
    road = sh("road", "road", rect(0, 100, 400, 300))
    d = sh("d", "area/direct", rect(100, 120, 250, 290))
    al = sh("al", "area/alternative", rect(200, 120, 350, 290))
    res = run([road, d, al])
    assert any(r["reason_code"] == "AREA_OVERLAP" and r["proposed_action"] == "SPLIT_PROPOSAL" and r["validation_status"] == "applied_ok"
               for r in res["records"])
    assert "AREA_OVERLAP" not in res["counts_after"]


def test_rule_unknown_code_flags():
    s = sh("s", "unobtainium", rect(0, 0, 10, 10))
    res = run([ROAD, s])
    assert any(r["reason_code"] == "LABEL_UNKNOWN" and r["proposed_action"] == "FLAG_REVIEW" for r in res["records"])


def test_rule_human_shape_with_error_is_only_flagged():
    lane = sh("l", "lane/single white", [100, 120, 104, 120, 130, 290, 126, 290], origin="human")
    res = run([ROAD, lane])
    assert next(s for s in res["shapes"] if s["id"] == "l")["points"] == lane["points"]
    assert all(r["proposed_action"] in S.READONLY_ACTIONS for r in res["records"])


def test_never_delete_human_even_if_forced():
    h = sh("h", "car", [10, 200, 80, 260], typ="rectangle", origin="human")
    ctx = ctx_of([ROAD, h])
    rec = S.new_record(1, 1, good(action="DELETE_DRAFT", shape_ids=["h"]), rule_id="R")
    assert S.validate_record(rec, ctx)                          # schema chan
    cur = {s["id"]: s for s in (ROAD, h)}
    with pytest.raises(EN.GuardError):                           # executor chan lan nua
        EN.execute(rec, cur, ctx)
    assert "h" in cur


def test_priority_score_formula_monotonic():
    car = sh("c", "car", [10, 200, 90, 280], typ="rectangle")
    err = [{"severity": "error", "code": "X"}]
    warn = [{"severity": "warn", "code": "X"}]
    p_err = RP.priority_score(err, 0.5, car, WH)
    assert p_err > RP.priority_score(warn, 0.5, car, WH)
    assert RP.priority_score(err, 0.1, car, WH) > RP.priority_score(err, 0.9, car, WH)
    assert RP.priority_score(err, 0.5, car, WH, votes=(0.1, 0.9)) > p_err
    assert 0 <= p_err <= 100


# ---------------- store + migration
def test_revisions_immutable_and_decisions_recorded():
    a = sh("a", "car", [10, 200, 80, 260], typ="rectangle", conf=5.0)
    b = sh("b", "car", [10, 200, 80, 260], typ="rectangle", conf=9.0)
    store = SS.ShapeStore(":memory:")
    res = run([ROAD, a, b], store=store)
    assert len(store.list(job=1)) == 4 and len(store.list(job=1, status="rejected")) == 1
    rej = store.list(job=1, status="rejected")[0]
    assert rej["stage"] == "llm_edit" and rej["parent_id"] is not None and rej["provenance"]["actor"] == "rule"
    assert store.get(rej["parent_id"])["status"] == "active"      # dong cu giu nguyen
    d = store.decisions(job=1)
    assert d and d[0]["calibrated"] is False and d[0]["approval_status"] == "pending"
    assert d[0]["before_revision"] == rej["parent_id"] and d[0]["after_revision"] == rej["id"]
    store.set_approval(d[0]["id"], "approved")
    assert store.decisions(job=1, approval="approved")
    assert res["counts_after"].get("DUPLICATE") is None


OLD_SCHEMA = SS.SCHEMA.split("-- migration cong them")[0]


def test_migration_keeps_old_db(tmp_path):
    p = tmp_path / "old.sqlite"
    db = sqlite3.connect(str(p))
    db.executescript(OLD_SCHEMA)
    db.execute("INSERT INTO shapes(job,frame,label,shape_type,points,stage,created_at) VALUES(1,1,'car','rectangle','[1,2,3,4]','rule_draft','t')")
    db.execute("INSERT INTO events(ts,kind,job,detail) VALUES('t','x',1,'{}')")
    db.commit()
    assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='decisions'").fetchone()
    db.close()
    st = SS.ShapeStore(p)
    assert st.db.execute("SELECT 1 FROM sqlite_master WHERE name='decisions'").fetchone()
    assert len(st.list(job=1)) == 1 and len(st.events(job=1)) == 1
    st.close()
    st = SS.ShapeStore(p)            # mo lai lan 2: idempotent
    assert st.list(job=1)[0]["label"] == "car"
    st.close()


# ---------------- VLM
def test_vlm_disabled_by_default():
    assert VA.enabled(RS) is False and VA.make_adjudicator(RS) is None


def test_ollama_down_gives_no_actions_and_no_error():
    ad = VA.OllamaAdjudicator({"url": "http://127.0.0.1:9", "model": "x", "timeout_s": 0.5})
    s = sh("s", "unobtainium", rect(0, 0, 10, 10))
    res = run([ROAD, s], adjudicator=ad)
    assert any(r["proposed_action"] == "FLAG_REVIEW" for r in res["records"])
    assert ad.last_error


def test_claude_session_pack_and_strict_reload(tmp_path):
    s = sh("s", "car", [10, 200, 80, 260], typ="rectangle")
    s2 = sh("t", "sky", rect(0, 0, 100, 50))
    ad = VA.ClaudeSessionAdjudicator(tmp_path)
    ctx = ctx_of([ROAD, s, s2])
    flagged = [S.new_record(1, 1, {"action": "FLAG_REVIEW", "shape_ids": ["s"], "reason_code": "SHAPE_TYPE", "params": {"note": "?"}}, rule_id="R")]
    f = ad.write_pack(ctx, flagged)
    doc = json.loads(f.read_text(encoding="utf8"))
    assert "s" in doc["valid_ids"]["shape_ids"] and "DELETE_DRAFT" in doc["allowed_actions"]
    resp = tmp_path / "frame_1" / "actions.json"
    resp.write_text(json.dumps([
        good(action="RELABEL", shape_ids=["s"], params={"label": "truck"}),                                            # hop le
        good(action="RELABEL", shape_ids=["ghost"], params={"label": "truck"}),                                        # id gia
        good(action="SNAP_EDGE", shape_ids=["s"], params={"roi_id": "roi:s", "band_px": 3, "points": [1, 2, 3, 4]}),   # toa do tho
    ]), encoding="utf8")
    out = ad.adjudicate(ctx, flagged)
    assert list(out) == [("s",)] and out[("s",)]["model_id"] == "claude_session" and out[("s",)]["calibrated"] is False
