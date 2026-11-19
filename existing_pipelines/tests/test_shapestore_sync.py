import copy
import json

import pytest

from cvat_tools.core import transactions as tx
from cvat_tools.core.client import Cvat, WriteNotAuthorized
from cvat_tools.core.shapestore import ShapeStore, vertex_deviation
from cvat_tools.sync import capture, push

LABELS = [{"id": 1, "name": "road"}, {"id": 2, "name": "car"}]


class FakeCvat(Cvat):
    """Client gia: khong HTTP, khong dung PAT; ghi chi qua bo nho."""

    def __init__(self, shapes=None, write_enabled=False):
        self.write_enabled = write_enabled
        self.shapes = copy.deepcopy(shapes or [])
        self.next_id = 100 + len(self.shapes)
        self.writes = 0

    def labels(self, job_id): return LABELS
    def annotations(self, job_id): return {"version": 0, "shapes": copy.deepcopy(self.shapes), "tracks": [], "tags": []}

    def create_shapes(self, job_id, shapes):
        self._need_write()
        self.writes += 1
        out = []
        for s in shapes:
            s = dict(copy.deepcopy(s), id=self.next_id)
            self.next_id += 1
            self.shapes.append(s)
            out.append(s)
        return {"shapes": out}

    def delete_shapes(self, job_id, shapes):
        self._need_write()
        self.writes += 1
        ids = {s["id"] for s in shapes}
        self.shapes = [s for s in self.shapes if s["id"] not in ids]


def cshape(i, frame, label_id, pts, type_="polygon", **kw):
    return {"id": i, "frame": frame, "label_id": label_id, "type": type_, "points": pts, "z_order": 0, "occluded": False,
            "source": "manual", "attributes": [], **kw}


SQ = [0, 0, 10, 0, 10, 10, 0, 10]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(tx, "RUNS", tmp_path / "runs")
    st = ShapeStore(tmp_path / "s.sqlite")
    yield st, tmp_path
    st.close()


def draft(st, frame=0, label="road", pts=SQ, stage="rule_draft", **kw):
    return st.add(5, frame, label, "polygon", pts, stage, conf=0.9, provenance={"rule": "r1"}, **kw)


def test_store_append_only_latest_diff(env):
    st, _ = env
    a = draft(st)
    b = st.add(5, 0, "road", "polygon", [1, 1, 11, 1, 11, 11, 1, 11], "llm_edit", parent_id=a)
    assert [r["id"] for r in st.latest(5)] == [b]
    assert len(st.list(job=5)) == 2 and st.get(a)["provenance"] == {"rule": "r1"}
    d = st.diff(a, b)
    assert d["points_changed"] and d["vertex_dev_px"] == pytest.approx(2 ** 0.5) and d["fields"]["stage"] == ("rule_draft", "llm_edit")
    assert not hasattr(st, "delete") and not hasattr(st, "remove")
    with pytest.raises(ValueError):
        st.add(5, 0, "road", "polygon", SQ, "bogus")
    with pytest.raises(ValueError):
        st.add(5, 0, "road", "polygon", SQ, "rule_draft", parent_id=999)
    assert vertex_deviation([0, 0, 2, 0], [0, 0, 2, 0, 1, 1]) >= 0  # khac so dinh: chamfer


def test_dry_run_does_not_write(env):
    st, tmp = env
    draft(st)
    cv = FakeCvat(write_enabled=True)
    r = push.push(cv, st, 5)                    # yes mac dinh False
    assert r["dry_run"] and r["n_to_add"] == 1 and cv.writes == 0 and cv.shapes == []
    assert not (tmp / "backups").exists() and st.list(synced=True) == []


def test_yes_needs_write_enabled(env):
    st, _ = env
    draft(st)
    with pytest.raises(WriteNotAuthorized):
        push.push(FakeCvat(write_enabled=False), st, 5, yes=True)


def test_push_adds_verifies_and_writes_back_cvat_id(env):
    st, tmp = env
    r1, r2 = draft(st, 0), draft(st, 1, "car", [0, 0, 5, 5])
    st.add(5, 1, "car", "rectangle", [0, 0, 5, 5], "llm_edit", parent_id=r2)  # latest cua r2 la rectangle nay
    human = FakeCvat([cshape(7, 9, 2, [1, 1, 2, 2], "rectangle")], write_enabled=True)
    r = push.push(human, st, 5, yes=True)
    assert r["added"] == 2 and r["verified"] and len(human.shapes) == 3
    assert all(s["source"] == "manual" for s in human.shapes)
    synced = st.list(job=5, synced=True)
    assert {x["id"] for x in synced} == {r1, r2 + 1}
    run = json.loads(open(r["run_log"], encoding="utf8").read())
    assert run["provenance"] == "ai_draft" and run["kind"] == "shapes" and sorted(run["created_ids"]) == sorted(x["cvat_id"] for x in synced)
    assert (tmp / "backups").exists() and 7 in {s["id"] for s in human.shapes}
    # chay lai: khong them trung
    r2nd = push.push(human, st, 5, yes=True)
    assert r2nd["n_to_add"] == 0 and len(human.shapes) == 3


def test_dedupe_against_cvat_and_in_batch(env):
    st, _ = env
    draft(st, 0, "road", SQ)
    draft(st, 0, "road", [0.3, 0, 10, 0.2, 10, 10, 0, 10])      # trung trong lo
    draft(st, 2, "road", SQ)                                    # trung shape da co tren CVAT
    draft(st, 3, "ghost", SQ)                                   # label khong co tren job
    cv = FakeCvat([cshape(50, 2, 1, SQ)], write_enabled=True)
    r = push.push(cv, st, 5, yes=True)
    reasons = sorted(s["reason"] for s in r["skipped"])
    assert reasons == ["label_khong_co_tren_job", "trung_trong_lo", "trung_voi_cvat"] and r["added"] == 1
    assert len(cv.shapes) == 2


def test_rollback_deletes_only_created_and_keeps_human_edits(env):
    st, _ = env
    a, b = draft(st, 0), draft(st, 1)
    human_shape = cshape(7, 4, 1, SQ)
    cv = FakeCvat([human_shape], write_enabled=True)
    r = push.push(cv, st, 5, yes=True)
    cid_a, cid_b = st.get(a)["cvat_id"], st.get(b)["cvat_id"]
    for s in cv.shapes:                                          # nguoi sua shape cua a
        if s["id"] == cid_a:
            s["points"] = [0, 0, 20, 0, 20, 20, 0, 20]
    capture.capture(cv, st, 5)
    res = push.rollback(cv, st, r["run_log"])
    assert res["deleted_ids"] == [cid_b] and res["kept"] == [cid_a]
    assert {s["id"] for s in cv.shapes} == {7, cid_a}           # shape nguoi con nguyen
    assert st.get(b)["cvat_id"] is None and len(st.list(job=5)) >= 3   # lich su khong mat


def test_capture_edit_delete_new_keep_and_stats(env):
    st, _ = env
    rows = {n: draft(st, i) for i, n in enumerate("kept edited deleted".split())}
    cv = FakeCvat(write_enabled=True)
    push.push(cv, st, 5, yes=True)
    cid = {n: st.get(r)["cvat_id"] for n, r in rows.items()}
    cv.write_enabled = False                                    # capture chi doc
    for s in cv.shapes:
        if s["id"] == cid["edited"]:
            s["points"] = [2, 0, 12, 0, 12, 10, 2, 10]          # lech 2px moi dinh
    cv.shapes = [s for s in cv.shapes if s["id"] != cid["deleted"]]
    cv.shapes.append(cshape(900, 8, 2, [0, 0, 4, 4], "rectangle"))
    res = capture.capture(cv, st, 5)
    assert (res["kept"], res["edited"], res["rejected"], res["new_human"]) == (1, 1, 1, 1)
    assert cv.writes == 1  # chi lan push, capture khong ghi
    hf = st.list(job=5, stage="human_final")
    assert len(hf) == 3
    edited = next(h for h in hf if h["parent_id"] == rows["edited"])
    assert edited["points"][0] == 2 and edited["provenance"]["source"] == "human"
    assert next(h for h in hf if h["parent_id"] == rows["deleted"])["status"] == "rejected"
    new = next(h for h in hf if h["cvat_id"] == 900)
    assert new["parent_id"] is None and new["label"] == "car"
    # idempotent
    res2 = capture.capture(cv, st, 5)
    assert (res2["edited"], res2["rejected"], res2["new_human"]) == (0, 0, 0)
    rows_ = capture.stats(st, 5)
    road = next(r for r in rows_ if r["label"] == "road")
    assert (road["n"], road["kept"], road["edited"], road["deleted"], road["new_human"]) == (3, 1, 1, 1, 0)
    assert road["mean_vertex_dev_px"] == pytest.approx(2.0)
    assert road["kept_pct"] == pytest.approx(100 / 3)
    assert "road" in capture.format_stats(rows_)
    # lich su capture lam latest = ban nguoi
    assert next(x for x in st.latest(5) if x["cvat_id"] == cid["edited"])["stage"] == "human_final"


def test_capture_never_writes(env):
    st, _ = env
    cv = FakeCvat([cshape(1, 0, 1, SQ)], write_enabled=False)
    capture.capture(cv, st, 5)                                  # neu goi ghi se raise WriteNotAuthorized
    assert cv.writes == 0 and len(st.list(job=5)) == 1
