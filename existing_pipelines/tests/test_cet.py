import json
from pathlib import Path

import numpy as np
import pytest

from cvat_tools.core import transactions as tx
from cvat_tools.core.cache import Cache, make_key
from cvat_tools.core.client import Cvat, CvatError, WriteNotAuthorized
from cvat_tools.quality import assist, review
from cvat_tools.tracking import engine
from cvat_tools.tracking import geometry as G
from cvat_tools.tracking import keyframes as K
from cvat_tools.tracking import writer


def walk(f0, f1, x0=0.0, dx=5.0):
    return {f: [x0 + dx * (f - f0), 50.0, x0 + dx * (f - f0) + 40, 150.0] for f in range(f0, f1 + 1)}


def zigzag(n=60):
    t = walk(0, n - 1)
    for f in range(20, 30):
        t[f] = [t[f][0], 50.0 + 3 * (f - 20), t[f][2], 150.0 + 3 * (f - 20)]
    for f in range(30, n):
        t[f] = [t[f][0], 50.0 + 27, t[f][2], 150.0 + 27]
    return t


# --- keyframe optimizer
def test_optimal_not_worse_than_dp_and_within_tol():
    t = zigzag()
    kf = K.optimal_keyframes(t, 2.0)
    assert len(kf) <= len(G.simplify_keyframes(t, 2.0))
    assert K.interp_error(t, kf)["max"] <= 2.0 + 1e-6


def test_forced_frames_kept_and_error_bound_holds():
    t = zigzag()
    kf = K.optimal_keyframes(t, 2.0, must_keep=[45])
    assert 45 in kf and K.interp_error(t, kf)["max"] <= 2.0 + 1e-6


def test_straight_walk_two_keyframes_and_entry_exit_events():
    p = K.plan_track(walk(0, 99), 1.0)
    assert p["keyframes"] == [0, 99] and p["error"]["max"] <= 1.0
    assert any("vao_khung" in r for r in p["events"][0])


def test_direction_change_event():
    t = walk(0, 20)
    t.update({f: [100 - 5 * (f - 20), 50.0, 140 - 5 * (f - 20), 150.0] for f in range(21, 41)})
    ev = K.must_keep_frames(t)
    assert any("doi_huong" in r for rs in ev.values() for r in rs)


def test_cvat_interpolation_matches_optimizer():
    t = zigzag()
    kf = K.optimal_keyframes(t, 2.0)
    trk = G.to_cvat_track(1, t, kf, 59)
    back = G.expand_cvat_track(trk, 59)
    assert max(G.max_dev(back[f], t[f]) for f in t) <= 2.0 + 0.02  # lam tron 2 chu so thap phan


# --- client: khong ghi neu chua duoc phep
def test_unauthorized_write_blocked(monkeypatch):
    cv = Cvat("http://x", "tok")
    called = []
    monkeypatch.setattr("requests.request", lambda *a, **k: called.append(1))
    for fn in (lambda: cv.create_tracks(1, []), lambda: cv.create_shapes(1, []),
               lambda: cv.delete_tracks(1, [{"id": 1}]), lambda: cv.delete_shapes(1, [{"id": 1}])):
        with pytest.raises(WriteNotAuthorized):
            fn()
    assert not called


def test_missing_pat():
    with pytest.raises(CvatError):
        Cvat("http://x", "")


class FakeCvat(Cvat):
    def __init__(self, write_enabled):
        super().__init__("http://x", "t", write_enabled=write_enabled)
        self.tracks = []
        self.next = 1

    def annotations(self, job):
        return {"shapes": [], "tracks": [dict(t) for t in self.tracks], "tags": []}

    def job(self, job):
        return {"stop_frame": 149}

    def labels(self, job):
        return [{"id": 7}]

    def _patch(self, job, action, body):
        self._need_write()
        if action == "create":
            for t in body["tracks"]:
                self.tracks.append({**t, "id": self.next})
                self.next += 1
        else:
            ids = {t["id"] for t in body["tracks"]}
            self.tracks = [t for t in self.tracks if t["id"] not in ids]

        class R:
            def json(s):
                return {}
        return R()


def _draft():
    return {"model": "m", "tracker": "t", "tracks": [{"name": "P1", "payload": G.to_cvat_track(7, walk(0, 40), [0, 40], 149)}]}


def test_apply_dry_run_backup_duplicate_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "BACKUPS", tmp_path / "b")
    monkeypatch.setattr(tx, "RUNS", tmp_path / "r")
    cv = FakeCvat(write_enabled=True)
    r = writer.apply(cv, 1, _draft(), yes=False)
    assert not r["applied"] and cv.tracks == []                          # dry-run
    r = writer.apply(cv, 1, _draft(), yes=True)
    assert r["applied"] and r["verified"] and len(cv.tracks) == 1 and Path(r["backup"]).exists()
    run = json.loads(Path(r["run"]).read_text())
    assert run["provenance"] == "ai_draft" and run["created_ids"] == [1]  # provenance AI tach khoi nhan nguoi
    r2 = writer.apply(cv, 1, _draft(), yes=True)                         # khong ghi trung
    assert not r2["applied"] and r2["skipped"] and len(cv.tracks) == 1
    cv.tracks.append({"id": 99, "shapes": [], "label_id": 7})            # track cua nguoi dung: khong bi rollback xoa
    writer.rollback(cv, r["run"])
    assert [t["id"] for t in cv.tracks] == [99]


def test_apply_yes_with_readonly_client_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "BACKUPS", tmp_path / "b")
    with pytest.raises(WriteNotAuthorized):
        writer.apply(FakeCvat(write_enabled=False), 1, _draft(), yes=True)


# --- cache key
def test_cache_key_changes_with_checkpoint_and_config(tmp_path):
    base = dict(source={"d": 1}, frames=[0, 9], det={"ckpt": "a", "imgsz": 640})
    assert make_key(**base) == make_key(**base)
    assert make_key(**{**base, "det": {"ckpt": "b", "imgsz": 640}}) != make_key(**base)
    assert make_key(**{**base, "det": {"ckpt": "a", "imgsz": 960}}) != make_key(**base)
    assert make_key(**{**base, "frames": [0, 10]}) != make_key(**base)
    c = Cache("t", tmp_path)
    c.put("k", {"a": 1})
    assert c.get("k")["value"] == {"a": 1} and c.get("zz") is None


# --- manual-assist
def _trk(id_, kfs, out_at=None, frame_out=None):
    sh = [{"frame": f, "points": b, "outside": False, "keyframe": True} for f, b in kfs]
    if out_at is not None:
        sh.append({"frame": out_at, "points": kfs[-1][1], "outside": True, "keyframe": True})
    return {"id": id_, "label_id": 7, "source": "manual", "shapes": sh}


def test_assist_detects_jump_gap_duplicate_sparse_and_does_not_modify():
    jump = _trk(1, [(0, [0, 0, 40, 100]), (10, [5, 0, 45, 100]), (11, [300, 0, 340, 100]), (20, [310, 0, 350, 100])])
    gap = _trk(2, [(0, [0, 0, 40, 100]), (20, [10, 0, 50, 100])], out_at=21)
    gap["shapes"] += [{"frame": 90, "points": [50, 0, 90, 100], "outside": False, "keyframe": True}]
    base = _trk(3, [(0, [500, 0, 540, 100]), (100, [600, 0, 640, 100])])
    dup = _trk(4, [(0, [501, 0, 541, 100]), (100, [601, 0, 641, 100])])
    before = json.dumps([jump, gap, base, dup])
    rep = assist.analyze([jump, gap, base, dup], 149)
    kinds = {(i["kind"], i["track"]) for i in rep["issues"]}
    assert ("bbox_jump", 1) in kinds and ("outside_gap", 2) in kinds and ("missing_keyframes", 3) in kinds
    assert any(i["kind"] == "duplicate_track" for i in rep["issues"])
    assert 10 in rep["reopen"] or 11 in rep["reopen"]
    assert json.dumps([jump, gap, base, dup]) == before


def test_assist_leading_outside_is_not_gap():
    t = {"id": 1, "shapes": [{"frame": 0, "points": [0, 0, 1, 1], "outside": True}, {"frame": 10, "points": [0, 0, 10, 10], "outside": False},
                             {"frame": 40, "points": [0, 0, 10, 10], "outside": False}]}
    assert not any(i["kind"] == "outside_gap" for i in assist.analyze([t], 149)["issues"])


# --- engine / review (khong can model)
def test_dets_shim_and_bytetrack_keeps_id_on_cached_dets():
    dets = {f: [[100 + 3 * f, 100, 140 + 3 * f, 200, 0.9]] for f in range(15)}
    r = engine.run_on_cache(dets, None, "bytetrack")
    assert len(r["tracks"]) == 1 and len(next(iter(r["tracks"].values()))) >= 12


def test_review_queue_flags_overlap_and_break_with_reasons():
    a, b = walk(0, 30), {f: [x + 2 for x in bx] for f, bx in walk(0, 30).items()}
    gap = walk(0, 30)
    del gap[15]
    q = review.build_queue({}, {"t1": {1: a, 2: b}, "t2": {1: gap}}, {})
    kinds = {r["kind"] for it in q for r in it["reasons"]}
    assert {"overlap", "track_break"} <= kinds and q[0]["risk"] >= q[-1]["risk"] and all(it["reasons"] for it in q)


def test_windows_path_with_spaces_and_unicode(tmp_path):
    c = Cache("t", tmp_path / "thu muc có dấu")
    c.put("k", {"x": 1})
    assert c.get("k")["value"] == {"x": 1}


def test_reid_evidence_never_relabels_and_osnet_blocked(tmp_path):
    from cvat_tools.tracking import reid

    class Src:
        def image(self, f):
            return np.full((200, 100, 3), 120 + f, np.uint8)
    tracks = {1: {0: [0, 0, 50, 100]}, 2: {5: [10, 0, 60, 100]}}
    ev = reid.evidence_for_events(tracks, [{"kind": "handoff", "frame_a": 0, "track_a": 1, "frame_b": 5, "track_b": 2},
                                           {"kind": "handoff", "frame_a": 0, "track_a": 1, "frame_b": 9, "track_b": 2}],
                                  reid.HsvEmbedder(), Src(), Cache("emb", tmp_path))
    assert ev[0]["auto_relabel"] is False and ev[0]["reid_cosine"] is not None and ev[1]["verdict"] == "khong ro"
    with pytest.raises(reid.Blocked):
        reid.OSNetEmbedder(tmp_path / "none.onnx")


def test_adaptive_hard_frames():
    from cvat_tools.detection.adaptive import hard_frames
    d = {f: [[0, 0, 40, 150, 0.9]] for f in range(10)}
    d[5] = [[0, 0, 40, 150, 0.3], [100, 0, 140, 40, 0.9]]
    h = hard_frames(d)
    assert 5 in h and 0 not in h
