"""apply_draft(replace_auto) và rollback chạy trên server thật: khóa hành vi bằng FakeCvat (không gọi mạng)."""
import copy
import json

import pytest

from cvat_tools.pose import week2

LABEL = {"id": 10, "name": "A", "sublabels": [{"id": 11, "name": "0"}, {"id": 12, "name": "1"}]}


def _shape(sid, frame, source, pts, occ=(False, False)):
    els = [{"id": sid * 10 + i, "type": "points", "frame": frame, "label_id": 11 + i, "group": 0, "source": source,
            "occluded": occ[i], "outside": False, "z_order": 0, "rotation": 0, "points": list(p), "attributes": []}
           for i, p in enumerate(pts)]
    return {"id": sid, "type": "skeleton", "frame": frame, "label_id": 10, "group": 0, "source": source, "occluded": False,
            "outside": False, "z_order": 0, "rotation": 0, "points": [], "attributes": [], "elements": els}


class FakeCvat:
    def __init__(self, shapes, assignee="me"):
        self.shapes, self.assignee, self.next_id, self.write_enabled = shapes, assignee, 1000, True

    def me(self): return {"username": "me"}
    def job(self, jid): return {"assignee": {"username": self.assignee}, "start_frame": 0}
    def labels(self, jid): return [LABEL]
    def annotations(self, jid): return {"shapes": copy.deepcopy(self.shapes), "tracks": [], "tags": []}

    def create_shapes(self, jid, shapes):
        for s in shapes:
            s = copy.deepcopy(s)
            s["id"], self.next_id = self.next_id, self.next_id + 1
            for e in s["elements"]:
                e["id"], self.next_id = self.next_id, self.next_id + 1
            self.shapes.append(s)

    def delete_shapes(self, jid, shapes):
        gone = {s["id"] for s in shapes}
        self.shapes = [s for s in self.shapes if s["id"] not in gone]


def _draft(frames):
    return {"frames": [{"frame": f, "people": [{"label": "A", "points": {"0": {"x": x, "y": y, "state": st}, "1": {"x": x + 5, "y": y, "state": "visible"}}}]}
                       for f, x, y, st in frames]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for d in ("backups", "drafts", "metrics"):
        (tmp_path / d).mkdir()
    # run cũ tạo shape 1 (frame 0, chưa sửa), 2 (frame 1, bị dời), 4 (frame 3, đổi cờ Q); shape 3 (frame 2) là source=file
    (tmp_path / "drafts" / "d1.json").write_text(json.dumps(_draft([(0, 10, 10, "visible"), (1, 20, 20, "visible"), (3, 40, 40, "visible")])))
    (tmp_path / "backups" / "run_1.json").write_text(json.dumps({"job": 1, "draft": "d1.json", "created_ids": [1, 2, 4]}))
    shapes = [_shape(1, 0, "auto", [(10, 10), (15, 10)]),
              _shape(2, 1, "auto", [(30, 30), (25, 20)]),                       # bạn đã dời điểm 0
              _shape(3, 2, "file", [(1, 1), (2, 2)]),
              _shape(4, 3, "auto", [(40, 40), (45, 40)], occ=(True, False))]   # bạn đã bấm Q
    (tmp_path / "drafts" / "d2.json").write_text(json.dumps(_draft([(f, 100 + f, 100, "visible") for f in range(5)])))
    fake = FakeCvat(shapes)
    monkeypatch.setattr(week2.Cvat, "from_env", classmethod(lambda cls, profile="edufun", write_enabled=False: fake))
    monkeypatch.setattr(week2, "job_folder", lambda job_id: tmp_path)
    return tmp_path, fake


def test_dry_run_counts_and_protection(env):
    tmp, fake = env
    r = week2.apply_draft(1, file=str(tmp / "drafts" / "d2.json"), yes=False, replace_auto=True)
    assert r["protected_frames"] == [1, 3]            # sửa vị trí và đổi cờ Q đều được bảo vệ
    assert r["delete_frames"] == [0] and r["to_delete"] == 1
    assert r["frames"] == [0, 4] and r["to_create"] == 2   # frame 2 là source=file: không thay
    assert len(fake.shapes) == 4 and r["applied"] is False


def test_apply_then_rollback_restores_original(env):
    tmp, fake = env
    before = {s["frame"]: s["source"] for s in fake.shapes}
    r = week2.apply_draft(1, file=str(tmp / "drafts" / "d2.json"), yes=True, replace_auto=True)
    assert r["applied"] and r["verified"] and r["after"] == 5
    by_frame = {s["frame"]: s for s in fake.shapes}
    assert by_frame[1]["elements"][0]["points"] == [30, 30]          # nhãn bạn sửa còn nguyên
    assert by_frame[2]["source"] == "file" and by_frame[0]["source"] == "auto"
    assert by_frame[0]["elements"][0]["points"][0] == 100.0            # frame 0 đã được thay bằng nháp mới
    rb = week2.rollback(r["run"], yes=True)
    assert rb["deleted"] == 2 and rb["restored"] == 1
    after = {s["frame"]: s["source"] for s in fake.shapes}
    assert after == before and fake.shapes and [s["elements"][0]["points"] for s in fake.shapes if s["frame"] == 0][0] == [10, 10]


def test_refuses_job_not_assigned_to_me(env):
    tmp, fake = env
    fake.assignee = "someone_else"
    with pytest.raises(PermissionError):
        week2.apply_draft(1, file=str(tmp / "drafts" / "d2.json"), yes=True, replace_auto=True)
    assert len(fake.shapes) == 4


def test_plain_apply_only_adds_and_skips_existing_frames(env):
    tmp, fake = env
    r = week2.apply_draft(1, file=str(tmp / "drafts" / "d2.json"), yes=True)
    assert r["to_delete"] == 0 and r["frames"] == [4] and len(fake.shapes) == 5
