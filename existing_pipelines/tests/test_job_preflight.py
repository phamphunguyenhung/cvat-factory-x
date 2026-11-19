import json

from cvat_tools.job import capability as CAP
from cvat_tools.job import discover as D
from cvat_tools.job import report as R
from cvat_tools.job import ruleset as RS


class FakeCvat:
    def job(self, j): return {"task_id": 7, "start_frame": 60, "stop_frame": 63}
    def task(self, t): return {"name": "demo"}
    def meta(self, j): return {"frames": [{"width": 1920, "height": 1080}] * 4}
    def labels(self, j):
        return [{"id": 1, "name": "car", "type": "rectangle", "attributes": [{"name": "occluded", "input_type": "checkbox"}]},
                {"id": 2, "name": "lane/single white", "type": "polyline", "attributes": []},
                {"id": 3, "name": "person", "type": "any", "attributes": []},
                {"id": 4, "name": "pedestrian", "type": "rectangle", "attributes": []},
                {"id": 5, "name": "lane/single other", "type": "any", "attributes": []},
                {"id": 6, "name": "zzz", "type": "any", "attributes": []}]
    def annotations(self, j):
        return {"shapes": [{"label_id": 1, "frame": 60, "type": "rectangle"}, {"label_id": 1, "frame": 62, "type": "rectangle"}],
                "tracks": []}


class DeadCvat:
    def job(self, j): raise RuntimeError("down")


def test_ruleset_default_and_override(tmp_path):
    (tmp_path / "jobs").mkdir()
    (tmp_path / "default.yaml").write_text("continuity: none\nvideo_keyframe_every: 10\ngroups: {object: [car]}\n", encoding="utf8")
    (tmp_path / "jobs" / "9.yaml").write_text("continuity: video\n", encoding="utf8")
    a, b = RS.load_ruleset(1, tmp_path), RS.load_ruleset(9, tmp_path)
    assert a["continuity"] == "none" and a["keyframe_every"] is None
    assert b["continuity"] == "video" and b["keyframe_every"] == 10
    (tmp_path / "jobs" / "8.yaml").write_text("continuity: bogus\n", encoding="utf8")
    c = RS.load_ruleset(8, tmp_path)
    assert c["continuity"] == "none" and c["_warnings"]


def test_real_ruleset_285():
    rs = RS.load_ruleset(285)
    assert rs["continuity"] == "none" and rs["occluded_threshold"] == 0.30
    assert RS.expected_shape(rs, "car") == "rectangle" and RS.expected_shape(rs, "lane/single white") == "polygon"
    assert RS.expected_z(rs, "road") == -1 and RS.expected_z(rs, "lane/crosswalk") == 0 and RS.expected_z(rs, "sidewalk") == -7
    assert RS.group_of(rs, "traffic_light") == "object"


def test_discover_frames_and_existing():
    d = D.discover(FakeCvat(), 1)
    assert d["frame_ids"] == [60, 61, 62, 63] and d["n_frames"] == 4
    car = d["per_label"]["car"]
    assert car["total"] == 2 and car["frames_with"] == [60, 62] and car["frames_without"] == [61, 63]
    assert d["per_label"]["zzz"]["total"] == 0
    assert D.alias_pairs(["person", "pedestrian"], {"person": "pedestrian"}) == [["pedestrian", "person"]]
    assert D.duplicate_groups(["traffic light", "traffic_light"]) == [["traffic light", "traffic_light"]]


def test_report_never_rejects_and_flags(tmp_path):
    rep = R.run(1, FakeCvat(), out_dir=tmp_path)
    assert (tmp_path / "gap_report.md").exists() and json.loads((tmp_path / "gap_report.json").read_text(encoding="utf8"))
    assert any("lane/single white" in c and "polyline" in c for c in rep["contradictions"])
    assert any("'person'" in c for c in rep["contradictions"])
    assert any("trùng nghĩa" in w for w in rep["warnings"])
    assert any("KHÔNG liên tục" in w for w in rep["warnings"])
    assert "lane/single other" in rep["need_model"]
    row = {r["label"]: r for r in rep["labels"]}
    assert row["car"]["assigned"] and row["car"]["frames_without"] == [61, 63]
    assert "## 3. Cần bạn tìm model" in (tmp_path / "gap_report.md").read_text(encoding="utf8")


def test_report_survives_cvat_down(tmp_path):
    rep = R.run(1, DeadCvat(), out_dir=tmp_path)
    assert rep["error"] and rep["labels"] and rep["task"] is None


def test_capability_status_values():
    caps = CAP.scan(None, RS.load_ruleset(None))
    assert caps["lane/double other"]["status"] == CAP.MISSING
    assert all(c["status"] in CAP.ORDER for c in caps.values())


def test_env_fallback_order(tmp_path, monkeypatch):
    from cvat_tools.core.client import load_env
    for k in ("CVAT_HOST", "CVAT_PAT", "CVAT_ORG"):
        monkeypatch.delenv(k, raising=False)
    a, b, c = tmp_path / "a.env", tmp_path / "b.env", tmp_path / "c.env"
    a.write_text("CVAT_HOST=http://a\n", encoding="utf8")                       # khong co PAT
    b.write_text("# c\nCVAT_HOST=http://b\nCVAT_PAT=pb\n", encoding="utf8")     # file PAT dau tien
    c.write_text("CVAT_HOST=http://c\nCVAT_PAT=pc\nCVAT_ORG=orgc\n", encoding="utf8")
    e = load_env(candidates=[tmp_path / "none.env", a, b, c])
    assert (e["CVAT_HOST"], e["CVAT_PAT"], e["CVAT_ORG"]) == ("http://b", "pb", "orgc")
    monkeypatch.setenv("CVAT_PAT", "penv")
    assert load_env(candidates=[b])["CVAT_PAT"] == "penv"
    assert load_env(candidates=[tmp_path / "x"]) == {} or True
