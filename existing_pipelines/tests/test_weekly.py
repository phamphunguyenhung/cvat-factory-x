import datetime as dt

import pytest

from cvat_tools.core import client
from cvat_tools.job import weekly


class FakeCvat:
    def job(self, jid):
        if jid == 2:
            raise client.CvatError("GET -> 404")
        return {"state": "in progress", "stage": "annotation", "task_id": 9, "start_frame": 0, "stop_frame": 49}


def test_current_week():
    assert weekly.current_week(dt.date(2026, 11, 16)) == "2026-W47"


def test_new_rejects_overwrite_and_placeholder_id(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "WEEKS", tmp_path)
    monkeypatch.setattr(weekly.paths, "DATA", tmp_path / "data")   # khong tao thu muc job that
    (tmp_path / "_template.yaml").write_text(
        'week: "{week}"\nprofile: edufun\njobs:\n  - {id: 0, kind: seg}\n', encoding="utf8")
    f = weekly.new_week("2026-W47")
    with pytest.raises(ValueError):
        weekly.new_week("2026-W47")          # khong ghi de
    with pytest.raises(ValueError):
        weekly.load_week("2026-W47")         # id=0 chua dien
    f.write_text('week: x\njobs:\n  - {id: 1, kind: seg}\n  - {id: 2, kind: det}\n', encoding="utf8")
    rows = weekly.status("2026-W47", client=FakeCvat())
    assert rows[0]["frames"] == 50 and rows[0]["stage"] == "annotation"
    assert "404" in rows[1]["error"]         # loi 1 job khong lam hong ca bang


def test_edufun_profile_does_not_use_local_pat(monkeypatch):
    monkeypatch.setattr(client, "load_env", lambda *a, **k: {"CVAT_PAT": "local", "CVAT_HOST": "http://localhost:8080", "CVAT_EDUFUN_PAT": "edu"})
    c = client.Cvat.from_env("edufun")
    assert c.host == "https://cvat.edufun.ai" and c.h["Authorization"] == "Bearer edu" and c.h["X-Organization"] == "cohort-5a"


def test_job_dir_separates_edufun_from_local():
    from cvat_tools.core import paths
    assert paths.job_dir("local", 285) == paths.DATA / "jobs" / "285"
    assert paths.job_dir("edufun", 285) == paths.DATA / "jobs" / "edufun" / "285"
