"""Agent Console cua dashboard (mock, khong can dataset/CVAT)."""
import json
import subprocess

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import dashboard.app as d  # noqa: E402


@pytest.fixture()
def c(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEMO_DIR", tmp_path)
    return TestClient(d.app)


def test_not_ready(c, monkeypatch):
    monkeypatch.setattr(d, "_cli_ready", lambda: False)
    r = c.post("/api/agent/ingest").json()
    assert r["ok"] is False and "chua san sang" in r["message"]
    assert c.post("/api/agent/hack").status_code == 404
    assert c.get("/api/job").json()["ready"] is False
    assert c.get("/api/queue").json()["ready"] is False


def test_no_session(c, monkeypatch):
    monkeypatch.setattr(d, "_cli_ready", lambda: True)
    assert c.post("/api/agent/prelabel").json()["ready"] is False


def test_run_mock(c, monkeypatch, tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"task_id": 7, "job_id": 9, "image_files": ["a", "b"]}))
    monkeypatch.setattr(d, "_cli_ready", lambda: True)
    seen = {}

    def fake(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return subprocess.CompletedProcess(argv, 0, stdout='{"x": 1}', stderr="")

    monkeypatch.setattr(d.subprocess, "run", fake)
    monkeypatch.setattr(d, "_shape_count", lambda s: (3, None))
    r = c.post("/api/agent/route").json()
    assert r["ok"] and r["json"] == {"x": 1}
    assert r["job_url"] == "http://localhost:8080/tasks/7/jobs/9"
    assert seen["argv"][1:] == ["-X", "utf8", "-m", "release_control.cli_live", "route"]
    assert not seen["kw"].get("shell") and seen["kw"]["timeout"] > 0
    j = c.get("/api/job").json()
    assert (j["frames"], j["shapes"], j["job_id"]) == (2, 3, 9)
    (tmp_path / "route.json").write_text(json.dumps({"queue": [{"frame": 1, "risk": 0.9}]}))
    assert c.get("/api/queue").json()["ready"] is True


def test_audit_and_index(c):
    a = c.get("/api/audit").json()
    assert a["ready"] or "chua san sang" in a["message"]
    assert c.get("/").status_code == 200
