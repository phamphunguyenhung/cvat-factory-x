import json

import pytest

from release_control.audit import AuditLog, WriteDenied, guard_write

SESSION = {"task_id": 7, "job_id": 9}


def test_chain_ok_and_tamper(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append("agent", "step", f"t{i}", inputs={"i": i})
    assert log.verify_chain() and len(log.read()) == 3 and len(log.read(limit=2)) == 2
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["target"] = "sua"
    lines[1] = json.dumps(row, sort_keys=True)
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify_chain() is False


def test_deleted_line_breaks_chain(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append("a", "x", i)
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "a.jsonl").write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
    assert log.verify_chain() is False


def test_secrets_redacted(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    e = log.append("a", "x", {"CVAT_PAT": "abc123secret"},
                   inputs={"password": "p", "nested": {"api_token": "tok", "ok": 1}, "path": "/x"})
    text = (tmp_path / "a.jsonl").read_text(encoding="utf-8")
    assert "abc123secret" not in text and '"tok"' not in text and '"p"' not in text
    assert e["inputs"]["nested"]["ok"] == 1 and e["inputs"]["path"] == "/x"


def test_guard_allows_local_session_task(tmp_path, monkeypatch):
    monkeypatch.delenv("RC_READ_ONLY", raising=False)
    guard_write("http://localhost:8080", 7, SESSION)
    guard_write("127.0.0.1:8080", "7", SESSION)


def test_guard_denies_and_audits(tmp_path, monkeypatch):
    monkeypatch.delenv("RC_READ_ONLY", raising=False)
    log = AuditLog(tmp_path / "a.jsonl")
    with pytest.raises(WriteDenied):
        guard_write("https://cvat.edufun.ai", 7, SESSION, audit=log)
    with pytest.raises(WriteDenied):
        guard_write("http://localhost:8080", 999, SESSION, audit=log)
    with pytest.raises(WriteDenied):
        guard_write("http://localhost.edufun.ai:8080", 7, SESSION, audit=log)
    monkeypatch.setenv("RC_READ_ONLY", "1")
    with pytest.raises(WriteDenied):
        guard_write("localhost", 7, SESSION, audit=log)
    rows = log.read()
    assert [r["action"] for r in rows] == ["DENIED"] * 4
    assert log.verify_chain()
