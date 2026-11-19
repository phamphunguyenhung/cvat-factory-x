import pytest

from release_control.demo_session import session as S


def test_assert_local_accepts_local():
    assert S.assert_local("http://localhost:8080/") == "http://localhost:8080"
    assert S.assert_local("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_assert_local_rejects_remote():
    with pytest.raises(S.DemoSessionError):
        S.assert_local("https://cvat.edufun.ai")


def test_labels_from_taxonomy():
    labels = S._labels_from_taxonomy()
    assert len(labels) == 80 and labels[0]["name"] == "person"


def test_put_denied_for_foreign_task(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SESSION_PATH", tmp_path / "session.json")
    (tmp_path / "session.json").write_text(
        '{"task_id": 1, "created_by": "release_control"}', encoding="utf-8")
    monkeypatch.delenv("RC_READ_ONLY", raising=False)
    with pytest.raises(Exception):
        S._guard({"cvat_host": "http://localhost:8080", "task_id": 999, "created_by": "release_control"}, "x")
