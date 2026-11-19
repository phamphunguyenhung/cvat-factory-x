"""Tests for release_control.cli and .env loader / read-only guard."""

from __future__ import annotations

import os

import pytest

from release_control import cli
from release_control.config import ReadOnlyError, guard_cvat_write, load_dotenv


def test_load_dotenv_no_override_no_secret_print(tmp_path, monkeypatch, capsys):
    f = tmp_path / ".env"
    f.write_text("# c\nRC_TEST_A=hello\nRC_TEST_B='quoted'\nRC_TEST_C=keep\n", encoding="utf-8")
    monkeypatch.setenv("RC_TEST_C", "existing")
    monkeypatch.delenv("RC_TEST_A", raising=False)
    monkeypatch.delenv("RC_TEST_B", raising=False)
    keys = load_dotenv(f)
    assert keys == ["RC_TEST_A", "RC_TEST_B"]
    assert os.environ["RC_TEST_B"] == "quoted"
    assert os.environ["RC_TEST_C"] == "existing"
    assert "hello" not in capsys.readouterr().out
    monkeypatch.delenv("RC_TEST_A", raising=False)
    monkeypatch.delenv("RC_TEST_B", raising=False)


def test_read_only_blocks_write(monkeypatch):
    monkeypatch.setenv("RC_READ_ONLY", "1")
    with pytest.raises(ReadOnlyError):
        guard_cvat_write("push")
    monkeypatch.setenv("RC_READ_ONLY", "0")
    guard_cvat_write("push")


def test_release_approve_requires_real_approver(capsys):
    rc = cli.main(["release", "approve", "nonexistent-release"])
    assert rc == 2
    assert "approver" in capsys.readouterr().out


def test_demo_reaches_validated_without_approval(capsys):
    rc = cli.main(["demo"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "DUNG o VALIDATED" in out
    assert "HOLD" in out and "simulated" in out
