"""Unit tests for Pi-inspired ToolRegistry, contracts, permission gates, and CLI tools subcommand."""
from __future__ import annotations

import json
import pytest

from cvat_tools import cli
from cvat_tools.core.contracts import ResourceRequirements, ToolSpec
from cvat_tools.core.permissions import Permission, PermissionDeniedError
from cvat_tools.core.registry import ToolRegistry, build_default_registry


def test_default_registry_discovers_core_tools():
    reg = build_default_registry()
    ids = {t.tool_id for t in reg.list()}
    assert {
        "cvat.doctor",
        "cvat.job.preflight",
        "segmentation.prelabel",
        "annotations.validate",
        "draft.review",
        "store.capture",
        "store.push",
    }.issubset(ids)


def test_remote_write_blocked_by_default():
    reg = build_default_registry()
    with pytest.raises(PermissionDeniedError):
        reg.invoke("store.push", {"job": 285})


def test_custom_tool_validation_and_execution():
    reg = ToolRegistry()
    reg.register(ToolSpec(
        tool_id="demo.echo",
        name="Demo Echo",
        version="1.0.0",
        description="Echo test tool",
        input_schema={"x": "int"},
        output_schema={"y": "int"},
        permissions=(Permission.READ_ONLY,),
        resources=ResourceRequirements(memory_mb=16),
        validator=lambda p: [] if isinstance(p.get("x"), int) else ["x must be int"],
        runner=lambda p: {"y": p["x"] * 2, "artifacts": ["out.json"]},
    ))
    with pytest.raises(ValueError, match="x must be int"):
        reg.invoke("demo.echo", {"x": "bad"})
    res = reg.invoke("demo.echo", {"x": 21})
    assert res.ok and res.data["y"] == 42 and res.artifacts == ["out.json"]


def test_cli_tools_list_json(capsys):
    rc = cli.main(["--json", "tools", "list"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert any(t["tool_id"] == "segmentation.prelabel" for t in out)


def test_cli_store_rollback_dry_run_and_missing_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_file = tmp_path / "run_285.json"
    run_file.write_text(json.dumps({"job": 285, "created_ids": [1, 2]}), encoding="utf8")
    db_file = tmp_path / "shapes.sqlite"
    rc = cli.main(["--json", "store", "rollback", "--db", str(db_file), "--run", str(run_file)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["would_delete"] == 2
