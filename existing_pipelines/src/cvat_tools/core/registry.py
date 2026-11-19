"""Minimal, agent-agnostic ToolRegistry for CVAT Tools v2.0."""
from __future__ import annotations

import os
import time
from typing import Any

from cvat_tools.core.contracts import ResourceRequirements, ToolResult, ToolSpec
from cvat_tools.core.permissions import DEFAULT_ALLOWED, Permission, check_permissions


def _rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


class ToolRegistry:
    """In-memory registry for independently discoverable CVAT tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.tool_id in self._tools:
            raise ValueError(f"Tool '{spec.tool_id}' da duoc dang ky.")
        self._tools[spec.tool_id] = spec

    def get(self, tool_id: str) -> ToolSpec:
        if tool_id not in self._tools:
            raise KeyError(f"Khong tim thay tool '{tool_id}'. Co: {sorted(self._tools)}")
        return self._tools[tool_id]

    def list(self) -> list[ToolSpec]:
        return [self._tools[k] for k in sorted(self._tools)]

    def invoke(
        self,
        tool_id: str,
        params: dict[str, Any] | None = None,
        *,
        allowed_permissions: frozenset[Permission] | set[Permission] = DEFAULT_ALLOWED,
    ) -> ToolResult:
        spec = self.get(tool_id)
        check_permissions(spec.permissions, allowed_permissions)
        payload = dict(params or {})
        warnings: list[str] = []
        if spec.validator is not None:
            errs = spec.validator(payload)
            if errs:
                raise ValueError(f"Input khong hop le cho '{tool_id}': {'; '.join(errs)}")
        if spec.runner is None:
            raise NotImplementedError(f"Tool '{tool_id}' chua gan runner.")
        t0 = time.perf_counter()
        out = spec.runner(payload) or {}
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        artifacts = list(out.pop("artifacts", [])) if isinstance(out, dict) and "artifacts" in out else []
        if isinstance(out, dict) and "warnings" in out:
            warnings.extend(out.pop("warnings"))
        return ToolResult(
            ok=True,
            tool_id=tool_id,
            data=out,
            warnings=warnings,
            artifacts=artifacts,
            elapsed_ms=elapsed_ms,
            rss_mb=_rss_mb(),
        )


def _require_int(key: str):
    def _v(p: dict[str, Any]) -> list[str]:
        v = p.get(key)
        return [] if isinstance(v, int) and v > 0 else [f"thieu hoac sai '{key}' (int > 0)"]
    return _v


def build_default_registry() -> ToolRegistry:
    """Register core CVAT tools lazily (no heavyweight CV imports at registration time)."""
    reg = ToolRegistry()

    def _run_doctor(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.core.client import Cvat
        cv = Cvat.from_env(p.get("profile", "local"))
        me = cv.me()
        return {"user": me.get("username"), "profile": p.get("profile", "local"), "rss_mb": _rss_mb()}

    def _run_preflight(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.core.client import Cvat
        from cvat_tools.job import report
        return report.run(p["job"], Cvat.from_env(p.get("profile", "local")), out_dir=p.get("out"))

    def _run_prelabel(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.pipeline.run import run_job
        return run_job(
            job_id=p["job"],
            profile=p.get("profile", "local"),
            frames_dir=p.get("frames_dir"),
            max_frames=p.get("max_frames"),
            dry_run=not bool(p.get("save_store", False)),
        )

    def _run_capture(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.core.client import Cvat
        from cvat_tools.core.shapestore import ShapeStore
        from cvat_tools.sync import capture
        return capture.capture(Cvat.from_env(p.get("profile", "local")), ShapeStore(p.get("db")), p["job"])

    def _run_push(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.core.client import Cvat
        from cvat_tools.core.shapestore import ShapeStore
        from cvat_tools.sync import push
        yes = bool(p.get("yes", False))
        return push.push(Cvat.from_env(p.get("profile", "local"), write_enabled=yes), ShapeStore(p.get("db")), p["job"], yes=yes)

    def _run_validate(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.core.shapestore import ShapeStore
        from cvat_tools.review.validators import validate_shapes
        rows = ShapeStore(p.get("db")).latest(p["job"])
        issues = validate_shapes(rows)
        return {"job": p["job"], "n_shapes": len(rows), "n_issues": len(issues), "issues": issues}

    def _run_review_pack(p: dict[str, Any]) -> dict[str, Any]:
        from cvat_tools.review.pack import build_pack
        return build_pack(job_id=p["job"], db_path=p.get("db"), out_dir=p.get("out"))

    reg.register(ToolSpec(
        tool_id="cvat.doctor",
        name="CVAT Doctor",
        version="2.0.0",
        description="Kiem tra moi truong, RAM, va ket noi CVAT chi doc (khong in PAT).",
        input_schema={"profile": "str (optional, default 'local')"},
        output_schema={"user": "str", "profile": "str", "rss_mb": "float|None"},
        permissions=(Permission.READ_ONLY,),
        resources=ResourceRequirements(memory_mb=64),
        runner=_run_doctor,
    ))
    reg.register(ToolSpec(
        tool_id="cvat.job.preflight",
        name="Job Preflight",
        version="2.0.0",
        description="Sinh GAP REPORT cho CVAT job (chi doc, khong tu choi job).",
        input_schema={"job": "int", "profile": "str", "out": "str|None"},
        output_schema={"job": "int", "files": "list[str]"},
        permissions=(Permission.READ_ONLY, Permission.LOCAL_WRITE),
        resources=ResourceRequirements(memory_mb=128),
        runner=_run_preflight,
        validator=_require_int("job"),
    ))
    reg.register(ToolSpec(
        tool_id="segmentation.prelabel",
        name="Segmentation & Lane Prelabel",
        version="2.0.0",
        description="Sinh nhap ai_draft/rule_draft cho job (chi doc CVAT, luu ShapeStore cuc bo).",
        input_schema={"job": "int", "profile": "str", "frames_dir": "str|None", "max_frames": "int|None"},
        output_schema={"job": "int", "frames": "int", "shapes": "int"},
        dependencies=("cvat_tools.segmentation", "cvat_tools.lanes"),
        permissions=(Permission.READ_ONLY, Permission.LOCAL_WRITE),
        resources=ResourceRequirements(memory_mb=2048, heavyweight_model=True),
        runner=_run_prelabel,
        validator=_require_int("job"),
    ))
    reg.register(ToolSpec(
        tool_id="annotations.validate",
        name="Annotations Validate",
        version="2.0.0",
        description="Kiem tra hinh hoc va quy uoc nhan tren ShapeStore cuc bo.",
        input_schema={"job": "int", "db": "str|None"},
        output_schema={"job": "int", "n_shapes": "int", "n_issues": "int"},
        permissions=(Permission.READ_ONLY,),
        resources=ResourceRequirements(memory_mb=128),
        runner=_run_validate,
        validator=_require_int("job"),
    ))
    reg.register(ToolSpec(
        tool_id="draft.review",
        name="Draft Review Pack",
        version="2.0.0",
        description="Tao goi review HTML/JSON de nguoi dung duyet draft truoc khi push.",
        input_schema={"job": "int", "db": "str|None", "out": "str|None"},
        output_schema={"out_dir": "str"},
        permissions=(Permission.LOCAL_WRITE,),
        resources=ResourceRequirements(memory_mb=256),
        runner=_run_review_pack,
        validator=_require_int("job"),
    ))
    reg.register(ToolSpec(
        tool_id="store.capture",
        name="Store Capture",
        version="2.0.0",
        description="Doc annotation CVAT (read-only) -> luu human_final vao ShapeStore.",
        input_schema={"job": "int", "profile": "str", "db": "str|None"},
        output_schema={"total": "int", "inserted": "int"},
        permissions=(Permission.READ_ONLY, Permission.LOCAL_WRITE),
        resources=ResourceRequirements(memory_mb=128),
        runner=_run_capture,
        validator=_require_int("job"),
    ))
    reg.register(ToolSpec(
        tool_id="store.push",
        name="Store Push",
        version="2.0.0",
        description="Day draft da duyet len CVAT (dry-run mac dinh; ghi that can yes=True + REMOTE_WRITE).",
        input_schema={"job": "int", "profile": "str", "db": "str|None", "yes": "bool"},
        output_schema={"dry_run": "bool", "n_to_add": "int"},
        permissions=(Permission.REMOTE_WRITE,),
        resources=ResourceRequirements(memory_mb=128),
        runner=_run_push,
        validator=_require_int("job"),
    ))
    return reg
