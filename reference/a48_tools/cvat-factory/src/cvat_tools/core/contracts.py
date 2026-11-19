"""Explicit tool contracts for the Pi-inspired CVAT tool kernel."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from cvat_tools.core.permissions import Permission


@dataclass(frozen=True)
class ResourceRequirements:
    memory_mb: int = 128
    requires_gpu: bool = False
    heavyweight_model: bool = False


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    dependencies: tuple[str, ...] = ()
    permissions: tuple[Permission, ...] = (Permission.READ_ONLY,)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    validator: Callable[[dict[str, Any]], list[str]] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "dependencies": list(self.dependencies),
            "permissions": [p.value for p in self.permissions],
            "resources": asdict(self.resources),
        }


@dataclass
class ToolResult:
    ok: bool
    tool_id: str
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    rss_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
