"""Domain models for the versioned Model Registry and Auto Model Selection Policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelProvenance:
    """Origin and audit trail for a registered model weight."""

    source: str = ""
    discovered_from: str = "manual"
    config_source: str = ""
    registered_at: str = ""
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelProvenance":
        if not data:
            return cls()
        return cls(
            source=str(data.get("source", "")),
            discovered_from=str(data.get("discovered_from", "manual")),
            config_source=str(data.get("config_source", "")),
            registered_at=str(data.get("registered_at", "")),
            aliases=tuple(data.get("aliases") or ()),
        )


@dataclass(frozen=True)
class ExpectedLabelSchema:
    """Label, geometry, and optional skeleton schema expected by a model."""

    classes: tuple[str, ...] = ()
    geometry_types: tuple[str, ...] = ()
    skeleton_keypoints: tuple[str, ...] = ()
    skeleton_edges: tuple[tuple[int, int], ...] = ()
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "geometry_types": list(self.geometry_types),
            "skeleton_keypoints": list(self.skeleton_keypoints),
            "skeleton_edges": [list(e) for e in self.skeleton_edges],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExpectedLabelSchema":
        if not data:
            return cls()
        return cls(
            classes=tuple(str(c) for c in (data.get("classes") or ())),
            geometry_types=tuple(str(g) for g in (data.get("geometry_types") or ())),
            skeleton_keypoints=tuple(str(k) for k in (data.get("skeleton_keypoints") or ())),
            skeleton_edges=tuple((int(e[0]), int(e[1])) for e in (data.get("skeleton_edges") or ()) if len(e) == 2),
            schema_version=str(data.get("schema_version", "1.0")),
        )


@dataclass(frozen=True)
class ModelRecord:
    """Versioned model metadata stored in the SQLite registry (weights never loaded at registration)."""

    model_id: str
    version: str = "1.0"
    architecture: str = "unknown"
    path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    framework: str = "unknown"
    supported_tasks: tuple[str, ...] = ()
    input_size: tuple[int, int] = (640, 640)
    cpu_supported: bool = True
    gpu_supported: bool = False
    ram_mb: int = 512
    vram_mb: int = 0
    expected_label_schema: ExpectedLabelSchema = field(default_factory=ExpectedLabelSchema)
    benchmark_status: str = "unverified"
    benchmark_metrics: dict[str, Any] = field(default_factory=dict)
    provenance: ModelProvenance = field(default_factory=ModelProvenance)
    license: str = "unknown"
    heavyweight: bool = False
    is_available: bool = True
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "architecture": self.architecture,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "framework": self.framework,
            "supported_tasks": list(self.supported_tasks),
            "input_size": list(self.input_size),
            "cpu_supported": self.cpu_supported,
            "gpu_supported": self.gpu_supported,
            "ram_mb": self.ram_mb,
            "vram_mb": self.vram_mb,
            "expected_label_schema": self.expected_label_schema.to_dict(),
            "benchmark_status": self.benchmark_status,
            "benchmark_metrics": dict(self.benchmark_metrics),
            "provenance": self.provenance.to_dict(),
            "license": self.license,
            "heavyweight": self.heavyweight,
            "is_available": self.is_available,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class HardwareProfile:
    """Available compute budget for model selection."""

    cpu_cores: int = 4
    ram_mb: int = 16384
    gpu_available: bool = False
    vram_mb: int = 0

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Lightweight hardware detection without importing torch/tensorflow."""
        import os

        cores = os.cpu_count() or 4
        ram_mb = 16384
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                ram_mb = max(512, int(stat.ullAvailPhys // (1024 * 1024)))
        except Exception:
            pass
        return cls(cpu_cores=cores, ram_mb=ram_mb, gpu_available=False, vram_mb=0)


@dataclass(frozen=True)
class CVATLabelDef:
    """Normalized CVAT label definition extracted from a task/job."""

    name: str
    geometry_type: str  # bbox/rectangle, polygon, mask, skeleton, points, tag, track
    keypoint_names: tuple[str, ...] = ()
    skeleton_edges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class DatasetStats:
    """Summary statistics of the target frames/job for model selection."""

    frame_count: int = 0
    mean_width: int = 0
    mean_height: int = 0
    low_light_ratio: float = 0.0
    small_object_ratio: float = 0.0


@dataclass(frozen=True)
class SelectionPolicyConfig:
    """Configurable weights and constraints for auto model selection."""

    weight_quality: float = 0.45
    weight_speed: float = 0.35
    weight_memory: float = 0.20
    max_ram_mb: int = 8192
    require_verified_benchmark: bool = False
    allow_unavailable_models: bool = False
    prefer_cpu_friendly: bool = True


@dataclass(frozen=True)
class ModelSelectionContext:
    """Full context passed to ModelSelector (never just a task name)."""

    labels: tuple[CVATLabelDef, ...] = ()
    dataset_stats: DatasetStats = field(default_factory=DatasetStats)
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    policy: SelectionPolicyConfig = field(default_factory=SelectionPolicyConfig)
    user_override_model_id: str | None = None
    force_override: bool = False
    task_hint: str = ""


@dataclass(frozen=True)
class SelectionCandidate:
    """Scored candidate model with transparent breakdown."""

    model_id: str
    total_score: float
    schema_score: float
    quality_score: float
    speed_score: float
    memory_score: float
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionDecision:
    """Outcome of model selection; stops at inspect_required/needs_review on unknown schemas."""

    status: Literal["selected", "inspect_required", "needs_review", "overridden"]
    selected_model_id: str | None
    reason: str
    candidates: tuple[SelectionCandidate, ...] = ()
    override_applied: bool = False
    warnings: tuple[str, ...] = ()
