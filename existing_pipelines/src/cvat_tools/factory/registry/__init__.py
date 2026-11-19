"""Phase 3 Model Registry, Bounded Scanner, and Auto Model Selection Policy."""
from cvat_tools.factory.registry.facade import ModelRegistry
from cvat_tools.factory.registry.models import (
    CVATLabelDef,
    DatasetStats,
    ExpectedLabelSchema,
    HardwareProfile,
    ModelProvenance,
    ModelRecord,
    ModelSelectionContext,
    SelectionCandidate,
    SelectionDecision,
    SelectionPolicyConfig,
)
from cvat_tools.factory.registry.scanner import (
    ModelScanner,
    ScanRoot,
    ScanSummary,
    compute_sha256,
    default_scan_roots,
)
from cvat_tools.factory.registry.selector import ModelSelector
from cvat_tools.factory.registry.store import ModelRegistryStore, SCHEMA_VERSION

__all__ = [
    "CVATLabelDef",
    "DatasetStats",
    "ExpectedLabelSchema",
    "HardwareProfile",
    "ModelProvenance",
    "ModelRecord",
    "ModelRegistry",
    "ModelRegistryStore",
    "ModelScanner",
    "ModelSelectionContext",
    "ModelSelector",
    "SCHEMA_VERSION",
    "ScanRoot",
    "ScanSummary",
    "SelectionCandidate",
    "SelectionDecision",
    "SelectionPolicyConfig",
    "compute_sha256",
    "default_scan_roots",
]
