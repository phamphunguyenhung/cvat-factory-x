"""High-level ModelRegistry facade unifying SQLite store, bounded scanner, and selection policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cvat_tools.core.paths import DATA, WORKSPACE
from cvat_tools.factory.registry.models import (
    ModelRecord,
    ModelSelectionContext,
    SelectionDecision,
)
from cvat_tools.factory.registry.scanner import ModelScanner, ScanRoot, ScanSummary
from cvat_tools.factory.registry.selector import ModelSelector
from cvat_tools.factory.registry.store import ModelRegistryStore


class ModelRegistry:
    """Unified lazy-loading facade for model discovery, SQLite metadata, and auto-selection."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root is not None else WORKSPACE
        resolved_db = Path(db_path) if db_path is not None else (DATA / "store" / "model_registry.sqlite")
        self.store = ModelRegistryStore(resolved_db)
        self.scanner = ModelScanner(self.store)
        self.selector = ModelSelector(self.store)

    def sync_config(self, yaml_path: str | Path | None = None) -> int:
        p = Path(yaml_path) if yaml_path is not None else (self.workspace_root / "config" / "models.yaml")
        return self.store.sync_from_yaml(p, workspace_root=self.workspace_root)

    def scan(
        self,
        roots: list[ScanRoot] | None = None,
        *,
        compute_hash: bool = True,
        sync_yaml: bool = True,
    ) -> ScanSummary:
        if sync_yaml:
            self.sync_config()
        return self.scanner.scan(roots=roots, workspace_root=self.workspace_root, compute_hash=compute_hash)

    def list_models(
        self,
        *,
        task: str | None = None,
        framework: str | None = None,
        only_available: bool = False,
    ) -> list[ModelRecord]:
        return self.store.list_models(task=task, framework=framework, only_available=only_available)

    def get(self, model_id: str) -> ModelRecord | None:
        return self.store.get(model_id)

    def register(self, record: ModelRecord, *, deduplicate_sha256: bool = True) -> tuple[ModelRecord, bool]:
        return self.store.register(record, deduplicate_sha256=deduplicate_sha256)

    def select(self, context: ModelSelectionContext) -> SelectionDecision:
        return self.selector.select(context)

    def update_benchmark(
        self,
        model_id: str,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        return self.store.update_benchmark(model_id, status=status, metrics=metrics)

    def load_runtime(self, model_id: str, *, allow_multi_heavy: bool = False) -> Any:
        return self.store.load_runtime(model_id, allow_multi_heavy=allow_multi_heavy)

    def unload_runtime(self, model_id: str) -> bool:
        return self.store.unload_runtime(model_id)

    def unload_all_runtime(self) -> int:
        return self.store.unload_all_runtime()
