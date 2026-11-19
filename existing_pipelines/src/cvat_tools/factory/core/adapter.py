"""Unified AnnotationAdapter interface and adapter registry for CVAT Annotation Factory."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from cvat_tools.factory.core.models import (
    AdapterContext,
    EvidencePacket,
    ImageInput,
    ModelSelection,
    PreprocessedBatch,
    Proposal,
    RawPrediction,
    TaskSchema,
    ValidationReport,
)


class AnnotationAdapter(ABC):
    """Unified interface implemented by Skeleton, Detection, Segmentation, and Tracking adapters."""

    adapter_name: str = "base"

    @abstractmethod
    def supports(self, task_schema: TaskSchema) -> bool:
        """Return True if this adapter can handle the normalized CVAT TaskSchema."""

    @abstractmethod
    def select_model(self, context: AdapterContext) -> ModelSelection:
        """Select the best available model for the task/hardware context without loading weights."""

    @abstractmethod
    def preprocess(
        self,
        images: Sequence[ImageInput],
        context: AdapterContext | None = None,
    ) -> PreprocessedBatch:
        """Prepare inference inputs while keeping display-enhanced images strictly separate."""

    @abstractmethod
    def predict(
        self,
        batch: PreprocessedBatch,
        model_selection: ModelSelection | None = None,
    ) -> list[RawPrediction]:
        """Run inference (or report honest capability gap) on a preprocessed batch."""

    @abstractmethod
    def convert_to_cvat(
        self,
        predictions: Sequence[RawPrediction],
        task_schema: TaskSchema,
        run_id: str = "",
    ) -> list[Proposal]:
        """Convert raw predictions into schema-aligned Proposals with CVAT-ready geometry."""

    @abstractmethod
    def validate(
        self,
        proposals: Sequence[Proposal],
        context: dict[str, Any] | None = None,
    ) -> list[ValidationReport]:
        """Validate proposals against schema, geometry, image alignment, and confidence rules."""

    @abstractmethod
    def refine(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Proposal]:
        """Apply bounded local refinement only to regions flagged for REFINE, preserving PASS regions."""

    @abstractmethod
    def render_evidence(
        self,
        proposals: Sequence[Proposal],
        validation_reports: Sequence[ValidationReport] | None = None,
        output_dir: str | Path | None = None,
    ) -> list[EvidencePacket]:
        """Render visual/diagnostic evidence packets for review and audit."""


class AdapterRegistry:
    """In-memory registry of AnnotationAdapter instances."""

    def __init__(self) -> None:
        self._adapters: dict[str, AnnotationAdapter] = {}

    def register(self, adapter: AnnotationAdapter) -> None:
        self._adapters[adapter.adapter_name] = adapter

    def get(self, adapter_name: str) -> AnnotationAdapter | None:
        return self._adapters.get(adapter_name)

    def list_names(self) -> list[str]:
        return sorted(self._adapters.keys())

    def matching(self, task_schema: TaskSchema) -> list[AnnotationAdapter]:
        return [a for a in self._adapters.values() if a.supports(task_schema)]
