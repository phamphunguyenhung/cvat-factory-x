"""Section 11 Agent Orchestrator with 0-call happy path and strict budget enforcement."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .budget import AgentBudget, BudgetTracker, BudgetUsage
from .providers import (
    AgentProvider,
    AgentRequest,
    AgentResponse,
    AntigravityReviewPacketProvider,
    NoneAgentProvider,
    ProviderUnavailableError,
)


@dataclass
class OrchestrationResult:
    """Result of running AgentOrchestrator over pipeline items."""
    run_id: str
    job_id: int
    happy_path: bool
    provider_used: str
    budget_usage: BudgetUsage
    escalated_count: int
    responses: list[AgentResponse] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "happy_path": self.happy_path,
            "provider_used": self.provider_used,
            "budget_usage": self.budget_usage.to_dict(),
            "escalated_count": self.escalated_count,
            "responses": [r.to_dict() for r in self.responses],
            "errors": list(self.errors),
        }


class AgentOrchestrator:
    """Coordinates deterministic checks first and invokes AgentProvider only on escalations."""

    def __init__(
        self,
        provider: AgentProvider | None = None,
        budget: AgentBudget | None = None,
        fallback_provider: AgentProvider | None = None,
    ) -> None:
        self.provider = provider or NoneAgentProvider()
        self.budget = budget or AgentBudget()
        self.fallback_provider = fallback_provider or AntigravityReviewPacketProvider()

    def run(
        self,
        job_id: int,
        escalated_items: list[dict[str, Any]],
        *,
        task: str = "review_escalated_annotations",
        context: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> OrchestrationResult:
        """Execute orchestration. Guarantees 0 agent calls when escalated_items is empty (happy path)."""
        rid = run_id or uuid.uuid4().hex[:12]
        tracker = BudgetTracker(self.budget)

        # Happy path: 0 agent calls
        if not escalated_items:
            tracker.usage.runtime_seconds = tracker.elapsed()
            return OrchestrationResult(
                run_id=rid,
                job_id=job_id,
                happy_path=True,
                provider_used="none",
                budget_usage=tracker.usage,
                escalated_count=0,
            )

        tracker.check_iteration()
        tracker.check_call_allowed(estimated_tokens=500)

        active_provider = self.provider
        if not active_provider.is_available():
            active_provider = self.fallback_provider

        request = AgentRequest(
            run_id=rid,
            job_id=job_id,
            task=task,
            items=list(escalated_items),
            context=dict(context or {}),
        )

        tracker.acquire_worker()
        try:
            try:
                resp = active_provider.invoke(request)
            except ProviderUnavailableError:
                active_provider = self.fallback_provider
                resp = active_provider.invoke(request)
            tracker.record_call(tokens_used=resp.tokens_used)
        finally:
            tracker.release_worker()

        return OrchestrationResult(
            run_id=rid,
            job_id=job_id,
            happy_path=False,
            provider_used=active_provider.name,
            budget_usage=tracker.usage,
            escalated_count=len(escalated_items),
            responses=[resp],
        )


def run_job_pipeline(
    job_id: int,
    *,
    mode: str = "full",
    provider: str = "none",
    dry_run: bool = True,
    runner: Callable[..., dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """High-level orchestration entrypoint for `cvat-factory run --job <id> --mode <mode>`."""
    rid = kwargs.get("run_id") or uuid.uuid4().hex[:12]
    if runner is not None:
        return runner(job_id=job_id, mode=mode, dry_run=dry_run, run_id=rid, **kwargs)
    return {
        "run_id": rid,
        "job_id": job_id,
        "mode": mode,
        "provider": provider,
        "dry_run": dry_run,
        "status": "ready",
        "escalated_count": 0,
    }


def resume_job(
    job_id: int,
    *,
    run_log_path: str | None = None,
    dry_run: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """High-level resume entrypoint for `cvat-factory resume --job <id>`."""
    return {
        "job_id": job_id,
        "run_log_path": run_log_path,
        "dry_run": dry_run,
        "status": "resumed" if run_log_path else "no_checkpoint",
        **kwargs,
    }
