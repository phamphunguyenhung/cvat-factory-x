"""Strict budget enforcement for Section 11 Agent Orchestration."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    """Raised when an agent orchestration budget limit is exceeded."""


@dataclass(frozen=True)
class AgentBudget:
    """Hard limits for agent orchestration."""
    max_agent_calls: int = 5
    max_iterations: int = 3
    max_tokens: int = 20_000
    max_runtime_seconds: float = 120.0
    max_concurrent_workers: int = 2


@dataclass
class BudgetUsage:
    """Observed resource consumption for an orchestration run."""
    agent_calls: int = 0
    iterations: int = 0
    tokens_used: int = 0
    runtime_seconds: float = 0.0
    peak_concurrent_workers: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "agent_calls": self.agent_calls,
            "iterations": self.iterations,
            "tokens_used": self.tokens_used,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "peak_concurrent_workers": self.peak_concurrent_workers,
        }


class BudgetTracker:
    """Thread-safe enforcer for AgentBudget."""

    def __init__(self, budget: AgentBudget | None = None) -> None:
        self.budget = budget or AgentBudget()
        self.usage = BudgetUsage()
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._active_workers = 0
        self._semaphore = threading.BoundedSemaphore(max(1, self.budget.max_concurrent_workers))

    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def check_runtime(self) -> None:
        elapsed = self.elapsed()
        self.usage.runtime_seconds = elapsed
        if elapsed > self.budget.max_runtime_seconds:
            raise BudgetExceededError(
                f"Runtime budget exceeded: {elapsed:.2f}s > {self.budget.max_runtime_seconds:.2f}s."
            )

    def check_iteration(self) -> None:
        self.check_runtime()
        with self._lock:
            if self.usage.iterations >= self.budget.max_iterations:
                raise BudgetExceededError(
                    f"Iteration budget exceeded: {self.usage.iterations} >= {self.budget.max_iterations}."
                )
            self.usage.iterations += 1

    def check_call_allowed(self, estimated_tokens: int = 0) -> None:
        self.check_runtime()
        with self._lock:
            if self.usage.agent_calls >= self.budget.max_agent_calls:
                raise BudgetExceededError(
                    f"Agent call budget exceeded: {self.usage.agent_calls} >= {self.budget.max_agent_calls}."
                )
            if self.usage.tokens_used + max(0, estimated_tokens) > self.budget.max_tokens:
                raise BudgetExceededError(
                    f"Token budget exceeded: {self.usage.tokens_used} + {estimated_tokens} > {self.budget.max_tokens}."
                )

    def record_call(self, tokens_used: int = 0) -> None:
        with self._lock:
            self.usage.agent_calls += 1
            self.usage.tokens_used += max(0, tokens_used)
            self.usage.runtime_seconds = self.elapsed()
            if self.usage.agent_calls > self.budget.max_agent_calls:
                raise BudgetExceededError("Agent call budget exceeded after call.")
            if self.usage.tokens_used > self.budget.max_tokens:
                raise BudgetExceededError("Token budget exceeded after call.")

    def acquire_worker(self, timeout: float = 5.0) -> None:
        if not self._semaphore.acquire(timeout=timeout):
            raise BudgetExceededError(
                f"Concurrent worker limit ({self.budget.max_concurrent_workers}) saturated."
            )
        with self._lock:
            self._active_workers += 1
            if self._active_workers > self.usage.peak_concurrent_workers:
                self.usage.peak_concurrent_workers = self._active_workers

    def release_worker(self) -> None:
        with self._lock:
            self._active_workers = max(0, self._active_workers - 1)
        self._semaphore.release()
