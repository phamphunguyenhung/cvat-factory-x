"""AgentProvider interface and implementations (None, Claude CLI, Antigravity Review Packet)."""
from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvat_tools.core import paths


class ProviderUnavailableError(RuntimeError):
    """Raised when a requested external agent CLI/provider is not available."""


@dataclass
class AgentRequest:
    """Structured prompt/context sent to an AgentProvider."""
    run_id: str
    job_id: int
    task: str
    items: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    max_tokens_hint: int = 2000


@dataclass
class AgentResponse:
    """Structured result returned by an AgentProvider."""
    provider: str
    handled: bool
    summary: str
    decisions: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    packet_path: str | None = None
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "handled": self.handled,
            "summary": self.summary,
            "decisions": list(self.decisions),
            "tokens_used": self.tokens_used,
            "packet_path": self.packet_path,
            "metadata": dict(self.metadata),
        }


class AgentProvider(ABC):
    """Abstract interface for escalation/review agent providers."""
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can execute in the current environment."""

    @abstractmethod
    def invoke(self, request: AgentRequest) -> AgentResponse:
        """Process an AgentRequest and return an AgentResponse."""


class NoneAgentProvider(AgentProvider):
    """Deterministic no-LLM provider (0 agent calls, flags escalations for human review)."""
    name = "none"

    def is_available(self) -> bool:
        return True

    def invoke(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            provider=self.name,
            handled=False,
            summary=f"Deterministic mode: {len(request.items)} item(s) require manual review.",
            decisions=[
                {"item_id": item.get("id", idx), "action": "human_review", "reason": "none_provider"}
                for idx, item in enumerate(request.items)
            ],
            tokens_used=0,
        )


class ClaudeCliAgentProvider(AgentProvider):
    """Real Claude CLI adapter. Checks real CLI availability; never simulates CLI if missing."""
    name = "claude_cli"

    def __init__(
        self,
        cli_bin: str = "claude",
        timeout_seconds: float = 60.0,
        verify_version: bool = True,
    ) -> None:
        self.cli_bin = cli_bin
        self.timeout_seconds = timeout_seconds
        self.verify_version = verify_version

    def resolve_binary(self) -> str | None:
        return shutil.which(self.cli_bin)

    def is_available(self) -> bool:
        exe = self.resolve_binary()
        if not exe:
            return False
        if not self.verify_version:
            return True
        try:
            proc = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def invoke(self, request: AgentRequest) -> AgentResponse:
        exe = self.resolve_binary()
        if not exe or not self.is_available():
            raise ProviderUnavailableError(
                f"Claude CLI '{self.cli_bin}' is not available in PATH or failed health check."
            )

        prompt = (
            f"Task: {request.task}\n"
            f"Job ID: {request.job_id}\n"
            f"Items ({len(request.items)}): {json.dumps(request.items, ensure_ascii=False)}\n"
            "Return concise JSON with keys: summary, decisions."
        )
        proc = subprocess.run(
            [exe, "-p", prompt, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude CLI failed (exit={proc.returncode}): {proc.stderr.strip()[:300]}"
            )

        raw = proc.stdout.strip()
        parsed_summary = raw[:500]
        decisions: list[dict[str, Any]] = []
        tokens_used = max(1, len(prompt) // 4 + len(raw) // 4)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                parsed_summary = str(data.get("result") or data.get("summary") or parsed_summary)
                if isinstance(data.get("decisions"), list):
                    decisions = data["decisions"]
                usage = data.get("usage") or {}
                if isinstance(usage, dict) and ("input_tokens" in usage or "output_tokens" in usage):
                    tokens_used = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        except json.JSONDecodeError:
            pass

        return AgentResponse(
            provider=self.name,
            handled=True,
            summary=parsed_summary,
            decisions=decisions,
            tokens_used=tokens_used,
            raw_output=raw,
        )


class AntigravityReviewPacketProvider(AgentProvider):
    """Fallback provider that emits a structured Antigravity review packet (JSON + Markdown) on disk."""
    name = "antigravity_packet"

    def __init__(self, packet_dir: Path | None = None) -> None:
        self.packet_dir = Path(packet_dir) if packet_dir else (paths.DATA / "review_packets")

    def is_available(self) -> bool:
        return True

    def invoke(self, request: AgentRequest) -> AgentResponse:
        self.packet_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"review_packet_job{request.job_id}_{request.run_id}_{ts}"
        json_path = self.packet_dir / f"{stem}.json"
        md_path = self.packet_dir / f"{stem}.md"

        packet = {
            "schema": "antigravity.review_packet.v1",
            "run_id": request.run_id,
            "job_id": request.job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": request.task,
            "item_count": len(request.items),
            "items": request.items,
            "context": request.context,
        }
        json_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            f"# Antigravity Review Packet — Job {request.job_id} (`{request.run_id}`)",
            "",
            f"- **Task:** {request.task}",
            f"- **Items requiring review:** {len(request.items)}",
            f"- **JSON Packet:** `{json_path}`",
            "",
            "## Escalated Items",
        ]
        for idx, item in enumerate(request.items[:20], 1):
            lines.append(f"{idx}. Frame `{item.get('frame', '?')}` | Label `{item.get('label', '?')}` | Reason: `{item.get('reason', 'review')}`")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return AgentResponse(
            provider=self.name,
            handled=False,
            summary=f"Generated Antigravity review packet for {len(request.items)} item(s): {json_path.name}",
            decisions=[
                {"item_id": item.get("id", idx), "action": "review_packet", "packet": str(json_path)}
                for idx, item in enumerate(request.items)
            ],
            tokens_used=0,
            packet_path=str(json_path),
            metadata={"markdown_path": str(md_path)},
        )


def select_provider(
    preferred: str = "none",
    *,
    allow_packet_fallback: bool = True,
    packet_dir: Path | None = None,
    claude_bin: str = "claude",
) -> AgentProvider:
    """Select an AgentProvider without ever simulating an unavailable CLI."""
    pref = preferred.lower().strip()
    if pref in ("none", "deterministic", "off"):
        return NoneAgentProvider()
    if pref in ("claude", "claude_cli", "claude-cli"):
        cli = ClaudeCliAgentProvider(cli_bin=claude_bin)
        if cli.is_available():
            return cli
        if allow_packet_fallback:
            return AntigravityReviewPacketProvider(packet_dir=packet_dir)
        raise ProviderUnavailableError(f"Requested provider '{preferred}' is unavailable.")
    if pref in ("antigravity", "antigravity_packet", "packet"):
        return AntigravityReviewPacketProvider(packet_dir=packet_dir)
    raise ValueError(f"Unknown agent provider: {preferred!r}")
