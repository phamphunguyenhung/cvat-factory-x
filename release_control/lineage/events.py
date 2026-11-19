"""OpenLineage-compatible event emitter and DB lineage edge recorder."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from release_control.db import ControlTowerDB
from release_control.models import LineageEdge


_VALID_EVIDENCE_STATUS = {"verified", "unverified"}
_OPENLINEAGE_SCHEMA_URL = (
    "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LineageTracker:
    """Records verified/unverified lineage edges in SQLite and OpenLineage JSONL."""

    def __init__(
        self,
        db: ControlTowerDB,
        openlineage_path: Path | str = Path("artifacts/lineage/openlineage.jsonl"),
        producer: str = "cvat-factory-x/release-control",
        namespace: str = "cvat-factory-x",
    ) -> None:
        self.db = db
        self.openlineage_path = Path(openlineage_path)
        self.producer = producer
        self.namespace = namespace

    def emit_run_event(
        self,
        event_type: str,
        job_name: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        evidence_status: str = "verified",
        facets: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit an OpenLineage-compatible RunEvent to the JSONL log."""
        if evidence_status not in _VALID_EVIDENCE_STATUS:
            raise ValueError(
                f"Invalid evidence_status {evidence_status!r}; expected one of {_VALID_EVIDENCE_STATUS}"
            )

        event: dict[str, Any] = {
            "eventTime": _utc_now_iso(),
            "eventType": event_type,
            "producer": self.producer,
            "schemaURL": _OPENLINEAGE_SCHEMA_URL,
            "run": {
                "runId": run_id or str(uuid.uuid4()),
                "facets": facets or {},
            },
            "job": {
                "namespace": self.namespace,
                "name": job_name,
            },
            "inputs": inputs or [],
            "outputs": outputs or [],
            "evidence_status": evidence_status,
        }

        self.openlineage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.openlineage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=False) + "\n")
        return event

    def record_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        evidence_status: str = "verified",
        metadata: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> LineageEdge:
        """Record a directed lineage edge in ControlTowerDB and emit an OpenLineage RunEvent."""
        if evidence_status not in _VALID_EVIDENCE_STATUS:
            raise ValueError(
                f"Invalid evidence_status {evidence_status!r}; expected one of {_VALID_EVIDENCE_STATUS}"
            )
        if not source_id.strip() or not target_id.strip():
            raise ValueError("source_id and target_id must be non-empty")

        edge_id = str(uuid.uuid4())
        edge_meta: dict[str, Any] = {
            **(metadata or {}),
            "source_type": source_type,
            "target_type": target_type,
            "evidence_status": evidence_status,
        }
        edge = LineageEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            actor=actor,
            metadata=edge_meta,
        )
        self.db.add_lineage_event(edge)

        self.emit_run_event(
            event_type="COMPLETE",
            job_name=relation,
            inputs=[{"namespace": source_type, "name": source_id}],
            outputs=[{"namespace": target_type, "name": target_id}],
            run_id=edge_id,
            evidence_status=evidence_status,
            facets={
                "edge": {
                    "edge_id": edge_id,
                    "actor": actor,
                    "metadata": edge_meta,
                }
            },
        )
        return edge

    def read_events(self) -> list[dict[str, Any]]:
        """Read all emitted OpenLineage RunEvents from the JSONL file."""
        if not self.openlineage_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.openlineage_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    events.append(json.loads(stripped))
        return events
