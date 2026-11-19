"""BFS lineage traversal and blast-radius impact analysis."""

from __future__ import annotations

from collections import deque
from typing import Any

from release_control.adapters.mlflow_adapter import MlflowTracker
from release_control.db import ControlTowerDB
from release_control.models import LineageEdge, ReleaseRecord, SnapshotRecord


class ImpactAnalyzer:
    """Computes forward/backward lineage and blast radius over DB edges and MLflow runs."""

    def __init__(
        self,
        db: ControlTowerDB,
        mlflow_tracker: MlflowTracker | None = None,
    ) -> None:
        self.db = db
        self.mlflow_tracker = mlflow_tracker

    def _recorded_edges(self, include_unverified: bool = False) -> list[LineageEdge]:
        edges = self.db.list_lineage_events()
        if include_unverified:
            return edges
        return [
            e
            for e in edges
            if e.metadata.get("evidence_status", "verified") != "unverified"
        ]

    def _mlflow_runs_for_digest(self, digest: str) -> list[dict[str, Any]]:
        if self.mlflow_tracker is None or not digest:
            return []
        return self.mlflow_tracker.query_runs_by_digest(digest)

    def _bfs(
        self,
        seed_ids: list[str],
        direction: str,
        include_unverified: bool = False,
    ) -> dict[str, Any]:
        edges = self._recorded_edges(include_unverified=include_unverified)
        snapshots_by_id = {s.snapshot_id: s for s in self.db.list_snapshots()}
        releases_by_id = {r.release_id: r for r in self.db.list_releases()}

        adj: dict[str, list[LineageEdge]] = {}
        for edge in edges:
            key = edge.source_id if direction == "forward" else edge.target_id
            adj.setdefault(key, []).append(edge)

        visited_nodes: dict[str, dict[str, Any]] = {}
        collected_edges: dict[str, dict[str, Any]] = {}
        queue: deque[tuple[str, int]] = deque()

        for sid in seed_ids:
            if sid not in visited_nodes:
                visited_nodes[sid] = {
                    "id": sid,
                    "type": self._infer_node_type(sid, snapshots_by_id, releases_by_id),
                    "depth": 0,
                }
                queue.append((sid, 0))

        while queue:
            current, depth = queue.popleft()
            for edge in adj.get(current, []):
                nxt = edge.target_id if direction == "forward" else edge.source_id
                nxt_type = (
                    edge.metadata.get("target_type")
                    if direction == "forward"
                    else edge.metadata.get("source_type")
                ) or self._infer_node_type(nxt, snapshots_by_id, releases_by_id)

                if edge.edge_id not in collected_edges:
                    collected_edges[edge.edge_id] = {
                        "edge_id": edge.edge_id,
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relation": edge.relation,
                        "evidence_status": edge.metadata.get(
                            "evidence_status", "verified"
                        ),
                        "metadata": edge.metadata,
                    }

                if nxt not in visited_nodes:
                    visited_nodes[nxt] = {
                        "id": nxt,
                        "type": nxt_type,
                        "depth": depth + 1,
                    }
                    queue.append((nxt, depth + 1))

        # Attach actual recorded MLflow runs when tracing forward from snapshots/digests/releases.
        mlflow_runs: list[dict[str, Any]] = []
        if direction == "forward" and self.mlflow_tracker is not None:
            candidate_digests: set[str] = set()
            for nid in visited_nodes:
                if nid in snapshots_by_id:
                    candidate_digests.add(snapshots_by_id[nid].manifest_sha256)
                elif nid in releases_by_id:
                    snap = snapshots_by_id.get(releases_by_id[nid].snapshot_id)
                    if snap is not None:
                        candidate_digests.add(snap.manifest_sha256)
                else:
                    candidate_digests.add(nid)

            seen_run_ids: set[str] = set()
            for digest in sorted(candidate_digests):
                for run in self._mlflow_runs_for_digest(digest):
                    rid = run["run_id"]
                    if rid in seen_run_ids:
                        continue
                    seen_run_ids.add(rid)
                    mlflow_runs.append(run)

                    run_node_id = f"mlflow_run:{rid}"
                    if run_node_id not in visited_nodes:
                        visited_nodes[run_node_id] = {
                            "id": run_node_id,
                            "type": "training_run",
                            "depth": 1,
                            "run_origin": run.get("run_origin", ""),
                        }
                    source_anchor = run.get("release_id") or digest
                    if source_anchor not in visited_nodes:
                        visited_nodes[source_anchor] = {
                            "id": source_anchor,
                            "type": self._infer_node_type(
                                source_anchor, snapshots_by_id, releases_by_id
                            ),
                            "depth": 0,
                        }
                    edge_key = f"mlflow:{source_anchor}->{run_node_id}"
                    collected_edges[edge_key] = {
                        "edge_id": edge_key,
                        "source_id": source_anchor,
                        "target_id": run_node_id,
                        "relation": "trained_in",
                        "evidence_status": "verified",
                        "metadata": {"run_origin": run.get("run_origin", "")},
                    }
                    model_name = run.get("model_name")
                    if model_name:
                        model_node_id = f"model:{model_name}"
                        if model_node_id not in visited_nodes:
                            visited_nodes[model_node_id] = {
                                "id": model_node_id,
                                "type": "model",
                                "depth": 2,
                            }
                        m_edge_key = f"mlflow:{run_node_id}->{model_node_id}"
                        collected_edges[m_edge_key] = {
                            "edge_id": m_edge_key,
                            "source_id": run_node_id,
                            "target_id": model_node_id,
                            "relation": "produced_model",
                            "evidence_status": "verified",
                            "metadata": {"run_origin": run.get("run_origin", "")},
                        }

        return {
            "root": seed_ids[0] if len(seed_ids) == 1 else seed_ids,
            "direction": direction,
            "nodes": list(visited_nodes.values()),
            "edges": list(collected_edges.values()),
            "mlflow_runs": mlflow_runs,
        }

    @staticmethod
    def _infer_node_type(
        node_id: str,
        snapshots_by_id: dict[str, SnapshotRecord],
        releases_by_id: dict[str, ReleaseRecord],
    ) -> str:
        if node_id in snapshots_by_id:
            return "snapshot"
        if node_id in releases_by_id:
            return "release"
        if node_id.startswith("mlflow_run:"):
            return "training_run"
        if node_id.startswith("model:"):
            return "model"
        return "entity"

    def trace_forward(
        self, node_id: str, include_unverified: bool = False
    ) -> dict[str, Any]:
        """Trace downstream artifacts from node_id via BFS."""
        return self._bfs(
            [node_id], direction="forward", include_unverified=include_unverified
        )

    def trace_backward(
        self, node_id: str, include_unverified: bool = False
    ) -> dict[str, Any]:
        """Trace upstream provenance of node_id via BFS."""
        return self._bfs(
            [node_id], direction="backward", include_unverified=include_unverified
        )

    def get_blast_radius(
        self, dataset_id_or_digest: str, include_unverified: bool = False
    ) -> dict[str, Any]:
        """Compute full downstream blast radius for a dataset, snapshot, release, or digest."""
        all_snapshots = self.db.list_snapshots()
        all_releases = self.db.list_releases()

        matched_snapshots = [
            s
            for s in all_snapshots
            if dataset_id_or_digest
            in {s.snapshot_id, s.manifest_sha256, s.dataset_name}
        ]
        matched_snapshot_ids = {s.snapshot_id for s in matched_snapshots}

        matched_releases = [
            r
            for r in all_releases
            if r.release_id == dataset_id_or_digest
            or r.dataset_name == dataset_id_or_digest
            or r.snapshot_id in matched_snapshot_ids
        ]

        seed_ids: list[str] = [dataset_id_or_digest]
        for s in matched_snapshots:
            if s.snapshot_id not in seed_ids:
                seed_ids.append(s.snapshot_id)
            if s.manifest_sha256 not in seed_ids:
                seed_ids.append(s.manifest_sha256)
        for r in matched_releases:
            if r.release_id not in seed_ids:
                seed_ids.append(r.release_id)

        bfs_result = self._bfs(
            seed_ids, direction="forward", include_unverified=include_unverified
        )
        node_ids = {n["id"] for n in bfs_result["nodes"]}

        affected_snapshots = [
            s.model_dump(mode="json")
            for s in all_snapshots
            if s.snapshot_id in node_ids or s.snapshot_id in matched_snapshot_ids
        ]
        affected_snap_id_set = {s["snapshot_id"] for s in affected_snapshots}

        affected_releases = [
            r.model_dump(mode="json")
            for r in all_releases
            if r.release_id in node_ids or r.snapshot_id in affected_snap_id_set
        ]

        actual_mlflow: list[dict[str, Any]] = []
        simulated_demo: list[dict[str, Any]] = []
        models_map: dict[str, dict[str, Any]] = {}

        for run in bfs_result["mlflow_runs"]:
            origin = run.get("run_origin", "")
            if origin == "actual_mlflow":
                actual_mlflow.append(run)
            elif origin == "simulated_demo":
                simulated_demo.append(run)
            else:
                actual_mlflow.append(run)

            model_name = run.get("model_name")
            if model_name:
                models_map.setdefault(
                    model_name,
                    {
                        "model_name": model_name,
                        "run_origins": set(),
                        "run_ids": [],
                    },
                )
                models_map[model_name]["run_origins"].add(origin or "unknown")
                models_map[model_name]["run_ids"].append(run["run_id"])

        for node in bfs_result["nodes"]:
            if node.get("type") == "model":
                m_name = node["id"].removeprefix("model:")
                models_map.setdefault(
                    m_name,
                    {
                        "model_name": m_name,
                        "run_origins": set(),
                        "run_ids": [],
                    },
                )

        models = [
            {
                "model_name": m["model_name"],
                "run_origins": sorted(m["run_origins"]),
                "run_ids": m["run_ids"],
            }
            for m in models_map.values()
        ]

        graph_nodes = [
            {
                "id": n["id"],
                "label": n["id"],
                "type": n["type"],
                "depth": n["depth"],
            }
            for n in bfs_result["nodes"]
        ]
        graph_edges = [
            {
                "id": e["edge_id"],
                "source": e["source_id"],
                "target": e["target_id"],
                "relation": e["relation"],
                "evidence_status": e["evidence_status"],
            }
            for e in bfs_result["edges"]
        ]

        return {
            "query": dataset_id_or_digest,
            "snapshots": affected_snapshots,
            "releases": affected_releases,
            "training_runs": {
                "actual_mlflow": actual_mlflow,
                "simulated_demo": simulated_demo,
            },
            "models": models,
            "graph": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }
