"""MLflow tracking adapter for release-linked training runs (never fabricates unrecorded runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class MlflowTracker:
    """Real MLflow tracker backed by a local file/SQLite store under artifacts/mlruns."""

    def __init__(
        self,
        tracking_uri: str | Path | None = None,
        experiment_name: str = "cvat-factory-x-releases",
    ) -> None:
        if tracking_uri is None:
            store_dir = Path("artifacts/mlruns").resolve()
            store_dir.mkdir(parents=True, exist_ok=True)
            self.tracking_uri = f"sqlite:///{(store_dir / 'mlflow.db').as_posix()}"
        elif isinstance(tracking_uri, Path) or (
            isinstance(tracking_uri, str) and "://" not in tracking_uri
        ):
            p = Path(tracking_uri).resolve()
            if p.suffix == ".db":
                p.parent.mkdir(parents=True, exist_ok=True)
                self.tracking_uri = f"sqlite:///{p.as_posix()}"
            else:
                p.mkdir(parents=True, exist_ok=True)
                self.tracking_uri = f"sqlite:///{(p / 'mlflow.db').as_posix()}"
        else:
            self.tracking_uri = tracking_uri

        self.experiment_name = experiment_name

    def _configure(self) -> MlflowClient:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        return MlflowClient(tracking_uri=self.tracking_uri)

    def log_simulated_training_run(
        self,
        release_id: str,
        dataset_digest: str,
        model_name: str,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        run_origin: str = "simulated_demo",
    ) -> dict[str, Any]:
        """Record a real MLflow run tagged explicitly with run_origin ('simulated_demo' or 'actual_mlflow')."""
        if run_origin not in {"simulated_demo", "actual_mlflow"}:
            raise ValueError(f"Unsupported run_origin: {run_origin!r}")

        self._configure()
        run_name = f"{release_id}-{model_name}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "run_origin": run_origin,
                    "release_id": release_id,
                    "dataset_digest": dataset_digest,
                    "model_name": model_name,
                }
            )
            log_params: dict[str, Any] = {
                "release_id": release_id,
                "dataset_digest": dataset_digest,
                "model_name": model_name,
            }
            if params:
                log_params.update(params)
            mlflow.log_params(log_params)
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

            info = run.info
            return {
                "run_id": info.run_id,
                "experiment_id": info.experiment_id,
                "release_id": release_id,
                "dataset_digest": dataset_digest,
                "model_name": model_name,
                "run_origin": run_origin,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "tracking_uri": self.tracking_uri,
            }

    def query_runs_by_digest(self, dataset_digest: str) -> list[dict[str, Any]]:
        """Query only actual recorded MLflow runs matching dataset_digest."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        exp = client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            return []

        safe_digest = dataset_digest.replace("'", "")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.dataset_digest = '{safe_digest}'",
            order_by=["attributes.start_time DESC"],
        )
        results: list[dict[str, Any]] = []
        for r in runs:
            results.append(
                {
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "run_origin": r.data.tags.get("run_origin", ""),
                    "release_id": r.data.tags.get("release_id", ""),
                    "dataset_digest": r.data.tags.get("dataset_digest", ""),
                    "model_name": r.data.tags.get("model_name", ""),
                    "metrics": dict(r.data.metrics),
                    "params": dict(r.data.params),
                }
            )
        return results
