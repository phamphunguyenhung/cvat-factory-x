"""Versioned SQLite Model Registry with SHA256 deduplication and lazy runtime loading."""
from __future__ import annotations

import gc
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from cvat_tools.factory.registry.models import (
    ExpectedLabelSchema,
    ModelProvenance,
    ModelRecord,
)

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ModelRegistryStore:
    """Versioned SQLite store for model metadata, provenance, benchmarks, and SHA256 deduplication."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._loaders: dict[str, Callable[[ModelRecord], Any]] = {}
        self._unloaders: dict[str, Callable[[Any], None]] = {}
        self._loaded: dict[str, Any] = {}
        self._active_heavy: str | None = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS registry_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = conn.execute("SELECT MAX(version) AS v FROM registry_migrations").fetchone()
            current = int(row["v"]) if row and row["v"] is not None else 0
            if current < 1:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY,
                        version TEXT NOT NULL,
                        architecture TEXT NOT NULL,
                        path TEXT NOT NULL,
                        sha256 TEXT NOT NULL DEFAULT '',
                        size_bytes INTEGER NOT NULL DEFAULT 0,
                        framework TEXT NOT NULL,
                        supported_tasks_json TEXT NOT NULL DEFAULT '[]',
                        input_width INTEGER NOT NULL DEFAULT 640,
                        input_height INTEGER NOT NULL DEFAULT 640,
                        cpu_supported INTEGER NOT NULL DEFAULT 1,
                        gpu_supported INTEGER NOT NULL DEFAULT 0,
                        ram_mb INTEGER NOT NULL DEFAULT 512,
                        vram_mb INTEGER NOT NULL DEFAULT 0,
                        expected_label_schema_json TEXT NOT NULL DEFAULT '{}',
                        benchmark_status TEXT NOT NULL DEFAULT 'unverified',
                        benchmark_metrics_json TEXT NOT NULL DEFAULT '{}',
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        license TEXT NOT NULL DEFAULT 'unknown',
                        heavyweight INTEGER NOT NULL DEFAULT 0,
                        is_available INTEGER NOT NULL DEFAULT 1,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_models_sha256
                        ON models(sha256) WHERE sha256 != '';

                    CREATE TABLE IF NOT EXISTS model_aliases (
                        alias_path TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        discovered_from TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
                    );
                    """
                )
                conn.execute(
                    "INSERT INTO registry_migrations (version, applied_at) VALUES (?, ?)",
                    (1, _utc_now()),
                )

    def schema_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(version) AS v FROM registry_migrations").fetchone()
            return int(row["v"]) if row and row["v"] is not None else 0

    def register(self, record: ModelRecord, *, deduplicate_sha256: bool = True) -> tuple[ModelRecord, bool]:
        """Insert or update a model record.

        When ``deduplicate_sha256`` is True and a non-empty SHA256 already exists under a
        different ``model_id``, the existing primary record is preserved and the new path is
        recorded as an alias instead of creating a duplicate entry.

        Returns ``(canonical_record, deduplicated)``.
        """
        now = record.updated_at or _utc_now()
        with self._connect() as conn:
            if deduplicate_sha256 and record.sha256:
                existing = conn.execute(
                    "SELECT * FROM models WHERE sha256 = ? LIMIT 1",
                    (record.sha256,),
                ).fetchone()
                if existing and existing["model_id"] != record.model_id:
                    canon_id = str(existing["model_id"])
                    if record.path:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO model_aliases (alias_path, model_id, sha256, discovered_from, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (record.path, canon_id, record.sha256, record.provenance.discovered_from, now),
                        )
                    prov = ModelProvenance.from_dict(json.loads(existing["provenance_json"] or "{}"))
                    aliases = set(prov.aliases)
                    if record.path and record.path != existing["path"]:
                        aliases.add(record.path)
                    if record.model_id:
                        aliases.add(f"alias_id:{record.model_id}")
                    merged_prov = ModelProvenance(
                        source=prov.source or record.provenance.source,
                        discovered_from=prov.discovered_from,
                        config_source=prov.config_source or record.provenance.config_source,
                        registered_at=prov.registered_at or now,
                        aliases=tuple(sorted(aliases)),
                    )
                    conn.execute(
                        "UPDATE models SET provenance_json = ?, updated_at = ? WHERE model_id = ?",
                        (json.dumps(merged_prov.to_dict()), now, canon_id),
                    )
                    return self._row_to_record(
                        conn.execute("SELECT * FROM models WHERE model_id = ?", (canon_id,)).fetchone()
                    ), True

            prov = record.provenance
            if not prov.registered_at:
                prov = ModelProvenance(
                    source=prov.source,
                    discovered_from=prov.discovered_from,
                    config_source=prov.config_source,
                    registered_at=now,
                    aliases=prov.aliases,
                )

            in_w, in_h = record.input_size if len(record.input_size) == 2 else (640, 640)
            conn.execute(
                """
                INSERT INTO models (
                    model_id, version, architecture, path, sha256, size_bytes, framework,
                    supported_tasks_json, input_width, input_height, cpu_supported, gpu_supported,
                    ram_mb, vram_mb, expected_label_schema_json, benchmark_status,
                    benchmark_metrics_json, provenance_json, license, heavyweight, is_available, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    version=excluded.version,
                    architecture=excluded.architecture,
                    path=excluded.path,
                    sha256=CASE WHEN excluded.sha256 != '' THEN excluded.sha256 ELSE models.sha256 END,
                    size_bytes=CASE WHEN excluded.size_bytes > 0 THEN excluded.size_bytes ELSE models.size_bytes END,
                    framework=excluded.framework,
                    supported_tasks_json=excluded.supported_tasks_json,
                    input_width=excluded.input_width,
                    input_height=excluded.input_height,
                    cpu_supported=excluded.cpu_supported,
                    gpu_supported=excluded.gpu_supported,
                    ram_mb=excluded.ram_mb,
                    vram_mb=excluded.vram_mb,
                    expected_label_schema_json=excluded.expected_label_schema_json,
                    benchmark_status=excluded.benchmark_status,
                    benchmark_metrics_json=excluded.benchmark_metrics_json,
                    provenance_json=excluded.provenance_json,
                    license=excluded.license,
                    heavyweight=excluded.heavyweight,
                    is_available=excluded.is_available,
                    updated_at=excluded.updated_at
                """,
                (
                    record.model_id,
                    record.version,
                    record.architecture,
                    record.path,
                    record.sha256,
                    int(record.size_bytes),
                    record.framework,
                    json.dumps(list(record.supported_tasks)),
                    int(in_w),
                    int(in_h),
                    1 if record.cpu_supported else 0,
                    1 if record.gpu_supported else 0,
                    int(record.ram_mb),
                    int(record.vram_mb),
                    json.dumps(record.expected_label_schema.to_dict()),
                    record.benchmark_status,
                    json.dumps(record.benchmark_metrics),
                    json.dumps(prov.to_dict()),
                    record.license,
                    1 if record.heavyweight else 0,
                    1 if record.is_available else 0,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM models WHERE model_id = ?", (record.model_id,)).fetchone()
            return self._row_to_record(row), False

    def get(self, model_id: str) -> ModelRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
            return self._row_to_record(row) if row else None

    def get_by_sha256(self, sha256: str) -> ModelRecord | None:
        if not sha256:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE sha256 = ? LIMIT 1", (sha256,)).fetchone()
            return self._row_to_record(row) if row else None

    def list_aliases(self, model_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT alias_path FROM model_aliases WHERE model_id = ? ORDER BY alias_path",
                (model_id,),
            ).fetchall()
            return [str(r["alias_path"]) for r in rows]

    def list_models(
        self,
        *,
        task: str | None = None,
        framework: str | None = None,
        only_available: bool = False,
    ) -> list[ModelRecord]:
        with self._connect() as conn:
            sql = "SELECT * FROM models WHERE 1=1"
            params: list[Any] = []
            if framework:
                sql += " AND framework = ?"
                params.append(framework)
            if only_available:
                sql += " AND is_available = 1"
            sql += " ORDER BY model_id"
            rows = conn.execute(sql, params).fetchall()
        records = [self._row_to_record(r) for r in rows]
        if task:
            records = [r for r in records if task in r.supported_tasks]
        return records

    def update_benchmark(
        self,
        model_id: str,
        *,
        status: str,
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT benchmark_metrics_json FROM models WHERE model_id = ?", (model_id,)).fetchone()
            if not row:
                return False
            merged = json.loads(row["benchmark_metrics_json"] or "{}")
            if metrics:
                merged.update(metrics)
            conn.execute(
                "UPDATE models SET benchmark_status = ?, benchmark_metrics_json = ?, updated_at = ? WHERE model_id = ?",
                (status, json.dumps(merged), _utc_now(), model_id),
            )
            return True

    def delete(self, model_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            return cur.rowcount > 0

    def sync_from_yaml(self, yaml_path: str | Path, *, workspace_root: Path | None = None) -> int:
        """Import model declarations from config/models.yaml without loading weights."""
        p = Path(yaml_path)
        if not p.is_file():
            return 0
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        models_cfg = raw.get("models") or {}
        count = 0
        for mid, cfg in models_cfg.items():
            if not isinstance(cfg, dict):
                continue
            task = str(cfg.get("task", "detect"))
            backends = cfg.get("backends") or {}
            framework = next(iter(backends), cfg.get("backend", "pytorch"))
            checkpoint = str(cfg.get("checkpoint", ""))
            imgsz = int((cfg.get("inference") or {}).get("imgsz", 640))
            mem_mb = int(cfg.get("memory_mb", 512))
            heavy = bool(cfg.get("heavyweight", mem_mb >= 800))
            classes = tuple(str(c) for c in (cfg.get("classes") or ()))
            geom_map = {
                "detect": ("bbox", "rectangle"),
                "segment": ("polygon", "mask"),
                "lane-segment": ("polygon", "mask"),
                "prompt-segment": ("polygon", "mask"),
                "pose": ("skeleton", "points"),
                "track": ("track", "bbox"),
            }
            geoms = geom_map.get(task, ("bbox",))

            bench_status = "unverified"
            for _, bval in backends.items():
                if isinstance(bval, str) and bval.startswith("supported"):
                    bench_status = "supported"
                    break

            ckpt_exists = False
            if checkpoint and workspace_root is not None and not checkpoint.startswith("huggingface:"):
                cand = Path(checkpoint)
                if not cand.is_absolute():
                    cand = workspace_root / checkpoint
                ckpt_exists = cand.is_file()

            rec = ModelRecord(
                model_id=str(mid),
                version=str(cfg.get("version", mid)),
                architecture=str(cfg.get("architecture", mid.split("-")[0])),
                path=checkpoint,
                sha256=str(cfg.get("sha256", "")),
                size_bytes=int(cfg.get("size_bytes", 0)),
                framework=str(framework),
                supported_tasks=(task,),
                input_size=(imgsz, imgsz),
                cpu_supported=True,
                gpu_supported=bool(cfg.get("gpu_supported", False)),
                ram_mb=mem_mb,
                vram_mb=int(cfg.get("vram_mb", 0)),
                expected_label_schema=ExpectedLabelSchema(
                    classes=classes,
                    geometry_types=geoms,
                ),
                benchmark_status=bench_status,
                benchmark_metrics={"benchmark_ref": cfg.get("benchmark", "")} if cfg.get("benchmark") else {},
                provenance=ModelProvenance(
                    source=str(cfg.get("source", "")),
                    discovered_from="config_yaml",
                    config_source=str(p),
                ),
                license=str(cfg.get("license", "unknown")),
                heavyweight=heavy,
                is_available=ckpt_exists or checkpoint.startswith("huggingface:"),
            )
            self.register(rec, deduplicate_sha256=False)
            count += 1
        return count

    # --- Lazy runtime loading (never called at CLI startup) ---

    def register_loader(
        self,
        model_id: str,
        loader: Callable[[ModelRecord], Any],
        unloader: Callable[[Any], None] | None = None,
    ) -> None:
        self._loaders[model_id] = loader
        if unloader is not None:
            self._unloaders[model_id] = unloader

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def load_runtime(self, model_id: str, *, allow_multi_heavy: bool = False) -> Any:
        if model_id in self._loaded:
            return self._loaded[model_id]
        rec = self.get(model_id)
        if rec is None:
            raise KeyError(f"Model '{model_id}' not found in registry.")
        if rec.heavyweight and not allow_multi_heavy and self._active_heavy and self._active_heavy != model_id:
            self.unload_runtime(self._active_heavy)
        if model_id not in self._loaders:
            raise NotImplementedError(f"No lazy loader registered for '{model_id}'.")
        obj = self._loaders[model_id](rec)
        self._loaded[model_id] = obj
        if rec.heavyweight:
            self._active_heavy = model_id
        return obj

    def unload_runtime(self, model_id: str) -> bool:
        obj = self._loaded.pop(model_id, None)
        if obj is None:
            return False
        unloader = self._unloaders.get(model_id)
        if unloader is not None:
            unloader(obj)
        if self._active_heavy == model_id:
            self._active_heavy = None
        gc.collect()
        return True

    def unload_all_runtime(self) -> int:
        ids = list(self._loaded)
        for mid in ids:
            self.unload_runtime(mid)
        return len(ids)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            model_id=str(row["model_id"]),
            version=str(row["version"]),
            architecture=str(row["architecture"]),
            path=str(row["path"]),
            sha256=str(row["sha256"]),
            size_bytes=int(row["size_bytes"]),
            framework=str(row["framework"]),
            supported_tasks=tuple(json.loads(row["supported_tasks_json"] or "[]")),
            input_size=(int(row["input_width"]), int(row["input_height"])),
            cpu_supported=bool(row["cpu_supported"]),
            gpu_supported=bool(row["gpu_supported"]),
            ram_mb=int(row["ram_mb"]),
            vram_mb=int(row["vram_mb"]),
            expected_label_schema=ExpectedLabelSchema.from_dict(
                json.loads(row["expected_label_schema_json"] or "{}")
            ),
            benchmark_status=str(row["benchmark_status"]),
            benchmark_metrics=json.loads(row["benchmark_metrics_json"] or "{}"),
            provenance=ModelProvenance.from_dict(json.loads(row["provenance_json"] or "{}")),
            license=str(row["license"]),
            heavyweight=bool(row["heavyweight"]),
            is_available=bool(row["is_available"]),
            updated_at=str(row["updated_at"]),
        )
