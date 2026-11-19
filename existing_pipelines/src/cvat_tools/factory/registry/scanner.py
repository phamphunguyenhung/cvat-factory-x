"""Bounded-depth local model weight scanner (repo, HF, PyTorch, Ultralytics, ONNX, OpenVINO)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from cvat_tools.factory.registry.models import (
    ExpectedLabelSchema,
    ModelProvenance,
    ModelRecord,
)
from cvat_tools.factory.registry.store import ModelRegistryStore

WEIGHT_EXTENSIONS = frozenset({
    ".pt",
    ".pth",
    ".onnx",
    ".xml",
    ".safetensors",
    ".tflite",
    ".task",
    ".engine",
})

SKIP_DIRS = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
})


@dataclass(frozen=True)
class ScanRoot:
    path: Path
    source_tag: str
    max_depth: int = 4


@dataclass
class ScanSummary:
    scanned_roots: list[str] = field(default_factory=list)
    files_found: int = 0
    registered: int = 0
    deduplicated: int = 0
    errors: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hex digest in bounded memory chunks."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def default_scan_roots(workspace_root: Path | None = None) -> list[ScanRoot]:
    """Return bounded-depth candidate roots for repo and standard ML framework caches."""
    roots: list[ScanRoot] = []
    home = Path.home()

    if workspace_root is not None:
        for sub, tag in (
            ("models", "repo_models"),
            ("checkpoints", "repo_checkpoints"),
            ("data/models", "repo_data_models"),
        ):
            p = workspace_root / sub
            if p.is_dir():
                roots.append(ScanRoot(p, tag, max_depth=4))

    hf_home = Path(os.environ.get("HF_HOME", str(home / ".cache" / "huggingface")))
    hf_hub = hf_home / "hub"
    if hf_hub.is_dir():
        roots.append(ScanRoot(hf_hub, "huggingface_cache", max_depth=4))

    torch_home = Path(os.environ.get("TORCH_HOME", str(home / ".cache" / "torch")))
    torch_ckpt = torch_home / "hub" / "checkpoints"
    if torch_ckpt.is_dir():
        roots.append(ScanRoot(torch_ckpt, "pytorch_cache", max_depth=3))

    for ultra_dir in (
        Path(os.environ["YOLO_CONFIG_DIR"]) if os.environ.get("YOLO_CONFIG_DIR") else None,
        home / ".config" / "Ultralytics",
        home / "AppData" / "Roaming" / "Ultralytics",
    ):
        if ultra_dir is not None and ultra_dir.is_dir():
            roots.append(ScanRoot(ultra_dir, "ultralytics_cache", max_depth=3))

    for cache_dir, tag in (
        (home / ".cache" / "openvino", "openvino_cache"),
        (home / ".cache" / "onnx", "onnx_cache"),
    ):
        if cache_dir.is_dir():
            roots.append(ScanRoot(cache_dir, tag, max_depth=3))

    return roots


def _iter_bounded(root: Path, max_depth: int) -> Iterable[Path]:
    """Walk directory tree up to ``max_depth`` without following symlinks."""
    if not root.is_dir():
        return
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in SKIP_DIRS:
                    continue
                if depth + 1 <= max_depth:
                    stack.append((entry, depth + 1))
            elif entry.is_file():
                yield entry


def _is_openvino_ir(xml_path: Path) -> bool:
    """OpenVINO IR consists of <name>.xml paired with <name>.bin."""
    return xml_path.suffix.lower() == ".xml" and xml_path.with_suffix(".bin").is_file()


def _infer_metadata(file_path: Path, source_tag: str, size_bytes: int) -> ModelRecord:
    """Infer framework, architecture, tasks, and resource estimates without loading weights."""
    ext = file_path.suffix.lower()
    name_low = file_path.stem.lower()
    path_low = str(file_path).lower()

    framework_map = {
        ".onnx": "onnx",
        ".xml": "openvino",
        ".pt": "ultralytics" if "yolo" in name_low or "ultralytics" in path_low else "pytorch",
        ".pth": "pytorch",
        ".safetensors": "huggingface",
        ".tflite": "tflite",
        ".task": "mediapipe",
        ".engine": "tensorrt",
    }
    framework = framework_map.get(ext, "unknown")

    # Architecture & task inference from filename/path
    arch = "unknown"
    tasks: tuple[str, ...] = ("detect",)
    geoms: tuple[str, ...] = ("bbox", "rectangle")
    classes: tuple[str, ...] = ()
    kps: tuple[str, ...] = ()
    input_size = (640, 640)

    if "yolo" in name_low:
        arch = "yolo"
        if "-pose" in name_low or "pose" in name_low:
            tasks = ("pose",)
            geoms = ("skeleton", "points")
        elif "-seg" in name_low:
            tasks = ("segment",)
            geoms = ("polygon", "mask")
        else:
            tasks = ("detect",)
            geoms = ("bbox", "rectangle")
    elif "segformer" in path_low or "mask2former" in path_low or "m2f" in name_low:
        arch = "segformer" if "segformer" in path_low else "mask2former"
        tasks = ("segment", "lane-segment") if "lane" in path_low else ("segment",)
        geoms = ("polygon", "mask")
        input_size = (1024, 1024) if "cityscapes" in path_low else (512, 512)
    elif "sam2" in path_low or "sam_" in name_low:
        arch = "sam2"
        tasks = ("prompt-segment", "segment")
        geoms = ("polygon", "mask")
        input_size = (1024, 1024)
    elif "face_landmarker" in name_low or "face50" in name_low or "2dfan" in name_low or "fan" in name_low:
        arch = "face_landmarker" if "landmarker" in name_low else "face_alignment"
        tasks = ("pose",)
        geoms = ("skeleton", "points")
        input_size = (256, 256)
    elif "rtmpose" in name_low or "hrnet" in name_low or "vitpose" in name_low:
        arch = "pose_topdown"
        tasks = ("pose",)
        geoms = ("skeleton", "points")
        input_size = (256, 192)
    elif "osnet" in name_low or "reid" in name_low:
        arch = "reid"
        tasks = ("track",)
        geoms = ("track", "bbox")
        input_size = (256, 128)

    # Read companion config.json if present (HF / local export) without loading weights
    cfg_json = file_path.parent / "config.json"
    if cfg_json.is_file():
        try:
            cfg = json.loads(cfg_json.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                id2label = cfg.get("id2label")
                if isinstance(id2label, dict) and id2label:
                    classes = tuple(str(v) for _, v in sorted(id2label.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])))
                archs = cfg.get("architectures")
                if isinstance(archs, list) and archs:
                    arch = str(archs[0])
        except Exception:
            pass

    size_mb = max(1, int(size_bytes // (1024 * 1024)))
    est_ram_mb = max(256, size_mb * 3)
    heavy = est_ram_mb >= 800

    safe_stem = file_path.stem.replace(" ", "_")
    model_id = f"{framework}:{safe_stem}"

    return ModelRecord(
        model_id=model_id,
        version="discovered",
        architecture=arch,
        path=str(file_path),
        sha256="",  # filled by scanner
        size_bytes=size_bytes,
        framework=framework,
        supported_tasks=tasks,
        input_size=input_size,
        cpu_supported=framework != "tensorrt",
        gpu_supported=framework in ("pytorch", "ultralytics", "onnx", "tensorrt"),
        ram_mb=est_ram_mb,
        vram_mb=est_ram_mb if framework == "tensorrt" else 0,
        expected_label_schema=ExpectedLabelSchema(
            classes=classes,
            geometry_types=geoms,
            skeleton_keypoints=kps,
        ),
        benchmark_status="unverified",
        provenance=ModelProvenance(
            source=str(file_path),
            discovered_from=source_tag,
        ),
        heavyweight=heavy,
        is_available=True,
    )


class ModelScanner:
    """Bounded-depth scanner that discovers local weights and deduplicates by SHA256."""

    def __init__(self, store: ModelRegistryStore) -> None:
        self.store = store

    def scan(
        self,
        roots: list[ScanRoot] | None = None,
        *,
        workspace_root: Path | None = None,
        compute_hash: bool = True,
    ) -> ScanSummary:
        scan_roots = roots if roots is not None else default_scan_roots(workspace_root)
        summary = ScanSummary()

        for sr in scan_roots:
            summary.scanned_roots.append(str(sr.path))
            for fpath in _iter_bounded(sr.path, sr.max_depth):
                ext = fpath.suffix.lower()
                if ext not in WEIGHT_EXTENSIONS:
                    continue
                if ext == ".xml" and not _is_openvino_ir(fpath):
                    continue
                summary.files_found += 1
                try:
                    size_bytes = fpath.stat().st_size
                    sha = compute_sha256(fpath) if compute_hash else ""
                    rec = _infer_metadata(fpath, sr.source_tag, size_bytes)
                    rec = ModelRecord(
                        **{**rec.__dict__, "sha256": sha}
                    )
                    canon, dedup = self.store.register(rec, deduplicate_sha256=bool(sha))
                    if dedup:
                        summary.deduplicated += 1
                    else:
                        summary.registered += 1
                    if canon.model_id not in summary.model_ids:
                        summary.model_ids.append(canon.model_id)
                except Exception as exc:
                    summary.errors.append(f"{fpath}: {exc}")

        return summary
