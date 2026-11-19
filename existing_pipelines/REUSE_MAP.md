# REUSE_MAP.md — Phase 0 Codebase Inventory & Classification

> **Generated:** 2026-11-18 | **Owner:** `@auditor` | **Scope:** `src/cvat_tools/`, `src/mtm/`, `harness/`, `scripts/`, `docs/guidelines/`, `data/`, `tests/`, and `~/.gemini/config/skills/cvat-image-first-sync/SKILL.md`.
> **Legend:**
> - **REUSE**: Production-ready, clean, agent-agnostic, or core SSOT; wrap or import directly without copying.
> - **REFACTOR**: Good domain logic/algorithm, but needs adaptation to Factory V3 interfaces, removal of hardcoded job/path assumptions, or BOM/encoding cleanup.
> - **REPLACE**: Superseded by Factory V3 (`src/cvat_tools/factory/`) or V3 `harness/` contracts, or monolithic/tightly coupled.
> - **DEPRECATE**: One-off Week 1/Week 2/Job 285/Job 1623/Job 1975 scripts, legacy wrappers (`mtm`), or superseded drafts; keep read-only for reference, do not build new pipelines on top.

---

## 1. `src/cvat_tools/` Modules

### 1.1 `core/` (Tool Kernel & Persistence)
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/core/client.py` | 136 | Shared CVAT REST client (`Cvat`, `load_env`, `load_profile`, `WriteNotAuthorized`) | **REUSE** | Clean, read-only by default (`write_enabled=False`); wrap in `factory/cvat/gateway.py`. |
| `src/cvat_tools/core/permissions.py` | 26 | `Permission` enum (`READ_ONLY`, `LOCAL_WRITE`, `REMOTE_WRITE`) & gate | **REUSE** | Enforces safety boundaries. |
| `src/cvat_tools/core/contracts.py` | 56 | `ToolSpec`, `ToolResult`, `ResourceRequirements` | **REUSE** | Core tool kernel contract. |
| `src/cvat_tools/core/registry.py` | 213 | `ToolRegistry` + `build_default_registry()` + RSS tracking | **REUSE** | Agent-agnostic CLI tool dispatch. |
| `src/cvat_tools/core/model_registry.py` | 129 | `ModelSpec`, `ModelRegistry` (lazy load, single-heavyweight eviction) | **REFACTOR** | Reuse eviction/lazy-load pattern; extend into SQLite-backed `factory/registry/` with SHA256 deduplication. |
| `src/cvat_tools/core/shapestore.py` | 231 | Local SQLite `ShapeStore` separating `human_final` from drafts | **REFACTOR** | Solid foundation; evolve schema in `factory/storage/` for 5-stage provenance (`original_annotation`, `model_proposal`, `refined_proposal`, `user_reviewed_revision`, `cvat_synchronized_revision`). |
| `src/cvat_tools/core/transactions.py` | 49 | Backup, run log, shape/track rollback | **REUSE** | Essential safety layer. |
| `src/cvat_tools/core/cache.py` | 38 | Content-addressed cache key (`file_hash`, `make_key`, `Cache`) | **REUSE** | Deterministic caching. |
| `src/cvat_tools/core/config.py` | 32 | YAML loader for projects/models/profiles | **REUSE** | Clean config loader. |
| `src/cvat_tools/core/paths.py` | 24 | Workspace path helpers (`job_dir`) | **REUSE** | No hardcoded credentials/jobs. |

### 1.2 `autolabel/` (Autolabel Factory V2)
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/autolabel/cache.py` | 101 | Incremental CVAT snapshot, SHA256 image cache, JSON checkpoints | **REFACTOR** | Good caching/checkpoint logic; migrate checkpointing into `factory/storage/` SQLite. |
| `src/cvat_tools/autolabel/qc.py` | 158 | Image stats, dark enhancement, region-level QC (`evaluate_frame_qc`) | **REFACTOR** | Useful image-aware QC cues; generalize away from Face50-only checks into `factory/preprocessing/` & `factory/quality/`. |
| `src/cvat_tools/autolabel/refine.py` | 258 | Selective region refinement + self-correction loop (max 3 attempts) | **REFACTOR** | Good selective-refinement pattern; wrap behind Skeleton `AnnotationAdapter`. |
| `src/cvat_tools/autolabel/sync.py` | 146 | Staleness-guarded minimal PATCH sync + read-back + auto-rollback | **REUSE** | Gold-standard safe sync (`ConcurrentEditConflict`); reuse/wrap in `factory/sync/`. |
| `src/cvat_tools/autolabel/cli.py` | 210 | Autolabel V2 CLI (`analyze|draft|refine|validate|sync|benchmark|report|run`) | **REPLACE** | Superseded by unified `cvat-factory` CLI (`factory/cli/`). |

### 1.3 `pose/` (Pose17 & Face50)
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/pose/face50.py` | 171 | FAN 2D iBUG-68 -> 50-pt mapping, spline resampling, `unrotate`, `FaceLandmarker` | **REUSE** | Wrap directly inside `factory/adapters/skeleton/` (do not copy-paste). |
| `src/cvat_tools/pose/face50_engine.py` | 836 | Face50 local-basis validation, `align_face50_clahe_fan`, surgical proposals, bold renderer, safe sync | **REUSE** | Primary engine for Face50; wrap in Skeleton adapter, extract general CLAHE+Gamma logic to `factory/preprocessing/`. |
| `src/cvat_tools/pose/person17.py` | 236 | YOLO11-pose + ViTPose-B + 4-rot + consensus + `enforce_image_lr` | **REUSE** | Wrap directly inside `factory/adapters/skeleton/`. |
| `src/cvat_tools/pose/pose17_validate.py` | 987 | 5-layer Pose17 validator, rule registry, case miner, guideline engine | **REUSE** | Comprehensive deterministic validator; wrap in Skeleton adapter. |
| `src/cvat_tools/pose/pose17_sync.py` | 549 | Pose17 flag-only proposal builder, bold VS Code renderer, staleness-guarded sync | **REUSE** | Reuse renderer (`open_in_vscode`, `_draw_bold_skeleton`) and safe sync guard. |
| `src/cvat_tools/pose/pose17_cli.py` | 323 | Pose17 standalone CLI | **DEPRECATE** | Keep for backwards compatibility; route new workflows through `cvat-factory`. |
| `src/cvat_tools/pose/gold.py` | 53 | Gold vs auto-draft diff loader | **REUSE** | Useful for evaluation/active learning. |
| `src/cvat_tools/pose/week2.py` | 387 | Week 2 `spm` wrapper for face50/person_vf17 | **DEPRECATE** | Superseded by `person17.py`, `face50.py`, and `face50_engine.py`. |

### 1.4 `detection/` (Object Detection & Night Enhancement)
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/detection/detector.py` | 101 | Lazy-load YOLO detector (`pytorch`/`onnx`/`openvino`) | **REUSE** | Clean multi-backend wrapper; wrap in Detection `AnnotationAdapter`. |
| `src/cvat_tools/detection/all_objects.py` | 392 | Multi-scale OpenVINO/YOLO + horizon tiling for vehicles/pedestrians/riders | **REFACTOR** | Strong CPU multi-scale & horizon tiling logic; generalize label mapping beyond Job 285. |
| `src/cvat_tools/detection/signs_lights.py` | 341 | Specialist detector for `traffic sign` / `traffic light` + bulb/glare cues | **REFACTOR** | Useful specialist cues & false-positive filters; integrate into Detection adapter. |
| `src/cvat_tools/detection/night_enhance.py` | 104 | CPU-only adaptive night/low-light CLAHE + Gamma variants | **REUSE** | Core building block for `factory/preprocessing/`. |
| `src/cvat_tools/detection/yolo_tile.py` | 107 | Tiled YOLO inference + `derive_riders` | **REFACTOR** | Merge tiling logic cleanly with `all_objects.py` in Detection adapter. |
| `src/cvat_tools/detection/cached.py` | 42 | Content-addressed detection cache | **REUSE** | Essential for fair tracker benchmarking & resumable runs. |
| `src/cvat_tools/detection/adaptive.py` | 46 | Hard-frame escalation heuristic | **REFACTOR** | Tightly coupled to MOT; generalize in `factory/preprocessing/` / adapter. |
| `src/cvat_tools/detection/frames.py` | 36 | Local dir / CVAT frame source | **REUSE** | Simple frame abstraction. |

### 1.5 `segmentation/` & `lanes/`
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/segmentation/inputspec.py` | 215 | `config/model_inputs.yaml` contract (`prepare_input`, `restore`, `enhance_dark`) | **REUSE** | Critical anti-distortion contract; wrap in `factory/preprocessing/` & adapters. |
| `src/cvat_tools/segmentation/onnx_seg.py` | 55 | SegFormer ONNX runner obeying `inputspec` | **REUSE** | Clean ONNX wrapper. |
| `src/cvat_tools/segmentation/evidence.py` | 205 | Unified stuff/lane evidence (`_B2`, `_B5`, `_M2F`, `_Lane`) | **REFACTOR** | Good multi-model evidence abstraction; wire into Segmentation adapter & `ModelRegistry`. |
| `src/cvat_tools/segmentation/geomrefine.py` | 634 | CPU polygon smoothing, `building` 0°/90° snap (`max_points=20, eps=1.5`), edge snap | **REUSE** | High-value deterministic geometry refiner. |
| `src/cvat_tools/segmentation/layers.py` | 177 | Z-order & occlusion resolution for segmentation | **REUSE** | Core layer composition logic. |
| `src/cvat_tools/segmentation/stuff_things.py` | 117 | `STUFF_OVER_THING` subtraction & overlap reporting | **REUSE** | Shared-boundary constraint enforcement. |
| `src/cvat_tools/segmentation/pole_rule.py` | 146 | Narrow 4-vertex rectangle extraction for `pole` | **REUSE** | Domain rule for thin vertical structures. |
| `src/cvat_tools/segmentation/sam_refine.py` | 108 | Lazy-load SAM2.1 / MobileSAM crop refinement | **REFACTOR** | Wire through `ModelRegistry` so single-heavyweight residency is enforced. |
| `src/cvat_tools/segmentation/labelmap.py` | 78 | Full stuff coverage & gap map | **REUSE** | Useful coverage diagnostics. |
| `src/cvat_tools/segmentation/inventory.py` | 61 | Per-frame label inventory summary | **REUSE** | Useful reporting helper. |
| `src/cvat_tools/lanes/framegen.py` | 629 | Per-frame `lane/*` + `area/*` generator (SegFormer-Lane + CV) | **REFACTOR** | Contains hard-won `rules.md §5` thresholds; modularize for Segmentation/Structure adapter. |
| `src/cvat_tools/lanes/detectors/{white,yellow,curb,crosswalk,__init__}.py` | ~700 | Dedicated classical CV lane/crosswalk/curb detectors | **REFACTOR** | Note: `curb.py` and `yellow.py` have UTF-8 BOM (`U+FEFF`); strip BOM & wrap in Segmentation/Structure adapter. |
| `src/cvat_tools/lanes/areadirect.py` | 99 | `area/direct` + `area/alternative` derivation | **REUSE** | Clean geometric derivation. |
| `src/cvat_tools/lanes/laneinterp.py` | 266 | Pure numpy/cv2 lane anchor interpolation | **REUSE** | Useful when anchor keyframes exist. |

### 1.6 `tracking/` & `src/mtm/`
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/tracking/engine.py` | 75 | Run Ultralytics trackers on cached detections | **REUSE** | Decouples detection from association. |
| `src/cvat_tools/tracking/draft.py` | 101 | Track draft builder (ByteTrack, BoT-SORT, TrackTrack, OC-SORT) | **REUSE** | Wrap in Tracking `AnnotationAdapter`. |
| `src/cvat_tools/tracking/geometry.py` | 189 | Track selection, gap fill, IoU, MOT 1.1 export | **REUSE** | Pure geometry, no network. |
| `src/cvat_tools/tracking/keyframes.py` | 126 | DP keyframe optimizer for CVAT linear interpolation | **REUSE** | Proven -81% keyframe reduction within `tol_px`. |
| `src/cvat_tools/tracking/reid.py` | 95 | Selective ReID (`HsvEmbedder`, `OSNetEmbedder` [weights missing]) | **REFACTOR** | `HsvEmbedder` works; `OSNetEmbedder` blocked on missing weights. |
| `src/cvat_tools/tracking/writer.py` | 53 | Safe track writer (dry-run, backup, dedupe, read-back) | **REUSE** | Safe remote track sync. |
| `src/cvat_tools/tracking/cli.py` | 218 | `mtm` CLI (`inspect|draft|bench|apply|check|export|rollback`) | **DEPRECATE** | Legacy CLI; keep for `tests/legacy_mtm/test_core.py`, route new work via `cvat-factory`. |
| `src/mtm/*` | ~200 | Legacy `mtm` package wrappers | **DEPRECATE** | Backwards-compat shim only. |

### 1.7 `review/`, `quality/`, `decision/`, `learning/`, `job/`, `pipeline/`, `benchmark/`, `cli.py`
| Path | Lines | Purpose | Classification | Rationale / Notes |
|---|---:|---|---|---|
| `src/cvat_tools/review/validators.py` | 400 | Ruleset-driven shape/geometry validator (`default.yaml`) | **REUSE** | Core validation engine. |
| `src/cvat_tools/review/pack.py` | 184 | Zoomed crop + coordinate grid review pack | **REUSE** | Evidence packet rendering. |
| `src/cvat_tools/review/edits.py` & `apply.py` | 173+272 | Structured JSON edit commands + edge-snapping apply | **REUSE** | Safe deterministic edit application. |
| `src/cvat_tools/quality/metrics_eval.py` | 161 | BBox IoU, Boundary F1, invalid geometry count | **REUSE** | Quantitative evaluation. |
| `src/cvat_tools/quality/review.py` & `assist.py` | 141+120 | Risk-ranked HTML review queue + manual track assist | **REUSE** | Human-in-the-loop review tooling. |
| `src/cvat_tools/decision/{action_schema,engine,rule_policy,vlm_adjudicator}.py` | ~650 | Hybrid deterministic-rule -> optional VLM -> geometry action engine | **REFACTOR** | Good decision/audit trail; align with `factory/` proposal/review states. |
| `src/cvat_tools/learning/{cvat_learner,convention_engine,yolo_exporter,types}.py` | ~1018 | Active-learning profile calibrator, Z-order conventions, YOLO exporter | **REUSE** | Valuable calibration & export utilities. |
| `src/cvat_tools/job/{discover,ruleset,capability,report,weekly}.py` | ~460 | Job preflight discovery, YAML ruleset merge, gap report | **REFACTOR** | `discover.py` & `ruleset.py` should be wrapped/extended in `factory/discovery/`. |
| `src/cvat_tools/pipeline/{run,cli_run,export,source,lanefilter,stuffpoly,targeteval}.py` | ~900 | Job 285 per-frame unified pipeline | **DEPRECATE** | Superseded by `harness/` V3 & `factory/` adapters. |
| `src/cvat_tools/benchmark/{ablation,detector_bench,tracker_bench,figures}.py` | ~389 | Subprocess-isolated CPU/RAM benchmarks | **REUSE** | Clean benchmarking harness. |
| `src/cvat_tools/cli.py` | 502 | `cvat-tools` v2 CLI entrypoint | **REUSE** | Keep stable for v2 users; `cvat-factory` (`factory/cli/`) is the v3 entrypoint. |

---

## 2. `harness/` (V3 Panoptic Prototype)
All 15 files (`harness/*.py`, ~9,500 lines) implement the V3 Object-First Panoptic Harness (`contracts.py`, `frame_profiler.py`, `label_router.py`, `model_registry.py`, `thing_engine.py`, `stuff_engine.py`, `structure_engine.py`, `temporal_engine.py`, `layer_composer.py`, `quality_evaluator.py`, `review_queue.py`, `cvat_adapter.py`, `experiment_registry.py`, `orchestrator.py`, `cli_v3.py`).
- **Classification:** **REFACTOR** — Reuse `contracts.py` (RLE codecs, typed contracts), `frame_profiler.py`, `label_router.py`, and `model_registry.py` patterns inside `src/cvat_tools/factory/` adapters rather than maintaining a parallel top-level `harness/` package.

---

## 3. `scripts/` Inventory & Classification

| Script | Lines | Classification | Rationale |
|---|---:|---|---|
| `scripts/fetch_checkpoints.py` | 13 | **REUSE** | Downloads `yolo26{n,s,l}.pt` into `models/checkpoints/`. |
| `scripts/run_detector_bench.py` | 23 | **REUSE** | Reproducible detector benchmark runner. |
| `scripts/train_from_cvat.py` | 187 | **REUSE** | Exports CVAT annotations to YOLO format + optional fine-tuning. |
| `scripts/create_sandbox_task.py` | 131 | **REUSE** | Safe sandbox task/job creator for testing. |
| `scripts/export_tools_zip.py` | 104 | **REUSE** | Secret-free portable packaging script. |
| `scripts/make_contact_sheets.py`, `render_skeleton_overlay.py`, `review_pack.py` | 58+70+67 | **REUSE** | Lightweight visual inspection helpers. |
| `scripts/eval_pose17.py`, `eval_face50.py`, `eval_all_bbox.py` | 78+69+655 | **REFACTOR** | Useful evaluation scripts; consolidate into `factory/quality/`. |
| `scripts/seg_input_bench.py`, `seg_model_bench.py`, `lanes_job285_eval.py` | 266+213+143 | **REFACTOR** | Benchmark scripts tied to Job 285; keep for regression checks. |
| `scripts/bbox_all_objects.py`, `cars_first.py`, `export_full_bbox_job285.py`, `run_grounding_dino_job285.py`, `compare_drafts.py`, `decide.py`, `build_case_index.py` | ~1700 | **DEPRECATE** | Job-285 / one-off experiment scripts superseded by `cvat_tools` & `factory/`. |
| `scripts/w2_inventory.py`, `w2_qc_rules.py`, `w2_face_propose.py`, `w2_face_sync.py`, `w2_final_report.py`, `w2_recovery_compare.py`, `w2_recovery_render.py`, `w2_review_image.py`, `w2_source_manual.py` | ~900 | **DEPRECATE** | Week 2 Job 1623/1975 ad-hoc scripts superseded by `pose17_*`, `face50_engine.py`, and `autolabel/`. |

---

## 4. `docs/guidelines/`, `data/`, `tests/`, and Skills

- **`docs/guidelines/`**:
  - `docs/guidelines/pose17/`: `POSE17_GENERAL_GUIDELINE.md`, `G06_POSE17_GUIDELINE.md`, `pose17_rules.csv` (`P17-G01`..`P17-G30`), `difficult_cases.csv`, `GUIDELINE_CHANGELOG.md` -> **REUSE** as SSOT ruleset for Pose17.
  - `docs/guidelines/face50/`: `FACE50_GUIDELINE.md`, `FACE50_RULES.csv` (`F50-R01`..`F50-R11`), `FACE50_DIFFICULT_CASES.csv` -> **REUSE** as SSOT ruleset for Face50.
  - `docs/QUY_UOC_VE_CHUNG.md`, `docs/QUY_UOC_VE_JOB285.md`, `docs/LAYER_POLICY.md`, `docs/MODEL_ROUTING.md` -> **REUSE** as SSOT for BBox/Seg/Traffic schemas.
- **`~/.gemini/config/skills/cvat-image-first-sync/SKILL.md`** (42 lines): **REUSE** as the operational safety & visual-review protocol (read-only snapshot -> rotation-aware validation -> zoom crop -> surgical proposal + review queue -> VS Code auto-open contact sheet -> staleness-guarded minimal PATCH + read-back).
- **`data/`**: Contains `autolabel_cache/job_1975/`, `benchmarks/` (`ablation.csv`, `detector_bench.csv`, `tracker_bench.csv`, `seg_model_bench_job285.json`, `refine_job285/`), `cache/detections/`, `datasets/job285_yolo/`, `exports/day05-mot17/`, `jobs/285/learned_profile.json`. Note: `data/store/` and `data/runs/` are not currently populated on disk.
- **`tests/`**: 38 test files (`tests/test_*.py` + `tests/legacy_mtm/test_core.py`) covering `autolabel_v2`, `face50`, `face50_engine`, `pose17_validate`, `person17`, `harness_v3`, `model_registry`, `tool_registry`, `shapestore_sync`, `night_enhance`, `seg_*`, `lanes_*`, `decision`, `review`, `cvat_learner`, `convention_engine`, `yolo_exporter`. **REUSE** as regression suite.
