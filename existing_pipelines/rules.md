# CVAT Tools Core Rules (`rules.md`)

> Core non-negotiable invariants for `cvat_tools`.
> Domain-specific thresholds live in `.agents/skills/*/references/` and are loaded on demand.

## 1. Agent-Agnostic & CLI-First
- Every `python -m cvat_tools ...` / `cvat-tools ...` command MUST run standalone without Pi, DeepSeek, or any LLM.
- `.pi/extensions/cvat-tools.ts` is a thin `--json` CLI wrapper only; never place inference logic inside it.

## 2. Data Safety & Read-Only Default
- **Read-only CVAT by default**: `store push`, `store rollback`, `pose apply`, and `w2_face_sync` MUST default to dry-run. Remote mutation requires explicit approval plus `--yes` and `write_enabled=True`.
- **Ground Truth vs Draft separation**: Store automated predictions in `ShapeStore` (`data/store/shapes.sqlite`) under `rule_draft` / `ai_draft` / `llm_edit`, never overwriting `human_final`.
- **Backup + Audit + Rollback + Staleness Guard**: Every real push MUST create a pre-write backup, write a run log (`data/runs/*.json` or `backups/run_*.json`), record `cvat_id`, verify read-back, and abort with `Stale proposal` if remote CVAT changed since `propose`.
- Keep `source="manual"` on approved shapes/elements pushed to CVAT (never `[AI]` tags).
- Preserve all human-placed coordinates during flag-only proposal updates (`before_el["points"] == proposed_el["points"]`).

## 3. 16 GB RAM & CPU-First Budget
- **Lazy imports**: Never import `torch`, `ultralytics`, `onnxruntime`, or `transformers` at CLI startup.
- **Single heavyweight model**: `ModelRegistry` (`src/cvat_tools/core/model_registry.py`) enforces `unload()` + `gc.collect()` before loading another heavy model.

## 4. Generalization & Visual QA
- Never copy or hardcode ground truth/frame-specific overrides from old jobs into `src/cvat_tools/`.
- When generating review images (`review_frames`, `proposed_corrections`, `before_after`), auto-open contact sheets in VS Code (`code.cmd --reuse-window`) before requesting sync approval.

## 5. Domain Rule References (Load On Demand)
- **Face50 & Pose17** (`face50_engine.py`, `pose17_sync.py`, Job 1975, Job 1623): see `.agents/skills/cvat-face50-pose17/references/face50-pose17-rules.md`.
- **Segmentation, Lanes, Night & Detection** (`geomrefine.py`, `lanes/framegen.py`, `detection/`): see `.agents/skills/cvat-seg-lanes/references/seg-lanes-rules.md`.
