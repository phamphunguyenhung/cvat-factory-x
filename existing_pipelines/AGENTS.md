# AGENTS.md — CVAT Workspace Harness

- **Root**: `D:\06-Learning\AIA-VINGROUP\learning\cvat` (Python >= 3.11, CPU-first, Windows 16 GB RAM).
- **Repo Map**: Read [docs/AGENT_MAP.md](docs/AGENT_MAP.md) first. Do not scan the repo recursively.
- **Core Rules**: Follow [rules.md](rules.md) for safety invariants. Load domain rules on demand from `.agents/skills/`.

## Task Router (Default: `FAST`)

| Mode | When to use | Discovery & Edits | Delegation | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **FAST** (default) | Isolated bug, 1-file edit, straightforward fix | <= 5 files via `AGENT_MAP.md` + `rg -n` | **0** subagents | Targeted `pytest` on affected test file(s) |
| **STANDARD** | Multi-file feature or moderate cross-file bug | Inspect direct deps once, short plan | <= **1** narrow specialist | Relevant `pytest` suite + `git diff` check |
| **DEEP** | Cross-module refactor, security/CVAT-sync risk, independent subtasks | Bounded discovery, checkpoint in `.agents/state/` | <= **3** concurrent specialists (disjoint files) | Targeted + regression `pytest` + dry-run sync check |

Escalate from `FAST` only when complexity, risk, or independent parallelism concretely requires it.

## Context Economy & Discovery
1. Start from `docs/AGENT_MAP.md`, `git status --short`, and `git diff --name-only`.
2. Read bounded line ranges (`StartLine`/`EndLine`). Never reread unchanged files.
3. Exclude `data/`, `models/`, `checkpoints/`, `reports/`, `artifacts/`, `**/__pycache__/`, and `week02/*/drafts/*.json` unless debugging a specific artifact.
4. Print only failing test output or concise pass summaries, never full logs.

## Safety & Git Policy
- **Read-only CVAT by default**: Remote writes (`store push`, `store rollback`, `pose apply`, `w2_face_sync`) require dry-run first, explicit user approval, and `--yes` + `write_enabled=True`.
- **Separation of truth**: Keep AI/rule drafts (`ai_draft`, `rule_draft`) separate from `human_final` in `data/store/shapes.sqlite`. Preserve `source="manual"` when pushing approved shapes.
- **Git**: Work only on the active feature branch. Never push, force-push, reset, or discard unrelated uncommitted changes.

## Checkpoint & Resume (`.agents/state/`)
- For multi-step `STANDARD`/`DEEP` tasks, persist state via `python scripts/harness_checkpoint.py save ...`.
- After context compaction: read the active `.agents/state/<task>.json` first, run `python scripts/harness_checkpoint.py validate <file>` to detect stale file hashes against disk, and inspect only changed/missing files. Never store tokens or secrets in checkpoints.

## On-Demand Skills
- `.agents/skills/cvat-face50-pose17/SKILL.md`: Face50 landmarks (Job 1975) & Pose17 skeleton (Job 1623) rules and sync guards.
- `.agents/skills/cvat-seg-lanes/SKILL.md`: Segmentation, `geomrefine.py`, lanes, night enhancement, and detection rules.
- `.pi/skills/cvat-workflow/SKILL.md`: Pi CLI wrapper workflow.
