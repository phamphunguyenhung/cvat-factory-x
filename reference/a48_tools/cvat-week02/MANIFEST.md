# SOURCE_SYNC_MANIFEST.md — Source Provenance & Integrity Manifest

> **Source Repository:** `D:\06-Learning\AIA-VINGROUP\learning` (`cvat/` & `cvat/week02/`)  
> **Source Commit:** `23f714dffbc6a84155dfe3c5e2398ec708e943fa` (`2026-11-18 18:59:29 +0700`)  
> **Owner:** Phạm Phú Nguyễn Hưng (`2A202604062` / `@phamphunguyenhung`, `@hungpixi`)

## Synced Files & Checksums

| Source Path (`learning/cvat/...`) | Destination (`A-048/...`) | SHA-256 | Test / Verification | Owner |
|---|---|---|---|---|
| `scripts/w2_qc_rules.py` | `tools/cvat/scripts/w2_qc_rules.py` | `F1ECFED4C7A319E88CEE281FEA00CCFA59590FEA8C1F736784389B6701EE1792` | Syntax + secret scan pass; runtime needs `cvat_tools` + `CVAT_TOKEN` | Phạm Phú Nguyễn Hưng |
| `scripts/w2_face_propose.py` | `tools/cvat/scripts/w2_face_propose.py` | `ACE39952EF9DC2CAFCAF83EBA805C89B4CECC1A9513AD42FF7419D9FB40DCE00` | Syntax + secret scan pass | Phạm Phú Nguyễn Hưng |
| `scripts/w2_face_sync.py` | `tools/cvat/scripts/w2_face_sync.py` | `D11D9B2C15FC75FDAA929158F10BB4493107AEEDBFF15D5502B314B0BEBE64CC` | Syntax + secret scan pass; used in R1+R2 sync | Phạm Phú Nguyễn Hưng |
| `scripts/eval_face50.py` | `tools/cvat/scripts/eval_face50.py` | `250BD393488AC3C5CBE631086F4640B8D4B82D879ACA89D9899EBF888D74160F` | Syntax + secret scan pass | Phạm Phú Nguyễn Hưng |
| `scripts/eval_pose17.py` | `tools/cvat/scripts/eval_pose17.py` | `07E5C33D6B4837A71CC0072490A223258119AD84630553098075CEDD2AB32141` | Syntax + secret scan pass | Phạm Phú Nguyễn Hưng |
| `week02/task473-face/job1975-face50/push_policy.json` | `tools/cvat/config/push_policy_job1975.json` | `748796A6C438A47834926188028480D2D0DF0777EECCAC13B0AA3A5C654A7433` | Valid JSON + secret scan pass | Phạm Phú Nguyễn Hưng |
| `week02/task402-pose/job1623-pose17/push_policy.json` | `tools/cvat/config/push_policy_job1623.json` | `2F0C37BCC93FB11288DA6398E89B2E645EF1D5B61AF21B0994FBFF4C9532F89F` | Valid JSON + secret scan pass | Phạm Phú Nguyễn Hưng |

## Excluded Artifacts (Rule R12)

- `.env`, API tokens, CVAT cookies
- `backups/`, `drafts/`, `models/`, `weights/`, `checkpoints/`
- Raw images and contact sheet JPG/PNG binaries (`data/`, `artifacts/*/evidence/`)
