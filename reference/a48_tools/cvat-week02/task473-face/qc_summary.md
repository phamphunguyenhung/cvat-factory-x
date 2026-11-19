# QC Summary — Job 1975 (Task 473 / Face50)

- **Source:** [`FACE50_QC_REPORT.md`](file:///D:/06-Learning/AIA-VINGROUP/learning/cvat/artifacts/face50_research/FACE50_QC_REPORT.md)
- **Current QC Status:** `readback_verified_not_human_approved` (not yet `Human Reviewed` or `Officially Accepted`)

---

## 1. Verified Metrics (Round 1 + Round 2)

| Metric | Verified Value | Details / Evidence |
|---|---:|---|
| Frame Coverage | **30 / 30** (`90..119`) | `210` shapes, `1,500` keypoints, `0` empty frames |
| Round 1 — PATCH Update | **72** keypoints (`6` frames, `12` shapes) | `f096` (20 V/Q), `f099` (kp 42), `f101` (kp 42), `f104` (16 V->Q), `f107` (33 V/Q), `f108` (kp 42) |
| Round 1 — Read-back Verification | **6 / 6** (`readback_mismatches = 0`) | `before_face50_sync_20261118_104118.json` & `run_face50_sync_20261118_104118.json` |
| Round 2 — PATCH Update | **49** keypoints (`4` frames, `7` shapes) | `f094` (4), `f095` (5), `f105` (7), `f118` (33) |
| Round 2 — POST Create (empty frames) | **100** keypoints (`2` frames, `14` shapes) | `f102` (50 kp, 7 shapes) & `f106` (50 kp, 7 shapes); shape IDs `1453124..1453137`, `source="manual"` |
| Round 2 — Read-back Verification | **6 / 6** (`readback_mismatches = 0`) | `before_face50_sync_20261118_141219.json` & `run_face50_sync_20261118_141219.json` |
| Total Synced (R1 + R2) | **221** keypoints (`12` frames) | `121` patched + `100` created ([`FACE50_CORRECTION_HISTORY.csv`](file:///D:/06-Learning/AIA-VINGROUP/learning/cvat/artifacts/face50_research/FACE50_CORRECTION_HISTORY.csv): 221 data rows) |
| Review Queue (held back) | **3** frames | `f100` (steep profile), `f111` (dark IR, `BROW_BELOW_EYE`), `f112` (very dark IR, small face `47x38`) |
| Post-sync `validate_face50_frame` (30 frames) | **0** errors on R2 frames | Total across 30 frames: **12 errors** (`7` in Review Queue `f100, f111, f112`; `5` baseline `LIP_SELF_INTERSECT` on `f090, f096, f101, f107, f114`), **4 warnings** |

---

## 2. Remaining Items Requiring Human Decision

1. **Review Queue (`f100`, `f111`, `f112`):** Intentionally not auto-overwritten to avoid guessing coordinates on steep profiles or very dark IR frames.
2. **Baseline `LIP_SELF_INTERSECT` (`f090`, `f096`, `f101`, `f107`, `f114`):** `5` baseline frames still carry lip self-intersection errors outside Round 2 scope.
3. **Human Approval Gate:** Status remains `readback_verified_not_human_approved` until human inspection of the contact sheets confirms acceptance.
