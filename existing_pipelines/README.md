# CVAT Autolabel Factory (V3) & Engineering Toolkit (CET V2)

Universal CLI (`cvat-factory`) and local-first CVAT annotation pipeline (**Face50**, **Pose17**, **Custom Skeleton**, **BBox Detection**, **Segmentation**, **Tracking**), with full backward compatibility for V2 (`cvat-tools`, `mtm`).

---

## 1. Windows PowerShell Quickstart (`cvat-factory`)

```powershell
cd D:\06-Learning\AIA-VINGROUP\learning\cvat

# 1. Cài đặt editable (không tải lại dependencies đã có)
pip install -e . --no-build-isolation --no-deps

# 2. Kiểm tra môi trường, RAM/CPU, lazy-loading & kết nối CVAT
cvat-factory doctor
cvat-factory doctor --offline --json

# 3. Cấu hình PAT an toàn (Windows Credential Manager / env, không lưu plaintext vào repo)
cvat-factory setup

# 4. Khám phá task/job & model cục bộ (không nạp weights khi khởi động)
cvat-factory discover
cvat-factory models list
cvat-factory models list --scan
cvat-factory jobs --mine

# 5. Kiểm tra schema, loại hình (polygon/rectangle/skeleton) & model phù hợp cho job
cvat-factory inspect --job 1623
cvat-factory inspect --job 1975

# 6. Chạy pipeline local-first (adaptive preprocessing + inference + QC + lưu SQLite draft)
cvat-factory run --job 1975 --mode auto
cvat-factory run --job 1623 --mode pose17 --limit 20

# 7. Sinh ảnh đối chiếu BEFORE/AFTER/OVERLAY & mở trực tiếp trong VS Code (tối đa 5 ca ưu tiên)
cvat-factory review --job 1975

# 8. Đồng bộ an toàn lên CVAT (mặc định dry-run; chỉ ghi thật khi có --yes & đủ quyền)
cvat-factory sync --job 1975
cvat-factory sync --job 1975 --yes
cvat-factory sync --job 1975 --rollback --latest

# 9. Đo benchmark V2 vs V3 & khôi phục phiên chạy bị gián đoạn
cvat-factory benchmark --job 1623 --limit 10 --compare-v2
cvat-factory resume --job 1975
```

---

## 2. Danh sách 11 lệnh `cvat-factory`

| Lệnh | Chức năng chính | Ghi chú an toàn |
| :--- | :--- | :--- |
| `doctor` | Kiểm tra Python, RAM/CPU, lazy-loading, CVAT auth, models | Hỗ trợ `--offline`, `--json` |
| `setup` | Tìm/lưu PAT qua Windows Credential Manager hoặc env | Không in PAT, không đọc cookie trình duyệt |
| `discover` | Liệt kê task/job truy cập được, coverage & model gợi ý | Read-only |
| `models list` | Liệt kê model trong SQLite registry (`--scan` để quét cache) | Lazy load, deduplicate theo SHA256 |
| `jobs` | Danh sách job kèm trạng thái local run để resume | Read-only |
| `inspect --job <id>` | Đọc label schema, geometry/skeleton, coverage, model phù hợp | Dừng ở inspect nếu schema lạ |
| `run --job <id> --mode <m>` | Chạy adaptive preprocessing, inference, QC & lưu draft SQLite | Không tự ghi lên CVAT |
| `review --job <id>` | Tạo ảnh BEFORE/AFTER/OVERLAY + crop zoom + HTML dashboard | Tự mở VS Code (`code.cmd --reuse-window`), tối đa 5 ca/lượt |
| `sync --job <id>` | Kiểm tra diff, staleness, human-edit preservation & PATCH lên CVAT | **Dry-run mặc định**; cần `--yes` để ghi thật |
| `benchmark --job <id>` | Đo runtime, latency, RAM, cache hit, geometry failures (V2 vs V3) | Tách riêng cold inference vs cache hit |
| `resume --job <id>` | Tiếp tục run/sync bị gián đoạn từ SQLite checkpoint | Idempotent |

---

## 3. Annotation Types & Quality Gate

- **Skeleton / Keypoints (`adapters/skeleton/`)**: Hỗ trợ **Face50** (7 nhóm / 50 điểm), **Pose17** (COCO 17 điểm), và **Custom Skeleton** (không hardcode số landmark, không trộn mapping Pose17/Face50).
- **Detection BBox (`adapters/detection/`)**: Taxonomy COCO & 7/10 lớp giao thông, selective tiling, duplicate/overlap & missing-object QC.
- **Segmentation (`adapters/segmentation/`)**: Semantic/instance mask & polygon, `mask_readback_iou`, `LAYER_ORDER`, polygon simplification & self-intersection check.
- **Tracking (`adapters/tracking/`)**: Keyframe selection, temporal/scale jump & ID-switch risk detection (không tự đổi `track_id` khi thiếu bằng chứng).
- **Quality Gate 3 mức (`quality/` + `refinement/`)**:
  - `PASS`: Đạt đủ 6 chiều (geometry, image alignment, schema, confidence, no regression, approval).
  - `REFINE`: Tự sửa cục bộ có giới hạn (tối đa 2–3 lần/region, không reset toàn bộ skeleton).
  - `REVIEW`: Đưa vào hàng đợi VS Code Review Dashboard để người dùng duyệt.

---

## 4. Kiểm thử & Benchmark

```powershell
# Chạy 25-item Test Matrix cho Factory V3 (dùng fixtures/mocks, không đụng live CVAT)
python -m pytest -q tests/factory/

# Chạy toàn bộ test suite (V2 + V3)
python -m pytest -q tests/

# Chạy benchmark V2 vs V3 (lưu JSON report)
python scripts/benchmark_factory_v3.py --job 1623 --limit 10 --output data/benchmarks/factory_v3_bench.json
```

Tham khảo chi tiết kiểm kê model & module:
- [MODEL_INVENTORY.md](MODEL_INVENTORY.md) — Danh mục weights, SHA256, kích thước, cache paths và trạng thái sẵn có.
- [REUSE_MAP.md](REUSE_MAP.md) — Phân loại `REUSE / REFACTOR / REPLACE / DEPRECATE` các module V2.

---

## 5. V2 Backward-Compatible Commands (`python -m cvat_tools`)

Workspace chung cho công cụ CVAT (Day 3 segmentation, Day 4 pose, Day 5 MOT). `week01/` và `guides/` là nháp/metrics cũ, **không bị đụng**.

```powershell
cd D:\06-Learning\AIA-VINGROUP\learning\cvat
python -m cvat_tools doctor
python -m cvat_tools tools list
python -m cvat_tools --json tools info segmentation.prelabel
python -m cvat_tools projects list
python -m cvat_tools models list
python -m cvat_tools store capture --job 285
python -m cvat_tools store push --job 285          # dry-run mặc định; thêm --yes để ghi
python -m cvat_tools store rollback --latest --job 285
python -m cvat_tools full-bbox --job 285 --open
python -m cvat_tools inspect --project day05-mot17                # chỉ đọc
python -m cvat_tools track assist --project day05-mot17           # manual-assist, không cần AI, chỉ đọc
python -m cvat_tools detect --project day05-mot17 --profile balanced --limit 60 --allow-ai-draft
python -m cvat_tools track bench --project day05-mot17 --limit 60 --allow-ai-draft
python -m cvat_tools track keyframes --project day05-mot17 --tracker bytetrack --limit 60 --allow-ai-draft
python -m cvat_tools track review --project day05-mot17 --limit 60 --allow-ai-draft
python -m mtm check --job 15                                      # wrapper tương thích
```

---

## 6. Cấu trúc thư mục & Quy chuẩn an toàn

- `src/cvat_tools/factory/`: Kiến trúc V3 (`cli/`, `core/`, `auth/`, `discovery/`, `cvat/`, `registry/`, `preprocessing/`, `adapters/`, `quality/`, `refinement/`, `review/`, `sync/`, `storage/`, `orchestration/`, `reporting/`, `benchmark/`).
- `src/cvat_tools/core`, `detection/`, `tracking/`, `quality/`, `pose/`, `segmentation/`: Các module V2 được tái sử dụng qua adapter.
- `src/mtm`: Wrapper `python -m mtm` (cùng code với `cvat_tools.tracking`).
- `config/`: `cvat.*.yaml`, `models.yaml`, `profiles.yaml`, `tracking.yaml`.

**Nguyên tắc vận hành:**
- **Read-only & Dry-run mặc định**: Không ghi lên CVAT nếu thiếu `--yes` và quyền ghi thực tế; không in PAT ra terminal/log.
- **Lazy model loading**: `cvat-factory` không import `torch`/`ultralytics`/`onnxruntime` khi khởi động CLI; chỉ giữ tối đa 1 heavyweight model trong RAM.
- **Tách biệt Ground Truth vs Draft**: SQLite store tách riêng 5 stage (`original_annotation`, `model_proposal`, `refined_proposal`, `user_reviewed_revision`, `cvat_synchronized_revision`), bảo toàn chỉnh sửa của con người (`human_final`) và rollback có kiểm tra xung đột (`conflict-aware`).
