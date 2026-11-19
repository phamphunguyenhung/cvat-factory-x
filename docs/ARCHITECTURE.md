# Kiến trúc Release Control Tower

Ghi chú: mô tả dựa trên cấu trúc thư mục `release_control/`; hành vi chi tiết cần đối chiếu mã nguồn.

## Control plane và data plane

```
 DATA PLANE (dữ liệu, chỉ đọc nguồn)         CONTROL PLANE (quyết định, bằng chứng)
 data_source (coco250, ...)  ----manifest--> contracts  (schema, taxonomy, release_policy)
 CVAT (đọc, RC_READ_ONLY)    ----adapter---> validators (structural/geometry/semantic/privacy)
 bản sao làm việc + snapshot <---restore---- versions   (snapshot, restore, digest)
                                             lineage    (events, impact)
                                             release    (gate, evidence)
                                             fault_lab  (inject, benchmark)
                                             db (SQLite) -> dashboard (FastAPI) / cli
```

Data plane: ảnh, annotation, snapshot. Control plane: hợp đồng dữ liệu, kết quả validate, quyết định HOLD/READY, lineage, evidence. Control plane không sửa nguồn.

## Module `release_control`

| Module | Vai trò |
| :-- | :-- |
| `adapters/` | `cvat_adapter`, `datumaro_adapter`, `mlflow_adapter`: nối với hệ ngoài |
| `contracts/` | `schema.yaml`, `taxonomy.yaml`, `release_policy.yaml` và `engine.py` |
| `validators/` | structural, geometry, semantic, privacy; `gate.py` gộp kết quả |
| `fault_lab/` | `inject.py` chèn lỗi T1–T8 có seed; `benchmark.py` đo phát hiện |
| `versions/` | `snapshot.py`, `restore.py` (kiểm digest) |
| `lineage/` | `events.py` ghi sự kiện, `impact.py` truy vết ảnh hưởng |
| `release/` | `gate.py` (QC/privacy bắt buộc), `evidence.py` |
| `scale/` | Hiện chỉ là khung, chưa có benchmark quy mô |
| `dashboard/` | Giao diện FastAPI (`app.py`, `static/`) |
| `db.py`, `models.py`, `config.py`, `cli.py`, `demo/run_e2e.py` | Lưu trữ, mô hình dữ liệu, cấu hình, CLI, demo end-to-end |

## Tái dùng và giả lập

- Tái dùng từ `existing_pipelines\` (pipeline CVAT sẵn có, xem `REUSE_MAP.md` ở đó): client/CLI CVAT. Mức tái dùng thực tế cần đối chiếu REUSE_MAP, chưa xác minh ở đây.
- Dataset: dataset thật của lab Day 10 (`data_source`), mở chỉ đọc.
- **Giả lập (simulated):** lỗi được chèn bởi fault_lab; run MLflow demo (`run_origin = simulated_demo`); mọi bước trong demo trên tập nhỏ. Không có review của người, ACK của consumer hay training run thật nếu chưa được ghi từ nguồn thật.
- Chưa đạt: quy mô 2 triệu frame (cần benchmark).
