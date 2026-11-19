# CVAT tuần 2 (cvat.edufun.ai, org cohort-5a, nhóm G06)

Mỗi tuần một thư mục, mỗi task một thư mục con (có `TASK.md`), mỗi job một thư mục con (có `SPEC.md`).
Danh sách job máy đọc nằm ở `../weeks/2026-W47.yaml`; guideline ở `../weeks/guidelines/w2-face50-pose17.md`.

| Task | Tên | Job của mình | Loại | Ghi chú |
|---|---|---|---|---|
| [473](task473-face/TASK.md) | `W2-FACE-G6-T1` | [1975](task473-face/job1975-face50/SPEC.md) (frame 90–119) | face landmark 50 điểm, 7 skeleton | mình là leader, task có 5 job |
| [402](task402-pose/TASK.md) | `week2-human-pose/G06` | [1623](task402-pose/job1623-pose17/SPEC.md) (frame 90–119) | pose COCO 17 điểm | chỉ thấy job này (task trả 403) |

Mỗi thư mục job: `backups/` (backup trước khi ghi lên CVAT, không commit), `drafts/` (nháp local), `metrics/`, `review/` (ảnh overlay, ghi chú QC).
Dữ liệu máy sinh (labels.json, overlay) nằm ở `../data/jobs/edufun/<job_id>/`.

Xem nhanh job: `python -m cvat_tools week status`.

## Công cụ nháp (local, `source=auto`, cần BTC cho phép dùng AI)
| Việc | Lệnh |
|---|---|
| Nháp mặt 50 điểm (FAN + map iBUG-68) | `python -m cvat_tools pose draft --job 1975` |
| Nháp pose 17 điểm (`--backend auto` = YOLO + ViTPose, điểm bất đồng bị đánh Q) | `python -m cvat_tools pose draft --job 1623` |
| Mất mạng: đọc cache local | thêm `--offline` (cần đã chạy draft online một lần, hoặc có `data/jobs/edufun/<job>/job_info.json`) |
| Đẩy lên CVAT (dry-run mặc định) | `python -m cvat_tools pose apply --job <id> --replace-auto` rồi thêm `--yes` |
| Hoàn tác một lần đẩy | `python -m cvat_tools pose rollback --run <job>/backups/run_*.json --yes` |
| Đo nháp so với nhãn bạn đã sửa | `python scripts/eval_face50.py --job 1975 --refresh-gold` / `python scripts/eval_pose17.py --job 1623 --refresh-gold` |
| Đóng gói công cụ | `python scripts/export_tools_zip.py --open` |

`push_policy.json` trong từng thư mục job liệt kê frame để vẽ tay (`skip_frames`) và nhóm điểm đánh Q cả nhóm (`q_labels`).
Chưa có gold cho pose: sửa tay 5–8 frame đại diện rồi chạy `eval_pose17.py --refresh-gold` để biết backend nào chính xác hơn.
