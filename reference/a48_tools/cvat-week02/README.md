# CVAT Tuần 2 — bộ công cụ

Tools của tuần 2, chia theo 2 task. Nhật ký: [Task 473 Face](../../nhat-ky-tuan/tuan-02/task-473-face_pham-phu-nguyen-hung.md) · [Task 402 Pose](../../nhat-ky-tuan/tuan-02/task-402-pose17_pham-phu-nguyen-hung.md).

| Thư mục / file | Task | Nội dung |
|---|---|---|
| [`task473-face/`](task473-face/) | **Task 473 — Job 1975**, Face 50 điểm, frame 90–119 | Công cụ đề xuất, đồng bộ và đo nháp mặt; cấu hình push; báo cáo QC và edge case |
| [`task402-pose/`](task402-pose/) | **Task 402 — Job 1623**, Pose 17 điểm | Công cụ đo nháp pose; cấu hình push |
| [`w2_qc_rules.py`](w2_qc_rules.py) | Dùng chung cho cả 2 job (1623, 1975) | Kiểm tra quy tắc hình học, chỉ đọc CVAT, không sửa gì |
| [`MANIFEST.md`](MANIFEST.md) | Chung | Nguồn gốc file, SHA-256, người sở hữu |

## Task 473 — Face 50 điểm (`task473-face/`)

| File | Làm gì |
|---|---|
| `w2_face_propose.py` | Sinh đề xuất 50 điểm cho 1 frame từ mốc giải phẫu đọc bằng mắt. Chỉ ghi file đề xuất, không ghi CVAT |
| `w2_face_sync.py` | Patch an toàn: chỉ cập nhật toạ độ của shape có sẵn, backup trước, đọc lại để so sánh, rollback nếu lỗi. Mặc định dry-run |
| `eval_face50.py` | Đo độ lệch nháp so với các điểm đã sửa tay trên CVAT (NME). Chỉ đọc CVAT |
| `push_policy_job1975.json` | Frame bỏ qua (`f100`, `f118`) và lý do; danh sách điểm gán `Q` |
| `edge_cases.md` | Các edge case đã ghi nhận của job 1975 |
| `qc_summary.md` | Kết quả QC: 30/30 frame, 221 keypoint đã sync (R1+R2), 3 frame còn trong Review Queue |

## Task 402 — Pose 17 điểm (`task402-pose/`)

| File | Làm gì |
|---|---|
| `eval_pose17.py` | Đo nháp pose 17 điểm so với điểm đã sửa tay trên CVAT (lệch px và OKS). Chỉ đọc CVAT |
| `push_policy_job1623.json` | Frame bỏ qua (`f100`, `f118`) và lý do |

## Lưu ý khi chạy

- Các script cần thư viện `cvat_tools` và biến môi trường `CVAT_TOKEN`. Không có token nào được lưu trong repo.
- Các script được copy từ repo gốc `learning/cvat/`, nên một số đường dẫn tương đối (`src/`, `reports/`) trỏ tới cấu trúc của repo gốc, chưa được điều chỉnh cho repo này.
- `w2_face_sync.py` import `w2_recovery_compare`, file này không có trong repo này.
- Không đưa ảnh, `.env`, cookie, backup, model hay checkpoint vào repo (xem mục R12 trong `MANIFEST.md`).
