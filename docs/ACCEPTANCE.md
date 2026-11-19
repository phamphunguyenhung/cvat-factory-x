# Ma trận nghiệm thu — Release Control Tower

Cột "Trạng thái" mặc định là **CHƯA XÁC MINH**. Người điều phối chỉ điền PASS/FAIL sau khi chạy thật và dán bằng chứng (đường dẫn log/report). Không điền theo suy đoán.

| # | Tiêu chí | Cách kiểm chứng (lệnh / artifact) | Trạng thái | Bằng chứng |
| :-- | :-- | :-- | :-- | :-- |
| 1 | Phát hiện 8 loại lỗi T1–T8 (thiếu ảnh, sai hash, class không hợp lệ, bbox ngoài ảnh, bbox drift, class swap, scene leak, lệch guideline) | `python -X utf8 -m release_control.cli benchmark`; `reports/` | CHƯA XÁC MINH | |
| 2 | Held-out: lỗi dùng để chỉnh ngưỡng khác với lỗi dùng để đánh giá; báo recall/precision trên tập held-out | benchmark report (tách seed/tập) | CHƯA XÁC MINH | |
| 3 | Lỗi ngữ nghĩa (T5–T8) không tự chứng nhận; kết quả HOLD | `cli validate` trên bản đã inject | CHƯA XÁC MINH | |
| 4 | Digest snapshot ổn định và xác minh được (SHA-256 manifest) | `cli snapshot`, tính lại digest | CHƯA XÁC MINH | |
| 5 | Restore về snapshot cũ cho đúng digest; sai digest thì từ chối | `cli restore` | CHƯA XÁC MINH | |
| 6 | Lineage chỉ chứa cạnh có bằng chứng (event/digest đã ghi); không suy diễn cạnh | `release_control/lineage`, DB demo | CHƯA XÁC MINH | |
| 7 | Từ chối release khi thiếu QC hoặc thiếu privacy approval | `release_control/release/gate.py`, `release_policy.yaml` | CHƯA XÁC MINH | |
| 8 | Không bịa review/ACK của người dùng cuối/training run; PUBLISHED khác ACCEPTED; run mô phỏng gắn nhãn `simulated_demo` | evidence + MLflow adapter | CHƯA XÁC MINH | |
| 9 | Dashboard mở được và hiển thị đúng trạng thái HOLD/READY | `uvicorn dashboard.app:app --port 8765` | CHƯA XÁC MINH | |
| 10 | Nguồn dữ liệu gốc (`data_source`) không bị ghi đè | so digest trước/sau demo | CHƯA XÁC MINH | |

## Mục tiêu chưa đạt (nêu thẳng)

- **Mục tiêu 2 triệu frame = CHƯA ĐẠT.** Demo chỉ chạy trên tập nhỏ (mặc định `--limit 30` từ COCO250). Module `scale/` chưa có benchmark thông lượng thật. Cần đo: thời gian validate/snapshot/restore theo số frame, bộ nhớ, và ngoại suy có kiểm chứng. Cho tới khi có số đo thì không được tuyên bố đạt.
- CVAT thật: mặc định chỉ đọc (`RC_READ_ONLY=1`); chưa xác minh kết nối live trong tài liệu này.
