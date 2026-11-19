# Kịch bản demo 5 phút

Dữ liệu: dataset thực của bài lab Day 10, `labs\K5-L23-Day10-Dataset-Release-Handover\data_source` (ví dụ `coco250`). Chạy: `scripts\run_demo.ps1` (hoặc `run_demo.sh`) rồi mở dashboard `python -X utf8 -m uvicorn dashboard.app:app --port 8765`. Con số cụ thể lấy từ output thật, không đọc từ kịch bản này.

| Phút | Bước | Làm gì | Nói gì / chỉ ra gì |
| :-- | :-- | :-- | :-- |
| 0:00–0:45 | 1. Load | Nạp subset dataset, tạo manifest và snapshot gốc | Dataset sạch có digest cố định; nguồn chỉ đọc |
| 0:45–1:45 | 2. Inject | Chèn lỗi T6 (class swap) và T5 (bbox drift) bằng fault_lab, có seed | Đây là lỗi ngữ nghĩa: file vẫn mở được, schema vẫn hợp lệ |
| 1:45–2:45 | 3. Validate / HOLD | Chạy validators trên bản lỗi | Release gate trả HOLD vì có lỗi mức critical; chỉ ra issue T5/T6 và frame bị ảnh hưởng |
| 2:45–3:30 | 4. Trace | Mở lineage/impact trên dashboard | Chỉ hiện cạnh có sự kiện đã ghi; run mô phỏng được gắn nhãn simulated |
| 3:30–4:15 | 5. Recover | Restore snapshot gốc, validate lại | Digest khớp snapshot gốc; trạng thái thoát HOLD nếu validate sạch (kiểm tra thực tế) |
| 4:15–5:00 | 6. Evidence | Mở release evidence/passport | Thiếu QC/privacy thì không release; không có ACK thì không ghi ACCEPTED |

Lưu ý khi trình bày: nếu một bước lỗi hoặc kết quả khác kỳ vọng thì nói đúng như vậy; không chỉnh số liệu. Mục tiêu 2 triệu frame chưa đạt (xem `ACCEPTANCE.md`).
