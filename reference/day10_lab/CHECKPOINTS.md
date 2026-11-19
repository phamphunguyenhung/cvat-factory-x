# Checkpoints — Day 10

Bài **nhóm**, khoảng **4 giờ**. Các mốc tính từ lúc bắt đầu, là điểm **tự kiểm tra** tại lớp, **không phải hạn nộp**.

| Mốc | Mục tiêu | Tự kiểm tra |
|---|---|---|
| 0:15 | **CP0** Chuẩn bị | Đã tải dữ liệu; `runs/r<seed>/delivery` có; đã chia việc trong `TEAM.md`. |
| 0:55 | **CP1** Lỗi luật | `score_findings.py`: lỗi luật F1 ≈ 1.0, trên ít nhất 2 seed (seed thêm tạo bằng `inject_release.py`) và lô sạch. |
| 1:50 | **CP2** Đối chiếu + xem ảnh | Biết số ứng viên trên lô sạch; hàng đợi đủ nhỏ để xem hết; `review_log.csv` xong cho seed luyện tập; công cụ xem ảnh sẵn sàng cho đề ẩn. |
| 2:05 | **CP3** Quyết định | Lineage có HOLD v1; `impact` ra `R-021: affected=True`. |
| 2:45 | **CP4** Bàn giao v2 | `check_handover.py` chỉ còn FAIL ở trace/security (nếu chưa làm). |
| 3:05 | **CP5** Truy vết | `trace.json` có ba khoá; `score_findings.py` có phần truy vết. |
| 3:15 | **CP6a** Bảo mật | `security.md` đủ ba mục; `check_handover.py` không còn FAIL. |
| 3:55 | **CP6b** Đề ẩn | Đã nộp `de_an/`; đã chạy `export_submission.py`, commit, push. |

## Cần hiểu ở từng mốc

### CP1
Vì sao `wrong_hash` phải tính lại sha256 thay vì kiểm định dạng? · `scene_leak` có target là gì? · Vì sao `guideline_mismatch` so với job chứ không so với các ảnh khác?

### CP2
Lô sạch có thể vẫn có ứng viên: vì sao? · Báo hết ứng viên thì precision và F1 đổi thế nào? · Detection ghi `car`, panoptic ghi `truck`: làm sao biết bên nào sai?

### CP3
Đổi dataset có sửa được model đã train không? · Vì sao ngưỡng là theo tỉ lệ ảnh lỗi chứ không theo số lỗi?

### CP4
Lỗi nào sửa được trong v2, lỗi nào phải gán lại? · Hash chứng minh gì, **không** chứng minh gì? · Vì sao không ghi đè v1?

### CP5
Tầng (a) và tầng (b) khác nhau thế nào? · Vì sao lấy guideline theo job?

### CP6
Thấy khuôn mặt rõ trong lô: vì sao **Hold + báo người phụ trách** chứ không tự làm mờ?

Chi tiết lệnh: [GUIDE.md](GUIDE.md).
