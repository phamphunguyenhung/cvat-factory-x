# CVAT tuần 2 — tooling cho task 473 (face 50 điểm)

Cập nhật: 19/11/2026. Phạm vi: task 473, job 1975 (frame 90–119, 30 frame, ảnh 640×480). Các số dưới đây lấy từ file trong repo `learning/cvat`, chưa đo lại.

## 1. Model đang dùng để sinh nháp

Pipeline `pose draft --job 1975` (chạy local trên CPU, không gọi LLM, không tốn tiền API):

| Bước | Thành phần | Ghi chú |
|---|---|---|
| Tiền xử lý | CLAHE (clip 3.0, lưới 8×8) + Gamma (0.65 ảnh màu, 0.55 ảnh IR khi luma < 50) | Giúp ảnh tối/IR |
| Phát hiện mặt | S3FD (`s3fd-619a316812.pth`, 85.7 MB) | Cắt khung mặt làm đầu vào cho FAN |
| Landmark | FAN 2DFAN4 qua `face-alignment 1.5.0`, ra 68 điểm iBUG rồi map về 50 điểm | Heavyweight, ~1.2 GB RAM |
| Xoay mặt | `try_rotations=True` (0/90/180/270°) | Frame xoay chạy lâu hơn |
| Hậu xử lý | `BrowEdgeSnap`, `InnerLipClamp` | Chỉ hình học, không gọi model |

Checkpoint nằm ở `~/.cache/torch/hub/checkpoints` (mirror `D:\AI_Cache\torch`): `2DFAN4-11f355bf06.pth.tar` (95.6 MB, có từ 16/11) và S3FD. `MODEL_INVENTORY.md` trong repo learning ghi thiếu 2DFAN4, bản đó đã lỗi thời.

## 2. Chi phí thời gian

- Khoảng **3.4 s/frame** (S3FD + FAN, CPU, 640×480, không `torch.compile`) theo `config/models.yaml`. 30 frame ≈ 1.7 phút khi không phải thử xoay.
- Lần chạy gần nhất xoay: 17 frame 0°, 8 frame 270°, 3 frame 90°, 1 frame 180°. Chưa có số đo thời gian thật cho lần này.
- Phát hiện được 29/30 frame (96.7%).
- Độ lệch so với nhãn hiện có trên CVAT: trung vị 3.2 px, trung bình 11.0 px, lớn nhất 331 px (28 frame so sánh được).
- Độ chính xác đã đo chỉ có frame 93: lỗi trung bình 4.8 px so với nhãn sửa tay. Một frame không đủ để kết luận.

## 3. Công cụ khác trong tuần

| Tool | Việc | Ghi chú |
|---|---|---|
| `cvat-week02/task473-face/` | Đề xuất, sync, đo nháp face; cấu hình push; QC | Sync mặc định dry-run, backup trước, rollback |
| `review-assign/assign_reviewers.py` | Chia reviewer chéo | Reviewer không bao giờ là annotator; cân bằng số job |
| `review-assign/apply_cvat.py` | Chia reviewer rồi đặt `stage=validation` + `assignee` cho job đã xong | Dry-run mặc định, `--yes` mới ghi, có backup và `--rollback`. Chỉ ghi job nêu trong `--jobs` |
| `self-critique` (repo riêng, `D:\06-Learning\AIA-VINGROUP\self-critique`) | Audit/refine/benchmark/report 6 tầng L1–L6 cho face50 và pose17 | Chỉ ghi file local; không ghi CVAT. Cần `cvat_tools` trong `PYTHONPATH` |

Trạng thái CVAT ngày 19/11 (đọc qua API): job 1975 và 1976 đã ở stage `validation`, reviewer đã gán (1975 → Long, 1976 → Hùng). Job 1972–1974 vẫn `annotation`, mới chỉ chia reviewer (1972 → Thịnh, 1973 → Việt, 1974 → Hưng), chưa ghi CVAT.

## 4. Kết quả self-critique trên job 1975 (offline snapshot)

- Audit 30 frame: 15 PASS, 10 REVIEW, 3 AUTO_FIX, 2 BLOCKED (f102, f106 chưa có nhãn).
- Refine dry-run: 6 sửa được chấp nhận, 4 rollback (reviewer từ chối vì lỗi vẫn còn); lỗi 24 → 18.
- Điểm nội bộ chỉ để so trước/sau, **không phải accuracy**. Gold face chỉ có frame 93 và frame đó không bị sửa.
- PASS nghĩa là qua kiểm tra tự động, chưa phải đã duyệt. Ví dụ f98 từng bị Haiku ghi nghi ngờ vẫn PASS.

## 5. Chưa có / việc nên làm

- Chưa đo thời gian thật từng frame và từng bước của pipeline.
- Nhật ký task 473 chưa có mục thời gian hay chi phí.
- Chưa có token/chi phí cho lượt review Haiku (frame 90–104) ghi trong `issues.md`.
- Cần thêm gold (sửa tay vài frame) mới nói được accuracy.
