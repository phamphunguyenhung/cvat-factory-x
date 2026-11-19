# Job 1623 (task 402) — week2-human-pose/G06, frame 90–119 (30 ảnh 1280×1024)

URL: https://cvat.edufun.ai/tasks/402/jobs/1623
Gán cho: mình. Trạng thái lúc đọc (16/11/2026): `new`, stage `annotation`.

## Phạm vi
- 1 skeleton `person_vf17` mỗi ảnh, 17 điểm thứ tự COCO (tên điểm trên CVAT là số 1–17). Chế độ Shape.
- R/L theo **ảnh** (BTC V1): điểm chẵn (R) luôn ở phải ảnh, điểm lẻ (L) ở trái ảnh, kể cả quay lưng/xoay 90°. Tay/chân che hẳn không suy ra được: v0 (outside); mặt luôn V hoặc Q. Ảnh trong cabin, có ảnh hồng ngoại xám.

## Hiện trạng
- Có sẵn 10 skeleton `source=file` ở frame 91, 92, 97, 98, 103, 104, 109, 110, 115, 116, nhiều điểm lệch khỏi khớp (nháp thô).
- 20 frame còn lại chưa có nhãn. Chưa biết được phép xóa hay sửa nháp không.

## Nháp tool (17/11/2026, local, chưa đẩy lại lên CVAT)
- `pose draft --backend auto`: YOLO11l + ViTPose-B, xoay 4 hướng cho ảnh nằm ngang (f103, f106, f112, f118 được xoay), điểm hai model bất đồng > 0.35 x thân bị đánh Q.
- Vẽ tay (policy): f100 (nhòe), f118 (gục mặt). 15 skeleton auto cũ trên CVAT (từ yolo11l thuần) sẽ được thay khi chạy `pose apply --replace-auto --yes`.
- Chưa có gold: sửa tay vài frame rồi `scripts/eval_pose17.py --refresh-gold`.

## Chưa làm
- Đẩy nháp mới lên CVAT: chờ mạng tới cvat.edufun.ai (DNS/TCP đang lỗi) và bạn duyệt.
- Xem nhãn: `python scripts/render_skeleton_overlay.py --job 1623 --profile edufun --frames 91`
