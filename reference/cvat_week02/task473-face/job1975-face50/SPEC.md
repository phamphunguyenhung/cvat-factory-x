# Job 1975 (task 473) — week2-human-face/G06, frame 90–119 (30 ảnh 640×480)

URL: https://cvat.edufun.ai/tasks/473/jobs/1975
Gán cho: mình. Trạng thái lúc đọc (16/11/2026): `new`, stage `annotation`.

## Phạm vi
- Mỗi ảnh đủ 7 skeleton: `longmaytrai`(0–4), `longmayphai`(5–9), `songmui`(10–13), `mattrai`(14–21), `matphai`(22–29), `moingoai`(30–41), `moitrong`(42–49). Chế độ Shape.
- Guideline: `../../../weeks/guidelines/w2-face50-pose17.md`. Chưa có guideline chính thức của BTC.

## Hiện trạng
- Có sẵn 70 skeleton `source=file` ở frame 91, 92, 97, 98, 103, 104, 109, 110, 115, 116 (7 × 10). Là nháp thô, điểm dồn thành cụm quanh mặt. Chưa biết được phép xóa hay sửa không (guideline mục 8 câu 3).
- 20 frame còn lại chưa có nhãn.

## Chưa làm / chờ quyết định
- Nháp AI cho 20 frame trống: chờ xác nhận có được dùng AI không (guideline mục 8 câu 4).
- Xem nhãn: `python scripts/render_skeleton_overlay.py --job 1975 --profile edufun --frames 91 --scale 2`
