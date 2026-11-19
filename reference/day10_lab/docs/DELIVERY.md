# Định dạng lô, các loại lỗi và cách chấm

## 1. Một lô gồm gì

```bash
python scripts/run_lab.py --seeds 11        # tạo runs/r11; seed đầu được gắn training run R-021
```

```text
runs/r11/delivery/
  images/*.jpg              # ảnh (có thể thiếu file)
  labels/*.txt              # nhãn YOLO: class_id x_center y_center width height (chuẩn hoá 0-1)
  masks/*.png               # mask panoptic: id segment = R + G*256 + B*256^2
  panoptic.json             # categories, images, annotations (segments_info: id, category_id, area, bbox, iscrowd)
  classes.txt               # 80 lớp detection, mỗi dòng "class_id tên"
  manifest.json             # mỗi ảnh: đường dẫn image/label/mask và sha256 của từng file
  metadata.json             # mỗi ảnh: scene_id, split, job_id, guideline_version
  jobs.csv                  # job_id, annotator, reviewer, guideline (phiên bản ghi ở đầu job)
  guideline/changelog.md    # v1.2 → v1.3 đổi quy tắc của một lớp
  incoming/ticket.md        # team AI báo lỗi sau khi train
  release.json              # release_id của lô v1
runs/r11/_private/          # ground truth: validator KHÔNG đọc
runs/lineage.jsonl          # nhật ký lineage của mọi lô
```

- **`classes.txt` và `panoptic.json` dùng cùng tên lớp.** Detection đánh số lớp 0–79; panoptic dùng `category_id` gốc của COCO. Ghép hai bên qua **tên lớp**.
- **Cảnh, job, người gán và người kiểm là giả lập** (COCO không có). Một cảnh là 5 ảnh liên tiếp, một job là 25 ảnh. Các job đầu theo guideline v1.2, các job sau theo v1.3.
- **`release_id`** = `lab250_` + 12 ký tự đầu của sha256 ghép `manifest.json` và `metadata.json`. Sửa v1 thì id đổi, và `check_handover.py` sẽ phát hiện.

## 2. Các loại lỗi

Đơn vị chấm là cặp `(type, target)`. `scene_leak` có target là `scene_id`; mọi loại khác có target là tên ảnh, không có đuôi. **Số lỗi mỗi loại thay đổi theo seed (có loại có thể vắng)**, và mỗi ảnh mang nhiều nhất một lỗi.

Đề ẩn dùng ảnh khác, đã qua xử lý (lật, nén lại), và **mức lỗi có thể khác lô luyện tập**: số lượng, độ lệch của box, tỉ lệ đổi sang lớp dễ nhầm. Validator dựa vào đúng các tham số trong `inject_release.py` thay vì đối chiếu hai nguồn sẽ không đứng vững.

| Nhóm | Loại | Nghĩa |
|---|---|---|
| Luật (detection) | `missing_image` | manifest có ảnh nhưng thiếu file |
| | `wrong_hash` | sha256 ghi trong manifest không phải của file đó |
| | `invalid_class` | class_id ngoài 0–79 |
| | `bbox_out_of_bounds` | box vượt ra ngoài ảnh |
| | `duplicate_box` | một box bị lặp: cùng lớp, cùng kích thước, lệch 1–2 pixel |
| | `scene_leak` | một cảnh có ảnh ở cả train và val |
| | `guideline_mismatch` | guideline của ảnh khác guideline ghi ở job của nó |
| Luật (panoptic) | `invalid_category_id` | category_id không có trong `categories` |
| | `thing_overlap` | `area` khai nhiều hơn số pixel thật của segment (bị vật khác đè) |
| Gán nhãn | `class_swap` | box detection bị gán nhầm sang lớp dễ nhầm |
| | `bbox_drift` | box detection bị lệch khỏi vật |
| | `missing_box` | detection thiếu box cho một vật đáng kể |
| | `category_swap` | segment panoptic bị gán nhầm sang lớp dễ nhầm |
| | `instance_merge` | hai vật cùng lớp bị gộp thành một segment |

Lỗi gán nhãn không chứng minh được bằng một file:
- **Ghép** box detection với segment panoptic của cùng ảnh. Lệch lớp, lệch vị trí, hoặc số vật không khớp là **ứng viên**.
- **Lô sạch cũng có thể có ứng viên**, vì COCO vốn lệch giữa hai nguồn (vật bị che, crowd). Số lượng tuỳ cách ghép. Muốn đo, tạo một lô sạch: `python scripts/inject_release.py --seed 5 --no-faults --out runs/clean`. Lô sạch dùng **50 ảnh khác** bộ luyện tập (pool `calib`), nên không so được với lô luyện tập.
- **Mở ảnh** để quyết định. Khi detection và panoptic bất đồng về lớp, ảnh cho biết bên nào sai (`class_swap` hay `category_swap`).

## 3. File nhóm nộp

`errors.json` (mẫu: `examples/errors.sample.json`):

```json
{"errors": [{"type": "class_swap", "target": "000000000139"}, {"type": "scene_leak", "target": "scene_557"}]}
```

`review_log.csv` (mẫu: `templates/review_log.csv`): `image,type,verdict,note`, với `verdict` là `confirmed` hoặc `rejected`. Mọi lỗi gán nhãn trong `errors.json` phải có một dòng `confirmed`.

`trace.json` (mẫu: `templates/trace.json`): `tier_a`, `tier_b` (tên ảnh), `notify` (mã người). "Có lớp bị đổi quy tắc" nghĩa là có ít nhất một **box detection** lớp đó trong `labels/` của lô.

Lô v2 bàn giao là sản phẩm **detection**: xem [GUIDE.md](../GUIDE.md) Pha 4 để biết ảnh nào vào rework, ảnh nào ở lại.

## 4. Chấm

```bash
python scripts/score_findings.py --run runs/r11
```

Lệnh in và ghi `report.md`: F1 tổng, F1 lỗi luật, F1 lỗi gán nhãn, F1 truy vết, **làm tròn một chữ số**. Đây là công cụ tự đo, không phải nơi dò đáp án (xem RULES). Giảng viên chấm đề ẩn bằng chế độ đầy đủ.

| Số | Cách tính |
|---|---|
| TP / FP / FN | cặp đúng / cặp báo thừa / cặp bỏ sót |
| precision | TP / (TP + FP) |
| recall | TP / (TP + FN) |
| F1 | trung bình điều hoà của precision và recall; đây là **rank F1** |

- Report tách hai nhóm `rule_detectable` và `cross_check_review`, và chấm riêng truy vết (`tier_a`, `tier_b`, `notify`).
