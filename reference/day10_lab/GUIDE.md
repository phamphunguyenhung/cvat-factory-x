# Hướng dẫn làm lab (từng pha)

Bài **nhóm 5–6 người**, khoảng **4 giờ**. Mốc tự kiểm tra: [CHECKPOINTS.md](CHECKPOINTS.md). Chấm điểm: [RUBRIC.md](RUBRIC.md). Định dạng lô và các loại lỗi: [docs/DELIVERY.md](docs/DELIVERY.md).

```
Pha 0  Chuẩn bị                          (~15')
Pha 1  Bắt lỗi luật                      (~40')   tự viết validator
Pha 2  Đối chiếu + xem ảnh               (~55')   ứng viên, review_log
Pha 3  Quyết định + lineage              (~15')
Pha 4  Bàn giao v2                       (~40')
Pha 5  Truy vết ticket                   (~20')
Pha 6  Bảo mật + đề ẩn                   (~50')   10' bảo mật, 40' đề ẩn
```

## Cách gõ lệnh

Mọi lệnh chạy **từ thư mục gốc repo**.

| | macOS / Linux | Windows (PowerShell) |
|---|---|---|
| Gọi Python | `python3` | `py` (hoặc `python`) |
| Nối hai lệnh | `a && b` | chạy từng lệnh riêng |
| Lỗi font tiếng Việt | | thêm `-X utf8`: `py -X utf8 scripts/...` |

Trong tài liệu, `python` nghĩa là lệnh gọi Python 3.10+ trên máy bạn.

## Tóm tắt một trang

| Pha | Việc | Lệnh chính | Sản phẩm (trong `runs/r<seed>/`) |
|---|---|---|---|
| 0 | Tải dữ liệu, tạo lô, chia việc | `download_public.py`, `run_lab.py` | |
| 1 | Validator bắt lỗi luật | `validator/validate.py` (nhóm viết), `score_findings.py` | `errors.json` |
| 2 | Đối chiếu det ↔ panoptic, xem ảnh | validator `--candidates` | `candidates.json`, `review_log.csv`, `errors.json` |
| 3 | Quyết định, lineage | `lineage.py` | `runs/lineage.jsonl` |
| 4 | Lô v2 | `pack_release.py`, `check_handover.py` | `handover/<release_id>_v2/` + `.zip` + `.sha256` |
| 5 | Truy vết ticket | `trace_query` tự viết hoặc bảng tính | `trace.json` |
| 6 | Bảo mật; đề ẩn | | `security.md`; bài nộp đề ẩn |

---

## Pha 0 — Chuẩn bị (15 phút)

1. Tải dữ liệu (một lần, khoảng 1 GB): `python scripts/download_public.py`
2. Mỗi nhóm chọn **một seed luyện tập** (ví dụ số nhóm × 1000 + 7), rồi tạo lô:
   ```bash
   python scripts/run_lab.py --seeds 3007
   ```
   Lệnh tạo `runs/r3007/` và ghi training run `R-021` vào lineage (chỉ lần chạy đầu). Muốn thêm seed để thử validator thì dùng `python scripts/inject_release.py --seed <X> --out runs/r<X>`, **không** chạy lại `run_lab.py` (tránh gắn thêm R-021 vào lineage).
3. Đọc [docs/DELIVERY.md](docs/DELIVERY.md) và mở thử `runs/r3007/delivery/`.
4. **Chia việc:** 1–2 người viết validator luật, 2 người viết phần đối chiếu và chuẩn bị review, 1–2 người làm bàn giao và truy vết. Ai cũng phải xem ảnh ở Pha 2.
5. Commit: `git add -A` rồi `git commit -m "P0: chia viec"`. Thư mục `runs/` và `data_source/` không commit (đã có `.gitignore`); xem [SUBMISSION.md](SUBMISSION.md) để biết file nào chép vào `submission/`.

---

## Pha 1 — Bắt lỗi luật (40 phút)

Viết `validator/validate.py`, chỉ dùng thư viện chuẩn, theo giao diện cố định:

```
python validator/validate.py <thư_mục_delivery> --out <errors.json> [--candidates <candidates.json>]
```

Validator **chỉ đọc** thư mục delivery được truyền vào. Không đọc `_private/`, `data_source/`, annotation COCO, và không dùng mạng. Được `import` hàm đọc PNG từ `scripts/pngutil.py` (thư viện chuẩn).

Chín loại lỗi luật (định nghĩa ở [docs/DELIVERY.md](docs/DELIVERY.md) mục 2):
- **File:** `missing_image`, `wrong_hash` (tự tính sha256 rồi so; đừng chỉ kiểm định dạng 64 ký tự).
- **Nhãn detection:** `invalid_class`, `bbox_out_of_bounds` (box có thể chỉ tràn mép một chút), `duplicate_box` (box lặp lại cùng lớp, **cùng kích thước**, chỉ lệch 1–2 pixel; COCO có cặp box tự nhiên chồng nhau tới IoU khoảng 0,66, nên đừng đặt ngưỡng thấp).
- **Metadata:** `scene_leak` (target là `scene_id`), `guideline_mismatch` (so với guideline của **job** trong `jobs.csv`).
- **Panoptic:** `invalid_category_id`, `thing_overlap` (đếm pixel của từng segment trong mask rồi so với `area`; `scripts/pngutil.py` có hàm đọc PNG).

Thử trên seed của nhóm và vài seed khác (tạo bằng `inject_release.py`, xem Pha 0):
```bash
python validator/validate.py runs/r3007/delivery --out runs/r3007/errors.json
python scripts/score_findings.py --run runs/r3007
```
Lỗi luật nên đạt F1 ≈ 1.0 trên mọi seed (`score_findings.py` chỉ in F1 làm tròn; đừng dùng nó để dò từng ảnh, xem RULES). Thử thêm lô sạch: `python scripts/inject_release.py --seed 5 --no-faults --out runs/clean` (50 ảnh **khác** bộ luyện tập); luật phải không báo gì.

Được dùng AI (xem [RULES.md](RULES.md)), nhưng ghi prompt vào `submission/prompts.md`. Mỗi người phải giải thích được một đoạn validator khi vấn đáp.

---

## Pha 2 — Đối chiếu detection ↔ panoptic, xem ảnh (55 phút)

Lỗi gán nhãn **không có nhãn gốc để so**. Nguồn đối chiếu duy nhất là nguồn nhãn còn lại của cùng ảnh.

1. **Ghép.** Đổi box YOLO sang pixel; ghép mỗi box với segment thing có bbox gần nhất (IoU). Ghép lớp qua **tên** (`classes.txt` ↔ `categories`).
   IoU giữa hai bbox **không đủ** để bắt `bbox_drift` và `instance_merge`. Tín hiệu tốt hơn là **pixel của mask**: với một cặp đã ghép đúng, gần như toàn bộ pixel của segment nằm trong box detection. Box bị lệch thì một phần pixel rơi ra ngoài; một segment có pixel nằm trong **hai** box cùng lớp là dấu hiệu bị gộp.
2. **Sinh ứng viên** vào `candidates.json`. Gợi ý các dấu hiệu:
   - Box và segment trùng vị trí nhưng khác lớp: `class_swap` hoặc `category_swap`.
   - Box lệch so với segment cùng lớp: `bbox_drift`.
   - Segment thing lớn mà không có box nào: `missing_box`.
   - Một segment phủ hai box cùng lớp: `instance_merge`.
3. **Đo nhiễu tự nhiên.** Chạy validator trên lô sạch (`runs/clean`, 50 ảnh khác bộ luyện tập). Ứng viên ở đây là lệch **tự nhiên** của COCO. Số này phụ thuộc cách ghép: ghép thô (chỉ IoU bbox, ngưỡng lỏng) ra nhiều ứng viên; ghép tốt có thể ra rất ít. Chọn ngưỡng sao cho hàng đợi đủ nhỏ để xem hết trong thời gian cho phép mà vẫn không bỏ sót lỗi, và ghi lý do vào `REPORT.md`.
4. **Xem ảnh.** Chia hàng đợi cho cả nhóm. Mở từng ảnh, nhìn vật, rồi ghi `review_log.csv` (`image,type,verdict,note`). Ghi chú phải nói **thấy gì**, ví dụ "xe tải mui kín, detection ghi car". Khi hai nguồn bất đồng về lớp, chính ảnh cho biết bên nào sai. Có ảnh nhỏ (khoảng 240×320): phóng to hoặc cắt vùng quanh vật để xem; nên viết một công cụ nhỏ vẽ box và viền segment lên ảnh (công cụ này được dùng Pillow, validator thì không). Ứng viên nào hoá ra là lệch tự nhiên thì ghi `rejected` kèm lý do.
5. **Chốt.** `errors.json` = lỗi luật + lỗi gán nhãn đã `confirmed`. Có thể dùng script gộp (validator ghi lỗi luật ra một file riêng, ví dụ `rules.json`):
   ```bash
   python scripts/merge_review.py --rules runs/r3007/rules.json --review runs/r3007/review_log.csv --out runs/r3007/errors.json
   python scripts/score_findings.py --run runs/r3007
   ```

**Tự hỏi:** vì sao báo hết ứng viên lại làm F1 **giảm**? (Ghi câu trả lời vào `submission/REPORT.md`.)

---

## Pha 3 — Quyết định và lineage (15 phút)

1. Tỉ lệ lỗi v1 = số mục trong `errors.json` / 250 ảnh. Mỗi ảnh mang nhiều nhất một lỗi; mỗi `scene_leak` tính là **một ảnh** (ảnh bị đặt sai split so với phần còn lại của cảnh). Theo ngưỡng Day 7, **trên 2% thì không Release**.
2. Ghi quyết định, rồi hỏi run nào bị ảnh hưởng:
   ```bash
   python scripts/lineage.py record --event decision --release-id <release_id v1> --decision HOLD
   python scripts/lineage.py impact --release-id <release_id v1>
   ```
   Dòng mong đợi: `R-021: affected=True action=review_or_retrain`. Đổi dataset không tự sửa model đã học, nên run đã train phải được xem lại hoặc train lại.

---

## Pha 4 — Bàn giao lô v2, không ghi đè v1 (40 phút)

Tạo `runs/r<seed>/handover/<release_id>_v2/`. **Không sửa gì trong `delivery/`.** Lô v2 là sản phẩm **detection** (ảnh + nhãn YOLO) giao cho team AI.

| Phần | Nội dung |
|---|---|
| `images/`, `labels/` | ảnh và nhãn **giữ nguyên** của v1, trừ ảnh phải gán lại |
| `rework_queue.csv` | `image,type,annotator,reviewer,action`: ảnh tách ra để gán lại, giao đúng người theo `jobs.csv` |
| `manifest.csv` | `file,sha256` cho mọi file trong `images/` và `labels/` (đường dẫn dạng `images/x.jpg`) |
| `label_summary.csv` | `class,count` theo tên lớp trong `classes.txt`, dòng cuối `TOTAL` |
| `truyvet.csv` | `image,scene_id,split,job_id,annotator,reviewer,guideline`: guideline lấy theo **job**, mỗi cảnh chỉ một split |
| `README.txt` | theo mẫu [templates/README.txt](templates/README.txt): số ảnh, số nhãn, QC v1 có mẫu số, kết luận v1 và v2 |

**Sửa trong v2, đưa vào rework, hay giữ nguyên?**
- **Sửa ngay trong v2** (không cần gán lại): `wrong_hash`, `guideline_mismatch` (guideline theo job), `scene_leak` (đưa cả cảnh về một split). Với `wrong_hash`, **mở ảnh kiểm trước**: ảnh còn nguyên (mở được, đúng nội dung) thì chỉ manifest sai, ghi hash mới; ảnh hỏng thì đưa vào rework.
- **Lỗi ở nhãn detection hoặc thiếu file** (`missing_image`, `invalid_class`, `bbox_out_of_bounds`, `duplicate_box`, `class_swap`, `bbox_drift`, `missing_box`): đưa ảnh vào `rework_queue.csv` và bỏ khỏi v2.
- **Lỗi chỉ ở panoptic** (`invalid_category_id`, `thing_overlap`, `category_swap`, `instance_merge`): nhãn detection của ảnh vẫn đúng nên ảnh **ở lại v2**. Ghi các ảnh này vào phần Ghi chú của `README.txt` để báo team panoptic.

Sau khi chốt:
```bash
python scripts/pack_release.py runs/r3007/handover/<release_id>_v2
python scripts/lineage.py record --event release --release-id <release_id>_v2 --parent <release_id>
python scripts/check_handover.py --run runs/r3007
```
Sửa mọi FAIL. Đổi file nào trong v2 sau khi đóng gói thì phải chạy lại `pack_release.py`.

---

## Pha 5 — Truy vết ticket của team AI (20 phút)

Đọc `incoming/ticket.md` và `guideline/changelog.md`, rồi ghi `runs/r<seed>/trace.json` (mẫu [templates/trace.json](templates/trace.json)):
- **`tier_a`:** ảnh trong các cảnh team AI nêu, thuộc job theo **guideline cũ**, có ít nhất một **box detection** (trong `labels/` của lô) thuộc **lớp bị đổi quy tắc**.
- **`tier_b`:** các ảnh khác cùng điều kiện (job guideline cũ, có lớp đó) mà chưa ai báo.
- **`notify`:** mã người gán và người kiểm của các job chứa ảnh tầng a và b.

Tiêu chí dùng phiên bản guideline **ghi ở job** (`jobs.csv`), không dùng guideline ghi ở từng ảnh, vì đó có thể chính là lỗi `guideline_mismatch`. `score_findings.py` chấm phần này riêng.

---

## Pha 6 — Bảo mật (10 phút) và đề ẩn (40 phút)

1. **Bảo mật:** điền `runs/r<seed>/security.md` theo mẫu [templates/security.md](templates/security.md): ca USB, năm việc được/không, ca khuôn mặt (kết luận **Hold + báo người phụ trách**). COCO có ảnh người thật: nêu ví dụ từ lô của nhóm.
2. **Đề ẩn (40 phút):** lúc giảng viên báo, vào tab **Releases** của repo đề, tải `de_an.zip` (hoặc nhận file từ LMS/nhóm chat), giải nén vào thư mục gốc repo nhóm, sẽ được thư mục `delivery/` (tên ảnh dạng `img_0001`). Rồi chạy:
   ```bash
   python validator/validate.py delivery --out submission/de_an/errors.json --candidates submission/de_an/candidates.json
   ```
   Xem ảnh ứng viên, ghi `submission/de_an/review_log.csv`, chốt `errors.json` (có thể dùng `merge_review.py`), làm `trace.json`, rồi commit và push (xem [SUBMISSION.md](SUBMISSION.md)). **Không commit thư mục `delivery/`** của đề ẩn (đã có trong `.gitignore`). Với 40 phút, validator và công cụ xem ảnh phải sẵn sàng **trước** khi nhận đề.

## Lỗi hay gặp

| Triệu chứng | Cách xử lý |
|---|---|
| Nhiều FP `wrong_hash` | Tính sha256 trên **bytes** của file, đọc ở chế độ `rb`. |
| `thing_overlap` báo cả lô | So với `area` trong `segments_info`; bỏ qua id 0 (nền); đếm theo id = R + G·256 + B·256². |
| Ứng viên quá nhiều | Đo trên lô sạch trước; dùng pixel mask thay vì chỉ IoU bbox; bỏ segment `iscrowd`. |
| `duplicate_box` báo nhầm ở mép ảnh | Box lặp chỉ lệch 1–2 pixel và cùng kích thước; cặp tự nhiên chồng nhau thường khác kích thước. Kiểm `duplicate_box` theo kích thước + độ lệch tâm, không chỉ IoU. |
| H5 FAIL với ảnh lỗi panoptic | Ảnh chỉ lỗi panoptic ở lại v2, không đưa vào `rework_queue`. |
| H16 FAIL | `tier_a` và `notify` không được rỗng; mã người phải có trong `jobs.csv`. |
| `class_swap` hay `category_swap`? | Mở ảnh: nguồn nào ghi đúng vật thì nguồn kia sai. |
| H1 FAIL (v1 bị sửa) | Đã sửa `delivery/`. Tạo lại lô bằng `run_lab.py` cùng seed, chỉ làm việc trong `handover/`. |
| H8 FAIL (sha256) | Sửa file sau khi lập `manifest.csv`: lập lại manifest, rồi `pack_release.py`. |
| H13 FAIL | Dòng `QC v1:` phải là `<số lỗi>/<250>`, đúng số trong `errors.json`. |
